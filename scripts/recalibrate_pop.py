"""#1 (retrain.md): post-hoc recalibration of the diffusion model's PoP.

The reliability test showed the PoP systematically under-forecasts ("20%" verifies
at ~55%). This fits a monotonic isotonic mapping g: raw_PoP -> calibrated_PoP per
threshold, on a CALIBRATION split of held-out cases, and validates on a disjoint
split (Brier + reliability before/after). Monotonic, so it doesn't change the
POD/FAR operating curve — it makes the probability numbers honest (and lowers Brier).

sklearn absent -> isotonic via PAVA (pool-adjacent-violators), numpy-only.
Saves the mapping to pop_calibration.npz (prob_grid + calibrated per threshold);
apply at inference with np.interp(raw_pop, prob_grid, calibrated[thr]).

Usage (from scripts/):  uv run python recalibrate_pop.py
"""
import io
import os
import sys
from contextlib import nullcontext, redirect_stdout

import numpy as np
import torch
from fire import Fire
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from ldcast.features.rust_data import RustRadarDataModule
from ldcast.models.diffusion import dpm_solver
from ldcast.visualization.plots import reverse_transform_R
from train_genforecast import setup_model

THRESHOLDS = (0.1, 1.0, 5.0)


def isotonic_fit(y, w):
    """Weighted isotonic regression (PAVA) on points already sorted by x.
    y = per-bin observed frequency, w = per-bin count. Returns monotonic fit."""
    vals, wts, cnts = [], [], []
    for yi, wi in zip(y, w):
        cv, cw, cc = float(yi), float(wi), 1
        while vals and vals[-1] > cv:
            pv, pw, pc = vals.pop(), wts.pop(), cnts.pop()
            cv = (cv * cw + pv * pw) / (cw + pw) if (cw + pw) else cv
            cw += pw; cc += pc
        vals.append(cv); wts.append(cw); cnts.append(cc)
    out = np.empty(len(y))
    k = 0
    for v, c in zip(vals, cnts):
        out[k:k + c] = v; k += c
    return out


def reliability(prob, outcome, nbins=11):
    edges = np.linspace(0, 1, nbins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, nbins - 1)
    mp, of = np.full(nbins, np.nan), np.full(nbins, np.nan)
    for b in range(nbins):
        sel = idx == b
        if sel.any():
            mp[b] = prob[sel].mean(); of[b] = outcome[sel].mean()
    return mp, of


def run(config="../config/train_rust.yaml",
        ckpt="../models/genforecast_rust/last.ckpt",
        per_bin_cases=30, scan_per_bin=250, ensemble_size=32, member_batch=16,
        num_diffusion_iters=20, px_per_case=8000, eval_seed=1234, out_dir=None):
    cfg = OmegaConf.load(config)
    if cfg.get("radar_root"):
        os.environ["DGMR_RADAR_ROOT"] = cfg.radar_root
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(ckpt)),
                                      "val_large_eval")
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)

    print("Loading data + model (EMA)...")
    dm = RustRadarDataModule(
        index_path=cfg.index_path, mode="diffusion",
        past_steps=cfg.past_steps, future_steps=cfg.future_steps,
        height=cfg.height, width=cfg.width,
        batch_size=cfg.genforecast_batch_size, num_workers=0,
        valid_frac=0.1, test_frac=0.0, split_mode="random", seed=42,
        use_weighted_sampler=cfg.use_weighted_sampler)
    dm.setup("fit")
    ldm, _ = setup_model(
        num_timesteps=cfg.future_steps // 4,
        autoenc_weights_fn=os.path.join(cfg.autoenc_dir, "state_dict.pt"),
        use_obs=True, use_nwp=False, model_dir=cfg.genforecast_dir,
        lr=cfg.genforecast_lr, precision=cfg.precision,
        optimizer_8bit=cfg.optimizer_8bit, max_epochs=1, limit_train_batches=1,
        limit_val_batches=1, scale_factor=1.0, gradient_clip_val=1.0,
        sample_every_n_epochs=1, max_hours=None, early_stopping_patience=0,
        accumulate_grad_batches=1, save_top_k=0)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    ldm.load_state_dict(sd, strict=True)
    if getattr(ldm, "model_ema", None) is not None:
        ldm.model_ema.copy_to(ldm.model)
    ldm.use_ema = False
    ldm = ldm.to(dev).eval()
    sampler = dpm_solver.DPMSolverSampler(ldm)
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if dev.type == "cuda" else nullcontext())
    with torch.no_grad(), amp:
        probe = torch.zeros(1, 1, cfg.future_steps, cfg.height, cfg.width, device=dev)
        gen_shape = tuple(ldm.autoencoder.encode(probe)[0].shape[1:]); del probe
    t_past = torch.arange(-int(dm.past_steps) + 1, 1, dtype=torch.float32,
                          device=dev).unsqueeze(0)

    # select per-bin wettest, split into calibration (even) / validation (odd)
    val_w, val_ds = dm.val_w, dm.valid_ds
    cal = {t: [[], []] for t in THRESHOLDS}   # [prob, outcome]
    vld = {t: [[], []] for t in THRESHOLDS}
    cases = []
    for w in sorted(np.unique(val_w).tolist()):
        rows = np.flatnonzero(val_w == w)[:scan_per_bin]
        scored = []
        for ridx in rows:
            past_t, future_t = val_ds[int(ridx)]
            truth = reverse_transform_R(future_t[0].float().numpy())
            scored.append((float((truth >= 1.0).mean()), past_t, truth))
        scored.sort(key=lambda x: -x[0])
        for j, (_, past_t, truth) in enumerate(scored[:per_bin_cases]):
            cases.append((past_t.unsqueeze(0), truth, "cal" if j % 2 == 0 else "vld"))
    print(f"  {len(cases)} cases ({sum(c[2]=='cal' for c in cases)} cal / "
          f"{sum(c[2]=='vld' for c in cases)} vld)\n")

    print("Sampling + collecting (prob, outcome) pairs...")
    for ci, (past, truth, grp) in enumerate(cases):
        members = []
        for start in range(0, ensemble_size, member_batch):
            cb = min(member_batch, ensemble_size - start)
            torch.manual_seed(eval_seed + 1000 * ci + start)
            if dev.type == "cuda":
                torch.cuda.manual_seed_all(eval_seed + 1000 * ci + start)
            past_rep = past.repeat(cb, 1, 1, 1, 1).to(dev)
            ldm._cond_cache = None
            with torch.no_grad(), amp, redirect_stdout(io.StringIO()):
                s, _ = sampler.sample(num_diffusion_iters, cb, gen_shape,
                                      [[past_rep, t_past.repeat(cb, 1)]],
                                      progbar=False, verbose=False)
                y = ldm.autoencoder.decode(s / ldm.scale_factor)
            for j in range(cb):
                members.append(reverse_transform_R(y[j, 0].float().cpu().numpy()))
        stack = np.stack(members)                       # (M,T,H,W)
        npix = truth.size
        pick = rng.integers(0, npix, min(px_per_case, npix))
        bucket = cal if grp == "cal" else vld
        for thr in THRESHOLDS:
            pop = (stack >= thr).mean(0).reshape(-1)[pick]
            out = (truth >= thr).reshape(-1)[pick].astype(np.float32)
            bucket[thr][0].append(pop); bucket[thr][1].append(out)
        if (ci + 1) % 50 == 0:
            print(f"  {ci+1}/{len(cases)}")

    # fit isotonic per threshold on calibration; validate on the held-out split
    grid = np.linspace(0, 1, 101)
    mapping = {}
    print("\n=== recalibration (fit on calib, metrics on held-out val) ===")
    print(f"{'thr':>5} | {'Brier raw':>10} {'Brier cal':>10} {'BSS':>7} "
          f"| max|reliab gap| raw->cal")
    for thr in THRESHOLDS:
        cp = np.concatenate(cal[thr][0]); co = np.concatenate(cal[thr][1])
        vp = np.concatenate(vld[thr][0]); vo = np.concatenate(vld[thr][1])
        # binned isotonic fit on calibration
        nb = 100
        edges = np.linspace(0, 1, nb + 1)
        bi = np.clip(np.digitize(cp, edges) - 1, 0, nb - 1)
        cnt = np.bincount(bi, minlength=nb).astype(float)
        obs = np.bincount(bi, weights=co, minlength=nb)
        nz = cnt > 0
        centers = (edges[:-1] + edges[1:]) / 2
        fit = isotonic_fit(obs[nz] / cnt[nz], cnt[nz])
        g = np.interp(grid, centers[nz], fit, left=fit[0], right=fit[-1])
        mapping[str(thr)] = g
        # validate
        vp_cal = np.interp(vp, grid, g)
        b_raw = float(((vp - vo) ** 2).mean())
        b_cal = float(((vp_cal - vo) ** 2).mean())
        bss = 1 - b_cal / b_raw if b_raw else float("nan")
        mp0, of0 = reliability(vp, vo)
        mp1, of1 = reliability(vp_cal, vo)
        gap0 = np.nanmax(np.abs(mp0 - of0)); gap1 = np.nanmax(np.abs(mp1 - of1))
        print(f"{thr:>5} | {b_raw:>10.4f} {b_cal:>10.4f} {bss:>+7.3f} "
              f"| {gap0:.3f} -> {gap1:.3f}")
        # plot before/after reliability
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(mp0, of0, "o-", label="raw PoP", color="tab:red")
        ax.plot(mp1, of1, "o-", label="recalibrated", color="tab:green")
        ax.set_xlabel("forecast P(rain)"); ax.set_ylabel("observed freq")
        ax.set_title(f"Reliability @{thr}mm (held-out)"); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"recalib_{thr}mm.png"), dpi=90)
        plt.close(fig)

    np.savez(os.path.join(out_dir, "pop_calibration.npz"),
             prob_grid=grid, **{f"thr_{t}": mapping[str(t)] for t in THRESHOLDS})
    print(f"\nSaved mapping -> {out_dir}/pop_calibration.npz  (+ recalib_*mm.png)")
    print("Apply at inference: cal = np.interp(raw_pop, prob_grid, thr_<thr>)")


if __name__ == "__main__":
    Fire(run)
