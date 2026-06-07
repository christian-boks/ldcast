"""Architecture A/B on the Aalborg held-out TEST set (retrain.md Tier-1 decision).

Scores, on the SAME clean Aalborg temporal-test cases, four forecasters over the
same 8-lead (+40 min) window:
  - prob   : the deterministic probabilistic nowcaster (BCE head, calibrated by
             construction) — models/prob_nowcaster_aalborg
  - diff   : the latent-diffusion ensemble PoP (raw fraction-of-members, and the
             all-Denmark-fit recalibration applied for a fair "calibrated" number)
  - eul    : Eulerian persistence (hold the last frame)
  - lag    : Lagrangian persistence (advect the last frame by phase-corr motion)

Decision metrics (NOT pooled CSI):
  1. Brier + BSS-vs-Eulerian, per threshold (obj 1/6: honest probability).
  2. Brier + BSS on GENESIS pixels (obj 4: TRUE growth) — pixels where advection
     both predicts dry AND cannot physically reach any t0 echo (dist > speed*lead).
     Rain appearing there is genuine formation, not a band moving in. This replaces
     the old dry@t0 "initiation" set, which conflated advection-in with genesis and
     made Eulerian's 0 an artifact (it just can't move).
  3. genesis / decay POD at the 0.5 operating point (interpretability).
Bootstrap 95% CIs over cases on the headline BSS (honest noise, not point reads).

CAVEAT: the diffusion model trained on the all-Denmark RANDOM split, so it has
seen these test days (leakage = an unfair advantage to diffusion). The
prob-nowcaster trained only on Aalborg temporal-train (clean). So if prob matches
or beats diff here, the conclusion is robust.

Usage (from scripts/):
    uv run python eval_aalborg_ab.py --n_cases=400 --ensemble_size=32
"""
import io
import os
import sys
import json
from collections import defaultdict
from contextlib import nullcontext, redirect_stdout
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from omegaconf import OmegaConf

from ldcast.features.rust_data import RustRadarDataModule
from ldcast.models.diffusion import dpm_solver
from ldcast.models.autoenc import autoenc, encoder
from ldcast.models.nowcast.prob_nowcast import ProbNowcastNet, ProbNowcaster
from ldcast.visualization.plots import reverse_transform_R

from train_genforecast import setup_model
from eval_persistence_baseline import estimate_motion
from scipy.ndimage import shift as nd_shift, distance_transform_edt

THRESHOLDS = (0.1, 1.0, 5.0)
METHODS = ("prob", "diff_raw", "diff_cal", "lag", "eul")
SAFETY_FACTOR = 1.5      # advection reach = SAFETY * |motion| * lead (generous)
GENESIS_MARGIN_PX = 8    # + cell-scale margin; pixels beyond reach = unreachable


def run(
    config="../config/train_rust.yaml",
    prob_ckpt="../models/prob_nowcaster_aalborg/last.ckpt",
    diff_ckpt="../models/genforecast_rust/last.ckpt",
    calib_npz="../models/genforecast_rust/val_large_eval/pop_calibration.npz",
    n_cases=400,
    scan_rows=5000,
    ensemble_size=32,
    eval_batch=24,
    num_diffusion_iters=20,
    region_center=(685, 852),
    region_radius=64,
    test_frac=0.1,
    valid_frac=0.1,
    eval_seed=1234,
    out_dir=None,
):
    cfg = OmegaConf.load(config)
    if cfg.get("radar_root"):
        os.environ["DGMR_RADAR_ROOT"] = cfg.radar_root
    out_dir = Path(out_dir or os.path.join(
        os.path.dirname(os.path.abspath(prob_ckpt)), "ab_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tp, Tf = int(cfg.past_steps), int(cfg.future_steps)
    print(f"prob ckpt : {prob_ckpt}")
    print(f"diff ckpt : {diff_ckpt}")
    print(f"region    : Aalborg center={region_center} r={region_radius}px (temporal TEST split)")
    print(f"design    : {n_cases} cases, {ensemble_size} members, {Tf} leads, EMA diffusion\n")

    # ---- data: Aalborg temporal TEST split (same recipe the prob model trained on) ----
    print("Loading Aalborg test split...")
    dm = RustRadarDataModule(
        index_path=cfg.index_path, mode="diffusion",
        past_steps=Tp, future_steps=Tf, height=cfg.height, width=cfg.width,
        batch_size=cfg.genforecast_batch_size, num_workers=0,
        valid_frac=valid_frac, test_frac=test_frac, split_mode="temporal",
        region_center=region_center, region_radius=region_radius,
        dedup_bin0=False, seed=42, use_weighted_sampler=False,
    )
    dm.setup()
    if dm.test_ds is None:
        sys.exit("test split is empty — check test_frac / region")
    test_ds, test_ts = dm.test_ds, dm.test_ts
    n_test = len(test_ds)
    print(f"  {n_test} test rows ({np.unique(test_ts // 86400).size} UTC days)")

    # ---- case selection: wettest n_cases over an evenly-spaced scan (spans days) ----
    print("Selecting wettest test cases...")
    stride = max(1, n_test // scan_rows)
    scan_idx = list(range(0, n_test, stride))
    scored = []
    for ridx in scan_idx:
        past_t, future_t = test_ds[ridx]
        truth = reverse_transform_R(future_t[0].float().numpy())     # (Tf,H,W)
        scored.append((float((truth >= 1.0).mean()), ridx, past_t, truth))
    scored.sort(key=lambda x: -x[0])
    cases = [{"past": p.unsqueeze(0).clone(), "truth": tr}
             for _, _, p, tr in scored[:n_cases]]
    C = len(cases)
    print(f"  {C} cases (wettest of {len(scan_idx)} scanned)\n")

    # ---- prob-nowcaster (frozen) ----
    print("Building prob-nowcaster...")
    ae_p = autoenc.AutoencoderKL(encoder.SimpleConvEncoder(), encoder.SimpleConvDecoder())
    net = ProbNowcastNet(ae_p, thresholds=THRESHOLDS, embed_dim=128,
                         analysis_depth=4, forecast_depth=4, output_patches=Tf // 4)
    pmodel = ProbNowcaster(net, thresholds=THRESHOLDS)
    pmodel.load_state_dict(
        torch.load(prob_ckpt, map_location="cpu", weights_only=False)["state_dict"],
        strict=True)
    pnet = pmodel.net.to(dev).eval()

    # ---- diffusion model (EMA, frozen) ----
    print("Building diffusion model (EMA)...")
    ldm, _ = setup_model(
        num_timesteps=Tf // 4,
        autoenc_weights_fn=os.path.join(cfg.autoenc_dir, "state_dict.pt"),
        use_obs=True, use_nwp=False, model_dir=cfg.genforecast_dir,
        lr=cfg.genforecast_lr, precision=cfg.precision,
        optimizer_8bit=cfg.optimizer_8bit, max_epochs=1, limit_train_batches=1,
        limit_val_batches=1, scale_factor=1.0, gradient_clip_val=1.0,
        sample_every_n_epochs=1, max_hours=None, early_stopping_patience=0,
        accumulate_grad_batches=1, save_top_k=0,
    )
    sd = torch.load(diff_ckpt, map_location="cpu", weights_only=False)["state_dict"]
    ldm.load_state_dict(sd, strict=True)
    if getattr(ldm, "model_ema", None) is not None:
        ldm.model_ema.copy_to(ldm.model)
    ldm.use_ema = False
    ldm = ldm.to(dev).eval()
    sampler = dpm_solver.DPMSolverSampler(ldm)
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if dev.type == "cuda" else nullcontext())
    with torch.no_grad(), amp:
        probe = torch.zeros(1, 1, Tf, cfg.height, cfg.width, device=dev)
        gen_shape = tuple(ldm.autoencoder.encode(probe)[0].shape[1:])
        del probe
    print(f"  latent gen_shape = {gen_shape}")

    # ---- calibration map (applied to diffusion PoP) ----
    cz = np.load(calib_npz)
    prob_grid = cz["prob_grid"]
    cal_map = {t: cz[f"thr_{t}"] for t in THRESHOLDS}

    t_relative = torch.arange(-Tp + 1, 1, dtype=torch.float32, device=dev).unsqueeze(0)

    @torch.no_grad()
    def prob_forecast(ci):
        """One forward -> P(rain>=thr) per pixel/lead, (n_thr, Tf, H, W)."""
        past = cases[ci]["past"].to(dev)
        with amp:
            logits = pnet([[past, t_relative]])
        return torch.sigmoid(logits)[0].float().cpu().numpy()

    # ---- accumulators ----
    brier = {me: {t: [0.0, 0] for t in THRESHOLDS} for me in METHODS}      # overall
    brier_gen = {me: {t: [0.0, 0] for t in THRESHOLDS} for me in METHODS}  # genesis pixels
    brier_dec = {me: {t: [0.0, 0] for t in THRESHOLDS} for me in METHODS}  # decay pixels
    gd_gen = {me: {t: [0, 0] for t in THRESHOLDS} for me in METHODS}       # caught,total @0.5
    gd_dec = {me: {t: [0, 0] for t in THRESHOLDS} for me in METHODS}
    # per-case (sse, npx) for bootstrap CIs
    pc = {me: {t: [] for t in THRESHOLDS} for me in METHODS}
    pc_gen = {me: {t: [] for t in THRESHOLDS} for me in METHODS}

    def add_brier(store, pcstore, me, thr, p_field, of, mask=None):
        sse = (p_field - of) ** 2
        if mask is not None:
            sse = sse[mask]
            n = int(mask.sum())
        else:
            n = of.size
        s = float(sse.sum())
        store[me][thr][0] += s
        store[me][thr][1] += n
        pcstore[me][thr].append((s, n))

    def finalize_case(ci, preds):
        truth = cases[ci]["truth"]                                  # (Tf,H,W) mm/h
        stack = np.stack(preds, axis=0)                             # (M,Tf,H,W)
        past_mmhr = reverse_transform_R(cases[ci]["past"][0, 0].float().numpy())
        t0 = past_mmhr[-1]
        dy, dx = estimate_motion(past_mmhr)
        eul = np.repeat(t0[None], truth.shape[0], axis=0)
        lag = np.stack([nd_shift(t0, (dy * (k + 1), dx * (k + 1)),
                                 order=1, mode="constant", cval=0.0)
                        for k in range(truth.shape[0])])
        P = prob_forecast(ci)                                       # (n_thr,Tf,H,W)
        Tf = truth.shape[0]
        speed = float(np.hypot(dy, dx))                             # px/step
        for ti, thr in enumerate(THRESHOLDS):
            o = truth >= thr                                        # (Tf,H,W) bool
            of = o.astype(np.float32)
            pop_raw = (stack >= thr).mean(axis=0).astype(np.float32)
            pop_cal = np.interp(pop_raw, prob_grid, cal_map[thr]).astype(np.float32)
            fields = {
                "prob": P[ti],
                "diff_raw": pop_raw,
                "diff_cal": pop_cal,
                "lag": (lag >= thr).astype(np.float32),
                "eul": (eul >= thr).astype(np.float32),
            }
            # advection-residual masks (per lead): genesis = advection both predicts
            # dry AND cannot reach any t0 echo; decay = advection predicts rain.
            dist0 = distance_transform_edt(~(t0 >= thr)).astype(np.float32)
            gen_m = np.zeros_like(o); dec_m = np.zeros_like(o)
            for k in range(Tf):
                reach = max(GENESIS_MARGIN_PX, SAFETY_FACTOR * speed * (k + 1))
                gen_m[k] = (lag[k] < thr) & (dist0 > reach)
                dec_m[k] = lag[k] >= thr
            gen_ev = gen_m & o                                      # genesis that rained
            dec_ev = dec_m & (~o)                                   # advected-wet that cleared
            ng_ev, nd_ev = int(gen_ev.sum()), int(dec_ev.sum())
            ng, nd = int(gen_m.sum()), int(dec_m.sum())
            for me, p in fields.items():
                add_brier(brier, pc, me, thr, p, of)
                se = (p - of) ** 2
                sg = float(se[gen_m].sum())
                brier_gen[me][thr][0] += sg; brier_gen[me][thr][1] += ng
                pc_gen[me][thr].append((sg, ng))
                brier_dec[me][thr][0] += float(se[dec_m].sum()); brier_dec[me][thr][1] += nd
                pb = p >= 0.5
                gd_gen[me][thr][0] += int((pb & gen_ev).sum()); gd_gen[me][thr][1] += ng_ev
                gd_dec[me][thr][0] += int((~pb & dec_ev).sum()); gd_dec[me][thr][1] += nd_ev

    # ---- cross-case batched diffusion sampling (slot-packing spans cases) ----
    print("Sampling diffusion ensemble...")
    slots = [(ci, mi) for ci in range(C) for mi in range(ensemble_size)]
    preds_buf = [[] for _ in range(C)]
    t_past = t_relative
    done = 0
    n_chunks = (len(slots) + eval_batch - 1) // eval_batch
    for k, start in enumerate(range(0, len(slots), eval_batch)):
        chunk = slots[start:start + eval_batch]
        cb = len(chunk)
        past_batch = torch.cat([cases[ci]["past"] for ci, _ in chunk], 0).to(dev)
        t_batch = t_past.repeat(cb, 1)
        seed = eval_seed + start
        torch.manual_seed(seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        ldm._cond_cache = None
        with torch.no_grad(), amp, redirect_stdout(io.StringIO()):
            s, _ = sampler.sample(num_diffusion_iters, cb, gen_shape,
                                  [[past_batch, t_batch]], progbar=False, verbose=False)
            y = ldm.autoencoder.decode(s / ldm.scale_factor)
        for j, (ci, _) in enumerate(chunk):
            preds_buf[ci].append(reverse_transform_R(y[j, 0].float().cpu().numpy()))
        while done < C and len(preds_buf[done]) == ensemble_size:
            finalize_case(done, preds_buf[done])
            preds_buf[done] = None
            done += 1
        if (k + 1) % 25 == 0 or k + 1 == n_chunks:
            print(f"  chunk {k+1}/{n_chunks}  cases done {done}/{C}")
    while done < C:
        if preds_buf[done] is not None and len(preds_buf[done]) == ensemble_size:
            finalize_case(done, preds_buf[done])
        done += 1

    _report(out_dir, C, ensemble_size, brier, brier_gen, brier_dec,
            gd_gen, gd_dec, pc, pc_gen)


def _bss_ci(pc, me, ref, thr, n_boot=2000, seed=0):
    """Bootstrap 95% CI for BSS(me vs ref) over cases. pc[m][thr]=list of (sse,npx)."""
    rng = np.random.default_rng(seed)
    a = np.array(pc[me][thr], dtype=np.float64)      # (C,2)
    b = np.array(pc[ref][thr], dtype=np.float64)
    if a.shape[0] < 2:
        return (float("nan"), float("nan"))
    n = a.shape[0]
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bm = a[idx, 0].sum() / max(a[idx, 1].sum(), 1)
        br = b[idx, 0].sum() / max(b[idx, 1].sum(), 1)
        vals.append(1 - bm / br if br else np.nan)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return (float(lo), float(hi))


def _brier_table(title, store, pc, summary_key, summary):
    print(f"\n=== {title} (lower=better; BSS vs Eulerian, 95% CI over cases) ===")
    summary[summary_key] = {}
    for thr in THRESHOLDS:
        be = store["eul"][thr][0] / max(store["eul"][thr][1], 1)
        rec = {}
        cells = [f"  {thr}mm |"]
        for me in ("prob", "diff_raw", "diff_cal", "lag", "eul"):
            bm = store[me][thr][0] / max(store[me][thr][1], 1)
            bss = 1 - bm / be if be else float("nan")
            rec[me] = {"brier": bm, "bss_vs_eul": bss}
            tag = me.replace("diff_", "d_")
            cells.append(f" {tag} {bm:.4f}({bss:+.3f})")
        # CIs on the two that matter
        for me in ("prob", "diff_cal"):
            lo, hi = _bss_ci(pc, me, "eul", thr)
            rec[me]["bss_ci"] = [lo, hi]
        summary[summary_key][str(thr)] = rec
        print("".join(cells))
        print(f"        BSS CI: prob [{rec['prob']['bss_ci'][0]:+.3f},"
              f"{rec['prob']['bss_ci'][1]:+.3f}]  "
              f"diff_cal [{rec['diff_cal']['bss_ci'][0]:+.3f},"
              f"{rec['diff_cal']['bss_ci'][1]:+.3f}]")


def _report(out_dir, C, M, brier, brier_gen, brier_dec, gd_gen, gd_dec, pc, pc_gen):
    summary = {"design": {"cases": C, "members": M}}
    _brier_table("BRIER overall (obj 1/6)", brier, pc, "brier", summary)
    print("\n  [GENESIS = advection predicts dry AND > 1.5*|motion|*lead px from any "
          "t0 echo:\n   rain there is true formation, not a band moving in. "
          "Persistence ≈ 0 here is FAIR, not an artifact.]")
    _brier_table("BRIER on GENESIS pixels (obj 4, TRUE growth)",
                 brier_gen, pc_gen, "brier_genesis", summary)

    # decay (advection predicts rain -> does the model anticipate clearing?)
    print("\n=== BRIER on DECAY pixels: advection predicts rain (BSS vs Lagrangian) ===")
    summary["brier_decay"] = {}
    for thr in THRESHOLDS:
        bl = brier_dec["lag"][thr][0] / max(brier_dec["lag"][thr][1], 1)
        rec = {}; cells = [f"  {thr}mm |"]
        for me in ("prob", "diff_raw", "diff_cal", "eul", "lag"):
            bm = brier_dec[me][thr][0] / max(brier_dec[me][thr][1], 1)
            bss = 1 - bm / bl if bl else float("nan")
            rec[me] = {"brier": bm, "bss_vs_lag": bss}
            cells.append(f" {me.replace('diff_', 'd_')} {bm:.4f}({bss:+.3f})")
        summary["brier_decay"][str(thr)] = rec
        print("".join(cells))

    print("\n=== GENESIS POD (rain from nowhere) / DECAY POD @0.5 ===")
    print("  (persistence = 0 on both by construction — it genuinely cannot forecast"
          " new rain or anticipate clearing)")
    summary["pod"] = {}
    for thr in THRESHOLDS:
        rec = {}; cells = [f"  {thr}mm |"]
        for me in ("prob", "diff_raw", "lag", "eul"):
            gc, gt = gd_gen[me][thr]; dc, dt = gd_dec[me][thr]
            gp = gc / gt if gt else float("nan")
            dp = dc / dt if dt else float("nan")
            rec[me] = {"genesis_pod": gp, "decay_pod": dp}
            cells.append(f" {me.replace('diff_raw', 'diff')}: gen {gp:.3f} dec {dp:.3f} |")
        summary["pod"][str(thr)] = rec
        print("".join(cells))

    # verdict: prob vs diffusion on probability + true-growth axes
    print("\n=== VERDICT: prob-nowcaster vs diffusion (the architecture A/B) ===")
    verdict = {}
    for thr in THRESHOLDS:
        bp = brier["prob"][thr][0] / max(brier["prob"][thr][1], 1)
        bdc = brier["diff_cal"][thr][0] / max(brier["diff_cal"][thr][1], 1)
        gbp = brier_gen["prob"][thr][0] / max(brier_gen["prob"][thr][1], 1)
        gbe = brier_gen["eul"][thr][0] / max(brier_gen["eul"][thr][1], 1)
        gbd = brier_gen["diff_cal"][thr][0] / max(brier_gen["diff_cal"][thr][1], 1)
        gp_p = gd_gen["prob"][thr][0] / max(gd_gen["prob"][thr][1], 1)
        gp_d = gd_gen["diff_raw"][thr][0] / max(gd_gen["diff_raw"][thr][1], 1)
        verdict[str(thr)] = {"brier_prob": bp, "brier_diff_cal": bdc,
                             "genesis_brier_prob": gbp, "genesis_brier_diff_cal": gbd,
                             "genesis_bss_prob_vs_persist": 1 - gbp / gbe if gbe else None,
                             "genesis_pod_prob": gp_p, "genesis_pod_diff": gp_d}
        print(f"  {thr}mm: overall Brier prob {bp:.4f} vs diff_cal {bdc:.4f}  |  "
              f"genesis: prob beats persistence by BSS "
              f"{1 - gbp / gbe if gbe else float('nan'):+.3f} "
              f"(diff_cal {1 - gbd / gbe if gbe else float('nan'):+.3f}); "
              f"gen-POD prob {gp_p:.3f} vs diff {gp_d:.3f}")
    summary["verdict"] = verdict

    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\nSaved: {out_dir}/summary.json")


if __name__ == "__main__":
    Fire(run)
