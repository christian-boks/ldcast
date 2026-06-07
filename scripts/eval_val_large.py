"""Large one-off forecast eval on the held-back (untrained) val crops.

WHY split_mode defaults to "random": the current checkpoint trained on the
RANDOM 90% (seed 42), so its true held-out data is the random 10% val. Do NOT
use the temporal split here — the model trained on most of those days. We accept
the neighbour-leakage (same-timestamp / +-10 min crops); this measures the
model's behaviour on data it didn't directly train on.

Optimised batching: instead of one sampler call per case, we pack (case x member)
slots into VRAM-filling chunks that SPAN cases, so every UNet forward is full
regardless of ensemble_size. Members are sampled cross-case; each case's metrics
are extracted and its raw predictions freed as soon as its ensemble completes, so
peak RAM stays bounded (~eval_batch predictions in flight, not C x M).

Eval uses the EMA weights (the deployable model) and DPM-Solver++ 20 steps
(validated corr >=0.98 vs PLMS-50, so no blur penalty).

Outputs to <model_dir>/val_large_eval/:
  - summary.json  : all aggregate metrics
  - per_case.npz  : per-case arrays (csi, crps, bin, month) for re-plotting
  - *.png         : per-case CSI distribution, per-lead curves, reliability,
                    PoP operating curve, FSS-vs-scale
and a table to stdout.

Usage (from scripts/):
    uv run python eval_val_large.py --per_bin_cases=50 --ensemble_size=32
"""
import io
import os
import sys
import json
from collections import defaultdict
from contextlib import nullcontext, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter, shift as nd_shift

from ldcast.features.rust_data import RustRadarDataModule
from ldcast.models.diffusion import dpm_solver
from ldcast.models.genforecast.monitor import SamplePredictionLogger
from ldcast.visualization.plots import reverse_transform_R

from train_genforecast import setup_model
from eval_persistence_baseline import estimate_motion

THRESHOLDS = (0.1, 1.0, 5.0)
FSS_SCALES = (1, 3, 9, 25, 51)        # neighbourhood box widths, pixels
POP_CUTOFFS = np.round(np.arange(0.05, 1.0, 0.05), 2)
REL_BINS = 11                          # reliability-diagram probability bins


# ----------------------------- small helpers -----------------------------
def csi_pod_far(h, m, f):
    h, m, f = float(h), float(m), float(f)
    csi = h / (h + m + f) if (h + m + f) else float("nan")
    pod = h / (h + m) if (h + m) else float("nan")
    far = f / (h + f) if (h + f) else float("nan")
    return csi, pod, far


def crps_ensemble(members, truth):
    """Fair CRPS estimator, averaged over all pixels/leads. members:(M,T,H,W)."""
    M = members.shape[0]
    xs = np.sort(members.reshape(M, -1), axis=0)        # (M, P)
    y = truth.reshape(-1)                               # (P,)
    term1 = np.abs(xs - y[None, :]).mean(axis=0)        # E|X-y|
    coeff = (2 * np.arange(1, M + 1) - M - 1)[:, None]  # sorted-form of E|X-X'|
    term2 = (coeff * xs).sum(axis=0) / (M * M)
    return float((term1 - term2).mean())


def fss_numden(fcst_bin, obs_bin, scale):
    """Return (sum (ff-of)^2, sum ff^2+of^2) over pixels for one case/scale."""
    ff = uniform_filter(fcst_bin.astype(np.float32), size=(1, scale, scale), mode="constant")
    of = uniform_filter(obs_bin.astype(np.float32), size=(1, scale, scale), mode="constant")
    return float(((ff - of) ** 2).sum()), float((ff * ff + of * of).sum())


def boot_ci(per_case_counts, idxs, n_boot=2000, seed=0):
    """Bootstrap 95% CI for CSI over cases. per_case_counts[i] = (h,m,f)."""
    rng = np.random.default_rng(seed)
    arr = np.array([per_case_counts[i] for i in idxs], dtype=np.float64)  # (n,3)
    if arr.shape[0] < 2:
        return (float("nan"), float("nan"))
    n = arr.shape[0]
    vals = []
    for _ in range(n_boot):
        s = arr[rng.integers(0, n, n)].sum(axis=0)
        d = s[0] + s[1] + s[2]
        vals.append(s[0] / d if d else np.nan)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return (float(lo), float(hi))


def season_of(month):
    return "warm(Apr-Sep)" if 4 <= month <= 9 else "cold(Oct-Mar)"


# ------------------------------- main -------------------------------
def run(
    config="../config/train_rust.yaml",
    ckpt="../models/genforecast_rust/last.ckpt",
    per_bin_cases=50,
    ensemble_size=32,
    eval_batch=24,
    num_diffusion_iters=20,
    scan_per_bin=250,
    eval_seed=1234,
    split_mode="random",   # the current model's TRUE held-out set
    out_dir=None,
):
    if not os.path.isfile(ckpt):
        sys.exit(f"checkpoint not found: {ckpt}")
    cfg = OmegaConf.load(config)
    if cfg.get("radar_root"):
        os.environ["DGMR_RADAR_ROOT"] = cfg.radar_root
    out_dir = Path(out_dir or os.path.join(
        os.path.dirname(os.path.abspath(ckpt)), "val_large_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"checkpoint : {ckpt}")
    print(f"split_mode : {split_mode}  (random = the model's real held-back 10%)")
    print(f"design     : {per_bin_cases}/bin x 11 bins, {ensemble_size} members, "
          f"eval_batch={eval_batch}, {num_diffusion_iters} steps, EMA\n")

    # ---- data (force the random split = the model's true holdout) ----
    print("Loading data...")
    dm = RustRadarDataModule(
        index_path=cfg.index_path, mode="diffusion",
        past_steps=cfg.past_steps, future_steps=cfg.future_steps,
        height=cfg.height, width=cfg.width,
        batch_size=cfg.genforecast_batch_size, num_workers=0,
        valid_frac=0.1, test_frac=0.0, split_mode=split_mode, seed=42,
        use_weighted_sampler=cfg.use_weighted_sampler,
    )
    dm.setup("fit")
    past_steps = int(dm.past_steps)

    # ---- model (EMA weights, frozen) ----
    print("Building model + loading weights (EMA)...")
    ldm, _ = setup_model(
        num_timesteps=cfg.future_steps // 4,
        autoenc_weights_fn=os.path.join(cfg.autoenc_dir, "state_dict.pt"),
        use_obs=True, use_nwp=False, model_dir=cfg.genforecast_dir,
        lr=cfg.genforecast_lr, precision=cfg.precision,
        optimizer_8bit=cfg.optimizer_8bit, max_epochs=1, limit_train_batches=1,
        limit_val_batches=1, scale_factor=1.0, gradient_clip_val=1.0,
        sample_every_n_epochs=1, max_hours=None, early_stopping_patience=0,
        accumulate_grad_batches=1, save_top_k=0,
    )
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    ldm.load_state_dict(sd, strict=True)
    if getattr(ldm, "model_ema", None) is not None:
        ldm.model_ema.copy_to(ldm.model)   # deploy the EMA weights (eval-only)
    ldm.use_ema = False                    # ema_scope now a no-op
    ldm = ldm.to(dev).eval()
    sampler = dpm_solver.DPMSolverSampler(ldm)
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if dev.type == "cuda" else nullcontext())

    # ---- case selection (per-bin wettest, recording timestamp) ----
    print("Selecting cases...")
    val_w, val_ts, val_ds = dm.val_w, dm.val_ts, dm.valid_ds
    bin_values = sorted(np.unique(val_w).tolist())
    cases = []  # dicts: bin, month, past[1,1,Tp,H,W], truth[T,H,W]
    for bin_idx, w in enumerate(bin_values):
        rows = np.flatnonzero(val_w == w)[:scan_per_bin]
        scored = []
        for ridx in rows:
            past_t, future_t = val_ds[int(ridx)]
            truth = reverse_transform_R(future_t[0].float().numpy())  # (T,H,W)
            scored.append((float((truth >= 1.0).mean()), int(ridx), past_t, truth))
        scored.sort(key=lambda x: -x[0])
        for _, ridx, past_t, truth in scored[:per_bin_cases]:
            month = datetime.fromtimestamp(int(val_ts[ridx]), tz=timezone.utc).month
            cases.append({"bin": bin_idx, "month": month,
                          "past": past_t.unsqueeze(0).clone(), "truth": truth})
    C = len(cases)
    num_bins = len(bin_values)
    print(f"  {C} cases ({per_bin_cases}/bin x {num_bins} bins), "
          f"{C * ensemble_size} samples to draw\n")

    with torch.no_grad(), amp:
        probe = torch.zeros(1, 1, cfg.future_steps, cfg.height, cfg.width, device=dev)
        gen_shape = tuple(ldm.autoencoder.encode(probe)[0].shape[1:])
        del probe
    print(f"  latent gen_shape = {gen_shape}")

    # ---- global accumulators ----
    bin_counts = {(b, t): [0, 0, 0] for b in range(num_bins) for t in THRESHOLDS}
    lead_counts = defaultdict(lambda: [0, 0, 0])             # (lead, thr)
    pm_counts = {t: [0, 0, 0] for t in THRESHOLDS}           # PM-mean det. map
    rel_obs = {t: np.zeros(REL_BINS) for t in THRESHOLDS}    # reliability
    rel_cnt = {t: np.zeros(REL_BINS) for t in THRESHOLDS}
    rel_psum = {t: np.zeros(REL_BINS) for t in THRESHOLDS}
    pop_counts = {(t, c): [0, 0, 0] for t in THRESHOLDS for c in POP_CUTOFFS}
    fss_num = {(t, s): 0.0 for t in THRESHOLDS for s in FSS_SCALES}
    fss_den = {(t, s): 0.0 for t in THRESHOLDS for s in FSS_SCALES}
    # per-case records (small): bin, month, per-thr (h,m,f), csi, crps
    pc_bin, pc_month, pc_crps = [], [], []
    pc_counts = {t: [] for t in THRESHOLDS}                  # list of (h,m,f) per case
    # growth/decay (obj 4) + Brier (obj 1/6): model (pm-mean) vs persistence
    GD_METHODS = ("model", "lag", "eul")
    gd = {me: {t: [0, 0, 0, 0] for t in THRESHOLDS} for me in GD_METHODS}  # [ic,it,dc,dt]
    brier = {me: {t: [0.0, 0] for t in THRESHOLDS} for me in GD_METHODS}   # [sse, n_px]

    def finalize_case(ci, preds):
        """Extract every metric contribution from a completed case, then free."""
        case = cases[ci]
        truth = case["truth"]                               # (T,H,W) mm/h
        stack = np.stack(preds, axis=0)                     # (M,T,H,W)
        pm = SamplePredictionLogger._pm_mean(stack)         # (T,H,W) det. map
        pc_bin.append(case["bin"]); pc_month.append(case["month"])
        pc_crps.append(crps_ensemble(stack, truth))
        # persistence forecasts for this case (model-vs-persistence tests)
        past_mmhr = reverse_transform_R(case["past"][0, 0].float().numpy())  # (Tp,H,W)
        t0 = past_mmhr[-1]
        dy, dx = estimate_motion(past_mmhr)
        eul = np.repeat(t0[None], truth.shape[0], axis=0)        # hold last frame
        lag = np.stack([nd_shift(t0, (dy * (k + 1), dx * (k + 1)),
                                 order=1, mode="constant", cval=0.0)
                        for k in range(truth.shape[0])])         # advect last frame
        for thr in THRESHOLDS:
            o = truth >= thr                          # (T,H,W)
            ob = o[None]                              # broadcast over members
            p = stack >= thr                          # (M,T,H,W)
            ht = (p & ob).sum(axis=(0, 2, 3))         # per-lead, pooled over M,H,W
            mt = ((~p) & ob).sum(axis=(0, 2, 3))
            ft = (p & ~ob).sum(axis=(0, 2, 3))
            h, m, f = int(ht.sum()), int(mt.sum()), int(ft.sum())
            for lt in range(truth.shape[0]):
                lc = lead_counts[(lt, thr)]
                lc[0] += int(ht[lt]); lc[1] += int(mt[lt]); lc[2] += int(ft[lt])
            bin_counts[(case["bin"], thr)][0] += h
            bin_counts[(case["bin"], thr)][1] += m
            bin_counts[(case["bin"], thr)][2] += f
            pc_counts[thr].append((h, m, f))
            # PM-mean deterministic-map CSI + FSS
            pmb = pm >= thr
            pm_counts[thr][0] += int((pmb & o).sum())
            pm_counts[thr][1] += int((~pmb & o).sum())
            pm_counts[thr][2] += int((pmb & ~o).sum())
            for s in FSS_SCALES:
                n, d = fss_numden(pmb, o, s)
                fss_num[(thr, s)] += n; fss_den[(thr, s)] += d
            # PoP probability field -> reliability + operating-curve sweep
            pop = (stack >= thr).mean(axis=0)               # (T,H,W) in [0,1]
            bidx = np.clip((pop * REL_BINS).astype(int), 0, REL_BINS - 1)
            np.add.at(rel_cnt[thr], bidx.ravel(), 1.0)
            np.add.at(rel_obs[thr], bidx.ravel(), o.ravel().astype(float))
            np.add.at(rel_psum[thr], bidx.ravel(), pop.ravel())
            for c in POP_CUTOFFS:
                dec = pop >= c
                pc = pop_counts[(thr, c)]
                pc[0] += int((dec & o).sum()); pc[1] += int((~dec & o).sum())
                pc[2] += int((dec & ~o).sum())
            # growth/decay (pm-mean vs persistence) + Brier (PoP vs persistence 0/1)
            t0b = t0 >= thr                                  # (H,W)
            fcsts = {"model": pmb, "lag": lag >= thr, "eul": eul >= thr}  # each (T,H,W)
            for lt in range(truth.shape[0]):
                init = (~t0b) & o[lt]                        # dry@t0 -> rain@lead
                diss = t0b & (~o[lt])                        # rain@t0 -> dry@lead
                it, dt = int(init.sum()), int(diss.sum())
                for me in GD_METHODS:
                    fb = fcsts[me][lt]
                    g = gd[me][thr]
                    g[0] += int((fb & init).sum()); g[1] += it
                    g[2] += int((~fb & diss).sum()); g[3] += dt
            of = o.astype(np.float32)                        # (T,H,W) 0/1 truth
            brier["model"][thr][0] += float(((pop - of) ** 2).sum())
            brier["model"][thr][1] += of.size
            brier["lag"][thr][0] += float((((lag >= thr).astype(np.float32) - of) ** 2).sum())
            brier["lag"][thr][1] += of.size
            brier["eul"][thr][0] += float((((eul >= thr).astype(np.float32) - of) ** 2).sum())
            brier["eul"][thr][1] += of.size

    # ---- cross-case batched sampling ----
    print("Sampling...")
    slots = [(ci, mi) for ci in range(C) for mi in range(ensemble_size)]
    preds_buf = [[] for _ in range(C)]
    t_past = torch.arange(-past_steps + 1, 1, dtype=torch.float32,
                          device=dev).unsqueeze(0)
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
        if (k + 1) % 20 == 0 or k + 1 == n_chunks:
            print(f"  chunk {k+1}/{n_chunks}  cases done {done}/{C}")
    while done < C:  # flush any stragglers (shouldn't happen with ordered slots)
        if preds_buf[done] is not None and len(preds_buf[done]) == ensemble_size:
            finalize_case(done, preds_buf[done])
        done += 1

    # ---- aggregate + report ----
    _report(out_dir, cases, num_bins, bin_counts, lead_counts, pm_counts,
            rel_obs, rel_cnt, rel_psum, pop_counts, fss_num, fss_den,
            pc_bin, pc_month, pc_crps, pc_counts, ensemble_size, per_bin_cases,
            gd, brier)


def _report(out_dir, cases, num_bins, bin_counts, lead_counts, pm_counts,
            rel_obs, rel_cnt, rel_psum, pop_counts, fss_num, fss_den,
            pc_bin, pc_month, pc_crps, pc_counts, M, per_bin_cases,
            gd, brier):
    pc_bin = np.array(pc_bin); pc_month = np.array(pc_month)
    pc_crps = np.array(pc_crps)
    summary = {"design": {"cases": len(cases), "members": M,
                          "per_bin_cases": per_bin_cases}}

    # overall + per-bin CSI/POD/FAR with bootstrap CIs (over cases)
    print("\n=== CSI / POD / FAR  (per-member pooled; 95% CI over cases) ===")
    print(f"{'bin':>5} {'n':>4} | "
          + " | ".join(f"{t}mm CSI[lo-hi]      POD   FAR" for t in THRESHOLDS))
    summary["per_bin"] = {}
    for b in range(num_bins):
        idxs = [i for i, bb in enumerate(pc_bin) if bb == b]
        cells = [f"{b:>5} {len(idxs):>4} |"]
        rec = {}
        for thr in THRESHOLDS:
            h, m, f = bin_counts[(b, thr)]
            csi, pod, far = csi_pod_far(h, m, f)
            lo, hi = boot_ci(pc_counts[thr], idxs)
            cells.append(f" {csi:.3f}[{lo:.3f}-{hi:.3f}] {pod:.2f} {far:.2f} |")
            rec[str(thr)] = {"csi": csi, "ci": [lo, hi], "pod": pod, "far": far,
                             "h": h, "m": m, "f": f}
        summary["per_bin"][b] = rec
        print("".join(cells))
    print("-" * 80)
    cells = [f"{'OVER':>5} {len(pc_bin):>4} |"]
    summary["overall"] = {}
    allidx = list(range(len(pc_bin)))
    for thr in THRESHOLDS:
        h = sum(bin_counts[(b, thr)][0] for b in range(num_bins))
        m = sum(bin_counts[(b, thr)][1] for b in range(num_bins))
        f = sum(bin_counts[(b, thr)][2] for b in range(num_bins))
        csi, pod, far = csi_pod_far(h, m, f)
        lo, hi = boot_ci(pc_counts[thr], allidx)
        cells.append(f" {csi:.3f}[{lo:.3f}-{hi:.3f}] {pod:.2f} {far:.2f} |")
        summary["overall"][str(thr)] = {"csi": csi, "ci": [lo, hi], "pod": pod, "far": far}
    print("".join(cells))

    # PM-mean deterministic map + CRPS
    print("\n=== ensemble products ===")
    summary["pm_mean"], summary["crps"] = {}, float(np.mean(pc_crps))
    for thr in THRESHOLDS:
        csi, pod, far = csi_pod_far(*pm_counts[thr])
        summary["pm_mean"][str(thr)] = {"csi": csi, "pod": pod, "far": far}
        print(f"  PM-mean CSI@{thr}mm = {csi:.3f}  (POD {pod:.2f} FAR {far:.2f})")
    print(f"  mean CRPS = {summary['crps']:.4f} mm/h")

    # season stratification (overall + per threshold)
    print("\n=== season (per-member pooled CSI) ===")
    summary["season"] = {}
    for seas in ("warm(Apr-Sep)", "cold(Oct-Mar)"):
        idxs = [i for i, mo in enumerate(pc_month) if season_of(mo) == seas]
        row = {"n_cases": len(idxs)}
        cells = [f"  {seas:>14} (n={len(idxs):>4}):"]
        for thr in THRESHOLDS:
            h = m = f = 0
            for i in idxs:
                hh, mm, ff = pc_counts[thr][i]; h += hh; m += mm; f += ff
            csi, pod, far = csi_pod_far(h, m, f)
            row[str(thr)] = {"csi": csi, "pod": pod, "far": far}
            cells.append(f"  {thr}mm CSI {csi:.3f}/POD {pod:.2f}")
        summary["season"][seas] = row
        print("".join(cells))

    # per-case distribution
    print("\n=== per-case CSI distribution (1.0mm; the bimodality check) ===")
    pc_csi1 = np.array([csi_pod_far(*c)[0] for c in pc_counts[1.0]])
    pct = np.nanpercentile(pc_csi1, [5, 25, 50, 75, 95])
    print(f"  p5={pct[0]:.3f} p25={pct[1]:.3f} med={pct[2]:.3f} "
          f"p75={pct[3]:.3f} p95={pct[4]:.3f}   frac>0.3={np.mean(pc_csi1>0.3):.2f}")
    summary["per_case_csi1_pct"] = pct.tolist()

    # FSS vs scale
    print("\n=== FSS vs neighbourhood scale (px) ===")
    summary["fss"] = {}
    for thr in THRESHOLDS:
        vals = []
        for s in FSS_SCALES:
            d = fss_den[(thr, s)]
            fss = 1 - fss_num[(thr, s)] / d if d else float("nan")
            vals.append(fss); summary["fss"][f"{thr}_{s}"] = fss
        print(f"  {thr}mm: " + "  ".join(f"{s}px={v:.3f}" for s, v in zip(FSS_SCALES, vals)))

    # PoP operating curve (overall) + reliability — saved to json, plotted below
    summary["pop_sweep"] = {str(t): {float(c): csi_pod_far(*pop_counts[(t, c)])[1:]
                                     for c in POP_CUTOFFS} for t in THRESHOLDS}

    # growth/decay (obj 4): model pm-mean vs persistence
    print("\n=== GROWTH/DECAY: model (pm-mean) vs persistence  (obj-4 test) ===")
    print("  init POD = caught rain growing from dry@t0;  diss = caught clearing")
    summary["growth_decay"] = {}
    for thr in THRESHOLDS:
        rec = {}; cells = [f"  {thr}mm |"]
        for me in ("model", "lag", "eul"):
            ic, it, dc, dt = gd[me][thr]
            ip = ic / it if it else float("nan")
            dp = dc / dt if dt else float("nan")
            rec[me] = {"init_pod": ip, "diss_pod": dp}
            cells.append(f" {me}: init {ip:.3f} diss {dp:.3f} |")
        summary["growth_decay"][str(thr)] = rec
        print("".join(cells))

    # Brier (obj 1/6): model PoP vs persistence 0/1 + skill score
    print("\n=== BRIER (lower=better) + skill vs persistence  (obj-1/6 test) ===")
    summary["brier"] = {}
    for thr in THRESHOLDS:
        bm = brier["model"][thr][0] / brier["model"][thr][1]
        bl = brier["lag"][thr][0] / brier["lag"][thr][1]
        be = brier["eul"][thr][0] / brier["eul"][thr][1]
        bss_l = 1 - bm / bl if bl else float("nan")
        bss_e = 1 - bm / be if be else float("nan")
        summary["brier"][str(thr)] = {"model": bm, "lag": bl, "eul": be,
                                      "bss_vs_lag": bss_l, "bss_vs_eul": bss_e}
        print(f"  {thr}mm Brier: model {bm:.4f}  lag {bl:.4f}  eul {be:.4f}"
              f"   BSS vs lag {bss_l:+.3f}  vs eul {bss_e:+.3f}")

    # ---- save artifacts ----
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    np.savez(out_dir / "per_case.npz", bin=pc_bin, month=pc_month,
             crps=pc_crps, csi1=pc_csi1)
    _plots(out_dir, pc_csi1, pc_bin, lead_counts, rel_obs, rel_cnt, rel_psum,
           pop_counts, fss_num, fss_den)
    print(f"\nSaved: {out_dir}/summary.json, per_case.npz, *.png")


def _plots(out_dir, pc_csi1, pc_bin, lead_counts, rel_obs, rel_cnt, rel_psum,
           pop_counts, fss_num, fss_den):
    # per-case distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pc_csi1, bins=30, color="steelblue", alpha=0.85)
    ax.set_xlabel("per-case CSI @1.0mm"); ax.set_ylabel("cases")
    ax.set_title("Per-case skill distribution (bimodal = regime structure)")
    fig.tight_layout(); fig.savefig(out_dir / "dist_csi1.png", dpi=90); plt.close(fig)

    # per-lead curves
    leads = sorted({lt for (lt, _) in lead_counts})
    fig, ax = plt.subplots(figsize=(7, 4))
    for thr in THRESHOLDS:
        y = [csi_pod_far(*lead_counts[(lt, thr)])[0] for lt in leads]
        ax.plot([(lt + 1) * 10 for lt in leads], y, marker="o", label=f"{thr}mm")
    ax.set_xlabel("lead (min)"); ax.set_ylabel("CSI"); ax.legend()
    ax.set_title("CSI vs lead time"); fig.tight_layout()
    fig.savefig(out_dir / "per_lead_csi.png", dpi=90); plt.close(fig)

    # reliability diagram
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for thr in THRESHOLDS:
        cnt = rel_cnt[thr]; ok = cnt > 0
        pmean = np.where(ok, rel_psum[thr] / np.maximum(cnt, 1), np.nan)
        obs = np.where(ok, rel_obs[thr] / np.maximum(cnt, 1), np.nan)
        ax.plot(pmean, obs, marker="o", label=f"{thr}mm")
    ax.set_xlabel("forecast P(rain)"); ax.set_ylabel("observed freq")
    ax.legend(); ax.set_title("Reliability (PoP calibration)")
    fig.tight_layout(); fig.savefig(out_dir / "reliability.png", dpi=90); plt.close(fig)

    # PoP operating curve (POD vs FAR), 0.1mm
    fig, ax = plt.subplots(figsize=(5, 5))
    for thr in THRESHOLDS:
        pods, fars = [], []
        for c in POP_CUTOFFS:
            _, pod, far = csi_pod_far(*pop_counts[(thr, c)])
            pods.append(pod); fars.append(far)
        ax.plot(fars, pods, marker=".", label=f"{thr}mm")
    ax.set_xlabel("FAR"); ax.set_ylabel("POD"); ax.legend()
    ax.set_title("PoP decision-threshold operating curve"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "pop_operating.png", dpi=90); plt.close(fig)

    # FSS vs scale
    fig, ax = plt.subplots(figsize=(7, 4))
    for thr in THRESHOLDS:
        y = [1 - fss_num[(thr, s)] / fss_den[(thr, s)] if fss_den[(thr, s)] else np.nan
             for s in FSS_SCALES]
        ax.plot(FSS_SCALES, y, marker="o", label=f"{thr}mm")
    ax.set_xlabel("neighbourhood (px)"); ax.set_ylabel("FSS"); ax.legend()
    ax.set_title("FSS vs scale"); fig.tight_layout()
    fig.savefig(out_dir / "fss_scale.png", dpi=90); plt.close(fig)


if __name__ == "__main__":
    Fire(run)
