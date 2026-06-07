"""Step 0 (retrain.md): Lagrangian-persistence baseline on the held-out val cases.

Objective 5 — the model must beat Lagrangian persistence. Objective 4 — it must
capture growth (initiation) / decay (dissipation), which advection can't.

This scores LAGRANGIAN persistence (single-vector phase-correlation advection;
pysteps/cv2 not installed, so motion is a global FFT estimate) and EULERIAN
persistence (hold last frame) on the SAME cases as eval_val_large.py (random
split = the model's true holdout, same selection), per threshold and per lead,
plus initiation/dissipation skill — and prints them next to the model's numbers
from val_large_eval/summary.json.

No model, no GPU: data + numpy/scipy only.

Usage (from scripts/):  uv run python eval_persistence_baseline.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from fire import Fire
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter, shift as nd_shift

from ldcast.features.rust_data import RustRadarDataModule
from ldcast.visualization.plots import reverse_transform_R

THRESHOLDS = (0.1, 1.0, 5.0)


def csi_pod_far(h, m, f):
    h, m, f = float(h), float(m), float(f)
    csi = h / (h + m + f) if (h + m + f) else float("nan")
    pod = h / (h + m) if (h + m) else float("nan")
    far = f / (h + f) if (h + f) else float("nan")
    return csi, pod, far


def estimate_motion(past):
    """(dy, dx) px/step that continues the field's motion, from the past frames.

    Phase correlation per consecutive pair, averaged. Returns 0 motion on a
    nearly-dry last frame (advection is meaningless there).
    """
    if float((past[-1] > 0.1).mean()) < 0.005:
        return 0.0, 0.0
    H, W = past[-1].shape
    win = np.hanning(H)[:, None] * np.hanning(W)[None, :]
    dys, dxs = [], []
    for a, b in zip(past[:-1], past[1:]):
        A = np.fft.fft2(gaussian_filter(a, 1.0) * win)
        B = np.fft.fft2(gaussian_filter(b, 1.0) * win)
        R = B * np.conj(A)   # peaks at +Δ = the a->b motion to continue
        R /= np.abs(R) + 1e-8
        r = np.fft.ifft2(R).real
        dy, dx = np.unravel_index(int(np.argmax(r)), r.shape)
        if dy > H // 2:
            dy -= H
        if dx > W // 2:
            dx -= W
        dys.append(dy); dxs.append(dx)
    dy, dx = float(np.mean(dys)), float(np.mean(dxs))
    cap = 20.0
    return float(np.clip(dy, -cap, cap)), float(np.clip(dx, -cap, cap))


def run(config="../config/train_rust.yaml", per_bin_cases=60, scan_per_bin=250,
        split_mode="random", out_dir=None):
    cfg = OmegaConf.load(config)
    if cfg.get("radar_root"):
        os.environ["DGMR_RADAR_ROOT"] = cfg.radar_root
    out_dir = Path(out_dir or "../models/genforecast_rust/val_large_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"split_mode : {split_mode}  (random = the model's true holdout)")
    print(f"design     : {per_bin_cases}/bin, same cases as eval_val_large\n")
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
    val_w, val_ds = dm.val_w, dm.valid_ds

    # same per-bin wettest selection as eval_val_large.py
    print("Selecting cases (matching eval_val_large)...")
    cases = []  # (past_mmhr [Tp,H,W], truth_mmhr [T,H,W])
    for w in sorted(np.unique(val_w).tolist()):
        rows = np.flatnonzero(val_w == w)[:scan_per_bin]
        scored = []
        for ridx in rows:
            past_t, future_t = val_ds[int(ridx)]
            truth = reverse_transform_R(future_t[0].float().numpy())
            scored.append((float((truth >= 1.0).mean()), int(ridx), past_t, truth))
        scored.sort(key=lambda x: -x[0])
        for _, ridx, past_t, truth in scored[:per_bin_cases]:
            past_mmhr = reverse_transform_R(past_t[0].float().numpy())
            cases.append((past_mmhr, truth))
    print(f"  {len(cases)} cases\n")

    T = cases[0][1].shape[0]
    # accumulators: counts[method][thr] = [h,m,f]; lead[method][(thr,lt)]=[h,m,f]
    methods = ("lagrangian", "eulerian")
    counts = {me: {t: [0, 0, 0] for t in THRESHOLDS} for me in methods}
    lead = {me: {(t, lt): [0, 0, 0] for t in THRESHOLDS for lt in range(T)}
            for me in methods}
    # growth/decay: init = dry@t0 & rain@lead ; diss = rain@t0 & dry@lead
    gd = {me: {t: [0, 0, 0, 0] for t in THRESHOLDS} for me in methods}
    #   [init_caught, init_total, diss_caught, diss_total]

    print("Scoring persistence...")
    for ci, (past, truth) in enumerate(cases):
        last = past[-1]
        dy, dx = estimate_motion(past)
        preds = {
            "eulerian": [last for _ in range(T)],
            "lagrangian": [nd_shift(last, (dy * (k + 1), dx * (k + 1)),
                                    order=1, mode="constant", cval=0.0)
                           for k in range(T)],
        }
        for me in methods:
            for lt in range(T):
                p_all = preds[me][lt]
                o_all = truth[lt]
                for thr in THRESHOLDS:
                    p, o = p_all >= thr, o_all >= thr
                    h = int((p & o).sum()); m = int((~p & o).sum())
                    f = int((p & ~o).sum())
                    counts[me][thr][0] += h; counts[me][thr][1] += m
                    counts[me][thr][2] += f
                    lc = lead[me][(thr, lt)]
                    lc[0] += h; lc[1] += m; lc[2] += f
                    # growth/decay vs t0=last
                    t0 = last >= thr
                    init = (~t0) & o          # dry@t0, rain@lead
                    diss = t0 & (~o)          # rain@t0, dry@lead
                    g = gd[me][thr]
                    g[0] += int((p & init).sum()); g[1] += int(init.sum())
                    g[2] += int((~p & diss).sum()); g[3] += int(diss.sum())
        if (ci + 1) % 100 == 0:
            print(f"  {ci+1}/{len(cases)}")

    # model numbers from the big eval
    model = None
    sj = out_dir / "summary.json"
    if sj.exists():
        model = json.load(open(sj))

    print("\n=== OVERALL skill: model vs persistence (same 660 held-out cases) ===")
    print(f"{'thr':>6} | {'MODEL pm-mean':>14} {'MODEL members':>14} "
          f"| {'LAGRANGIAN':>11} {'EULERIAN':>11}")
    print(f"{'':>6} | {'CSI  POD':>14} {'CSI  POD':>14} "
          f"| {'CSI  POD':>11} {'CSI  POD':>11}")
    for thr in THRESHOLDS:
        lag = csi_pod_far(*counts["lagrangian"][thr])
        eul = csi_pod_far(*counts["eulerian"][thr])
        mpm = model["pm_mean"].get(str(thr)) if model else None
        mmem = model["overall"].get(str(thr)) if model else None
        def cp(d, k="csi"):
            return f"{d[k]:.3f} {d['pod']:.2f}" if d else "  --   -- "
        print(f"{thr:>6} | {cp(mpm):>14} {cp(mmem):>14} "
              f"| {lag[0]:.3f} {lag[1]:.2f} | {eul[0]:.3f} {eul[1]:.2f}")

    print("\n=== per-lead CSI (Lagrangian persistence) ===")
    print("  thr    " + "  ".join(f"+{(lt+1)*10:>3}m" for lt in range(T)))
    for thr in THRESHOLDS:
        vals = [csi_pod_far(*lead["lagrangian"][(thr, lt)])[0] for lt in range(T)]
        print(f"  {thr:<5} " + "  ".join(f"{v:.3f}" for v in vals))

    print("\n=== GROWTH / DECAY skill of persistence (the bar for obj 4) ===")
    print("  initiation POD = caught rain that grew from dry@t0;  "
          "dissipation = caught clearing")
    print(f"{'thr':>6} | {'LAG init POD':>13} {'EUL init POD':>13} "
          f"| {'LAG diss':>9} {'EUL diss':>9}")
    for thr in THRESHOLDS:
        out = []
        for me in ("lagrangian", "eulerian"):
            ic, it, dc, dt = gd[me][thr]
            out.append((ic / it if it else float("nan"),
                        dc / dt if dt else float("nan")))
        print(f"{thr:>6} | {out[0][0]:>13.3f} {out[1][0]:>13.3f} "
              f"| {out[0][1]:>9.3f} {out[1][1]:>9.3f}")

    # save
    summary = {
        "lagrangian": {str(t): dict(zip(("csi", "pod", "far"),
                       csi_pod_far(*counts["lagrangian"][t]))) for t in THRESHOLDS},
        "eulerian": {str(t): dict(zip(("csi", "pod", "far"),
                     csi_pod_far(*counts["eulerian"][t]))) for t in THRESHOLDS},
        "growth_decay": {me: {str(t): gd[me][t] for t in THRESHOLDS} for me in methods},
    }
    with open(out_dir / "persistence_baseline.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\nSaved: {out_dir}/persistence_baseline.json")


if __name__ == "__main__":
    Fire(run)
