"""Prob-nowcaster Brier on the FAIR INTERIOR (the 2026-06-02 confound fix).

Companion to eval_steps_baseline.py. The wettest-400 A/B scored the FULL 128px
domain, which penalises advection for dry inflow it can't see (21-27% of the
domain at +30-40 min) and flatters persistence-like models. The fair comparison
scores only the INTERIOR: pixels whose advective source (current pos - motion*lead)
was inside the crop at t0 — the region a radar-only forecast could possibly get
right. The mask is forecaster-agnostic (defined by t0 motion + lead, same single
global vector as the genesis/decay masks), so prob/STEPS/persistence are judged on
identical pixels.

This re-derives the SAME cases as eval_aalborg_ab.py and runs ONLY the prob model
(one forward/case — no 29-min diffusion). It writes prob's full-domain Brier (a
cross-check that must match ab_eval/summary.json) and its interior Brier (the new
honest number) to prob_interior.json, which eval_steps_baseline.py loads for the
side-by-side. Run in the MAIN venv with .venv/bin/python (NOT uv run — that wipes
the maturin dgmr_py loader).

Usage (from scripts/):
    ../.venv/bin/python eval_prob_interior.py
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from omegaconf import OmegaConf
from scipy.ndimage import shift as nd_shift, distance_transform_edt

from ldcast.features.rust_data import RustRadarDataModule
from ldcast.models.autoenc import autoenc, encoder
from ldcast.models.nowcast.prob_nowcast import ProbNowcastNet, ProbNowcaster
from ldcast.visualization.plots import reverse_transform_R
from eval_persistence_baseline import estimate_motion

THRESHOLDS = (0.1, 1.0, 5.0)
METHODS = ("prob", "lag_sv", "eul")
SAFETY_FACTOR = 1.5
GENESIS_MARGIN_PX = 8


def interior_mask(dy, dx, Tf, H, W):
    """(Tf,H,W) bool: pixel's advective source (pos - motion*lead) is inside crop.

    Forecaster-agnostic — identical for every method (uses the single global t0
    vector, as the genesis/decay masks do). speed=0 -> whole domain is interior.
    """
    ys = np.arange(H)[:, None].astype(np.float64)
    xs = np.arange(W)[None, :].astype(np.float64)
    m = np.zeros((Tf, H, W), dtype=bool)
    for k in range(Tf):
        sy = ys - dy * (k + 1)
        sx = xs - dx * (k + 1)
        m[k] = (sy >= 0) & (sy <= H - 1) & (sx >= 0) & (sx <= W - 1)
    return m


def run(
    config="../config/train_rust.yaml",
    prob_ckpt="../models/prob_nowcaster_aalborg/last.ckpt",
    n_cases=400,
    scan_rows=5000,
    region_center=(685, 852),
    region_radius=64,
    test_frac=0.1,
    valid_frac=0.1,
    out=None,
):
    cfg = OmegaConf.load(config)
    if cfg.get("radar_root"):
        os.environ["DGMR_RADAR_ROOT"] = cfg.radar_root
    out = Path(out or "../models/prob_nowcaster_aalborg/ab_eval/prob_interior.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tp, Tf = int(cfg.past_steps), int(cfg.future_steps)

    print("Loading Aalborg test split (same recipe as eval_aalborg_ab)...")
    dm = RustRadarDataModule(
        index_path=cfg.index_path, mode="diffusion",
        past_steps=Tp, future_steps=Tf, height=cfg.height, width=cfg.width,
        batch_size=cfg.genforecast_batch_size, num_workers=0,
        valid_frac=valid_frac, test_frac=test_frac, split_mode="temporal",
        region_center=tuple(region_center), region_radius=region_radius,
        dedup_bin0=False, seed=42, use_weighted_sampler=False,
    )
    dm.setup()
    test_ds = dm.test_ds
    n_test = len(test_ds)
    stride = max(1, n_test // scan_rows)
    scored = []
    for ridx in range(0, n_test, stride):
        past_t, future_t = test_ds[ridx]
        truth = reverse_transform_R(future_t[0].float().numpy())
        scored.append((float((truth >= 1.0).mean()), past_t, truth))
    scored.sort(key=lambda x: -x[0])
    sel = scored[:n_cases]
    C = len(sel)
    print(f"  {C} cases\n")

    print("Building prob-nowcaster...")
    ae = autoenc.AutoencoderKL(encoder.SimpleConvEncoder(), encoder.SimpleConvDecoder())
    net = ProbNowcastNet(ae, thresholds=THRESHOLDS, embed_dim=128,
                         analysis_depth=4, forecast_depth=4, output_patches=Tf // 4)
    pmodel = ProbNowcaster(net, thresholds=THRESHOLDS)
    pmodel.load_state_dict(
        torch.load(prob_ckpt, map_location="cpu", weights_only=False)["state_dict"],
        strict=True)
    pnet = pmodel.net.to(dev).eval()
    t_rel = torch.arange(-Tp + 1, 1, dtype=torch.float32, device=dev).unsqueeze(0)

    # accumulators: full-domain (cross-check) + interior (headline)
    brier = {m: {t: [0.0, 0] for t in THRESHOLDS} for m in METHODS}
    brier_int = {m: {t: [0.0, 0] for t in THRESHOLDS} for m in METHODS}
    pc_int = {m: {t: [] for t in THRESHOLDS} for m in METHODS}

    def add(store, pcs, me, thr, field, of, mask):
        se = ((field - of) ** 2)[mask]
        s, n = float(se.sum()), int(mask.sum())
        store[me][thr][0] += s
        store[me][thr][1] += n
        if pcs is not None:
            pcs[me][thr].append((s, n))

    print("Scoring prob (one forward/case)...")
    for ci, (_, past_t, truth) in enumerate(sel):
        past_mmhr = reverse_transform_R(past_t[0].float().numpy())   # (Tp,H,W)
        t0 = past_mmhr[-1]
        dy, dx = estimate_motion(past_mmhr)
        int_m = interior_mask(dy, dx, Tf, truth.shape[1], truth.shape[2])
        full_m = np.ones_like(int_m)
        eul = np.repeat(t0[None], Tf, axis=0)
        lag_sv = np.stack([nd_shift(t0, (dy * (k + 1), dx * (k + 1)), order=1,
                                    mode="constant", cval=0.0) for k in range(Tf)])
        with torch.no_grad():
            past = past_t.unsqueeze(0).to(dev)                       # (1,1,Tp,H,W)
            P = torch.sigmoid(pnet([[past, t_rel]]))[0].float().cpu().numpy()
        for ti, thr in enumerate(THRESHOLDS):
            of = (truth >= thr).astype(np.float32)
            fields = {"prob": P[ti],
                      "lag_sv": (lag_sv >= thr).astype(np.float32),
                      "eul": (eul >= thr).astype(np.float32)}
            for me, p in fields.items():
                add(brier, None, me, thr, p, of, full_m)
                add(brier_int, pc_int, me, thr, p, of, int_m)
        if (ci + 1) % 100 == 0 or ci + 1 == C:
            print(f"  {ci+1}/{C}")

    def fin(store):
        return {str(t): {m: {"brier": store[m][t][0] / max(store[m][t][1], 1),
                             "npx": store[m][t][1]} for m in METHODS}
                for t in THRESHOLDS}
    summary = {"design": {"cases": C}, "brier": fin(brier),
               "brier_interior": fin(brier_int)}
    # BSS vs eulerian on the interior + the per-case lists (for STEPS-side CI parity)
    for t in THRESHOLDS:
        be = summary["brier_interior"][str(t)]["eul"]["brier"]
        for m in METHODS:
            bm = summary["brier_interior"][str(t)][m]["brier"]
            summary["brier_interior"][str(t)][m]["bss_vs_eul"] = (
                1 - bm / be if be else float("nan"))

    print("\n=== prob: FULL-domain Brier (must match ab_eval/summary.json) ===")
    ab = Path("../models/prob_nowcaster_aalborg/ab_eval/summary.json")
    abj = json.load(open(ab)) if ab.exists() else None
    for t in THRESHOLDS:
        f = summary["brier"][str(t)]["prob"]["brier"]
        ref = abj["brier"][str(t)]["prob"]["brier"] if abj else float("nan")
        d = abs(f - ref) if abj else float("nan")
        print(f"  {t}mm  prob full {f:.4f}  A/B {ref:.4f}  Δ{d:.5f}"
              f"{'  <-- MISMATCH' if abj and d >= 5e-4 else ''}")
    print("\n=== prob: FAIR-INTERIOR Brier (BSS vs Eulerian-interior) ===")
    for t in THRESHOLDS:
        r = summary["brier_interior"][str(t)]
        print(f"  {t}mm  prob {r['prob']['brier']:.4f}({r['prob']['bss_vs_eul']:+.3f})"
              f"  eul {r['eul']['brier']:.4f}  lag_sv {r['lag_sv']['brier']:.4f}"
              f"  | interior px/case ~{r['eul']['npx']//C}")

    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    Fire(run)
