"""3-way paired A/B: 128 vs 256 vs 384-context prob-nowcaster (centre-128 ground).

Generalises eval_prob_ctx_ab.py. Per test case, loads the windows at pad 0/64/128
(→ 128²/256²/384² inputs, all CENTRED on the same 128² crop, all must load clean),
runs each model on its native input size, crops every output to the centre 128², and
interior-scores all three on the IDENTICAL centre-128 truth (overall + per-lead). The
256→384 delta is the test of whether pushing context past 256² still pays.

Run in MAIN venv (.venv/bin/python, NOT uv run), GPU free. From scripts/:
    DGMR_RADAR_ROOT=/opt/radar_data ../.venv/bin/python eval_prob_ctx3.py
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from omegaconf import OmegaConf
from scipy.ndimage import shift as nd_shift

import dgmr_py
from ldcast.features.rust_data import mmhr_rainrate_transform, _load_ldcast_index
from ldcast.visualization.plots import reverse_transform_R
from eval_persistence_baseline import estimate_motion
from eval_prob_interior import interior_mask
from eval_prob_ctx_ab import _load_model

THRESHOLDS = (0.1, 1.0, 5.0)
BASE = 128


def run(config="../config/train_rust.yaml",
        ckpt128="../models/prob_nowcaster_aalborg/last.ckpt",
        ckpt256="../models/prob_nowcaster_aalborg_ctx256/epoch=24-val_brier=0.0143.ckpt",
        ckpt384="../models/prob_nowcaster_aalborg_ctx384/epoch=25-val_brier=0.0150.ckpt",
        n_cases=400, scan_rows=5000, region_center=(685, 852), region_radius=64,
        test_frac=0.1, valid_frac=0.1, max_nocoverage_frac=0.05, out=None):
    cfg = OmegaConf.load(config)
    if cfg.get("radar_root"):
        os.environ["DGMR_RADAR_ROOT"] = cfg.radar_root
    out = Path(out or "../models/prob_nowcaster_aalborg_ctx384/ctx3_ab.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tp, Tf = int(cfg.past_steps), int(cfg.future_steps)
    MODELS = [("128", ckpt128, 0), ("256", ckpt256, 64), ("384", ckpt384, 128)]
    pads = [p for _, _, p in MODELS]

    (_, _, (ts, x, y, _)) = _load_ldcast_index(
        cfg.index_path, valid_frac, 42, test_frac, "temporal",
        tuple(region_center), region_radius, dedup_bin0=False)
    cache = dgmr_py.FrameCache(64); tfm = mmhr_rainrate_transform()

    def load_win(i, pad):
        H = BASE + 2 * pad
        e = dgmr_py.make_entry(int(ts[i]), int(x[i]) - pad, int(y[i]) - pad)
        past, future = dgmr_py.load_sample(e, cache, Tp, Tf, H, H, False)
        full = np.concatenate([past, future], axis=1)
        if float(np.mean(full < 0.0)) > max_nocoverage_frac:
            raise RuntimeError("nocov")
        full = tfm(full).astype(np.float32)
        return full[:, :Tp], full[:, Tp:]

    print(f"test catalog {ts.size}; pairing all 3 windows (pad 0/64/128)...")
    stride = max(1, ts.size // scan_rows)
    cands = []; skip = 0
    for i in range(0, ts.size, stride):
        try:
            data = {p: load_win(i, p) for p in pads}     # {pad:(past,future)}
        except RuntimeError:
            skip += 1; continue
        truth = reverse_transform_R(data[0][1][0])       # (Tf,128,128) from pad-0 future
        cands.append((float((truth >= 1.0).mean()), {p: data[p][0] for p in pads}, truth))
    cands.sort(key=lambda c: -c[0])
    cases = cands[:n_cases]; C = len(cases)
    print(f"  {C} paired cases (skipped {skip} where any window was off-radar)\n")

    nets = {lbl: _load_model(ck, dev) for lbl, ck, _ in MODELS}
    padmap = {lbl: p for lbl, _, p in MODELS}
    METHODS = [lbl for lbl, _, _ in MODELS]
    t_rel = torch.arange(-Tp + 1, 1, dtype=torch.float32, device=dev).unsqueeze(0)

    bi = {m: {t: [0.0, 0] for t in THRESHOLDS} for m in METHODS}
    bl = {m: {(t, k): [0.0, 0] for t in THRESHOLDS for k in range(Tf)} for m in METHODS}
    pcase = {m: {t: [] for t in THRESHOLDS} for m in METHODS}

    @torch.no_grad()
    def pop(net, past_np):
        pt = torch.from_numpy(past_np).unsqueeze(0).to(dev)
        return torch.sigmoid(net([[pt, t_rel]]))[0].float().cpu().numpy()

    print("Scoring 3 models on identical centre-128 ground...")
    for ci, (_, pasts, truth) in enumerate(cases):
        dy, dx = estimate_motion(reverse_transform_R(pasts[0][0]))
        int_m = interior_mask(dy, dx, Tf, BASE, BASE)
        P = {}
        for lbl in METHODS:
            pad = padmap[lbl]
            Pf = pop(nets[lbl], pasts[pad])                 # (3,Tf,H,H)
            P[lbl] = Pf[:, :, pad:pad + BASE, pad:pad + BASE] if pad else Pf
        for ti, thr in enumerate(THRESHOLDS):
            of = (truth >= thr).astype(np.float32)
            for lbl in METHODS:
                se = (P[lbl][ti] - of) ** 2
                s, n = float(se[int_m].sum()), int(int_m.sum())
                bi[lbl][thr][0] += s; bi[lbl][thr][1] += n
                pcase[lbl][thr].append((s, n))
                for k in range(Tf):
                    mk = int_m[k]
                    bl[lbl][(thr, k)][0] += float(se[k][mk].sum())
                    bl[lbl][(thr, k)][1] += int(mk.sum())
        if (ci + 1) % 100 == 0 or ci + 1 == C:
            print(f"  {ci+1}/{C}")

    _report(out, C, Tf, bi, bl, pcase, METHODS)


def _paired_ci(pcase, a, b, thr, n_boot=2000, seed=0):
    """95% CI for Brier(a) - Brier(b) over cases (positive => b better than a)."""
    rng = np.random.default_rng(seed)
    A = np.array(pcase[a][thr]); B = np.array(pcase[b][thr]); n = A.shape[0]
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d.append(A[idx, 0].sum() / max(A[idx, 1].sum(), 1)
                 - B[idx, 0].sum() / max(B[idx, 1].sum(), 1))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _report(out, C, Tf, bi, bl, pcase, METHODS):
    def b(store, m, key):
        return store[m][key][0] / max(store[m][key][1], 1)

    print("=== INTERIOR Brier on centre-128 (lower=better) — 128 vs 256 vs 384 ===")
    print(f"  {'thr':>4} | {'128':>8} {'256':>8} {'384':>8} | "
          f"{'128→256':>16} {'256→384':>16}")
    summary = {"design": {"cases": C}, "interior": {}, "per_lead": {}}
    for thr in THRESHOLDS:
        b128, b256, b384 = b(bi, "128", thr), b(bi, "256", thr), b(bi, "384", thr)
        lo1, hi1 = _paired_ci(pcase, "128", "256", thr)    # +ve => 256 better
        lo2, hi2 = _paired_ci(pcase, "256", "384", thr)    # +ve => 384 better
        summary["interior"][str(thr)] = {
            "128": b128, "256": b256, "384": b384,
            "d_128_256": b128 - b256, "ci_128_256": [lo1, hi1],
            "d_256_384": b256 - b384, "ci_256_384": [lo2, hi2]}
        tag = "384 wins" if lo2 > 0 else ("256 wins" if hi2 < 0 else "tie")
        print(f"  {thr:>4} | {b128:>8.4f} {b256:>8.4f} {b384:>8.4f} | "
              f"{b128-b256:+.4f}[{lo1:+.3f},{hi1:+.3f}] "
              f"{b256-b384:+.4f}[{lo2:+.3f},{hi2:+.3f}] {tag}")

    print("\n=== per-lead INTERIOR Brier @0.1 / 1.0 mm (128 / 256 / 384) ===")
    for thr in (0.1, 1.0):
        print(f"  {thr}mm  lead:  " + "  ".join(f"+{(k+1)*10:>3}m" for k in range(Tf)))
        for lbl in METHODS:
            vals = [b(bl, lbl, (thr, k)) for k in range(Tf)]
            print(f"        {lbl:>4}: " + "  ".join(f"{v:.3f}" for v in vals))
            summary["per_lead"].setdefault(str(thr), {})[lbl] = vals

    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    Fire(run)
