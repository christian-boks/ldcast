"""Paired A/B: 128-input prob-nowcaster vs 256-input (context) prob-nowcaster.

The 2026-06-02 diagnosis: at +30-40 min, 21-27% of the 128px crop is dry inflow
the model can't see. This tests the fix — a model that sees a 256px window and
predicts the centre 128px. PERFECTLY PAIRED: for each test case we load a 128px
window (truth + baseline input) AND the 256px window centred on it (context-model
input), so both models are scored on the IDENTICAL centre-128 ground truth; the
only difference is how much context the input carried.

Both are interior-scored (pixels whose advective source was inside the *128px*
crop — the region where the baseline could compete) so the question is sharp: does
the wider input recover Brier on exactly the pixels the baseline loses to inflow?
Reports overall + per-lead (the +30-40 min leads are the use case), with a paired
bootstrap CI on the per-case Brier difference.

Run in the MAIN venv with .venv/bin/python (NOT uv run) AFTER training stops (it
needs the GPU). Usage (from scripts/):
    ../.venv/bin/python eval_prob_ctx_ab.py --prob256_ckpt=<best ctx256 ckpt>
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
from ldcast.features.rust_data import mmhr_rainrate_transform
from ldcast.models.autoenc import autoenc, encoder
from ldcast.models.nowcast.prob_nowcast import ProbNowcastNet, ProbNowcaster
from ldcast.visualization.plots import reverse_transform_R
from eval_persistence_baseline import estimate_motion
from eval_prob_interior import interior_mask

THRESHOLDS = (0.1, 1.0, 5.0)
BASE = 128                       # target/output crop size
PAD = 64                         # -> 256px context window


def _load_model(ckpt, dev):
    ae = autoenc.AutoencoderKL(encoder.SimpleConvEncoder(), encoder.SimpleConvDecoder())
    net = ProbNowcastNet(ae, thresholds=THRESHOLDS, embed_dim=128,
                         analysis_depth=4, forecast_depth=4, output_patches=2)
    ProbNowcaster(net, thresholds=THRESHOLDS).load_state_dict(
        torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"])
    return net.to(dev).eval()


def run(
    config="../config/train_rust.yaml",
    prob128_ckpt="../models/prob_nowcaster_aalborg/last.ckpt",
    prob256_ckpt="../models/prob_nowcaster_aalborg_ctx256/last.ckpt",
    n_cases=400,
    scan_rows=5000,
    region_center=(685, 852),
    region_radius=64,
    test_frac=0.1,
    valid_frac=0.1,
    max_nocoverage_frac=0.05,
    out=None,
):
    cfg = OmegaConf.load(config)
    if cfg.get("radar_root"):
        os.environ["DGMR_RADAR_ROOT"] = cfg.radar_root
    out = Path(out or "../models/prob_nowcaster_aalborg_ctx256/ctx_ab.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tp, Tf = int(cfg.past_steps), int(cfg.future_steps)

    # ---- test catalog (same temporal split as everything else) ----
    from ldcast.features.rust_data import _load_ldcast_index
    (_, _, (ts, x, y, _)) = _load_ldcast_index(
        cfg.index_path, valid_frac, 42, test_frac, "temporal",
        tuple(region_center), region_radius, dedup_bin0=False)
    print(f"test catalog: {ts.size} entries")

    cache = dgmr_py.FrameCache(64)
    tf = mmhr_rainrate_transform()

    def load_win(i, pad):
        """(1,Tp,H,W),(1,Tf,H,W) mm/h-space for the (BASE+2pad) window at entry i."""
        H = BASE + 2 * pad
        entry = dgmr_py.make_entry(int(ts[i]), int(x[i]) - pad, int(y[i]) - pad)
        past, future = dgmr_py.load_sample(entry, cache, Tp, Tf, H, H, False)
        full = np.concatenate([past, future], axis=1)
        if float(np.mean(full < 0.0)) > max_nocoverage_frac:
            raise RuntimeError("off-radar")
        full = tf(full).astype(np.float32)
        return full[:, :Tp], full[:, Tp:]

    # ---- paired selection: only entries where BOTH windows load cleanly ----
    print("Scanning + pairing (128 & 256 windows must both load)...")
    stride = max(1, ts.size // scan_rows)
    cands = []
    skipped = 0
    for i in range(0, ts.size, stride):
        try:
            p128, f128 = load_win(i, 0)
            p256, _ = load_win(i, PAD)
        except RuntimeError:
            skipped += 1
            continue
        truth = reverse_transform_R(f128[0])                 # (Tf,128,128) mm/h
        cands.append((float((truth >= 1.0).mean()), p128, p256, truth))
    cands.sort(key=lambda c: -c[0])
    cases = cands[:n_cases]
    C = len(cases)
    print(f"  {C} paired cases (skipped {skipped} for off-radar/gap in either window)\n")

    net128 = _load_model(prob128_ckpt, dev)
    net256 = _load_model(prob256_ckpt, dev)
    t_rel = torch.arange(-Tp + 1, 1, dtype=torch.float32, device=dev).unsqueeze(0)
    o = PAD                                                   # centre-crop offset (256->128)

    METHODS = ("prob256", "prob128", "lag_sv", "eul")
    # overall-interior + per-lead-interior accumulators + per-case for paired CI
    bi = {m: {t: [0.0, 0] for t in THRESHOLDS} for m in METHODS}
    bl = {m: {(t, k): [0.0, 0] for t in THRESHOLDS for k in range(Tf)} for m in METHODS}
    pcase = {m: {t: [] for t in THRESHOLDS} for m in METHODS}   # (sse,n) per case, interior

    @torch.no_grad()
    def pop(net, past_np):
        past = torch.from_numpy(past_np).unsqueeze(0).to(dev)   # (1,1,Tp,H,W)
        return torch.sigmoid(net([[past, t_rel]]))[0].float().cpu().numpy()

    print("Scoring...")
    for ci, (_, p128, p256, truth) in enumerate(cases):
        t0 = reverse_transform_R(p128[0])[-1]                # (128,128) mm/h
        dy, dx = estimate_motion(reverse_transform_R(p128[0]))
        int_m = interior_mask(dy, dx, Tf, BASE, BASE)        # (Tf,128,128) bool
        eul = np.repeat(t0[None], Tf, axis=0)
        lag_sv = np.stack([nd_shift(t0, (dy * (k + 1), dx * (k + 1)), order=1,
                                    mode="constant", cval=0.0) for k in range(Tf)])
        P256 = pop(net256, p256)[:, :, o:o + BASE, o:o + BASE]   # (3,Tf,128,128)
        P128 = pop(net128, p128)                                  # (3,Tf,128,128)
        for ti, thr in enumerate(THRESHOLDS):
            of = (truth >= thr).astype(np.float32)
            fields = {"prob256": P256[ti], "prob128": P128[ti],
                      "lag_sv": (lag_sv >= thr).astype(np.float32),
                      "eul": (eul >= thr).astype(np.float32)}
            for m, p in fields.items():
                se = (p - of) ** 2
                s_in = float(se[int_m].sum()); n_in = int(int_m.sum())
                bi[m][thr][0] += s_in; bi[m][thr][1] += n_in
                pcase[m][thr].append((s_in, n_in))
                for k in range(Tf):
                    mk = int_m[k]
                    bl[m][(thr, k)][0] += float(se[k][mk].sum())
                    bl[m][(thr, k)][1] += int(mk.sum())
        if (ci + 1) % 100 == 0 or ci + 1 == C:
            print(f"  {ci+1}/{C}")

    _report(out, C, Tf, bi, bl, pcase, METHODS)


def _paired_ci(pcase, a, b, thr, n_boot=2000, seed=0):
    """Bootstrap 95% CI for Brier(b) - Brier(a) over cases (positive => a better)."""
    rng = np.random.default_rng(seed)
    A = np.array(pcase[a][thr]); B = np.array(pcase[b][thr])
    n = A.shape[0]
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ba = A[idx, 0].sum() / max(A[idx, 1].sum(), 1)
        bb = B[idx, 0].sum() / max(B[idx, 1].sum(), 1)
        d.append(bb - ba)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _report(out, C, Tf, bi, bl, pcase, METHODS):
    def b(store, m, key):
        return store[m][key][0] / max(store[m][key][1], 1)

    print("\n=== INTERIOR Brier on the centre 128px (lower=better); paired ===")
    print(f"  {'thr':>4} | {'prob256(ctx)':>13} {'prob128(base)':>13} "
          f"{'eul':>8} {'lag_sv':>8} | Δ(base-ctx) [95% CI]")
    summary = {"design": {"cases": C}, "interior": {}}
    for thr in THRESHOLDS:
        b256, b128 = b(bi, "prob256", thr), b(bi, "prob128", thr)
        lo, hi = _paired_ci(pcase, "prob256", "prob128", thr)
        summary["interior"][str(thr)] = {
            "prob256": b256, "prob128": b128, "eul": b(bi, "eul", thr),
            "lag_sv": b(bi, "lag_sv", thr), "delta_base_minus_ctx": b128 - b256,
            "delta_ci": [lo, hi]}
        flag = "  <-- ctx wins" if lo > 0 else ("  <-- base wins" if hi < 0 else "  (overlaps 0)")
        print(f"  {thr:>4} | {b256:>13.4f} {b128:>13.4f} {b(bi,'eul',thr):>8.4f} "
              f"{b(bi,'lag_sv',thr):>8.4f} | {b128-b256:+.4f} [{lo:+.4f},{hi:+.4f}]{flag}")

    print("\n=== per-lead INTERIOR Brier @0.1 / 1.0 mm (ctx vs base) ===")
    summary["per_lead"] = {}
    for thr in (0.1, 1.0):
        print(f"  {thr}mm  lead:    " + "  ".join(f"+{(k+1)*10:>3}m" for k in range(Tf)))
        c = [b(bl, "prob256", (thr, k)) for k in range(Tf)]
        bb = [b(bl, "prob128", (thr, k)) for k in range(Tf)]
        print(f"        ctx  : " + "  ".join(f"{v:.3f}" for v in c))
        print(f"        base : " + "  ".join(f"{v:.3f}" for v in bb))
        print(f"        Δ    : " + "  ".join(f"{bb[k]-c[k]:+.3f}" for k in range(Tf)))
        summary["per_lead"][str(thr)] = {"ctx": c, "base": bb}

    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    Fire(run)
