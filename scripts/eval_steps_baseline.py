"""The honest advection bar: pysteps STEPS ensemble vs the prob/diffusion models.

Runs in the isolated Python-3.12 venv (.venv-steps) — pure numpy/scipy/pysteps,
no torch/rust. Reads the cases steps_extract_cases.py dumped (the SAME wettest
Aalborg held-out TEST cases eval_aalborg_ab.py scored) and scores three
advection-based forecasters the A/B never had:

  steps     : pysteps STEPS ensemble PoP (dense Lucas-Kanade flow + 6-level scale
              cascade + stochastic perturbations) — the operational gold-standard
              *probabilistic* advection. The fair ensemble-vs-ensemble bar.
  lag_lk    : deterministic dense-LK semi-Lagrangian extrapolation, scored 0/1 —
              proper optical-flow advection (the A/B's `lag` was a single global
              FFT vector; this is what actually competes).
  lag_nbhd  : neighbourhood probability from lag_lk (Gaussian-smoothed exceedance)
              — calibrated-by-spread "persistence as a probability".

HEADLINE = the FAIR INTERIOR (2026-06-02 confound fix). The full 128px domain
penalises advection for dry inflow it can't see (21-27% of the domain at +30-40
min) and flatters persistence. The interior keeps only pixels whose advective
source (current pos - motion*lead) was inside the crop at t0 — the region a
radar-only forecast could possibly get right. The mask is forecaster-agnostic
(single global t0 vector, as the genesis/decay masks), so STEPS, prob, and
persistence are judged on identical pixels. prob's interior Brier comes from
eval_prob_interior.py (prob_interior.json); the eulerian-interior Brier is
recomputed here and asserted to match it — proof the masks/cases line up.

Scoring is otherwise byte-identical to eval_aalborg_ab.py (same THRESHOLDS,
genesis/decay masks, Brier/BSS, bootstrap CI). kmperpixel for the DMI mosaic is
unconfirmed; it only scales STEPS' stochastic velocity perturbation (lag_lk /
lag_nbhd are pixel-unit invariant). Exposed as --kmperpixel.

Usage (from scripts/, the 3.12 venv):
    ../.venv-steps/bin/python eval_steps_baseline.py --limit 12      # smoke
    ../.venv-steps/bin/python eval_steps_baseline.py                 # full 400
"""
import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, shift as nd_shift, distance_transform_edt

from pysteps import nowcasts
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.utils import transformation

THRESHOLDS = (0.1, 1.0, 5.0)
METHODS = ("steps", "lag_lk", "lag_nbhd", "lag_sv", "eul")
SAFETY_FACTOR = 1.5          # advection reach = SAFETY * |motion| * lead  (== A/B)
GENESIS_MARGIN_PX = 8
DB_ZERO, DB_THR = -15.0, -10.0   # dB_transform(threshold=0.1) -> 0.1 mm/h == -10 dB


def estimate_motion(past):
    """Single global phase-correlation vector == eval_persistence_baseline; the
    genesis/decay/interior masks are defined from this so they match the A/B."""
    if float((past[-1] > 0.1).mean()) < 0.005:
        return 0.0, 0.0
    H, W = past[-1].shape
    win = np.hanning(H)[:, None] * np.hanning(W)[None, :]
    dys, dxs = [], []
    for a, b in zip(past[:-1], past[1:]):
        A = np.fft.fft2(gaussian_filter(a, 1.0) * win)
        B = np.fft.fft2(gaussian_filter(b, 1.0) * win)
        R = B * np.conj(A)
        R /= np.abs(R) + 1e-8
        r = np.fft.ifft2(R).real
        dy, dx = np.unravel_index(int(np.argmax(r)), r.shape)
        if dy > H // 2:
            dy -= H
        if dx > W // 2:
            dx -= W
        dys.append(dy); dxs.append(dx)
    dy, dx = float(np.mean(dys)), float(np.mean(dxs))
    return float(np.clip(dy, -20.0, 20.0)), float(np.clip(dx, -20.0, 20.0))


def interior_mask(dy, dx, Tf, H, W):
    """(Tf,H,W) bool: advective source (pos - motion*lead) is inside the crop.
    Forecaster-agnostic; identical formula to eval_prob_interior.interior_mask."""
    ys = np.arange(H)[:, None].astype(np.float64)
    xs = np.arange(W)[None, :].astype(np.float64)
    m = np.zeros((Tf, H, W), dtype=bool)
    for k in range(Tf):
        sy = ys - dy * (k + 1)
        sx = xs - dx * (k + 1)
        m[k] = (sy >= 0) & (sy <= H - 1) & (sx >= 0) & (sx <= W - 1)
    return m


def _bss_ci(pc, me, ref, thr, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.array(pc[me][thr], dtype=np.float64)
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


def steps_forecast(past_mmhr, Tf, n_members, kmperpixel, timestep, seed):
    """STEPS ensemble (M,Tf,H,W) mm/h + deterministic LK extrapolation (Tf,H,W).
    Mirrors ldcast.models.benchmarks.pysteps.predict_sample (dB space, 6 cascades,
    nonparametric noise, bps vel-pert, incremental mask) but timestep=10 min."""
    H, W = past_mmhr.shape[-2:]
    R, _ = transformation.dB_transform(past_mmhr.astype(np.float64),
                                       threshold=0.1, zerovalue=DB_ZERO)
    R[~np.isfinite(R)] = DB_ZERO
    if (R <= DB_ZERO).all():
        zero = np.zeros((Tf, H, W), dtype=np.float32)
        return np.repeat(zero[None], n_members, axis=0), zero
    with redirect_stdout(io.StringIO()):
        V = dense_lucaskanade(R)
        det_db = nowcasts.get_method("extrapolation")(R[-1], V, Tf)
    det = np.nan_to_num(transformation.dB_transform(det_db, threshold=DB_THR,
                                                    inverse=True)[0], nan=0.0).astype(np.float32)
    try:
        with redirect_stdout(io.StringIO()):
            ens_db = nowcasts.get_method("steps")(
                R, V, Tf, n_ens_members=n_members, n_cascade_levels=6,
                precip_thr=DB_THR, kmperpixel=kmperpixel, timestep=timestep,
                noise_method="nonparametric", vel_pert_method="bps",
                mask_method="incremental", seed=seed, num_workers=4)
        ens = np.nan_to_num(transformation.dB_transform(ens_db, threshold=DB_THR,
                                                        inverse=True)[0], nan=0.0).astype(np.float32)
    except (ValueError, RuntimeError):
        ens = np.repeat(det[None], n_members, axis=0)
    return ens, det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="../models/prob_nowcaster_aalborg/ab_eval/steps_cases.npz")
    ap.add_argument("--ab_summary", default="../models/prob_nowcaster_aalborg/ab_eval/summary.json")
    ap.add_argument("--prob_interior", default="../models/prob_nowcaster_aalborg/ab_eval/prob_interior.json")
    ap.add_argument("--out", default="../models/prob_nowcaster_aalborg/ab_eval/steps_summary.json")
    ap.add_argument("--n_members", type=int, default=32)
    ap.add_argument("--kmperpixel", type=float, default=2.0)
    ap.add_argument("--nbhd_sigma", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    z = np.load(args.cases)
    past_all, truth_all = z["past"], z["truth"]
    timestep = int(z["timestep_min"])
    C = past_all.shape[0] if not args.limit else min(args.limit, past_all.shape[0])
    Tf = truth_all.shape[1]
    print(f"cases {C}/{past_all.shape[0]} | {Tf} leads | timestep {timestep}min | "
          f"members {args.n_members} | kmperpixel {args.kmperpixel}\n")

    brier = {m: {t: [0.0, 0] for t in THRESHOLDS} for m in METHODS}        # full domain
    brier_int = {m: {t: [0.0, 0] for t in THRESHOLDS} for m in METHODS}    # fair interior
    brier_gen = {m: {t: [0.0, 0] for t in THRESHOLDS} for m in METHODS}
    brier_dec = {m: {t: [0.0, 0] for t in THRESHOLDS} for m in METHODS}
    pc = {m: {t: [] for t in THRESHOLDS} for m in METHODS}
    pc_int = {m: {t: [] for t in THRESHOLDS} for m in METHODS}
    pc_gen = {m: {t: [] for t in THRESHOLDS} for m in METHODS}
    gd_gen = {m: {t: [0, 0] for t in THRESHOLDS} for m in METHODS}
    gd_dec = {m: {t: [0, 0] for t in THRESHOLDS} for m in METHODS}

    def add(store, pcs, me, thr, field, of, mask=None):
        se = (field - of) ** 2
        if mask is not None:
            se = se[mask]; n = int(mask.sum())
        else:
            n = of.size
        s = float(se.sum())
        store[me][thr][0] += s; store[me][thr][1] += n
        if pcs is not None:
            pcs[me][thr].append((s, n))

    for ci in range(C):
        past, truth = past_all[ci], truth_all[ci]
        H, W = truth.shape[-2:]
        t0 = past[-1]
        dy, dx = estimate_motion(past)
        speed = float(np.hypot(dy, dx))
        int_m = interior_mask(dy, dx, Tf, H, W)
        eul = np.repeat(t0[None], Tf, axis=0)
        lag_sv = np.stack([nd_shift(t0, (dy * (k + 1), dx * (k + 1)), order=1,
                                    mode="constant", cval=0.0) for k in range(Tf)])
        ens, det = steps_forecast(past, Tf, args.n_members, args.kmperpixel,
                                  timestep, args.seed + ci)
        for thr in THRESHOLDS:
            o = truth >= thr
            of = o.astype(np.float32)
            dist0 = distance_transform_edt(~(t0 >= thr)).astype(np.float32)
            gen_m = np.zeros_like(o); dec_m = np.zeros_like(o)
            for k in range(Tf):
                reach = max(GENESIS_MARGIN_PX, SAFETY_FACTOR * speed * (k + 1))
                gen_m[k] = (lag_sv[k] < thr) & (dist0 > reach)
                dec_m[k] = lag_sv[k] >= thr
            gen_ev, dec_ev = gen_m & o, dec_m & (~o)
            det_ex = (det >= thr).astype(np.float32)
            fields = {
                "steps": (ens >= thr).mean(axis=0).astype(np.float32),
                "lag_lk": det_ex,
                "lag_nbhd": np.clip(np.stack(
                    [gaussian_filter(det_ex[k], args.nbhd_sigma) for k in range(Tf)]
                ), 0.0, 1.0).astype(np.float32),
                "lag_sv": (lag_sv >= thr).astype(np.float32),
                "eul": (eul >= thr).astype(np.float32),
            }
            for me, p in fields.items():
                add(brier, pc, me, thr, p, of)
                add(brier_int, pc_int, me, thr, p, of, mask=int_m)
                add(brier_gen, pc_gen, me, thr, p, of, mask=gen_m)
                add(brier_dec, None, me, thr, p, of, mask=dec_m)
                pb = p >= 0.5
                gd_gen[me][thr][0] += int((pb & gen_ev).sum())
                gd_gen[me][thr][1] += int(gen_ev.sum())
                gd_dec[me][thr][0] += int((~pb & dec_ev).sum())
                gd_dec[me][thr][1] += int(dec_ev.sum())
        if (ci + 1) % 25 == 0 or ci + 1 == C:
            print(f"  scored {ci+1}/{C}")

    report(args, C, brier, brier_int, brier_gen, brier_dec,
           pc, pc_int, pc_gen, gd_gen, gd_dec)


def _b(store, me, thr):
    return store[me][thr][0] / max(store[me][thr][1], 1)


def report(args, C, brier, brier_int, brier_gen, brier_dec,
           pc, pc_int, pc_gen, gd_gen, gd_dec):
    ab = json.load(open(args.ab_summary)) if Path(args.ab_summary).exists() else None
    pj = json.load(open(args.prob_interior)) if Path(args.prob_interior).exists() else None

    if not args.limit:
        print("\n=== CROSS-CHECK (must match → same cases & masks) ===")
        if ab:
            for thr in THRESHOLDS:
                for ours, th in (("eul", "eul"), ("lag_sv", "lag")):
                    d = abs(_b(brier, ours, thr) - ab["brier"][str(thr)][th]["brier"])
                    print(f"  full   {thr}mm {ours:>6}: Δ{d:.5f}"
                          f"{'  <-- MISMATCH' if d >= 5e-4 else ''}")
        if pj:
            for thr in THRESHOLDS:
                d = abs(_b(brier_int, "eul", thr)
                        - pj["brier_interior"][str(thr)]["eul"]["brier"])
                print(f"  interior {thr}mm    eul: Δ{d:.5f}"
                      f"{'  <-- MISMATCH' if d >= 5e-4 else ''}")

    def bss(store, me, thr, ref):
        bm, br = _b(store, me, thr), _b(store, ref, thr)
        return bm, (1 - bm / br if br else float("nan"))

    print("\n############ HEADLINE: FAIR-INTERIOR Brier (BSS vs Eulerian-interior) "
          "############")
    print("  (only pixels whose advective source was inside the crop — advection "
          "no longer\n   penalised for unseen inflow; prob from eval_prob_interior.py)")
    print(f"  {'thr':>4} | {'STEPS':>17} {'lag_lk':>15} {'lag_nbhd':>15} "
          f"{'prob-nowcast':>15} {'eul':>9}")
    summary = {"design": {"cases": C, "members": args.n_members,
                          "kmperpixel": args.kmperpixel}, "brier_interior": {}}
    for thr in THRESHOLDS:
        rec = {}
        row = [f"  {thr:>4} |"]
        for me in ("steps", "lag_lk", "lag_nbhd"):
            bm, b = bss(brier_int, me, thr, "eul")
            rec[me] = {"brier": bm, "bss_vs_eul": b}
            row.append(f" {bm:.4f}({b:+.3f})")
        rec["steps"]["bss_ci"] = _bss_ci(pc_int, "steps", "eul", thr)
        if pj:
            pr = pj["brier_interior"][str(thr)]["prob"]
            row.append(f" {pr['brier']:.4f}({pr['bss_vs_eul']:+.3f})")
        row.append(f" {_b(brier_int,'eul',thr):.4f}")
        summary["brier_interior"][str(thr)] = rec
        print("".join(row))
        lo, hi = rec["steps"]["bss_ci"]
        print(f"         STEPS BSS CI [{lo:+.3f},{hi:+.3f}]")

    print("\n=== full-domain Brier (CONFOUNDED by inflow — reference only) ===")
    summary["brier_full"] = {}
    for thr in THRESHOLDS:
        rec = {}
        row = [f"  {thr:>4} |"]
        for me in ("steps", "lag_lk", "lag_nbhd"):
            bm, b = bss(brier, me, thr, "eul")
            rec[me] = {"brier": bm, "bss_vs_eul": b}
            row.append(f" {bm:.4f}({b:+.3f})")
        if ab:
            for me in ("prob", "diff_cal"):
                r = ab["brier"][str(thr)][me]
                row.append(f" {me} {r['brier']:.4f}({r['bss_vs_eul']:+.3f})")
        summary["brier_full"][str(thr)] = rec
        print("".join(row))

    print("\n=== GENESIS / DECAY POD @0.5 (full domain) ===")
    summary["pod"] = {}
    for thr in THRESHOLDS:
        rec = {}
        row = [f"  {thr:>4} |"]
        for me in ("steps", "lag_lk"):
            gc, gt = gd_gen[me][thr]; dc, dt = gd_dec[me][thr]
            gp = gc / gt if gt else float("nan")
            dp = dc / dt if dt else float("nan")
            rec[me] = {"genesis_pod": gp, "decay_pod": dp}
            row.append(f" {me}: gen {gp:.3f} dec {dp:.3f} |")
        if ab and "pod" in ab:
            r = ab["pod"][str(thr)]["prob"]
            row.append(f" prob: gen {r['genesis_pod']:.3f} dec {r['decay_pod']:.3f}")
        summary["pod"][str(thr)] = rec
        print("".join(row))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
