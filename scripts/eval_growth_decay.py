"""Growth (genesis) & decay (dissipation) skill of the prob-nowcaster — obj 4.

The part advection physically cannot do. On the wettest Aalborg test cases scores
the 256² core (and the 128² model + persistence as references) on:

  GENESIS pixels  = advection predicts dry (lag<thr) AND > 1.5·|motion|·lead + 8px
                    from any t0 echo  → rain there is true formation, not a band
                    moving in (the 06-01 corrected definition).
  DECAY pixels    = advection predicts rain (lag>=thr) → does the model anticipate
                    clearing where it actually clears?

Metrics per threshold (+ per-lead at 1.0mm):
  - Brier on the event pixels, BSS vs the right bar (genesis: vs Eulerian≈climatology;
    decay: vs Lagrangian, which keeps the rain by construction).
  - POD@0.5: genesis caught (rain-from-nowhere with PoP≥0.5); clearing caught
    (decay pixel that cleared with PoP<0.5).
  - DISCRIMINATION (the cleanest signal): mean PoP on event vs non-event pixels —
    does the model put MORE probability where genesis happens / LESS where rain clears?
  Bootstrap 95% CI over cases on the 256 genesis/decay Brier-BSS.

Paired loading (pad 0 = truth+128 input+masks, pad 64 = 256 input), identical
centre-128 ground. Run in MAIN venv (.venv/bin/python), GPU free. From scripts/:
    DGMR_RADAR_ROOT=/opt/radar_data ../.venv/bin/python eval_growth_decay.py
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from omegaconf import OmegaConf
from scipy.ndimage import shift as nd_shift, distance_transform_edt

import dgmr_py
from ldcast.features.rust_data import mmhr_rainrate_transform, _load_ldcast_index
from ldcast.visualization.plots import reverse_transform_R
from eval_persistence_baseline import estimate_motion
from eval_prob_ctx_ab import _load_model

THRESHOLDS = (0.1, 1.0, 5.0)
BASE, SAFETY, MARGIN = 128, 1.5, 8
METHODS = ("prob256", "prob128", "lag", "eul")


def run(config="../config/train_rust.yaml",
        ckpt256="../models/prob_nowcaster_aalborg_ctx256/epoch=24-val_brier=0.0143.ckpt",
        ckpt128="../models/prob_nowcaster_aalborg/last.ckpt",
        n_cases=400, scan_rows=5000, region_center=(685, 852), region_radius=64,
        test_frac=0.1, valid_frac=0.1, max_nocoverage_frac=0.05, out=None):
    cfg = OmegaConf.load(config)
    if cfg.get("radar_root"):
        os.environ["DGMR_RADAR_ROOT"] = cfg.radar_root
    out = Path(out or "../models/prob_nowcaster_aalborg_ctx256/growth_decay.json")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tp, Tf = int(cfg.past_steps), int(cfg.future_steps)
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

    print(f"test catalog {ts.size}; pairing 128 & 256 windows...")
    stride = max(1, ts.size // scan_rows)
    cands = []
    for i in range(0, ts.size, stride):
        try:
            p0, f0 = load_win(i, 0); p64, _ = load_win(i, 64)
        except RuntimeError:
            continue
        truth = reverse_transform_R(f0[0])
        cands.append((float((truth >= 1.0).mean()), p0, p64, truth))
    cands.sort(key=lambda c: -c[0])
    cases = cands[:n_cases]; C = len(cases)
    print(f"  {C} paired cases\n")

    net256 = _load_model(ckpt256, dev); net128 = _load_model(ckpt128, dev)
    t_rel = torch.arange(-Tp + 1, 1, dtype=torch.float32, device=dev).unsqueeze(0)

    @torch.no_grad()
    def pop(net, past_np):
        pt = torch.from_numpy(past_np).unsqueeze(0).to(dev)
        return torch.sigmoid(net([[pt, t_rel]]))[0].float().cpu().numpy()

    # accumulators
    def mk():
        return {m: {t: [0.0, 0] for t in THRESHOLDS} for m in METHODS}
    bg, bd = mk(), mk()                                   # Brier on genesis / decay pixels
    bg_lead = {m: {(t, k): [0.0, 0] for t in THRESHOLDS for k in range(Tf)} for m in METHODS}
    bd_lead = {m: {(t, k): [0.0, 0] for t in THRESHOLDS for k in range(Tf)} for m in METHODS}
    pod_g = {m: {t: [0, 0] for t in THRESHOLDS} for m in METHODS}   # caught, total events
    pod_d = {m: {t: [0, 0] for t in THRESHOLDS} for m in METHODS}
    disc_g = {m: {t: [0.0, 0, 0.0, 0] for t in THRESHOLDS} for m in METHODS}  # popEv,nEv, popNon,nNon
    disc_d = {m: {t: [0.0, 0, 0.0, 0] for t in THRESHOLDS} for m in METHODS}
    pc_g = {m: {t: [] for t in THRESHOLDS} for m in METHODS}  # per-case (sse,n) for CI
    pc_d = {m: {t: [] for t in THRESHOLDS} for m in METHODS}

    print("Scoring growth/decay...")
    for ci, (_, p0, p64, truth) in enumerate(cases):
        past0 = reverse_transform_R(p0[0]); t0 = past0[-1]
        dy, dx = estimate_motion(past0); speed = float(np.hypot(dy, dx))
        lag = np.stack([nd_shift(t0, (dy * (k + 1), dx * (k + 1)), order=1,
                                 mode="constant", cval=0.0) for k in range(Tf)])
        eul = np.repeat(t0[None], Tf, axis=0)
        P256 = pop(net256, p64)[:, :, 64:192, 64:192]
        P128 = pop(net128, p0)
        for ti, thr in enumerate(THRESHOLDS):
            o = truth >= thr
            dist0 = distance_transform_edt(~(t0 >= thr)).astype(np.float32)
            gen_m = np.zeros_like(o); dec_m = np.zeros_like(o)
            for k in range(Tf):
                reach = max(MARGIN, SAFETY * speed * (k + 1))
                gen_m[k] = (lag[k] < thr) & (dist0 > reach)
                dec_m[k] = lag[k] >= thr
            of = o.astype(np.float32)
            fields = {"prob256": P256[ti], "prob128": P128[ti],
                      "lag": (lag >= thr).astype(np.float32),
                      "eul": (eul >= thr).astype(np.float32)}
            for m, p in fields.items():
                se = (p - of) ** 2
                # genesis Brier (+per-case, +per-lead)
                sg = float(se[gen_m].sum()); ng = int(gen_m.sum())
                bg[m][thr][0] += sg; bg[m][thr][1] += ng; pc_g[m][thr].append((sg, ng))
                sd = float(se[dec_m].sum()); nd = int(dec_m.sum())
                bd[m][thr][0] += sd; bd[m][thr][1] += nd; pc_d[m][thr].append((sd, nd))
                for k in range(Tf):
                    bg_lead[m][(thr, k)][0] += float(se[k][gen_m[k]].sum())
                    bg_lead[m][(thr, k)][1] += int(gen_m[k].sum())
                    bd_lead[m][(thr, k)][0] += float(se[k][dec_m[k]].sum())
                    bd_lead[m][(thr, k)][1] += int(dec_m[k].sum())
                # events
                gen_ev = gen_m & o; gen_non = gen_m & (~o)
                dec_ev = dec_m & (~o); dec_non = dec_m & o
                pb = p >= 0.5
                pod_g[m][thr][0] += int((pb & gen_ev).sum()); pod_g[m][thr][1] += int(gen_ev.sum())
                pod_d[m][thr][0] += int((~pb & dec_ev).sum()); pod_d[m][thr][1] += int(dec_ev.sum())
                disc_g[m][thr][0] += float(p[gen_ev].sum()); disc_g[m][thr][1] += int(gen_ev.sum())
                disc_g[m][thr][2] += float(p[gen_non].sum()); disc_g[m][thr][3] += int(gen_non.sum())
                disc_d[m][thr][0] += float(p[dec_ev].sum()); disc_d[m][thr][1] += int(dec_ev.sum())
                disc_d[m][thr][2] += float(p[dec_non].sum()); disc_d[m][thr][3] += int(dec_non.sum())
        if (ci + 1) % 100 == 0 or ci + 1 == C:
            print(f"  {ci+1}/{C}")

    _report(out, C, Tf, bg, bd, bg_lead, bd_lead, pod_g, pod_d, disc_g, disc_d, pc_g, pc_d)


def _bss_ci(pc, m, ref, thr, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.array(pc[m][thr], float); b = np.array(pc[ref][thr], float)
    v = []
    for _ in range(n_boot):
        idx = rng.integers(0, a.shape[0], a.shape[0])
        bm = a[idx, 0].sum() / max(a[idx, 1].sum(), 1)
        br = b[idx, 0].sum() / max(b[idx, 1].sum(), 1)
        v.append(1 - bm / br if br else np.nan)
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def _report(out, C, Tf, bg, bd, bg_lead, bd_lead, pod_g, pod_d, disc_g, disc_d, pc_g, pc_d):
    def br(s, m, k):
        return s[m][k][0] / max(s[m][k][1], 1)
    summ = {"design": {"cases": C}}

    print("=== GROWTH (genesis): Brier on true-formation pixels, BSS vs Eulerian ===")
    print(f"  {'thr':>4} | {'prob256':>16} {'prob128':>16} | {'POD@.5 256/128':>16} "
          f"| {'mean PoP rained/dry (256)':>26}")
    summ["genesis"] = {}
    for thr in THRESHOLDS:
        be = br(bg, "eul", thr)
        b256, b128 = br(bg, "prob256", thr), br(bg, "prob128", thr)
        s256 = 1 - b256 / be if be else float("nan"); s128 = 1 - b128 / be if be else float("nan")
        lo, hi = _bss_ci(pc_g, "prob256", "eul", thr)
        g = pod_g["prob256"][thr]; g1 = pod_g["prob128"][thr]
        pod256 = g[0] / g[1] if g[1] else float("nan"); pod128 = g1[0] / g1[1] if g1[1] else float("nan")
        d = disc_g["prob256"][thr]
        pev = d[0] / d[1] if d[1] else float("nan"); pno = d[2] / d[3] if d[3] else float("nan")
        summ["genesis"][str(thr)] = {"brier256": b256, "bss256": s256, "bss256_ci": [lo, hi],
                                     "brier128": b128, "bss128": s128,
                                     "pod256": pod256, "pod128": pod128,
                                     "popPoP_rained": pev, "poP_dry": pno, "n_events": d[1]}
        print(f"  {thr:>4} | {b256:.4f}({s256:+.3f})  {b128:.4f}({s128:+.3f}) | "
              f"{pod256:.3f} / {pod128:.3f}  | {pev:.3f} vs {pno:.3f}  (n={d[1]})")
        print(f"        256 BSS 95% CI [{lo:+.3f}, {hi:+.3f}]")

    print("\n=== DECAY (dissipation): Brier on advected-rain pixels, BSS vs Lagrangian ===")
    print(f"  {'thr':>4} | {'prob256':>16} {'prob128':>16} | {'clearPOD 256/128':>16} "
          f"| {'mean PoP cleared/stayed (256)':>28}")
    summ["decay"] = {}
    for thr in THRESHOLDS:
        bl = br(bd, "lag", thr)
        b256, b128 = br(bd, "prob256", thr), br(bd, "prob128", thr)
        s256 = 1 - b256 / bl if bl else float("nan"); s128 = 1 - b128 / bl if bl else float("nan")
        lo, hi = _bss_ci(pc_d, "prob256", "lag", thr)
        d0 = pod_d["prob256"][thr]; d1 = pod_d["prob128"][thr]
        cp256 = d0[0] / d0[1] if d0[1] else float("nan"); cp128 = d1[0] / d1[1] if d1[1] else float("nan")
        dd = disc_d["prob256"][thr]
        pev = dd[0] / dd[1] if dd[1] else float("nan"); pno = dd[2] / dd[3] if dd[3] else float("nan")
        summ["decay"][str(thr)] = {"brier256": b256, "bss256": s256, "bss256_ci": [lo, hi],
                                   "brier128": b128, "bss128": s128,
                                   "clearpod256": cp256, "clearpod128": cp128,
                                   "poP_cleared": pev, "poP_stayed": pno, "n_events": dd[1]}
        print(f"  {thr:>4} | {b256:.4f}({s256:+.3f})  {b128:.4f}({s128:+.3f}) | "
              f"{cp256:.3f} / {cp128:.3f}  | {pev:.3f} vs {pno:.3f}  (n={dd[1]})")

    print("\n=== per-lead skill at 1.0mm (BSS): genesis vs Eul, decay vs Lag ===")
    summ["per_lead_1mm"] = {}
    for name, sl, ref in (("genesis", bg_lead, "eul"), ("decay", bd_lead, "lag")):
        gen = [1 - br(sl, "prob256", (1.0, k)) / max(br(sl, ref, (1.0, k)), 1e-9) for k in range(Tf)]
        summ["per_lead_1mm"][name] = gen
        print(f"  {name:>8}: " + "  ".join(f"{v:+.2f}" for v in gen))

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(summ, fh, indent=2, default=float)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    Fire(run)
