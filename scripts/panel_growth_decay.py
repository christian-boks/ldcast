"""Judge-for-yourself growth & decay examples — LOW-MOTION cases where the rain
area changes IN PLACE, so it reads as formation / dissipation, not advection.

Selects (over the Aalborg test cases, |motion| small so it isn't just sliding):
  growth = t0 has a seed of rain, total rain area GROWS >= 2x over +10..+80
  decay  = t0 raining, total rain area SHRINKS to <= 0.45x
Renders per mode: Truth | Model PoP | Lagrangian advection (8 leads) + a wet-area-
vs-lead plot — truth grows/shrinks; does the model follow? Lagrangian stays flat
(advection can't create or destroy rain), so it's the bar.

Run in MAIN venv (.venv/bin/python), GPU free. From scripts/:
    DGMR_RADAR_ROOT=/opt/radar_data ../.venv/bin/python panel_growth_decay.py
"""
import os
import numpy as np
import torch
from fire import Fire
from omegaconf import OmegaConf
from scipy.ndimage import shift as nd_shift
import matplotlib; matplotlib.use("Agg")
from matplotlib import pyplot as plt

import dgmr_py
from ldcast.features.rust_data import mmhr_rainrate_transform, _load_ldcast_index
from ldcast.visualization.plots import reverse_transform_R, plot_precip_image
from eval_persistence_baseline import estimate_motion
from eval_prob_ctx_ab import _load_model

BASE, PAD, INTERVAL = 128, 64, 10


def run(config="../config/train_rust.yaml",
        ckpt256="../models/prob_nowcaster_aalborg_ctx256/epoch=24-val_brier=0.0143.ckpt",
        thr=0.1, scan_rows=4000, speed_max=5.0,
        grow_ratio=2.0, decay_ratio=0.45,
        region_center=(685, 852), region_radius=64, test_frac=0.1, valid_frac=0.1):
    cfg = OmegaConf.load(config)
    if cfg.get("radar_root"):
        os.environ["DGMR_RADAR_ROOT"] = cfg.radar_root
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
        if float(np.mean(full < 0.0)) > 0.05:
            raise RuntimeError("nocov")
        full = tfm(full).astype(np.float32)
        return full[:, :Tp], full[:, Tp:]

    print(f"Selecting LOW-MOTION (|v|<{speed_max}) in-place growth/decay cases...")
    stride = max(1, ts.size // scan_rows)
    best = {"growth": (None, None), "decay": (None, None)}
    for i in range(0, ts.size, stride):
        try:
            p0, f0 = load_win(i, 0)
        except RuntimeError:
            continue
        truth = reverse_transform_R(f0[0]); past0 = reverse_transform_R(p0[0])
        dy, dx = estimate_motion(past0)
        if float(np.hypot(dy, dx)) > speed_max:
            continue
        w0 = float((past0[-1] >= thr).mean())
        wend = float((truth[-1] >= thr).mean())
        if w0 < 1e-3:
            continue
        ratio = wend / w0
        if 0.02 <= w0 <= 0.20 and ratio >= grow_ratio and wend >= 0.06:
            if best["growth"][0] is None or ratio > best["growth"][0]:
                best["growth"] = (ratio, i)
        if 0.10 <= w0 <= 0.45 and ratio <= decay_ratio:
            if best["decay"][0] is None or ratio < best["decay"][0]:
                best["decay"] = (ratio, i)

    net = _load_model(ckpt256, dev)
    t_rel = torch.arange(-Tp + 1, 1, dtype=torch.float32, device=dev).unsqueeze(0)
    thr_idx = (0.1, 1.0, 5.0).index(thr)

    @torch.no_grad()
    def pop(past_np):
        pt = torch.from_numpy(past_np).unsqueeze(0).to(dev)
        return torch.sigmoid(net([[pt, t_rel]]))[0].float().cpu().numpy()[thr_idx]

    leads = np.arange(1, Tf + 1) * INTERVAL
    for mode in ("growth", "decay"):
        ratio, i = best[mode]
        if i is None:
            print(f"  no clean {mode} case found (try relaxing speed_max / ratio)")
            continue
        p0, f0 = load_win(i, 0); p64, _ = load_win(i, PAD)
        truth = reverse_transform_R(f0[0]); past0 = reverse_transform_R(p0[0]); t0 = past0[-1]
        dy, dx = estimate_motion(past0)
        lag = np.stack([nd_shift(t0, (dy * (k + 1), dx * (k + 1)), order=1,
                                 mode="constant", cval=0.0) for k in range(Tf)])
        Pc = pop(p64)[:, PAD:PAD + BASE, PAD:PAD + BASE]
        w0 = float((t0 >= thr).mean())
        a_truth = [float((truth[k] >= thr).mean()) for k in range(Tf)]
        a_lag = [float((lag[k] >= thr).mean()) for k in range(Tf)]
        a_model = [float((Pc[k] >= 0.5).mean()) for k in range(Tf)]
        print(f"  {mode}: i={i} area t0={w0:.3f} -> +80={a_truth[-1]:.3f} ({ratio:.1f}x)")

        fig = plt.figure(figsize=(2.0 * Tf, 8.4))
        gs = fig.add_gridspec(4, Tf, height_ratios=[2, 2, 2, 2.4], hspace=0.12)
        spatial = [("Truth", "rain", truth), ("Model PoP", "prob", Pc),
                   ("Lagrangian", "rain", lag)]
        for r, (name, kind, arr) in enumerate(spatial):
            for k in range(Tf):
                ax = fig.add_subplot(gs[r, k]); ax.set_xticks([]); ax.set_yticks([])
                if kind == "rain":
                    plot_precip_image(ax, arr[k].copy())
                else:
                    ax.imshow(arr[k], cmap="magma", vmin=0, vmax=1)
                if r == 0:
                    ax.set_title(f"+{(k + 1) * INTERVAL}min", fontsize=9)
                if k == 0:
                    ax.set_ylabel(name, fontsize=11)
        axL = fig.add_subplot(gs[3, :])
        axL.axhline(w0, ls=":", color="gray", lw=1, label="t0 rain area")
        axL.plot(leads, a_truth, "-o", color="black", lw=2.2, label="Truth")
        axL.plot(leads, a_model, "-o", color="#2ca02c", lw=2.2, label="Model (PoP≥0.5)")
        axL.plot(leads, a_lag, "-o", color="#8c564b", lw=2.2, label="Lagrangian (advection)")
        axL.set_xlabel("lead time [min]"); axL.set_ylabel(f"rain-area frac (≥{thr:g}mm)")
        axL.set_xticks(leads); axL.grid(alpha=0.3); axL.legend(fontsize=9, ncol=4, loc="upper center")
        verb = "GROWS" if mode == "growth" else "SHRINKS"
        fig.suptitle(f"{mode.upper()} — low-motion ({np.hypot(dy,dx):.0f} px/step) Aalborg case, ≥{thr:g} mm/h: "
                     f"truth rain area {verb} {w0:.3f}→{a_truth[-1]:.3f}. "
                     f"Does the model (green) follow? Lagrangian (brown) can't.", fontsize=10.5)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = f"../models/prob_nowcaster_aalborg_ctx256/example_{mode}.png"
        fig.savefig(out, dpi=110); print("  saved", out)


if __name__ == "__main__":
    Fire(run)
