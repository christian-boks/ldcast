"""Side-by-side IMAGES: 128 baseline vs 256-context prob-nowcaster on one case.

Picks a fast-advecting Aalborg test case (where inflow is large) and renders, all 8
leads as columns:
  Truth (centre) | Baseline 128² PoP | Context 256² PoP (centre) | Context PoP (FULL
  256² window, with the 128² the baseline sees boxed)
The last row is the mechanism: rain in the collar OUTSIDE the dashed box is what the
baseline can't see and the context model exploits. Cyan = truth >= thr.

Run in MAIN venv (.venv/bin/python, NOT uv run). Usage (from scripts/):
    DGMR_RADAR_ROOT=/opt/radar_data ../.venv/bin/python panel_ctx.py --rank=0
"""
import os
import numpy as np
import torch
from fire import Fire
from omegaconf import OmegaConf
import matplotlib; matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

import dgmr_py
from ldcast.features.rust_data import mmhr_rainrate_transform, _load_ldcast_index
from ldcast.visualization.plots import reverse_transform_R, plot_precip_image
from eval_persistence_baseline import estimate_motion
from eval_prob_ctx_ab import _load_model

THR, THR_IDX, BASE, PAD, INTERVAL = 0.1, 0, 128, 64, 10


def run(config="../config/train_rust.yaml",
        prob128_ckpt="../models/prob_nowcaster_aalborg/last.ckpt",
        prob256_ckpt="../models/prob_nowcaster_aalborg_ctx256/epoch=24-val_brier=0.0143.ckpt",
        rank=0, scan_rows=3000, wet_lo=0.10, wet_hi=0.30,
        speed_min=11.0, speed_max=16.0,
        region_center=(685, 852), region_radius=64, out=None):
    cfg = OmegaConf.load(config)
    if cfg.get("radar_root"):
        os.environ["DGMR_RADAR_ROOT"] = cfg.radar_root
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tp, Tf = int(cfg.past_steps), int(cfg.future_steps)
    (_, _, (ts, x, y, _)) = _load_ldcast_index(
        cfg.index_path, 0.1, 42, 0.1, "temporal",
        tuple(region_center), region_radius, dedup_bin0=False)
    cache = dgmr_py.FrameCache(64); tf = mmhr_rainrate_transform()

    def load_win(i, pad):
        H = BASE + 2 * pad
        e = dgmr_py.make_entry(int(ts[i]), int(x[i]) - pad, int(y[i]) - pad)
        past, future = dgmr_py.load_sample(e, cache, Tp, Tf, H, H, False)
        full = np.concatenate([past, future], axis=1)
        if float(np.mean(full < 0.0)) > 0.05:
            raise RuntimeError("nocov")
        full = tf(full).astype(np.float32)
        return full[:, :Tp], full[:, Tp:]

    print("Selecting a fast-advecting case...")
    stride = max(1, ts.size // scan_rows)
    cands = []
    for i in range(0, ts.size, stride):
        try:
            p128, f128 = load_win(i, 0)
        except RuntimeError:
            continue
        truth = reverse_transform_R(f128[0]); past = reverse_transform_R(p128[0])
        wet = float((truth >= THR).mean())
        if not (wet_lo <= wet <= wet_hi):
            continue
        dy, dx = estimate_motion(past); spd = float(np.hypot(dy, dx))
        if speed_min <= spd <= speed_max:
            cands.append((spd, i, wet))
    cands.sort(key=lambda c: -c[0])
    spd, i, wet = cands[rank]
    print(f"  i={i} speed={spd:.1f}px/step wet={wet:.2f} (rank {rank}/{len(cands)})")

    p128, f128 = load_win(i, 0)
    p256, f256 = load_win(i, PAD)
    truth = reverse_transform_R(f128[0])           # (Tf,128,128)
    truth256 = reverse_transform_R(f256[0])        # (Tf,256,256)
    net128 = _load_model(prob128_ckpt, dev)
    net256 = _load_model(prob256_ckpt, dev)
    t_rel = torch.arange(-Tp + 1, 1, dtype=torch.float32, device=dev).unsqueeze(0)

    @torch.no_grad()
    def pop(net, past_np):
        pt = torch.from_numpy(past_np).unsqueeze(0).to(dev)
        return torch.sigmoid(net([[pt, t_rel]]))[0].float().cpu().numpy()
    P128 = pop(net128, p128)[THR_IDX]              # (Tf,128,128)
    P256f = pop(net256, p256)[THR_IDX]             # (Tf,256,256)
    P256c = P256f[:, PAD:PAD + BASE, PAD:PAD + BASE]
    Pdiff = P256c - P128                           # >0 (red): context raises PoP

    rows = [("Truth", "rain", truth, None),
            ("Baseline 128²", "prob", P128, truth),
            ("Context 256²", "prob", P256c, truth),
            ("Δ  ctx − base", "diff", Pdiff, truth),
            ("Context full 256²", "full", P256f, truth256)]
    fig, axes = plt.subplots(5, Tf, figsize=(2.0 * Tf, 2.0 * 5))
    for r, (name, kind, arr, contour) in enumerate(rows):
        for k in range(Tf):
            ax = axes[r, k]; ax.set_xticks([]); ax.set_yticks([])
            if kind == "rain":
                plot_precip_image(ax, arr[k].copy())
            elif kind == "diff":
                ax.imshow(arr[k], cmap="bwr", vmin=-0.6, vmax=0.6)
                ax.contour(contour[k] >= THR, levels=[0.5], colors="black", linewidths=0.5)
            else:
                ax.imshow(arr[k], cmap="magma", vmin=0, vmax=1)
                if contour is not None:
                    ax.contour(contour[k] >= THR, levels=[0.5], colors="cyan", linewidths=0.6)
                if kind == "full":
                    ax.add_patch(Rectangle((PAD, PAD), BASE, BASE, fill=False,
                                           edgecolor="white", lw=1.3, ls="--"))
            if r == 0:
                ax.set_title(f"+{(k + 1) * INTERVAL} min", fontsize=10)
        # stagger alternate row labels horizontally so the (rotated) text can't
        # overlap into neighbouring rows
        axes[r, 0].set_ylabel(name, fontsize=11, rotation=90, va="center",
                              labelpad=8 if r % 2 == 0 else 30)
    fig.suptitle(f"128² baseline vs 256² context — fast-advecting Aalborg case "
                 f"(motion {spd:.0f} px/step, wet {wet:.2f}); cyan = truth ≥{THR:g} mm/h; "
                 f"dashed box = the 128² the baseline sees", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=0.075, hspace=0.08)
    out = out or "../models/prob_nowcaster_aalborg_ctx256/panel_ctx.png"
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    Fire(run)
