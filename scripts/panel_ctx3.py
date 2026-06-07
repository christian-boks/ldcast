"""3-way per-case IMAGES: 128 vs 256 vs 384-context prob-nowcaster on one case.

Picks a fast-advecting Aalborg case and renders, all 8 leads as columns:
  Truth (centre) | 128² PoP | 256² PoP | 384² PoP | Δ(384−256)
All PoP rows are the centre-128 of each model's native window, so they're directly
comparable. The 256 and 384 rows look near-identical (context saturated); the Δ row
(red = 384 higher) shows the small differences emerging at long leads. Cyan = truth.

Run in MAIN venv (.venv/bin/python, NOT uv run), GPU free. From scripts/:
    DGMR_RADAR_ROOT=/opt/radar_data ../.venv/bin/python panel_ctx3.py --rank=0
"""
import os
import numpy as np
import torch
from fire import Fire
from omegaconf import OmegaConf
import matplotlib; matplotlib.use("Agg")
from matplotlib import pyplot as plt

import dgmr_py
from ldcast.features.rust_data import mmhr_rainrate_transform, _load_ldcast_index
from ldcast.visualization.plots import reverse_transform_R, plot_precip_image
from eval_persistence_baseline import estimate_motion
from eval_prob_ctx_ab import _load_model

THR, THR_IDX, BASE, INTERVAL = 0.1, 0, 128, 10


def run(config="../config/train_rust.yaml",
        ckpt128="../models/prob_nowcaster_aalborg/last.ckpt",
        ckpt256="../models/prob_nowcaster_aalborg_ctx256/epoch=24-val_brier=0.0143.ckpt",
        ckpt384="../models/prob_nowcaster_aalborg_ctx384/epoch=25-val_brier=0.0150.ckpt",
        rank=0, scan_rows=3000, wet_lo=0.10, wet_hi=0.30,
        speed_min=13.0, speed_max=18.0,
        region_center=(685, 852), region_radius=64, out=None):
    cfg = OmegaConf.load(config)
    if cfg.get("radar_root"):
        os.environ["DGMR_RADAR_ROOT"] = cfg.radar_root
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tp, Tf = int(cfg.past_steps), int(cfg.future_steps)
    (_, _, (ts, x, y, _)) = _load_ldcast_index(
        cfg.index_path, 0.1, 42, 0.1, "temporal",
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

    print("Selecting a fast-advecting case (all 3 windows must load)...")
    stride = max(1, ts.size // scan_rows)
    cands = []
    for i in range(0, ts.size, stride):
        try:
            p0, f0 = load_win(i, 0)
            load_win(i, 128)                      # ensure the 384 window also loads
        except RuntimeError:
            continue
        truth = reverse_transform_R(f0[0]); past = reverse_transform_R(p0[0])
        wet = float((truth >= THR).mean())
        if not (wet_lo <= wet <= wet_hi):
            continue
        dy, dx = estimate_motion(past); spd = float(np.hypot(dy, dx))
        if speed_min <= spd <= speed_max:
            cands.append((spd, i, wet))
    cands.sort(key=lambda c: -c[0])
    spd, i, wet = cands[rank]
    print(f"  i={i} speed={spd:.1f}px/step wet={wet:.2f} (rank {rank}/{len(cands)})")

    p0, f0 = load_win(i, 0); p64, _ = load_win(i, 64); p128, _ = load_win(i, 128)
    truth = reverse_transform_R(f0[0])
    net = {"128": _load_model(ckpt128, dev), "256": _load_model(ckpt256, dev),
           "384": _load_model(ckpt384, dev)}
    t_rel = torch.arange(-Tp + 1, 1, dtype=torch.float32, device=dev).unsqueeze(0)

    @torch.no_grad()
    def pop(m, past_np):
        pt = torch.from_numpy(past_np).unsqueeze(0).to(dev)
        return torch.sigmoid(net[m]([[pt, t_rel]]))[0].float().cpu().numpy()[THR_IDX]
    P128 = pop("128", p0)
    P256 = pop("256", p64)[:, 64:192, 64:192]
    P384 = pop("384", p128)[:, 128:256, 128:256]
    Pdiff = P384 - P256

    rows = [("Truth", "rain", truth, None),
            ("128² PoP", "prob", P128, truth),
            ("256² PoP", "prob", P256, truth),
            ("384² PoP", "prob", P384, truth),
            ("Δ 384−256", "diff", Pdiff, truth)]
    fig, axes = plt.subplots(5, Tf, figsize=(2.0 * Tf, 2.0 * 5))
    for r, (name, kind, arr, contour) in enumerate(rows):
        for k in range(Tf):
            ax = axes[r, k]; ax.set_xticks([]); ax.set_yticks([])
            if kind == "rain":
                plot_precip_image(ax, arr[k].copy())
            elif kind == "diff":
                ax.imshow(arr[k], cmap="bwr", vmin=-0.5, vmax=0.5)
                ax.contour(contour[k] >= THR, levels=[0.5], colors="black", linewidths=0.5)
            else:
                ax.imshow(arr[k], cmap="magma", vmin=0, vmax=1)
                ax.contour(contour[k] >= THR, levels=[0.5], colors="cyan", linewidths=0.6)
            if r == 0:
                ax.set_title(f"+{(k + 1) * INTERVAL} min", fontsize=10)
        axes[r, 0].set_ylabel(name, fontsize=11, rotation=90, va="center",
                              labelpad=8 if r % 2 == 0 else 30)
    fig.suptitle(f"128 vs 256 vs 384 context — fast-advecting Aalborg case "
                 f"(motion {spd:.0f} px/step, wet {wet:.2f}); cyan = truth ≥{THR:g} mm/h. "
                 f"256² & 384² look near-identical (context saturated); Δ row red = 384 higher PoP",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=0.075, hspace=0.08)
    out = out or "../models/prob_nowcaster_aalborg_ctx384/panel_ctx3.png"
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    Fire(run)
