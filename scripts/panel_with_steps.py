"""Render the side-by-side panel for ONE case WITH a pysteps STEPS row added.

Reads ab_eval/panel_fields.npz (dumped by plot_side_by_side.py in the main venv:
truth, nowcaster P, diffusion PoP + PM-mean, and the case's past rain rates), runs
STEPS on that SAME case here (3.12 venv, pysteps), and renders, all 8 lead steps as
columns:
  Truth | Nowcaster P | Diffusion PoP | STEPS PoP | Diffusion PM-mean | STEPS PM-mean
so the three PROBABILITY fields (rows 2-4) and the two sharp ensemble fields
(rows 5-6) sit next to each other. Cyan = truth >= thr.

Usage (from scripts/):  ../.venv-steps/bin/python panel_with_steps.py
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from eval_steps_baseline import steps_forecast


def pm_mean(stack):
    """Probability-matched ensemble mean (Ebert 2001): keep the ensemble-mean's
    spatial ranking, reassign pixel values from the pooled member distribution —
    restores sharp peaks the plain mean blurs. stack (M,T,H,W) -> (T,H,W)."""
    M, T, H, W = stack.shape
    n = H * W
    out = np.empty((T, H, W), np.float32)
    for t in range(T):
        ens = stack[:, t].reshape(M, n)
        pooled = np.sort(ens.reshape(-1))
        q = pooled[np.linspace(0, pooled.size - 1, n).round().astype(int)]
        ranks = np.argsort(np.argsort(ens.mean(0)))
        out[t] = q[ranks].reshape(H, W)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="../models/prob_nowcaster_aalborg/ab_eval/panel_fields.npz")
    ap.add_argument("--out", default="../models/prob_nowcaster_aalborg/ab_eval/panel_with_steps.png")
    ap.add_argument("--members", type=int, default=32)
    ap.add_argument("--kmperpixel", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    past_mmhr, truth = z["past_mmhr"], z["truth"]
    P, pop, pm = z["P"], z["pop"], z["pm"]
    thr, wet, interval = float(z["thr"]), float(z["wet"]), int(z["interval"])
    info = str(z["info"])
    Tf = truth.shape[0]

    print(f"case: {info}; running STEPS ({args.members} members)...")
    ens, _ = steps_forecast(past_mmhr, Tf, args.members, args.kmperpixel,
                            interval, args.seed)
    steps_pop = (ens >= thr).mean(axis=0).astype(np.float32)
    steps_pm = pm_mean(ens)

    rows = [("Truth", "rain", truth),
            (f"Nowcaster P≥{thr:g}", "prob", P),
            (f"Diffusion PoP≥{thr:g}", "prob", pop),
            (f"STEPS PoP≥{thr:g}", "prob", steps_pop),
            ("Diffusion PM-mean", "rain", pm),
            ("STEPS PM-mean", "rain", steps_pm)]
    rain_cmap = plt.get_cmap("turbo").with_extremes(bad="white")

    fig, axes = plt.subplots(len(rows), Tf, figsize=(2.0 * Tf, 2.0 * len(rows)))
    im_rain = im_prob = None
    for r, (name, kind, arr) in enumerate(rows):
        for k in range(Tf):
            ax = axes[r, k]
            ax.set_xticks([]); ax.set_yticks([])
            if kind == "rain":
                im_rain = ax.imshow(np.ma.masked_less(arr[k], 0.1), cmap=rain_cmap,
                                    norm=LogNorm(vmin=0.1, vmax=20))
            else:
                im_prob = ax.imshow(arr[k], cmap="magma", vmin=0, vmax=1)
                ax.contour(truth[k] >= thr, levels=[0.5], colors="cyan", linewidths=0.6)
            if r == 0:
                ax.set_title(f"+{(k + 1) * interval} min", fontsize=10)
        axes[r, 0].set_ylabel(name, fontsize=10)

    fig.suptitle(f"Nowcaster vs diffusion vs pysteps STEPS — one Aalborg test case "
                 f"({info}); cyan = truth ≥{thr:g} mm/h", fontsize=12)
    fig.tight_layout(rect=(0.01, 0.04, 1, 0.97))
    fig.colorbar(im_rain, cax=fig.add_axes([0.30, 0.015, 0.22, 0.012]),
                 orientation="horizontal").set_label("rain rate [mm/h]", fontsize=8)
    fig.colorbar(im_prob, cax=fig.add_axes([0.62, 0.015, 0.22, 0.012]),
                 orientation="horizontal").set_label(f"P(rain ≥{thr:g})", fontsize=8)
    fig.savefig(args.out, dpi=110)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
