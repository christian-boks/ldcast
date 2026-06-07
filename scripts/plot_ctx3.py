"""Plot the 3-way context A/B (128/256/384) per-lead interior Brier — the crossover.

Reads ctx3_ab.json. Run in .venv-steps: ../.venv-steps/bin/python plot_ctx3.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("../models/prob_nowcaster_aalborg_ctx384/ctx3_ab.json"))
leads = np.arange(1, 9) * 10
COL = {"128": "#8c564b", "256": "#2ca02c", "384": "#1f77b4"}
LAB = {"128": "128² (baseline)", "256": "256² context", "384": "384² context"}
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
for ax, thr in zip(axes, ("0.1", "1.0")):
    ax.axvspan(30, 40, color="gold", alpha=0.18, label="decision horizon (+30-40 min)")
    for lbl in ("128", "256", "384"):
        ax.plot(leads, d["per_lead"][thr][lbl], "-o", color=COL[lbl], lw=2, ms=5, label=LAB[lbl])
    iv = d["interior"][thr]
    ax.set_title(f"≥ {thr} mm/h   ·   overall interior Brier  "
                 f"128 {iv['128']:.3f} → 256 {iv['256']:.3f} → 384 {iv['384']:.3f}", fontsize=10.5)
    ax.set_xlabel("lead time [min]"); ax.grid(alpha=0.3); ax.set_xticks(leads)
axes[0].set_ylabel("interior Brier on the centre 128px  (lower = better)")
axes[0].legend(loc="upper left", fontsize=9, framealpha=0.95)
fig.suptitle("Pushing context further: 256² → 384² — gains move to the LONG leads, with a small "
             "short-lead cost\n256² is the sweet spot for the +30-40 min use case; 384² only pulls "
             "ahead at +50-80 min (light rain), and is worse at 5 mm", fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.92))
out = "../models/prob_nowcaster_aalborg_ctx384/ctx3_ab.png"
fig.savefig(out, dpi=140); print("saved", out)
