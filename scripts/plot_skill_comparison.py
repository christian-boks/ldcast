"""Bar-chart the prob/diffusion/STEPS/advection Brier-skill comparison.

Reads the three result JSONs (ab_eval/{summary,prob_interior,steps_summary}.json)
and renders BSS-vs-Eulerian as grouped bars, full-domain | fair-interior, so the
inflow-confound removal is visible. Bar label = raw Brier (lower=better).
Run in the 3.12 venv (has matplotlib): ../.venv-steps/bin/python plot_skill_comparison.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = "../models/prob_nowcaster_aalborg/ab_eval/"
ab = json.load(open(B + "summary.json"))
pj = json.load(open(B + "prob_interior.json"))
st = json.load(open(B + "steps_summary.json"))
T = ["0.1", "1.0", "5.0"]
TLAB = ["0.1 mm", "1.0 mm", "5.0 mm"]

METHODS = [
    ("prob",  "prob-nowcaster",     "#2ca02c"),
    ("diff",  "diffusion (recal)",  "#ff7f0e"),
    ("steps", "STEPS (gold std)",   "#9467bd"),
    ("lk",    "LK advection",       "#8c8c8c"),
]


def full(m, t):
    d = {"prob": ab["brier"][t]["prob"], "diff": ab["brier"][t]["diff_cal"],
         "steps": st["brier_full"][t]["steps"], "lk": st["brier_full"][t]["lag_lk"]}[m]
    return d["bss_vs_eul"], d["brier"]


def interior(m, t):
    if m == "diff":
        return None, None
    d = {"prob": pj["brier_interior"][t]["prob"],
         "steps": st["brier_interior"][t]["steps"],
         "lk": st["brier_interior"][t]["lag_lk"]}[m]
    return d["bss_vs_eul"], d["brier"]


fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
PANELS = [("Full domain", "penalises advection for inflow it can't see", full),
          ("Fair interior", "inflow removed — the honest comparison", interior)]
x = np.arange(len(T))
w = 0.2
for ax, (title, sub, fn) in zip(axes, PANELS):
    for i, (key, label, color) in enumerate(METHODS):
        xs = x + (i - 1.5) * w
        for xp, t in zip(xs, T):
            bss, br = fn(key, t)
            if bss is None:
                ax.text(xp, 0.015, "n/a", ha="center", va="bottom", fontsize=7,
                        rotation=90, color="gray")
                continue
            ax.bar(xp, bss, w, color=color, edgecolor="black", linewidth=0.4,
                   label=label if t == "0.1" else None, zorder=3)
            ax.text(xp, bss + (0.014 if bss >= 0 else -0.014), f"{br:.3f}",
                    ha="center", va="bottom" if bss >= 0 else "top", fontsize=6.8)
    ax.axhline(0, color="black", lw=1.1, zorder=4)
    ax.set_title(f"{title}\n{sub}", fontsize=10.5)
    ax.set_xticks(x); ax.set_xticklabels(TLAB)
    ax.set_xlabel("rain threshold")
    ax.grid(axis="y", alpha=0.3, zorder=0)
axes[0].set_ylabel("Brier Skill Score vs Eulerian\n(higher = better; 0 = ties 'hold last frame')")
axes[0].annotate("Eulerian baseline", (2.32, 0), (2.0, -0.22), fontsize=8,
                 arrowprops=dict(arrowstyle="->", lw=0.8))
axes[0].legend(loc="lower left", fontsize=8.5, framealpha=0.95)
fig.suptitle("Probability-of-rain skill — prob-nowcaster vs diffusion vs pysteps STEPS vs advection\n"
             "400 wettest Aalborg held-out test cases · 32 members · bar label = raw Brier (lower=better)",
             fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
out = B + "skill_comparison.png"
fig.savefig(out, dpi=145)
print("saved", out)
