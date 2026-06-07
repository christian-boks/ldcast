"""Plot the 256-context vs 128-baseline paired A/B (per-lead interior Brier).

Reads ctx_ab.json (eval_prob_ctx_ab.py) and shows, at 0.1 & 1.0 mm, the interior
Brier per lead for base (128 input) vs ctx (256 input) with the gap shaded — the
gap widens with lead time, recovering the inflow loss. Run in .venv-steps:
    ../.venv-steps/bin/python plot_ctx_ab.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

J = "../models/prob_nowcaster_aalborg_ctx256/ctx_ab.json"
d = json.load(open(J))
leads = np.arange(1, 9) * 10
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
for ax, thr in zip(axes, ("0.1", "1.0")):
    base = np.array(d["per_lead"][thr]["base"])
    ctx = np.array(d["per_lead"][thr]["ctx"])
    ov = d["interior"][thr]
    ax.axvspan(30, 40, color="gold", alpha=0.18, label="decision horizon (+30-40 min)")
    ax.fill_between(leads, ctx, base, color="#2ca02c", alpha=0.18)
    ax.plot(leads, base, "-o", color="#8c564b", lw=2, ms=5, label="128² input (baseline)")
    ax.plot(leads, ctx, "-o", color="#2ca02c", lw=2, ms=5, label="256² input (context)")
    ax.set_title(f"≥ {thr} mm/h   ·   overall interior Brier "
                 f"{ov['prob128']:.3f} → {ov['prob256']:.3f} "
                 f"({100*(ov['prob128']-ov['prob256'])/ov['prob128']:+.0f}%)", fontsize=11)
    ax.set_xlabel("lead time [min]")
    ax.grid(alpha=0.3)
    ax.set_xticks(leads)
axes[0].set_ylabel("interior Brier on the centre 128px  (lower = better)")
axes[0].legend(loc="upper left", fontsize=9, framealpha=0.95)
fig.suptitle("Wider input context recovers the inflow loss — paired A/B on identical "
             "centre-128 ground (400 wettest Aalborg test cases)\n"
             "the ctx gain grows with lead, tracking the unseen-inflow fraction "
             "(~0 near-term → 21-27% at +30-40 min → 55% at +80)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.92))
out = "../models/prob_nowcaster_aalborg_ctx256/ctx_ab.png"
fig.savefig(out, dpi=140)
print("saved", out)
