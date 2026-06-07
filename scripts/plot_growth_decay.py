"""Plot growth/decay skill of the 256 core (reads growth_decay.json).
Run in .venv-steps: ../.venv-steps/bin/python plot_growth_decay.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("../models/prob_nowcaster_aalborg_ctx256/growth_decay.json"))
T = ["0.1", "1.0", "5.0"]
x = np.arange(len(T)); w = 0.38
fig, (axg, axd) = plt.subplots(1, 2, figsize=(12, 5))

# GROWTH: mean PoP where genesis happens vs where it stays dry (256)
ev = [d["genesis"][t]["popPoP_rained"] for t in T]
non = [d["genesis"][t]["poP_dry"] for t in T]
axg.bar(x - w/2, ev, w, color="#2ca02c", label="genesis pixels that DID rain")
axg.bar(x + w/2, non, w, color="#c7c7c7", label="genesis pixels that stayed dry")
for i, t in enumerate(T):
    axg.text(i, max(ev[i], non[i]) + 0.02,
             f"POD {d['genesis'][t]['pod256']:.2f}\n(128: {d['genesis'][t]['pod128']:.2f})",
             ha="center", fontsize=8)
axg.set_title("GROWTH (genesis): model PoP where rain forms vs where it doesn't\n"
              "big gap = it anticipates formation; POD = fraction of genesis caught @0.5", fontsize=9.5)
axg.set_ylabel("mean forecast P(rain)"); axg.set_xticks(x); axg.set_xticklabels([f"≥{t}mm" for t in T])
axg.set_ylim(0, 1); axg.legend(fontsize=8, loc="upper right"); axg.grid(axis="y", alpha=0.3)

# DECAY: mean PoP where advected rain clears vs where it stays (256)
clr = [d["decay"][t]["poP_cleared"] for t in T]
sty = [d["decay"][t]["poP_stayed"] for t in T]
axd.bar(x - w/2, clr, w, color="#1f77b4", label="advected-rain pixels that CLEARED")
axd.bar(x + w/2, sty, w, color="#c7c7c7", label="advected-rain pixels that stayed wet")
for i, t in enumerate(T):
    axd.text(i, max(clr[i], sty[i]) + 0.02,
             f"clearPOD {d['decay'][t]['clearpod256']:.2f}", ha="center", fontsize=8)
axd.set_title("DECAY (dissipation): model PoP where rain clears vs where it stays\n"
              "lower on 'cleared' = it anticipates clearing (Lagrangian keeps it → fails)", fontsize=9.5)
axd.set_xticks(x); axd.set_xticklabels([f"≥{t}mm" for t in T]); axd.set_ylim(0, 1)
axd.legend(fontsize=8, loc="upper left"); axd.grid(axis="y", alpha=0.3)

fig.suptitle("256² prob-nowcaster — growth & decay skill (obj 4), 400 Aalborg test cases", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = "../models/prob_nowcaster_aalborg_ctx256/growth_decay.png"
fig.savefig(out, dpi=140); print("saved", out)
