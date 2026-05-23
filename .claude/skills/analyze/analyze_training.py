#!/usr/bin/env python
"""Summarize the current LDCast training run from its TensorBoard logs.

Auto-discovers the most recently written run under <repo>/models/*/tb/version_*,
reads its scalar curves, and prints a deterministic, structured summary that the
/analyze skill turns into a plain-language verdict. Read-only: never touches
checkpoints or training.

Usage (from anywhere — paths resolve relative to the repo, not the CWD):
    uv run python .claude/skills/analyze/analyze_training.py
    uv run python .claude/skills/analyze/analyze_training.py --run models/autoenc_rust/tb/version_1
"""
from __future__ import annotations

import argparse
import glob
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO = Path(__file__).resolve().parents[3]  # .../.claude/skills/analyze/ -> repo root

# (primary metric tracked by checkpoints/early-stopping, [secondary metrics]) per stage
STAGE_METRICS = {
    "autoencoder": ("val_rec_loss", ["val_loss", "val_kl_loss"]),
    "diffusion": ("val_loss_ema", ["val_loss"]),
}


def rel(p: Path):
    try:
        return p.resolve().relative_to(REPO)
    except Exception:
        return p


def find_runs(models_dir: Path):
    """All version_* run dirs that contain event files, newest write first."""
    latest: dict[Path, float] = {}
    for ev in glob.glob(str(models_dir / "*" / "tb" / "version_*" / "*tfevents*")):
        d = Path(ev).parent
        mt = os.path.getmtime(ev)
        if d not in latest or mt > latest[d]:
            latest[d] = mt
    return sorted(((mt, d) for d, mt in latest.items()), reverse=True)


def load(run: Path):
    ea = EventAccumulator(str(run), size_guidance={"scalars": 1_000_000})
    ea.Reload()
    out = {}
    for t in ea.Tags().get("scalars", []):
        s = ea.Scalars(t)
        out[t] = (np.array([e.step for e in s]),
                  np.array([e.value for e in s], dtype=float))
    return out


def detect_stage(tags):
    if "val_rec_loss" in tags:
        return "autoencoder"
    if "val_loss_ema" in tags:
        return "diffusion"
    return "unknown"


def fmt(v, n=4):
    return [round(float(x), n) for x in v]


def find_ckpts(model_dir: Path, key: str):
    import re
    rx = re.compile(re.escape(key) + r"=(\d+\.\d+)")
    found = []
    for p in glob.glob(str(model_dir / "*.ckpt")):
        m = rx.search(os.path.basename(p))
        if m:
            found.append((float(m.group(1)), os.path.basename(p)))
    found.sort()
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None,
                    help="specific version_* dir to analyze (default: most recent)")
    ap.add_argument("--models-dir", default=str(REPO / "models"))
    args = ap.parse_args()
    models_dir = Path(args.models_dir)

    runs = find_runs(models_dir)
    if not runs:
        print(f"No TensorBoard runs found under {rel(models_dir)}/*/tb/version_*")
        print("Has training started, and is it writing to models/<dir>/tb?")
        return

    if args.run:
        run = Path(args.run).resolve()
        evs = glob.glob(str(run / "*tfevents*"))
        if not evs:
            print(f"No event files in {run}")
            return
        mt = max(os.path.getmtime(e) for e in evs)
    else:
        mt, run = runs[0]

    model_dir = run.parent.parent  # models/<name>/tb/version_X -> models/<name>
    age_min = (time.time() - mt) / 60
    status = "ACTIVE (writing now)" if age_min < 5 else f"idle/finished (last write {age_min:.0f} min ago)"

    data = load(run)
    tags = list(data)
    stage = detect_stage(tags)

    print("=" * 72)
    print(f"RUN:    {rel(run)}")
    print(f"STAGE:  {stage}")
    print(f"STATUS: {status}")
    if "epoch" in data:
        print(f"EPOCH:  {data['epoch'][1][-1]:.0f}")
    print("=" * 72)

    if len(runs) > 1:
        print("\nOther runs present (newest first):")
        for m, d in runs:
            mark = "  <- analyzing" if d == run else ""
            print(f"  {time.strftime('%Y-%m-%d %H:%M', time.localtime(m))}  {rel(d)}{mark}")

    print("\n--- STABILITY ---")
    bad = [t for t in tags if len(data[t][1]) and not np.isfinite(data[t][1]).all()]
    if bad:
        print(f"  !!! NON-FINITE (NaN/Inf) values in: {bad}")
        print("      -> the run diverged. This is the known instability; resume from the")
        print("         last good checkpoint (gradient clipping should prevent recurrence).")
    else:
        print("  all logged scalars finite (no NaN/Inf)")

    primary, secondary = STAGE_METRICS.get(stage, (tags[0] if tags else None, []))

    print("\n--- PRIMARY METRIC (checkpoint / early-stopping target) ---")
    if primary in data:
        _, v = data[primary]
        print(f"  {primary}: n={len(v)}")
        print(f"    per-epoch: {fmt(v)}")
        print(f"    best={v.min():.4f} (epoch idx {int(v.argmin())}/{len(v)-1})  latest={v[-1]:.4f}")
        if len(v) >= 6:
            k = max(2, len(v) // 3)
            improving = v[-k:].min() < v[:k].min() - 1e-9
            print(f"    whole-run: first-{k} best={v[:k].min():.4f}  vs  last-{k} best={v[-k:].min():.4f}"
                  f"  -> {'IMPROVING' if improving else 'flat / plateau'}")
            kk = min(8, len(v))
            slope = float(np.polyfit(np.arange(kk), v[-kk:], 1)[0])
            pct = slope / abs(v[-kk:].mean()) * 100
            lab = "PLATEAU (flat)" if abs(pct) < 0.5 else ("still improving" if slope < 0 else "WORSENING")
            since_best = (len(v) - 1) - int(v.argmin())
            print(f"    recent:    slope over last {kk} ep = {slope:+.5f}/ep ({pct:+.2f}%/ep) -> {lab}")
            print(f"    epochs since best: {since_best}  (early-stop fires after `patience` with no new best)")
    else:
        print(f"  (metric '{primary}' not found; tags present: {tags})")

    if secondary:
        print("\n--- SECONDARY METRICS (per epoch) ---")
        for t in secondary:
            if t in data:
                print(f"  {t}: {fmt(data[t][1])}")

    print("\n--- TRAIN LOSS (per-step, noisy; read the binned trend, not raw spikes) ---")
    if "train_loss" in data:
        st, v = data["train_loss"]
        print(f"  n={len(v)}  first={v[0]:.4f}  last={v[-1]:.4f}  min={v.min():.4f}  max={v.max():.4f}")
        nb = min(12, len(v))
        for i, ix in enumerate(np.array_split(np.arange(len(v)), nb)):
            seg = v[ix]
            print(f"    bin{i:2d}  step {int(st[ix][0]):6d}-{int(st[ix][-1]):6d}"
                  f"  mean={seg.mean():.4f}  min={seg.min():.4f}  max={seg.max():.4f}")
        print(f"  last 15 raw: {fmt(v[-15:])}")
    else:
        print("  (no train_loss tag)")

    print("\n--- CHECKPOINTS (best by filename metric) ---")
    cks = find_ckpts(model_dir, primary) if primary else []
    if cks:
        for val, name in cks[:5]:
            print(f"    {val:.4f}  {name}")
    else:
        print(f"    (no '{primary}=' checkpoints in {rel(model_dir)})")
    last = model_dir / "last.ckpt"
    if last.exists():
        print(f"    last.ckpt @ {time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(last)))}")


if __name__ == "__main__":
    main()
