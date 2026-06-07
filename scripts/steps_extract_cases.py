"""Extract the SAME Aalborg held-out test cases eval_aalborg_ab.py scores, to .npz.

pysteps has no wheel for this repo's Python 3.14, so STEPS runs in an isolated
Python-3.12 venv (.venv-steps) that has no torch / no rust loader. This script
runs in the MAIN venv (which has the rust data loader), reproduces the A/B's
exact case selection — wettest `n_cases` over an evenly-spaced scan of the Aalborg
temporal-TEST split — and dumps past+truth rain-rate fields (mm/h) so the STEPS
baseline can score the identical cases offline.

Selection here is byte-identical to eval_aalborg_ab.run (same dm config, scan
stride, wetness sort) so steps_summary lines up with ab_eval/summary.json. The
eulerian/single-vector-lag Brier the STEPS script recomputes is the cross-check
that the case sets actually match.

Usage (from scripts/, MAIN venv):
    uv run python steps_extract_cases.py --n_cases=400
"""
import os
from pathlib import Path

import numpy as np
from fire import Fire
from omegaconf import OmegaConf

from ldcast.features.rust_data import RustRadarDataModule
from ldcast.visualization.plots import reverse_transform_R

THRESHOLDS = (0.1, 1.0, 5.0)


def run(
    config="../config/train_rust.yaml",
    n_cases=400,
    scan_rows=5000,
    region_center=(685, 852),
    region_radius=64,
    test_frac=0.1,
    valid_frac=0.1,
    timestep_min=10,                 # DMI radar cadence (journal 2026-05-30 fix)
    out=None,
):
    cfg = OmegaConf.load(config)
    if cfg.get("radar_root"):
        os.environ["DGMR_RADAR_ROOT"] = cfg.radar_root
    out = Path(out or "../models/prob_nowcaster_aalborg/ab_eval/steps_cases.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    Tp, Tf = int(cfg.past_steps), int(cfg.future_steps)

    print("Loading Aalborg test split (same recipe as eval_aalborg_ab)...")
    dm = RustRadarDataModule(
        index_path=cfg.index_path, mode="diffusion",
        past_steps=Tp, future_steps=Tf, height=cfg.height, width=cfg.width,
        batch_size=cfg.genforecast_batch_size, num_workers=0,
        valid_frac=valid_frac, test_frac=test_frac, split_mode="temporal",
        region_center=tuple(region_center), region_radius=region_radius,
        dedup_bin0=False, seed=42, use_weighted_sampler=False,
    )
    dm.setup()
    if dm.test_ds is None:
        raise SystemExit("test split is empty — check test_frac / region")
    test_ds, test_ts = dm.test_ds, dm.test_ts
    n_test = len(test_ds)
    print(f"  {n_test} test rows ({np.unique(test_ts // 86400).size} UTC days)")

    print("Selecting wettest cases (identical scan/sort to the A/B)...")
    stride = max(1, n_test // scan_rows)
    scan_idx = list(range(0, n_test, stride))
    scored = []
    for ridx in scan_idx:
        past_t, future_t = test_ds[ridx]
        truth = reverse_transform_R(future_t[0].float().numpy())     # (Tf,H,W) mm/h
        scored.append((float((truth >= 1.0).mean()), ridx, past_t, truth))
    scored.sort(key=lambda x: -x[0])
    sel = scored[:n_cases]

    past_all = np.stack(
        [reverse_transform_R(p.float().numpy())[0] for _, _, p, _ in sel]
    ).astype(np.float32)                                             # (C,Tp,H,W)
    truth_all = np.stack([tr for _, _, _, tr in sel]).astype(np.float32)  # (C,Tf,H,W)
    print(f"  {len(sel)} cases  past{past_all.shape} truth{truth_all.shape}")

    np.savez_compressed(
        out, past=past_all, truth=truth_all,
        thresholds=np.array(THRESHOLDS, dtype=np.float32),
        timestep_min=np.int64(timestep_min),
        past_steps=np.int64(Tp), future_steps=np.int64(Tf),
        n_scanned=np.int64(len(scan_idx)),
    )
    print(f"\nSaved: {out}  ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    Fire(run)
