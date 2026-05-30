"""One-off post-training eval: CSI / POD / FAR per LDCast intensity bin.

Runs the same bin-stratified eval the in-training monitor would run, against
a saved checkpoint. Writes per-case PNGs to
<model_dir>/post_train_eval/eval_pngs/step_<NNNNNN>/ and prints a compact
table to stdout.

Usage (from scripts/):
    uv run python eval_pod_far.py
        --config=../config/train_rust.yaml
        [--ckpt=../models/genforecast_rust/last.ckpt]
        [--use_ema=False] [--per_bin_cases=2] [--ensemble_size=4]
"""
import os
import sys
from pathlib import Path

import torch
from fire import Fire
from omegaconf import OmegaConf

from ldcast.features.rust_data import RustRadarDataModule
from ldcast.models.genforecast.monitor import SamplePredictionLogger

from train_genforecast import setup_model


class _MockExperiment:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, tag, value, global_step=None):
        self.scalars[tag] = float(value)

    def add_figure(self, tag, fig, global_step=None):
        import matplotlib.pyplot as plt
        plt.close(fig)


class _MockLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.experiment = _MockExperiment()


class _MockTrainer:
    def __init__(self, datamodule, log_dir, step):
        self.datamodule = datamodule
        self.global_step = step
        self.current_epoch = 0
        self.is_global_zero = True
        self.sanity_checking = False
        self.logger = _MockLogger(log_dir)


def run(
    config="../config/train_rust.yaml",
    ckpt="../models/genforecast_rust/last.ckpt",
    use_ema=False,
    per_bin_cases=2,
    ensemble_size=4,
    num_diffusion_iters=20,
    sampler="dpmpp",
    eval_seed=1234,
    scan_per_bin=32,
    out_dir=None,
):
    if not os.path.isfile(ckpt):
        sys.exit(f"checkpoint not found: {ckpt}")

    cfg = OmegaConf.load(config)
    if cfg.get("radar_root"):
        os.environ["DGMR_RADAR_ROOT"] = cfg.radar_root
    out_dir = out_dir or os.path.join(
        os.path.dirname(os.path.abspath(ckpt)), "post_train_eval"
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"checkpoint: {ckpt}")
    print(f"output:     {out_dir}/eval_pngs/")
    print(f"use_ema:    {use_ema}")
    print(f"cases:      {per_bin_cases} per bin x 11 bins x {ensemble_size} members "
          f"({sampler}, {num_diffusion_iters} steps)")
    print()

    print("Loading data...")
    dm = RustRadarDataModule(
        index_path=cfg.index_path,
        mode="diffusion",
        past_steps=cfg.past_steps,
        future_steps=cfg.future_steps,
        height=cfg.height,
        width=cfg.width,
        batch_size=cfg.genforecast_batch_size,
        num_workers=0,            # no training, no loader workers needed
        valid_frac=0.1,
        seed=42,
        use_weighted_sampler=cfg.use_weighted_sampler,
    )
    dm.setup("fit")

    print("Building model...")
    ldm, _ = setup_model(
        num_timesteps=cfg.future_steps // 4,
        autoenc_weights_fn=os.path.join(cfg.autoenc_dir, "state_dict.pt"),
        use_obs=True,
        use_nwp=False,
        model_dir=cfg.genforecast_dir,
        lr=cfg.genforecast_lr,
        precision=cfg.precision,
        optimizer_8bit=cfg.optimizer_8bit,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        scale_factor=1.0,
        gradient_clip_val=1.0,
        sample_every_n_epochs=1,
        max_hours=None,
        early_stopping_patience=0,
        accumulate_grad_batches=1,
        save_top_k=0,
    )

    print(f"Loading weights from {ckpt}...")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    ldm.load_state_dict(sd, strict=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ldm = ldm.to(dev).eval()
    ldm.use_ema = use_ema   # baseline; the monitor will override per-call

    step = int(ckpt.split("step=")[-1].split(".")[0]) if "step=" in ckpt else 999999
    monitor = SamplePredictionLogger(
        every_n_epochs=1,
        num_diffusion_iters=num_diffusion_iters,
        sample_hw=cfg.height,        # full crop, not preview
        use_ema=use_ema,
        per_bin_cases=per_bin_cases,
        ensemble_size=ensemble_size,
        sampler=sampler,
        eval_seed=eval_seed,
        scan_per_bin=scan_per_bin,
    )

    trainer = _MockTrainer(dm, out_dir, step)
    print(f"\nRunning eval ({per_bin_cases * 11 * ensemble_size} samples)...")
    monitor._log_forecast(trainer, ldm, trainer.logger)

    scalars = trainer.logger.experiment.scalars
    if not scalars:
        sys.exit("no scalars captured; check the log output above for errors")

    bins = sorted({
        int(t.split("bin")[-1])
        for t in scalars if "/bin" in t
    })

    print()
    print(f"=== POD / FAR / CSI at step {step} "
          f"(use_ema={use_ema}, sampler={sampler}/{num_diffusion_iters}, "
          f"n={per_bin_cases * 11 * ensemble_size}) ===\n")
    print("                 |       0.1 mm/h         |        1.0 mm/h        |        5.0 mm/h        ")
    print("           bin   |   POD     FAR     CSI  |   POD     FAR     CSI  |   POD     FAR     CSI  ")
    print("           ----  |  -----   -----  -----  |  -----   -----  -----  |  -----   -----  -----  ")

    def fmt(v):
        return "  --  " if v is None else f" {v:.3f}"

    for b in bins:
        row = [f"           bin{b:02d}  | "]
        for thr in (0.1, 1.0, 5.0):
            for k in ("pod", "far", "csi"):
                tag = f"val/{k}_{thr}mm/bin{b:02d}"
                row.append(fmt(scalars.get(tag)))
            row.append(" | ")
        print("".join(row))

    print("           ----  | ----------------------- | ----------------------- | ----------------------")
    row = ["           OVER  | "]
    for thr in (0.1, 1.0, 5.0):
        for k in ("pod", "far", "csi"):
            tag = f"val/{k}_{thr}mm/overall"
            row.append(fmt(scalars.get(tag)))
        row.append(" | ")
    print("".join(row))

    print()
    print(f"PNGs: {out_dir}/eval_pngs/step_{step:06d}/")
    print("MAE / RMSE (overall):  "
          f"mae={scalars.get('val/mae_mmhr/overall', float('nan')):.4f} mm/hr  "
          f"rmse={scalars.get('val/rmse_mmhr/overall', float('nan')):.4f} mm/hr")


if __name__ == "__main__":
    Fire(run)
