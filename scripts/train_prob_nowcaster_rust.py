"""Train the deterministic probabilistic nowcaster (retrain.md #2 architecture A/B).

A directly-optimised P(rain>=thr) nowcaster (AFNO backbone + frozen AE + BCE head)
— the cheap, calibrated-by-construction alternative we're testing against the
diffusion model's Brier-skill + initiation POD. Same data/split as the diffusion
model (random split, weighted sampler) for an apples-to-apples comparison.

Usage (from scripts/):
    DGMR_RADAR_ROOT=/opt/radar_data uv run python train_prob_nowcaster_rust.py \\
        --index_path=/opt/radar_data/index_ldcast_128.txt \\
        --autoenc_weights_fn=../models/autoenc_rust/state_dict.pt \\
        --height=128 --width=128 --batch_size=16
"""
import gc
import os

import torch
from fire import Fire
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from ldcast.features.rust_data import RustRadarDataModule
from ldcast.models.autoenc import autoenc, encoder
from ldcast.models.nowcast.prob_nowcast import ProbNowcastNet, ProbNowcaster

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def train(
    index_path=None,
    autoenc_weights_fn=None,
    past_steps=4,
    future_steps=8,
    height=128,
    width=128,
    input_pad=0,            # >0 loads (height+2*pad)^2 context windows (e.g. 64 -> 256^2 input)
    output_crop=None,       # supervise only the centre output_crop^2 (= the target crop)
    batch_size=16,
    num_workers=8,
    cache_capacity=64,
    valid_frac=0.1,
    test_frac=0.1,
    split_mode="temporal",       # clean Aalborg train/val/test (no day leakage)
    region_center=(685, 852),    # Aalborg pixel; None = all Denmark
    region_radius=64,            # 64 px -> the 16 crops containing Aalborg
    dedup_bin0=False,           # natural rain frequency -> calibrated by construction
    seed=42,
    use_weighted_sampler=False,  # natural distribution (no heavy-rain oversampling)
    max_nocoverage_frac=0.05,
    model_dir="../models/prob_nowcaster_rust",
    lr=1e-3,
    embed_dim=128,
    analysis_depth=4,
    forecast_depth=4,
    ckpt_path=None,
    init_weights=None,       # warm-start WEIGHTS only (fine-tune a new geometry); not a full resume
    precision="bf16-mixed",
    max_epochs=1000,
    limit_train_batches=4000,
    limit_val_batches=50,
    max_hours=None,
    early_stopping_patience=0,
    accumulate_grad_batches=1,
    save_top_k=3,
):
    if index_path is None:
        index_path = os.environ["DGMR_RADAR_INDEX"]
    if autoenc_weights_fn is None and init_weights is None:
        raise SystemExit("--autoenc_weights_fn=<path> is required (or --init_weights=<ckpt> to warm-start)")
    assert future_steps % 4 == 0, "future_steps must be divisible by 4 (AE time ratio)"
    thresholds = (0.1, 1.0, 5.0)

    print(f"Loading data from {index_path} (split_mode={split_mode})...")
    dm = RustRadarDataModule(
        index_path=index_path, mode="diffusion",
        past_steps=past_steps, future_steps=future_steps,
        height=height, width=width, batch_size=batch_size,
        num_workers=num_workers, cache_capacity=cache_capacity,
        valid_frac=valid_frac, test_frac=test_frac, split_mode=split_mode,
        region_center=region_center, region_radius=region_radius,
        dedup_bin0=dedup_bin0, seed=seed,
        use_weighted_sampler=use_weighted_sampler,
        max_nocoverage_frac=max_nocoverage_frac,
        input_pad=input_pad,
    )

    print("Building probabilistic nowcaster (frozen AE + AFNO backbone + BCE head)...")
    ae = autoenc.AutoencoderKL(encoder.SimpleConvEncoder(), encoder.SimpleConvDecoder())
    if autoenc_weights_fn is not None:
        ae.load_state_dict(torch.load(autoenc_weights_fn))
    net = ProbNowcastNet(
        ae, thresholds=thresholds, embed_dim=embed_dim,
        analysis_depth=analysis_depth, forecast_depth=forecast_depth,
        output_patches=future_steps // 4,
    )
    model = ProbNowcaster(net, thresholds=thresholds, lr=lr, output_crop=output_crop)
    # Warm-start from a prior checkpoint's WEIGHTS only (incl. the frozen AE) — for
    # fine-tuning into a new geometry (ctx256 -> 512), where Lightning's ckpt_path
    # (a full resume of optimizer/epoch) would choke on the changed data config.
    if init_weights is not None:
        sd = torch.load(init_weights, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(sd, strict=True)
        print(f"warm-started (weights only) from {init_weights}")
    gc.collect()

    os.makedirs(model_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir=model_dir, name="tb")
    callbacks = [
        ModelCheckpoint(dirpath=model_dir, monitor="val_brier", mode="min",
                        save_top_k=save_top_k, save_last=True,
                        filename="{epoch}-{val_brier:.4f}"),
        LearningRateMonitor(),
    ]
    if early_stopping_patience > 0:
        callbacks.append(EarlyStopping(monitor="val_brier", mode="min",
                                       patience=early_stopping_patience))
    trainer_kwargs = dict(
        max_epochs=max_epochs, precision=precision,
        accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
        callbacks=callbacks, logger=logger,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=50,
    )
    if limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = limit_train_batches
    if limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = limit_val_batches
    if max_hours is not None:
        trainer_kwargs["max_time"] = {"hours": max_hours}
    trainer = pl.Trainer(**trainer_kwargs)

    print("Starting training...")
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    Fire(train)
