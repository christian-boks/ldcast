"""Train the LDCast diffusion model against dgmr-rs radar data via dgmr-py.

Requires an autoencoder state_dict produced by train_autoenc_rust.py first.
The trainer saves Lightning .ckpt files; extract a state_dict with:

    python -c "import torch, glob; \\
        ckpt = sorted(glob.glob('../models/autoenc_rust/*.ckpt'))[-1]; \\
        sd = torch.load(ckpt, map_location='cpu')['state_dict']; \\
        torch.save(sd, '../models/autoenc_rust/state_dict.pt')"

Usage (from this directory):
    DGMR_RADAR_ROOT=/path/to/radar_data DGMR_RADAR_INDEX=/path/to/index.txt \\
        uv run python train_genforecast_rust.py \\
            --autoenc_weights_fn=../models/autoenc_rust/state_dict.pt \\
            --height=256 --width=256 --batch_size=8 \\
            --model_dir=../models/genforecast_rust
"""
import gc
import os

from fire import Fire
from omegaconf import OmegaConf

from ldcast.features.rust_data import RustRadarDataModule

from train_genforecast import setup_model


def train(
    index_path=None,
    autoenc_weights_fn=None,
    past_steps=4,
    future_steps=12,
    height=256,
    width=256,
    full_frame=False,
    batch_size=8,
    num_workers=4,
    cache_capacity=64,
    valid_frac=0.1,
    seed=42,
    use_weighted_sampler=True,
    model_dir="../models/genforecast_rust",
    lr=1e-4,
    ckpt_path=None,
    precision="bf16-mixed",
    optimizer_8bit=False,
    max_epochs=1000,
    limit_train_batches=None,
    limit_val_batches=None,
):
    if index_path is None:
        index_path = os.environ["DGMR_RADAR_INDEX"]
    if autoenc_weights_fn is None:
        raise SystemExit("--autoenc_weights_fn=<path to autoenc state_dict.pt> is required")
    assert future_steps % 4 == 0, "future_steps must be divisible by 4 (autoenc time ratio)"

    print(f"Loading data from {index_path}...")
    dm = RustRadarDataModule(
        index_path=index_path,
        mode="diffusion",
        past_steps=past_steps,
        future_steps=future_steps,
        height=height,
        width=width,
        full_frame=full_frame,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_capacity=cache_capacity,
        valid_frac=valid_frac,
        seed=seed,
        use_weighted_sampler=use_weighted_sampler,
    )

    print("Setting up model...")
    ldm, trainer = setup_model(
        num_timesteps=future_steps // 4,
        autoenc_weights_fn=autoenc_weights_fn,
        use_obs=True,
        use_nwp=False,
        model_dir=model_dir,
        lr=lr,
        precision=precision,
        optimizer_8bit=optimizer_8bit,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
    )
    gc.collect()

    print("Starting training...")
    trainer.fit(ldm, datamodule=dm, ckpt_path=ckpt_path)


def main(config=None, **kwargs):
    cfg = OmegaConf.load(config) if config else {}
    cfg.update(kwargs)
    train(**cfg)


if __name__ == "__main__":
    Fire(main)
