"""Train the LDCast autoencoder against dgmr-rs radar data via dgmr-py.

Usage (from this directory):
    DGMR_RADAR_ROOT=/path/to/radar_data DGMR_RADAR_INDEX=/path/to/index.txt \
        uv run python train_autoenc_rust.py \
            --height=256 --width=256 --batch_size=16 \
            --model_dir=../models/autoenc_rust
"""
import gc
import os

from fire import Fire
from omegaconf import OmegaConf

from ldcast.features.rust_data import RustRadarDataModule
from ldcast.models.autoenc import encoder, training


def train(
    index_path=None,
    past_steps=4,
    future_steps=8,
    height=256,
    width=256,
    full_frame=False,
    batch_size=16,
    num_workers=4,
    cache_capacity=64,
    valid_frac=0.1,
    seed=42,
    use_weighted_sampler=True,
    model_dir="../models/autoenc_rust",
    ckpt_path=None,
    precision="bf16-mixed",
    max_epochs=1000,
    limit_train_batches=None,
    limit_val_batches=None,
):
    if index_path is None:
        index_path = os.environ["DGMR_RADAR_INDEX"]

    print(f"Loading data from {index_path}...")
    dm = RustRadarDataModule(
        index_path=index_path,
        mode="autoenc",
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
    enc = encoder.SimpleConvEncoder()
    dec = encoder.SimpleConvDecoder()
    autoencoder, trainer = training.setup_autoenc_training(
        encoder=enc, decoder=dec, model_dir=model_dir, precision=precision,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
    )
    gc.collect()

    print("Starting training...")
    trainer.fit(autoencoder, datamodule=dm, ckpt_path=ckpt_path)


def main(config=None, **kwargs):
    cfg = OmegaConf.load(config) if config else {}
    cfg.update(kwargs)
    train(**cfg)


if __name__ == "__main__":
    Fire(main)
