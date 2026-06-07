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

import torch
from fire import Fire
from omegaconf import OmegaConf

from ldcast.features.rust_data import RustRadarDataModule

from train_genforecast import setup_model

# Training runs millions of fixed-shape steps, so let cuDNN pick the fastest
# conv algos once and reuse them, and allow TF32 for the fp32 regions (the AFNO
# spectral path self-casts to fp32). The first few steps are slightly slower
# while cuDNN searches; amortized away over a full run. (Inference does the
# opposite -- one unique-shape forward -- so predict_rust.py leaves these off.)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def train(
    index_path=None,
    autoenc_weights_fn=None,
    past_steps=4,
    future_steps=8,
    height=256,
    width=256,
    full_frame=False,
    batch_size=8,
    num_workers=4,
    cache_capacity=64,
    valid_frac=0.1,
    test_frac=0.0,
    split_mode="random",
    seed=42,
    use_weighted_sampler=False,
    max_nocoverage_frac=0.05,
    region_center=None,        # (cx,cy) Aalborg pixel; None = all Denmark
    region_radius=64,
    input_pad=0,               # >0 loads (height+2*pad)^2 windows centred on the crop (192 -> 512)
    dedup_bin0=True,
    init_weights=None,         # warm-start WEIGHTS only (accepts .ckpt or .pt); not a full resume
    strict_weights=True,
    use_ema=True,              # False frees the ~2.7 GB EMA shadow (+ its val-time copy) for tight VRAM
    model_dir="../models/genforecast_rust",
    lr=1e-4,
    ckpt_path=None,
    precision="bf16-mixed",
    optimizer_8bit=False,
    use_checkpoint=False,
    max_epochs=1000,
    limit_train_batches=None,
    limit_val_batches=None,
    scale_factor=1.0,
    gradient_clip_val=1.0,
    sample_every_n_epochs=1,
    max_hours=None,
    early_stopping_patience=6,
    accumulate_grad_batches=1,
    save_top_k=3,
    cudnn_benchmark=True,      # set False for tight-VRAM runs: benchmark picks workspace-heavy convs
):
    torch.backends.cudnn.benchmark = cudnn_benchmark
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
        test_frac=test_frac,
        split_mode=split_mode,
        seed=seed,
        use_weighted_sampler=use_weighted_sampler,
        max_nocoverage_frac=max_nocoverage_frac,
        region_center=region_center,
        region_radius=region_radius,
        input_pad=input_pad,
        dedup_bin0=dedup_bin0,
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
        use_checkpoint=use_checkpoint,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        scale_factor=scale_factor,
        gradient_clip_val=gradient_clip_val,
        sample_every_n_epochs=sample_every_n_epochs,
        max_hours=max_hours,
        early_stopping_patience=early_stopping_patience,
        accumulate_grad_batches=accumulate_grad_batches,
        save_top_k=save_top_k,
    )
    gc.collect()

    if init_weights is not None:
        print(f"Warm-starting (weights only) from {init_weights}...")
        sd = torch.load(init_weights, map_location="cpu")
        sd = sd.get("state_dict", sd)   # accept a Lightning .ckpt or a raw state_dict .pt
        ldm.load_state_dict(sd, strict=strict_weights)

    if not use_ema:
        ldm.use_ema = False     # drop the EMA shadow + its val-time param copy to fit VRAM
        ldm.model_ema = None
        gc.collect()
        torch.cuda.empty_cache()

    print("Starting training...")
    trainer.fit(ldm, datamodule=dm, ckpt_path=ckpt_path)


def main(config=None, **kwargs):
    cfg = OmegaConf.load(config) if config else {}
    cfg.update(kwargs)
    train(**cfg)


if __name__ == "__main__":
    Fire(main)
