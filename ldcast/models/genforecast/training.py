from datetime import timedelta

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from ..diffusion import diffusion
from . import monitor


def setup_genforecast_training(
    model,
    autoencoder,
    context_encoder,
    model_dir,
    lr=1e-4,
    precision=None,
    optimizer_8bit=False,
    max_epochs=1000,
    limit_train_batches=None,
    limit_val_batches=None,
    scale_factor=1.0,
    gradient_clip_val=1.0,
    sample_every_n_epochs=1,
    max_hours=None,
    early_stopping_patience=6,
):
    ldm = diffusion.LatentDiffusion(model, autoencoder,
        context_encoder=context_encoder, lr=lr,
        optimizer_8bit=optimizer_8bit, scale_factor=scale_factor,
        gradient_clip_val=gradient_clip_val)

    num_gpus = torch.cuda.device_count()
    accelerator = "gpu" if (num_gpus > 0) else "cpu"
    devices = torch.cuda.device_count() if (accelerator == "gpu") else 1

    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        filename="{epoch}-{val_loss_ema:.4f}",
        monitor="val_loss_ema",
        every_n_epochs=1,
        save_top_k=1,  # reduced from 3: each 670M-param ckpt is ~6 GB; 16 GB free disk
        save_last=True,  # also keep last.ckpt -> clean resume via --ckpt_path
    )
    callbacks = [checkpoint]
    if early_stopping_patience and early_stopping_patience > 0:
        callbacks.append(pl.callbacks.EarlyStopping(
            "val_loss_ema", patience=early_stopping_patience,
            verbose=True, check_finite=True))
    if sample_every_n_epochs > 0:
        # periodically log a ground-truth-vs-forecast image grid to TensorBoard
        callbacks.append(monitor.SamplePredictionLogger(
            every_n_epochs=sample_every_n_epochs))

    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        strategy=('ddp' if num_gpus > 1 else 'auto'),
        callbacks=callbacks,
        logger=TensorBoardLogger(save_dir=model_dir, name="tb"),
        # gradient clipping is done manually in LatentDiffusion.on_before_optimizer_step
        # (Lightning's automatic clip is incompatible with the fused AdamW)
    )
    if precision is not None:
        trainer_kwargs["precision"] = precision
    if limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = limit_train_batches
    if limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = limit_val_batches
    if max_hours is not None:
        trainer_kwargs["max_time"] = timedelta(hours=max_hours)
    trainer = pl.Trainer(**trainer_kwargs)

    return (ldm, trainer)
