import warnings
from datetime import timedelta

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from ..diffusion import diffusion
from . import monitor

# torch's pytree LeafSpec deprecation, surfaced via PL's _pytree shim on recent torch.
# Third-party and benign; silence it so it doesn't spam every training run's logs.
warnings.filterwarnings("ignore", message=r".*LeafSpec.*deprecated.*")


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
        every_n_epochs=1,
        save_top_k=1,  # one rolling ckpt per epoch (~6.7 GB). NOTE: save_top_k=0 +
                       # save_last=True is a SILENT NO-OP in Lightning 2.x -- last.ckpt
                       # is only written when a top-k ckpt is also saved that same step,
                       # so save_top_k=0 means no checkpoint is ever written mid-run.
                       # monitor=None keeps just the latest (a "best" by val_loss_ema,
                       # an eps-MSE, doesn't track forecast quality anyway); resume picks
                       # the newest ckpt.
        monitor=None,
        save_last=False,
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
