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
    accumulate_grad_batches=1,
    save_top_k=3,
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
        filename="{epoch}-{step}",
        # Keep the most-recent `save_top_k` checkpoints. We monitor "step" (always
        # injected by Lightning, strictly increasing) with mode="max" rather than a
        # quality metric: val_loss_ema (eps-MSE) doesn't track forecast quality, and
        # the val/csi_* metrics are add_scalar'd (not self.log'd), so neither is a
        # usable monitor -- pick the best checkpoint offline (eval_genforecast / metrics).
        # NOTE: monitor=None + save_top_k>1 raises in Lightning 2.x, hence "step".
        monitor="step",
        mode="max",
        every_n_epochs=1,
        save_top_k=save_top_k,  # default 3; -1 keeps all. Each diffusion ckpt is ~6.3 GB.
        save_last=True,         # last.ckpt for clean resume (_resume_ckpt prefers it)
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
        accumulate_grad_batches=accumulate_grad_batches,
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
