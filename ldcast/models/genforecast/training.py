import pytorch_lightning as pl
import torch

from ..diffusion import diffusion


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
):
    ldm = diffusion.LatentDiffusion(model, autoencoder,
        context_encoder=context_encoder, lr=lr,
        optimizer_8bit=optimizer_8bit)

    num_gpus = torch.cuda.device_count()
    accelerator = "gpu" if (num_gpus > 0) else "cpu"
    devices = torch.cuda.device_count() if (accelerator == "gpu") else 1

    early_stopping = pl.callbacks.EarlyStopping(
        "val_loss_ema", patience=6, verbose=True, check_finite=False
    )
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        filename="{epoch}-{val_loss_ema:.4f}",
        monitor="val_loss_ema",
        every_n_epochs=1,
        save_top_k=3
    )
    callbacks = [early_stopping, checkpoint]

    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        strategy=('ddp' if num_gpus > 1 else 'auto'),
        callbacks=callbacks,
    )
    if precision is not None:
        trainer_kwargs["precision"] = precision
    if limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = limit_train_batches
    if limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = limit_val_batches
    trainer = pl.Trainer(**trainer_kwargs)

    return (ldm, trainer)
