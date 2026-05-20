import pytorch_lightning as pl
import torch

from . import autoenc


def setup_autoenc_training(
    encoder,
    decoder,
    model_dir,
    precision=None,
    max_epochs=1000,
    limit_train_batches=None,
    limit_val_batches=None,
):
    autoencoder = autoenc.AutoencoderKL(encoder, decoder)

    num_gpus = torch.cuda.device_count()
    accelerator = "gpu" if (num_gpus > 0) else "cpu"
    devices = torch.cuda.device_count() if (accelerator == "gpu") else 1

    early_stopping = pl.callbacks.EarlyStopping(
        "val_rec_loss", patience=6, verbose=True
    )
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        filename="{epoch}-{val_rec_loss:.4f}",
        monitor="val_rec_loss",
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

    return (autoencoder, trainer)
