from datetime import timedelta

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from . import autoenc
from . import monitor


def setup_autoenc_training(
    encoder,
    decoder,
    model_dir,
    precision=None,
    max_epochs=1000,
    limit_train_batches=None,
    limit_val_batches=None,
    sample_every_n_epochs=1,
    max_hours=None,
    early_stopping_patience=6,
):
    autoencoder = autoenc.AutoencoderKL(encoder, decoder)

    num_gpus = torch.cuda.device_count()
    accelerator = "gpu" if (num_gpus > 0) else "cpu"
    devices = torch.cuda.device_count() if (accelerator == "gpu") else 1

    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        filename="{epoch}-{val_rec_loss:.4f}",
        monitor="val_rec_loss",
        every_n_epochs=1,
        save_top_k=3,
        save_last=True,  # also keep last.ckpt -> clean resume via --ckpt_path
    )
    callbacks = [checkpoint]
    if early_stopping_patience and early_stopping_patience > 0:
        callbacks.append(pl.callbacks.EarlyStopping(
            "val_rec_loss", patience=early_stopping_patience, verbose=True))
    if sample_every_n_epochs > 0:
        # periodically log an input-vs-reconstruction grid to TensorBoard
        callbacks.append(monitor.ReconstructionLogger(
            every_n_epochs=sample_every_n_epochs))

    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        strategy=('ddp' if num_gpus > 1 else 'auto'),
        callbacks=callbacks,
        logger=TensorBoardLogger(save_dir=model_dir, name="tb"),
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

    return (autoencoder, trainer)
