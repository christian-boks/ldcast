"""TensorBoard sample-prediction logger for diffusion forecast training.

`val_loss_ema` (an eps-prediction MSE) barely moves and says little about sample
quality, so this callback periodically samples the model on one fixed validation
case and logs a ground-truth-vs-prediction image grid to TensorBoard -- the
training-time analogue of running a forecast to eyeball progress.

Memory note (16 GB GPU): training already holds model + EMA + optimizer
(~10.8 GB), so a full-res EMA sample OOMs. By default this samples with the
LIVE weights (no EMA-store backup) on a center-CROP of the case -- both cut the
footprint, and the current model's forecast is at least as informative as the
lagging EMA early in training. Set use_ema=True / sample_hw=None if you have
headroom.
"""
import io
from contextlib import redirect_stdout, nullcontext

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import torch
import pytorch_lightning as pl

from ..diffusion import plms
from ...visualization.plots import plot_precip_image, reverse_transform_R


class SamplePredictionLogger(pl.Callback):
    def __init__(self, every_n_epochs=1, num_diffusion_iters=50, max_leadtimes=4,
                 sample_hw=128, use_ema=False):
        super().__init__()
        self.every_n_epochs = max(int(every_n_epochs), 0)
        self.num_diffusion_iters = num_diffusion_iters
        self.max_leadtimes = max_leadtimes
        self.sample_hw = sample_hw  # center-crop preview to this size (None = full)
        self.use_ema = use_ema
        self._case = None  # one fixed (x, y) from the val set, grabbed once

    def _crop(self, t):
        if self.sample_hw is None:
            return t
        (H, W) = (t.shape[-2], t.shape[-1])
        (h, w) = (min(self.sample_hw, H), min(self.sample_hw, W))
        return t[..., (H - h) // 2:(H - h) // 2 + h, (W - w) // 2:(W - w) // 2 + w]

    def _fixed_case(self, trainer):
        if self._case is None:
            (x, y) = next(iter(trainer.datamodule.val_dataloader()))
            # crop only the spatial frame tensor; leave the timestep vector
            x = [[self._crop(past[:1]).clone(), t_past[:1].clone()]
                 for (past, t_past) in x]
            self._case = (x, self._crop(y[:1]).clone())
        return self._case

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if (self.every_n_epochs == 0 or trainer.sanity_checking
                or not trainer.is_global_zero
                or trainer.current_epoch % self.every_n_epochs != 0):
            return
        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            return
        try:
            self._log_forecast(trainer, pl_module, logger)
        except Exception as e:  # monitoring must never crash a training run
            print(f"[SamplePredictionLogger] skipped: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()

    def _log_forecast(self, trainer, pl_module, logger):
        dev = pl_module.device
        (x, y) = self._fixed_case(trainer)
        x = [[t.to(dev) for t in pair] for pair in x]
        y = y.to(dev)

        # latent shape from the deterministic frozen encoder
        gen_shape = tuple(pl_module.autoencoder.encode(y)[0].shape[1:])
        sampler = plms.PLMSSampler(pl_module)
        amp = (torch.autocast("cuda", dtype=torch.bfloat16)
               if dev.type == "cuda" else nullcontext())
        prev_use_ema = pl_module.use_ema
        pl_module.use_ema = self.use_ema  # False: live weights, skip the +2.7 GB EMA store
        try:
            with amp, redirect_stdout(io.StringIO()):
                (s, _) = sampler.sample(self.num_diffusion_iters, y.shape[0],
                                        gen_shape, x, progbar=False, verbose=False)
                y_pred = pl_module.autoencoder.decode(s / pl_module.scale_factor)
        finally:
            pl_module.use_ema = prev_use_ema
            pl_module._cond_cache = None  # drop the cached fixed-case context

        truth = reverse_transform_R(y[0, 0].float().cpu().numpy())  # (T,H,W) mm/h
        pred = reverse_transform_R(y_pred[0, 0].float().cpu().numpy())

        T = truth.shape[0]
        n = min(T, self.max_leadtimes)
        cols = [round(i * (T - 1) / (n - 1)) for i in range(n)] if n > 1 else [T - 1]
        fig, axs = plt.subplots(2, len(cols), figsize=(3 * len(cols), 6.5),
                                squeeze=False, constrained_layout=True)
        im = None
        for c, t in enumerate(cols):
            plot_precip_image(axs[0, c], truth[t].copy())
            im = plot_precip_image(axs[1, c], pred[t].copy())
            axs[0, c].set_title(f"+{t + 1}", fontsize=9)
        axs[0, 0].set_ylabel("truth", fontsize=10)
        axs[1, 0].set_ylabel("prediction", fontsize=10)
        if im is not None:
            fig.colorbar(im, ax=axs, shrink=0.8, label="mm/h")
        logger.experiment.add_figure("val/forecast", fig, global_step=trainer.global_step)
        plt.close(fig)
