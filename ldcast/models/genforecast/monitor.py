"""TensorBoard sample-prediction logger for diffusion forecast training.

`val_loss_ema` (an eps-prediction MSE) barely moves and says little about sample
quality, so this callback periodically samples the model on a few fixed validation
cases and logs (a) a ground-truth-vs-prediction image grid and (b) quality scalars
-- CSI at rain thresholds plus mean-abs / RMS error in mm/h -- to TensorBoard. It is
the training-time analogue of scoring a forecast to track progress, and it rides the
validation that already runs each epoch (no separate eval job).

Memory note (16 GB GPU): training already holds model + EMA + optimizer (~10.8 GB),
so a full-res EMA sample OOMs. By default this samples with the LIVE weights (no
EMA-store backup) on a center-CROP of the case -- both cut the footprint, and the
current model's forecast is at least as informative as the lagging EMA early in
training. Set use_ema=True / sample_hw=None if you have headroom.

The K scored cases (the wettest found in a one-time scan) are cached and reused every
epoch, and sampling runs under a fixed seed (RNG restored afterwards), so the metric
trend reflects the model rather than case choice or sampling noise.
"""
import io
from contextlib import redirect_stdout, nullcontext

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl

from ..diffusion import plms
from ...visualization.plots import plot_precip_image, reverse_transform_R


class SamplePredictionLogger(pl.Callback):
    def __init__(self, every_n_epochs=1, num_diffusion_iters=50, max_leadtimes=4,
                 sample_hw=128, use_ema=False, num_eval_cases=4, eval_seed=1234):
        super().__init__()
        self.every_n_epochs = max(int(every_n_epochs), 0)
        self.num_diffusion_iters = num_diffusion_iters
        self.max_leadtimes = max_leadtimes
        self.sample_hw = sample_hw  # center-crop preview to this size (None = full)
        self.use_ema = use_ema
        self.num_eval_cases = max(int(num_eval_cases), 1)
        self.eval_seed = eval_seed
        self._cases = None  # K fixed (x, y) from the val set, grabbed once

    def _crop(self, t):
        if self.sample_hw is None:
            return t
        (H, W) = (t.shape[-2], t.shape[-1])
        (h, w) = (min(self.sample_hw, H), min(self.sample_hw, W))
        return t[..., (H - h) // 2:(H - h) // 2 + h, (W - w) // 2:(W - w) // 2 + w]

    def _fixed_cases(self, trainer, scan_batches=32):
        if self._cases is None:
            # The leading val crops are often bone-dry (-> blank grids, undefined CSI).
            # Scan a bounded number of batches and keep the num_eval_cases with the
            # wettest future, so the scored cases actually contain rain. Done once and
            # cached, so the same cases are scored every epoch (a comparable trend).
            scored = []  # (wetness, x_pairs, y) per sample
            for bi, (x, y) in enumerate(trainer.datamodule.val_dataloader()):
                wet = y.flatten(1).mean(1)  # per-sample mean rain in the future
                for i in range(y.shape[0]):
                    xi = [[self._crop(past[i:i + 1]).clone(), t_past[i:i + 1].clone()]
                          for (past, t_past) in x]
                    scored.append((float(wet[i]), xi, self._crop(y[i:i + 1]).clone()))
                if bi + 1 >= scan_batches:
                    break
            scored.sort(key=lambda s: s[0], reverse=True)
            self._cases = [(xi, yi) for (_, xi, yi) in scored[:self.num_eval_cases]]
        return self._cases

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
        cases = self._fixed_cases(trainer)
        sampler = plms.PLMSSampler(pl_module)
        amp = (torch.autocast("cuda", dtype=torch.bfloat16)
               if dev.type == "cuda" else nullcontext())
        thresholds = (0.1, 1.0, 5.0)
        counts = {thr: [0, 0, 0] for thr in thresholds}  # hits, misses, false alarms
        abs_err = sq_err = n_px = 0.0
        first = None  # (truth, pred) of the wettest case, for the image grid

        prev_use_ema = pl_module.use_ema
        pl_module.use_ema = self.use_ema  # False: live weights, skip the +2.7 GB EMA store
        # Seed sampling so the metric tracks the model (not noise); restore both the CPU
        # and CUDA RNG afterwards so training's stochasticity is left untouched.
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            torch.manual_seed(self.eval_seed)
            for ci, (x, y) in enumerate(cases):
                x = [[t.to(dev) for t in pair] for pair in x]
                y = y.to(dev)
                gen_shape = tuple(pl_module.autoencoder.encode(y)[0].shape[1:])
                with amp, redirect_stdout(io.StringIO()):
                    (s, _) = sampler.sample(self.num_diffusion_iters, y.shape[0],
                                            gen_shape, x, progbar=False, verbose=False)
                    y_pred = pl_module.autoencoder.decode(s / pl_module.scale_factor)
                pl_module._cond_cache = None  # drop cached conditioning before next case
                truth = reverse_transform_R(y[0, 0].float().cpu().numpy())  # (T,H,W) mm/h
                pred = reverse_transform_R(y_pred[0, 0].float().cpu().numpy())
                for thr in thresholds:
                    (p, o) = (pred >= thr, truth >= thr)
                    counts[thr][0] += int((p & o).sum())     # hits
                    counts[thr][1] += int((~p & o).sum())    # misses
                    counts[thr][2] += int((p & ~o).sum())    # false alarms
                diff = pred - truth
                abs_err += float(np.abs(diff).sum())
                sq_err += float((diff ** 2).sum())
                n_px += diff.size
                if ci == 0:
                    first = (truth, pred)
        finally:
            pl_module.use_ema = prev_use_ema
            pl_module._cond_cache = None
            torch.set_rng_state(rng_cpu)
            if rng_cuda is not None:
                torch.cuda.set_rng_state_all(rng_cuda)

        step = trainer.global_step
        for thr in thresholds:
            (h, m, f) = counts[thr]
            if h + m + f:  # skip a threshold with no rain that heavy in truth or pred
                logger.experiment.add_scalar(f"val/csi_{thr}mm", h / (h + m + f),
                                             global_step=step)
        if n_px:
            logger.experiment.add_scalar("val/mae_mmhr", abs_err / n_px, global_step=step)
            logger.experiment.add_scalar("val/rmse_mmhr", (sq_err / n_px) ** 0.5,
                                         global_step=step)

        # Image grid for the wettest case (case 0), as before.
        (truth, pred) = first
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
        logger.experiment.add_figure("val/forecast", fig, global_step=step)
        plt.close(fig)
