"""TensorBoard reconstruction logger for autoencoder training.

`val_rec_loss` is a fairly honest quality number, but a picture shows *how* the
autoencoder fails (smeared cells, clipped peaks). Every N epochs, reconstruct
one fixed validation case -- `decode(encode(x))`, the autoencoder alone, no
diffusion model -- and log an input-vs-reconstruction grid to TensorBoard.
"""
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import torch
import pytorch_lightning as pl

from ...visualization.plots import plot_precip_image, reverse_transform_R


class ReconstructionLogger(pl.Callback):
    def __init__(self, every_n_epochs=1, max_frames=4):
        super().__init__()
        self.every_n_epochs = max(int(every_n_epochs), 0)
        self.max_frames = max_frames
        self._x = None  # one fixed input frame-stack from the val set

    def _fixed_input(self, trainer):
        if self._x is None:
            (x, _y) = next(iter(trainer.datamodule.val_dataloader()))
            while isinstance(x, (list, tuple)):  # unwrap [(frames, t_rel)] -> frames
                x = x[0]
            # the first val crop is often bone-dry (-> an all-white plot); pick the
            # wettest sample in the batch so the grid actually shows rain.
            wettest = int(x.flatten(1).mean(1).argmax())
            self._x = x[wettest:wettest + 1].clone()
        return self._x

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
            self._log_reconstruction(trainer, pl_module, logger)
        except Exception as e:  # monitoring must never crash a training run
            print(f"[ReconstructionLogger] skipped: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()

    def _log_reconstruction(self, trainer, pl_module, logger):
        x = self._fixed_input(trainer).to(pl_module.device)
        x_hat = pl_module(x, sample_posterior=False)[0]  # decode(encode(x)), deterministic

        inp = reverse_transform_R(x[0, 0].float().cpu().numpy())     # (T,H,W) mm/h
        rec = reverse_transform_R(x_hat[0, 0].float().cpu().numpy())

        T = inp.shape[0]
        n = min(T, self.max_frames)
        cols = [round(i * (T - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
        fig, axs = plt.subplots(2, len(cols), figsize=(3 * len(cols), 6.5),
                                squeeze=False, constrained_layout=True)
        im = None
        for c, t in enumerate(cols):
            plot_precip_image(axs[0, c], inp[t].copy())
            im = plot_precip_image(axs[1, c], rec[t].copy())
            axs[0, c].set_title(f"t{t}", fontsize=9)
        axs[0, 0].set_ylabel("input", fontsize=10)
        axs[1, 0].set_ylabel("reconstruction", fontsize=10)
        if im is not None:
            fig.colorbar(im, ax=axs, shrink=0.8, label="mm/h")
        logger.experiment.add_figure("val/reconstruction", fig, global_step=trainer.global_step)
        plt.close(fig)
