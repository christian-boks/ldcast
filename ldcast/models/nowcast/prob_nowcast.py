"""Deterministic *probabilistic* nowcaster (retrain.md #2, the architecture A/B).

Same AFNO backbone the diffusion model conditions on (`AFNONowcastNetBase`:
encode past via the frozen AE -> analysis AFNO -> temporal transformer ->
forecast AFNO), but with a decode head that outputs **P(rain >= thr) per pixel
per lead** for several thresholds, trained with BCE. One forward, calibrated by
construction (proper scoring rule) — the cheap alternative we're testing against
the diffusion model's Brier-skill + initiation POD.

Targets are formed in the normalized log-rain space the AE works in:
reverse_transform_R = 10^(R*std+mean) with mean=-0.051, std=0.528, so a mm/h
threshold maps to norm = (log10(thr) - mean)/std.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from .nowcast import AFNONowcastNetBase
from ..autoenc.encoder import SimpleConvDecoder

_MEAN, _STD = -0.051, 0.528   # matches visualization.plots.reverse_transform_R


def _thr_norm(thr_mmhr):
    return (np.log10(thr_mmhr) - _MEAN) / _STD


class ProbNowcastNet(nn.Module):
    """AFNO nowcaster backbone -> per-threshold probability-logit fields."""

    def __init__(self, autoencoder, thresholds=(0.1, 1.0, 5.0),
                 embed_dim=128, analysis_depth=4, forecast_depth=4,
                 output_patches=2):
        super().__init__()
        self.backbone = AFNONowcastNetBase(
            autoencoder=[autoencoder], embed_dim=embed_dim,
            analysis_depth=analysis_depth, forecast_depth=forecast_depth,
            input_patches=(1,), input_size_ratios=(1,),
            output_patches=output_patches, train_autoenc=False,
        )
        eo = self.backbone.embed_dim_out
        # mirror the AE decode path: features -> 64 ch -> upsample decoder -> n_thr
        self.out_proj = nn.Conv3d(eo, 64, kernel_size=1)
        self.head = SimpleConvDecoder(in_dim=len(thresholds))
        self.n_thr = len(thresholds)

    def forward(self, x):
        z = self.backbone(x)        # (B, eo, output_patches, h, w)
        z = self.out_proj(z)        # (B, 64, output_patches, h, w)
        return self.head(z)         # (B, n_thr, T, H, W) logits


class ProbNowcaster(pl.LightningModule):
    def __init__(self, net, thresholds=(0.1, 1.0, 5.0), lr=1e-3, output_crop=None):
        super().__init__()
        self.net = net
        self.thresholds = tuple(thresholds)
        self.register_buffer(
            "thr_norm",
            torch.tensor([_thr_norm(t) for t in thresholds], dtype=torch.float32))
        self.lr = lr
        # When the input is padded for context (rust_data input_pad>0), the net
        # outputs the full padded H/W; supervise/score only the centre output_crop
        # (= the original target crop), whose pixels saw full surrounding context.
        self.output_crop = output_crop

    def _crop(self, t):
        if self.output_crop is None or t.shape[-1] == self.output_crop:
            return t
        o = (t.shape[-1] - self.output_crop) // 2
        return t[..., o:o + self.output_crop, o:o + self.output_crop]

    def _targets(self, y):
        # y: (B,1,T,H,W) normalized -> (B,n_thr,T,H,W) binary rain>=thr
        return (y[:, 0:1] >= self.thr_norm.view(1, -1, 1, 1, 1)).float()

    def training_step(self, batch, batch_idx):
        (x, y) = batch
        logits = self._crop(self.net(x))
        loss = F.binary_cross_entropy_with_logits(logits, self._crop(self._targets(y)))
        self.log("train_loss", loss)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        (x, y) = batch
        logits = self._crop(self.net(x))
        targets = self._crop(self._targets(y))
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        prob = torch.sigmoid(logits)
        log = {"on_step": False, "on_epoch": True, "prog_bar": True}
        self.log("val_loss", loss, **log)
        self.log("val_brier", ((prob - targets) ** 2).mean(), **log)
        # per-threshold Brier (lower=better; the obj-1/6 signal)
        for i, thr in enumerate(self.thresholds):
            self.log(f"val_brier_{thr}mm",
                     ((prob[:, i] - targets[:, i]) ** 2).mean(),
                     on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                betas=(0.5, 0.9), weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=3, factor=0.25)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "monitor": "val_brier"}}
