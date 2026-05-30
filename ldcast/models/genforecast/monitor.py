"""TensorBoard sample-prediction logger for diffusion forecast training.

`val_loss_ema` (an eps-prediction MSE) plateaus early and says little about
sample quality, so this callback runs a stratified forecast eval every val
epoch and logs (a) CSI / MAE / RMSE per LDCast intensity bin to TensorBoard,
plus an overall summary, and (b) a ground-truth-vs-prediction image grid.
It also dumps one PNG per case to <log_dir>/eval_pngs/step_<NNNNNN>/ so
ensembles are inspectable offline (truth row + N ensemble member rows).

Sampling cost per val epoch (defaults: 2 cases/bin x 11 bins x 4 members x
DPM-Solver++ 20 steps): ~88 small samples, ~2 min at 128^2 on a 16 GB GPU.

The case set is selected once and cached, so trends across epochs reflect
the model -- not which cases happened to land in any particular val batch.

Memory note (16 GB GPU): training already holds model + EMA + optimizer
(~10.8 GB) and sampling peaks ~15.2 GB. We evaluate the EMA weights (the ones
worth deploying -- the live Adam iterate bounces epoch-to-epoch and produced a
sawtooth CSI), swapping them in for the whole sampling loop ONCE with the live
weights backed up to CPU (`_ema_weights`), so the EMA eval costs ~0 extra GPU
memory at the peak instead of LitEma.store()'s GPU clone. Sampling stays on a
center-CROP (sample_hw) to keep the footprint down; set sample_hw=None if you
have headroom, use_ema=False to eval the live weights instead.
"""
import io
from collections import defaultdict
from contextlib import contextmanager, redirect_stdout, nullcontext
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl

from ..diffusion import dpm_solver, plms, uni_pc
from ...visualization.plots import plot_precip_image, reverse_transform_R


_SAMPLERS = {
    "plms": plms.PLMSSampler,
    "dpmpp": dpm_solver.DPMSolverSampler,
    "unipc": uni_pc.UniPCSampler,
}


class SamplePredictionLogger(pl.Callback):
    def __init__(self, every_n_epochs=1, num_diffusion_iters=20, max_leadtimes=4,
                 sample_hw=128, use_ema=True, per_bin_cases=2,
                 ensemble_size=24, sampler="dpmpp", eval_seed=1234,
                 scan_per_bin=32, pop_threshold=0.1,
                 ensemble_batch=None, evict_optimizer=False):
        super().__init__()
        if sampler not in _SAMPLERS:
            raise ValueError(
                f"unknown sampler {sampler!r}; choose from {list(_SAMPLERS)}"
            )
        self.every_n_epochs = max(int(every_n_epochs), 0)
        self.num_diffusion_iters = int(num_diffusion_iters)
        self.max_leadtimes = int(max_leadtimes)
        self.sample_hw = sample_hw  # center-crop preview (None = full)
        self.use_ema = use_ema
        self.per_bin_cases = max(int(per_bin_cases), 1)
        self.ensemble_size = max(int(ensemble_size), 1)
        # Max ensemble members sampled in ONE batched forward (None = all at
        # once). Members share conditioning, so batching collapses per-member
        # overhead -- N members run in ~the wall-clock of 1, up to the VRAM
        # ceiling. Chunk below that ceiling to raise ensemble_size without OOM.
        self.ensemble_batch = (max(int(ensemble_batch), 1)
                               if ensemble_batch else None)
        # Also offload the 8-bit optimizer state to CPU during eval (~1.3 GB
        # more headroom). Opt-in: bnb quantized state round-tripped GPU<->CPU is
        # value-preserving in principle but less battle-tested than the EMA/grad
        # eviction, so default off. Enable if you need the extra room.
        self.evict_optimizer = bool(evict_optimizer)
        self.scan_per_bin = max(int(scan_per_bin), self.per_bin_cases)
        self.pop_threshold = float(pop_threshold)  # "any rain" thr for PoP maps
        self.sampler = sampler
        self.eval_seed = int(eval_seed)
        self._cases = None         # list of (bin_idx, case_in_bin, past_b, future_b)
        self._past_steps = None    # cached from datamodule on first val

    @contextmanager
    def _ema_weights(self, trainer, pl_module):
        """Run the eval on the EMA weights, with only the EMA model on the GPU.

        The sampling forward pass needs just the UNet (here loaded with the EMA
        weights) + the tiny autoencoder/context encoder -- everything else
        resident is training-only machinery. On enter we therefore:
          1. back the live params up to CPU, copy the EMA shadow -> model (GPU);
          2. offload the now-redundant EMA shadow buffers to CPU (~2.7 GB);
          3. drop gradients (rebuilt by the next backward);
          4. (opt-in) offload the optimizer state to CPU (~1.3 GB);
        roughly doubling the headroom available for a batched ensemble. On exit
        every piece is restored so the next training step is unaffected. The
        transfers are sub-second; the eval is minutes. No-op if EMA unavailable.
        """
        ema = getattr(pl_module, "model_ema", None)
        if not (self.use_ema and getattr(pl_module, "use_ema", False)
                and ema is not None):
            yield
            return
        # 1. live params -> CPU; EMA shadow -> model (in place, GPU)
        backup = {n: p.detach().cpu().clone()
                  for n, p in pl_module.model.named_parameters()
                  if p.requires_grad}
        ema.copy_to(pl_module.model)
        prev = pl_module.use_ema
        pl_module.use_ema = False               # silence apply_model's ema_scope

        # 2. offload the EMA shadow buffers to CPU (values now live in `model`)
        shadow_dev = {}
        for n, b in ema.named_buffers():
            if b.is_cuda:
                shadow_dev[n] = b.device
                b.data = b.data.cpu()
        # 3. drop gradients (the next training backward repopulates them)
        for p in pl_module.model.parameters():
            p.grad = None
        # 4. (opt-in) offload optimizer state to CPU
        opt_moved = []
        if self.evict_optimizer:
            try:
                for opt in getattr(trainer, "optimizers", []) or []:
                    for st in opt.state.values():
                        for k, v in st.items():
                            if torch.is_tensor(v) and v.is_cuda:
                                opt_moved.append((st, k, v.device))
                                st[k] = v.cpu()
            except Exception as e:
                print(f"[SamplePredictionLogger] optimizer offload skipped: "
                      f"{type(e).__name__}: {e}")
        torch.cuda.empty_cache()
        try:
            yield
        finally:
            # restore in reverse: optimizer -> EMA shadow -> live params
            for st, k, dev in opt_moved:
                st[k] = st[k].to(dev)
            for n, b in ema.named_buffers():
                if n in shadow_dev:
                    b.data = b.data.to(shadow_dev[n])
            pl_module.use_ema = prev
            with torch.no_grad():
                params = dict(pl_module.model.named_parameters())
                for n, b in backup.items():
                    params[n].data.copy_(b.to(params[n].device))
            del backup
            torch.cuda.empty_cache()

    def _crop(self, t):
        if self.sample_hw is None:
            return t
        (H, W) = (t.shape[-2], t.shape[-1])
        (h, w) = (min(self.sample_hw, H), min(self.sample_hw, W))
        return t[..., (H - h) // 2:(H - h) // 2 + h, (W - w) // 2:(W - w) // 2 + w]

    def _fixed_cases(self, trainer):
        """Pick per_bin_cases cases from each LDCast intensity bin via dm.val_w.

        For each bin, scan scan_per_bin candidates and pick the top
        per_bin_cases by fraction of future pixels >= 1 mm/h on the cropped
        view. This filters out radar-clutter cases where a single extreme pixel
        lifts the 99th percentile into a high bin while the rest of the crop is
        dry -- those scored zero on all CSI thresholds.
        """
        if self._cases is not None:
            return self._cases
        dm = trainer.datamodule
        val_ds = dm.valid_ds
        val_w = dm.val_w
        self._past_steps = int(dm.past_steps)
        bin_values = sorted(np.unique(val_w).tolist())
        cases = []
        for bin_idx, w in enumerate(bin_values):
            row_idxs = np.flatnonzero(val_w == w)[: self.scan_per_bin]
            scored = []  # (score, past_b, future_b)
            for ridx in row_idxs:
                past_t, future_t = val_ds[int(ridx)]
                past_b = self._crop(past_t.unsqueeze(0)).clone()
                future_b = self._crop(future_t.unsqueeze(0)).clone()
                future_mmhr = reverse_transform_R(
                    future_b[0].float().cpu().numpy()
                )
                score = float((future_mmhr >= 1.0).mean())
                scored.append((score, past_b, future_b))
            scored.sort(key=lambda x: -x[0])
            for case_in_bin, (_, past_b, future_b) in enumerate(
                scored[: self.per_bin_cases]
            ):
                cases.append((bin_idx, int(case_in_bin), past_b, future_b))
        self._cases = cases
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
        if not cases:
            return
        sampler = _SAMPLERS[self.sampler](pl_module)
        amp = (torch.autocast("cuda", dtype=torch.bfloat16)
               if dev.type == "cuda" else nullcontext())
        thresholds = (0.1, 1.0, 5.0)
        num_bins = max(b for b, _, _, _ in cases) + 1

        # Per-(bin, threshold) hit/miss/FA accumulators, summed over case+member.
        counts = {(b, thr): [0, 0, 0]
                  for b in range(num_bins) for thr in thresholds}
        # Per-(lead_time, threshold) hit/miss/FA, summed over bin+case+member.
        # Answers "how far out is the forecast usable" -- the pooled-over-time
        # CSI/POD washes strong near-term steps together with weak late ones.
        lt_counts = defaultdict(lambda: [0, 0, 0])     # key (t, thr)
        # Per-bin ensemble-mean MAE / RMSE accumulators.
        mae = {b: [0.0, 0] for b in range(num_bins)}   # (abs_err_sum, n_px)
        sq  = {b: [0.0, 0] for b in range(num_bins)}   # (sq_err_sum, n_px)

        step = trainer.global_step
        png_dir = Path(logger.log_dir) / "eval_pngs" / f"step_{step:06d}"
        png_dir.mkdir(parents=True, exist_ok=True)

        t_past = torch.arange(
            -self._past_steps + 1, 1, dtype=torch.float32, device=dev
        ).unsqueeze(0)  # [1, T_past]

        # For the TB summary image, pick the wettest case after the loop.
        wettest = None       # (bin_idx, case_in_bin, truth, pred0)
        wettest_score = -float("inf")

        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ema_ctx = self._ema_weights(trainer, pl_module)  # EMA + free training VRAM
        ema_ctx.__enter__()
        try:
            for ci, (bin_idx, case_in_bin, past_b, future_b) in enumerate(cases):
                pl_module._cond_cache = None  # clear between cases; reuse across members
                past_b = past_b.to(dev)
                future_b = future_b.to(dev)
                gen_shape = tuple(pl_module.autoencoder.encode(future_b)[0].shape[1:])
                truth_mmhr = reverse_transform_R(
                    future_b[0, 0].float().cpu().numpy()
                )  # (T, H, W) mm/h
                case_preds = []

                # Sample the ensemble in batched chunks. Members share the same
                # conditioning (only the noise differs), so we repeat the past
                # along the batch dim and let one sampler call denoise all `cb`
                # members at once -- N members cost ~1 member's wall-clock up to
                # the VRAM ceiling. One RNG seed per chunk; members within a
                # chunk get distinct noise rows from the [cb, ...] draw.
                members = self.ensemble_size
                mb = self.ensemble_batch or members
                for start in range(0, members, mb):
                    cb = min(mb, members - start)
                    seed = self.eval_seed + 1000 * ci + start
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    past_rep = past_b.repeat(cb, *([1] * (past_b.dim() - 1)))
                    t_rep = t_past.repeat(cb, 1)
                    pl_module._cond_cache = None
                    with amp, redirect_stdout(io.StringIO()):
                        (s, _) = sampler.sample(
                            self.num_diffusion_iters,
                            cb, gen_shape, [[past_rep, t_rep]],
                            progbar=False, verbose=False,
                        )
                        y_pred = pl_module.autoencoder.decode(
                            s / pl_module.scale_factor
                        )
                    for j in range(cb):
                        case_preds.append(
                            reverse_transform_R(
                                y_pred[j, 0].float().cpu().numpy()
                            )
                        )

                # CSI / POD / FAR: count each member's prediction once.
                for pred_mmhr in case_preds:
                    for thr in thresholds:
                        p, o = pred_mmhr >= thr, truth_mmhr >= thr
                        # (T, H, W) -> per-lead-time counts by summing H, W only
                        hits_t = (p & o).sum(axis=(1, 2))
                        miss_t = (~p & o).sum(axis=(1, 2))
                        fa_t = (p & ~o).sum(axis=(1, 2))
                        counts[(bin_idx, thr)][0] += int(hits_t.sum())
                        counts[(bin_idx, thr)][1] += int(miss_t.sum())
                        counts[(bin_idx, thr)][2] += int(fa_t.sum())
                        for t in range(pred_mmhr.shape[0]):
                            lt = lt_counts[(t, thr)]
                            lt[0] += int(hits_t[t])
                            lt[1] += int(miss_t[t])
                            lt[2] += int(fa_t[t])

                # ensemble mean -> per-bin MAE/RMSE
                pred_mean = np.mean(np.stack(case_preds, axis=0), axis=0)
                diff = pred_mean - truth_mmhr
                mae[bin_idx][0] += float(np.abs(diff).sum())
                mae[bin_idx][1] += diff.size
                sq[bin_idx][0]  += float((diff ** 2).sum())
                sq[bin_idx][1]  += diff.size

                # save per-case ensemble PNG
                try:
                    self._save_case_png(
                        png_dir / f"bin{bin_idx:02d}_case{case_in_bin}.png",
                        truth_mmhr, case_preds,
                    )
                except Exception as e:
                    print(f"[SamplePredictionLogger] PNG skip "
                          f"bin{bin_idx:02d}_case{case_in_bin}: "
                          f"{type(e).__name__}: {e}")

                # PoP map + decision strip (reuses the ensemble members already
                # sampled -- zero extra diffusion cost). Only for cases that
                # actually contain rain; on a dry crop PoP is uninformative.
                if float((truth_mmhr >= self.pop_threshold).mean()) > 0.0:
                    try:
                        self._save_pop_png(
                            png_dir / f"pop_bin{bin_idx:02d}_case{case_in_bin}.png",
                            truth_mmhr, case_preds, self.pop_threshold,
                        )
                    except Exception as e:
                        print(f"[SamplePredictionLogger] PoP PNG skip "
                              f"bin{bin_idx:02d}_case{case_in_bin}: "
                              f"{type(e).__name__}: {e}")

                # track wettest case for the TB summary image
                score = float(truth_mmhr.mean())
                if score > wettest_score:
                    wettest_score = score
                    wettest = (bin_idx, case_in_bin, truth_mmhr, case_preds[0])
        finally:
            ema_ctx.__exit__(None, None, None)   # restore live weights from CPU
            pl_module._cond_cache = None
            torch.set_rng_state(rng_cpu)
            if rng_cuda is not None:
                torch.cuda.set_rng_state_all(rng_cuda)

        # Emit per-bin and overall CSI / POD / FAR scalars to TB.
        # POD = hits / (hits + misses)             -- recall; higher = fewer missed rain events
        # FAR = false_alarms / (hits + false_alarms) -- "how often a rain forecast was wrong";
        #                                              lower = better, but high FAR is acceptable
        #                                              if POD stays high ("will I get wet" use case)
        for thr in thresholds:
            for b in range(num_bins):
                h, m, f_ = counts[(b, thr)]
                if h + m + f_:
                    logger.experiment.add_scalar(
                        f"val/csi_{thr}mm/bin{b:02d}",
                        h / (h + m + f_), global_step=step,
                    )
                if h + m:
                    logger.experiment.add_scalar(
                        f"val/pod_{thr}mm/bin{b:02d}",
                        h / (h + m), global_step=step,
                    )
                if h + f_:
                    logger.experiment.add_scalar(
                        f"val/far_{thr}mm/bin{b:02d}",
                        f_ / (h + f_), global_step=step,
                    )
            h = sum(counts[(b, thr)][0] for b in range(num_bins))
            m = sum(counts[(b, thr)][1] for b in range(num_bins))
            f_ = sum(counts[(b, thr)][2] for b in range(num_bins))
            if h + m + f_:
                csi_overall = h / (h + m + f_)
                logger.experiment.add_scalar(
                    f"val/csi_{thr}mm/overall", csi_overall, global_step=step,
                )
                # Mirror the csi_1.0mm overall to a self.log'd metric so the
                # best-by-CSI ModelCheckpoint can monitor it (add_scalar above
                # is invisible to trainer.callback_metrics). Slash-free name
                # keeps the checkpoint filename clean.
                if thr == 1.0:
                    pl_module.log(
                        "val_csi", csi_overall,
                        on_step=False, on_epoch=True, sync_dist=True,
                    )
            if h + m:
                logger.experiment.add_scalar(
                    f"val/pod_{thr}mm/overall",
                    h / (h + m), global_step=step,
                )
            if h + f_:
                logger.experiment.add_scalar(
                    f"val/far_{thr}mm/overall",
                    f_ / (h + f_), global_step=step,
                )

        # Per-lead-time CSI / POD / FAR (summed across bins+cases+members),
        # in their own `_lead` TB namespace so each is a clean chart of T lines.
        # Radar cadence is 10 min/step, so lt<NN> = +(NN+1)*10 min into the
        # future (lt00 = +10 min nearest, increasing = further out).
        leads = sorted({t for (t, _) in lt_counts})
        for thr in thresholds:
            for t in leads:
                h, m, f_ = lt_counts[(t, thr)]
                if h + m + f_:
                    logger.experiment.add_scalar(
                        f"val/csi_{thr}mm_lead/lt{t:02d}",
                        h / (h + m + f_), global_step=step,
                    )
                if h + m:
                    logger.experiment.add_scalar(
                        f"val/pod_{thr}mm_lead/lt{t:02d}",
                        h / (h + m), global_step=step,
                    )
                if h + f_:
                    logger.experiment.add_scalar(
                        f"val/far_{thr}mm_lead/lt{t:02d}",
                        f_ / (h + f_), global_step=step,
                    )

        # Per-bin and overall MAE / RMSE.
        for b in range(num_bins):
            ae, n = mae[b]
            sqe, _ = sq[b]
            if n:
                logger.experiment.add_scalar(
                    f"val/mae_mmhr/bin{b:02d}", ae / n, global_step=step,
                )
                logger.experiment.add_scalar(
                    f"val/rmse_mmhr/bin{b:02d}", (sqe / n) ** 0.5, global_step=step,
                )
        ae_total = sum(mae[b][0] for b in range(num_bins))
        n_total  = sum(mae[b][1] for b in range(num_bins))
        sq_total = sum(sq[b][0]  for b in range(num_bins))
        if n_total:
            logger.experiment.add_scalar(
                "val/mae_mmhr/overall", ae_total / n_total, global_step=step,
            )
            logger.experiment.add_scalar(
                "val/rmse_mmhr/overall", (sq_total / n_total) ** 0.5,
                global_step=step,
            )

        # TB summary image: wettest case, member 0.
        if wettest is not None:
            try:
                self._log_wettest_image(logger, step, wettest)
            except Exception as e:
                print(f"[SamplePredictionLogger] TB image skip: "
                      f"{type(e).__name__}: {e}")

    @staticmethod
    def _pm_mean(stack):
        """Probability-matched ensemble mean (Ebert 2001), per lead time.

        Takes the spatial *pattern* of the plain ensemble mean (which is
        smooth but blurred and under-intense) and reassigns pixel values from
        the pooled intensity distribution of all members, so peak rates are
        restored while the field stays denoised. stack: (M, T, H, W) -> (T, H, W).
        """
        M, T, H, W = stack.shape
        out = np.empty((T, H, W), dtype=stack.dtype)
        for t in range(T):
            field = stack[:, t]                         # (M, H, W)
            order = np.argsort(field.mean(axis=0).ravel())  # ranks of mean field
            pooled = np.sort(field.ravel())             # all M*H*W intensities
            sampled = pooled[(M - 1)::M][:H * W]         # H*W, distribution-matched
            flat = np.empty(H * W, dtype=stack.dtype)
            flat[order] = sampled                       # largest value -> wettest px
            out[t] = flat.reshape(H, W)
        return out

    def _save_case_png(self, path, truth, preds):
        """One PNG: truth + member rows + PM-mean row, x T columns (lead times)."""
        T = truth.shape[0]
        pm = self._pm_mean(np.stack(preds, axis=0))     # (T, H, W)
        n_rows = 1 + len(preds) + 1                      # truth + members + PM mean
        fig, axs = plt.subplots(
            n_rows, T,
            figsize=(1.5 * T, 2.0 * n_rows),
            squeeze=False, constrained_layout=True,
        )
        im = None
        for t in range(T):
            im = plot_precip_image(axs[0, t], truth[t].copy())
            axs[0, t].set_title(f"+{(t + 1) * 10}min", fontsize=8)
            for r, pred in enumerate(preds):
                im = plot_precip_image(axs[r + 1, t], pred[t].copy())
            im = plot_precip_image(axs[n_rows - 1, t], pm[t].copy())
        axs[0, 0].set_ylabel("truth", fontsize=10)
        for r in range(len(preds)):
            axs[r + 1, 0].set_ylabel(f"member {r}", fontsize=10)
        axs[n_rows - 1, 0].set_ylabel("PM mean", fontsize=10)
        if im is not None:
            fig.colorbar(im, ax=axs, shrink=0.8, label="mm/h")
        fig.savefig(path, dpi=80)
        plt.close(fig)

    def _save_pop_png(self, path, truth, preds, threshold):
        """PoP map (truth + PM mean + probability per lead time) + decision strip.

        Probability of precipitation = fraction of ensemble members with rain
        >= threshold, per pixel per lead time. With `ensemble_size` members the
        granularity is 1/ensemble_size (4 -> 25% steps). The decision strip is
        the per-lead-time rain probability at the wettest truth pixel -- the
        "should I wait N minutes" readout. 10 min/radar step.
        """
        stack = np.stack(preds, axis=0)              # (M, T, H, W)
        M = stack.shape[0]
        pop = (stack >= threshold).mean(axis=0)      # (T, H, W) in [0, 1]
        pm = self._pm_mean(stack)                    # (T, H, W) mm/h
        T = truth.shape[0]
        leads = [(t + 1) * 10 for t in range(T)]
        # wettest truth pixel (summed over time) for the decision strip
        flat = truth.sum(axis=0)
        py, px = np.unravel_index(int(flat.argmax()), flat.shape)

        fig = plt.figure(figsize=(1.7 * T, 8.0), constrained_layout=True)
        gs = fig.add_gridspec(4, T, height_ratios=[2, 2, 2, 1.6])
        rate_axes, pop_axes = [], []   # rate_axes share the mm/h colorbar
        im_t = im_p = None
        for t in range(T):
            ax0 = fig.add_subplot(gs[0, t])
            im_t = plot_precip_image(ax0, truth[t].copy())
            ax0.set_title(f"+{leads[t]}min", fontsize=8)
            rate_axes.append(ax0)
            axm = fig.add_subplot(gs[1, t])
            im_t = plot_precip_image(axm, pm[t].copy())
            rate_axes.append(axm)
            ax1 = fig.add_subplot(gs[2, t])
            im_p = ax1.imshow(pop[t], vmin=0, vmax=1, cmap="viridis")
            ax1.plot(px, py, "rx", markersize=6)  # mark the strip location
            ax1.set_xticks([]); ax1.set_yticks([])
            pop_axes.append(ax1)
            if t == 0:
                ax0.set_ylabel("truth (mm/h)", fontsize=9)
                axm.set_ylabel("PM mean (mm/h)", fontsize=9)
                ax1.set_ylabel(f"P(rain≥{threshold}) [n={M}]", fontsize=9)
        if im_t is not None:
            fig.colorbar(im_t, ax=rate_axes, shrink=0.8, label="mm/h",
                         location="right")
        if im_p is not None:
            fig.colorbar(im_p, ax=pop_axes, shrink=0.8, label="probability",
                         location="right")
        # decision strip across the full bottom row
        axs = fig.add_subplot(gs[3, :])
        for thr in (0.1, 1.0, 5.0):
            cnt = (stack[:, :, py, px] >= thr).sum(axis=0)   # (T,) members hitting
            axs.plot(leads, cnt / M * 100, marker="o", markersize=4, label=f"≥{thr}")
            if thr == threshold:  # label the "any rain" line with member counts
                for x, c in zip(leads, cnt):
                    axs.annotate(f"{int(c)}/{M}", (x, c / M * 100),
                                 textcoords="offset points", xytext=(0, 5),
                                 ha="center", fontsize=6)
        axs.set_ylim(0, 100)
        axs.set_xlabel("lead time (min)", fontsize=9)
        axs.set_ylabel("P(rain) %", fontsize=9)
        axs.set_title(f"decision strip @ wettest pixel ({px},{py}) — n={M} members",
                      fontsize=9)
        axs.grid(True, alpha=0.3)
        axs.legend(fontsize=8, ncol=3)
        fig.savefig(path, dpi=80)
        plt.close(fig)

    def _log_wettest_image(self, logger, step, wettest):
        bin_idx, case_in_bin, truth, pred = wettest
        T = truth.shape[0]
        n = min(T, self.max_leadtimes)
        cols = ([round(i * (T - 1) / (n - 1)) for i in range(n)]
                if n > 1 else [T - 1])
        fig, axs = plt.subplots(
            2, len(cols), figsize=(3 * len(cols), 6.5),
            squeeze=False, constrained_layout=True,
        )
        im = None
        for c, t in enumerate(cols):
            plot_precip_image(axs[0, c], truth[t].copy())
            im = plot_precip_image(axs[1, c], pred[t].copy())
            axs[0, c].set_title(f"+{(t + 1) * 10}min", fontsize=9)
        axs[0, 0].set_ylabel(f"truth (bin{bin_idx:02d})", fontsize=10)
        axs[1, 0].set_ylabel("prediction (m0)", fontsize=10)
        if im is not None:
            fig.colorbar(im, ax=axs, shrink=0.8, label="mm/h")
        logger.experiment.add_figure("val/forecast", fig, global_step=step)
        plt.close(fig)
