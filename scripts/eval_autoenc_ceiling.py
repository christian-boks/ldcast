"""Autoencoder reconstruction-ceiling test (no training, no diffusion).

The diffusion model generates a latent and decodes it through the FROZEN
stage-1 autoencoder, so the AE's own round-trip is a hard upper bound on the
whole pipeline's forecast skill: the diffusion model can never produce a
sharper field than `decode(encode(truth))`. This script measures that ceiling
directly -- encode then decode the ground-truth FUTURE field (posterior MEAN,
the AE's best-case, noise-free reconstruction) and score it with the SAME
cases (`monitor._fixed_cases`) and the SAME CSI/POD/FAR math the in-training
monitor uses. So the numbers are directly comparable to `val/csi_*mm`.

Reading the result:
  - recon csi_5 ~ the diffusion model's ~0.01  -> the AE is the ceiling. No
    diffusion-side change (more epochs, rain-weighted loss) can lift heavy rain;
    only a stage-1 AE retrain can.
  - recon csi_5 >> 0.01 (e.g. > 0.3)           -> the AE round-trips heavy rain
    fine; the bottleneck is the diffusion model, so a rain-weighted diffusion
    loss is the lever.

Usage (from scripts/):
    uv run python eval_autoenc_ceiling.py
        [--config=../config/train_rust.yaml]
        [--ckpt=../models/genforecast_rust/last.ckpt]
        [--sample_posterior=False]   # True = score a posterior SAMPLE (matches
                                     #        the training loss) instead of mean
"""
import os
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from omegaconf import OmegaConf

from ldcast.features.rust_data import RustRadarDataModule
from ldcast.models.genforecast.monitor import SamplePredictionLogger
from ldcast.visualization.plots import plot_precip_image, reverse_transform_R

from train_genforecast import setup_model


class _MockTrainer:
    """_fixed_cases only reads trainer.datamodule (.valid_ds/.val_w/.past_steps)."""
    def __init__(self, datamodule):
        self.datamodule = datamodule


def _csi_pod_far(h, m, f):
    csi = h / (h + m + f) if (h + m + f) else None
    pod = h / (h + m) if (h + m) else None
    far = f / (h + f) if (h + f) else None
    return csi, pod, far


def run(
    config="../config/train_rust.yaml",
    ckpt="../models/genforecast_rust/last.ckpt",
    sample_posterior=False,
    per_bin_cases=2,
    scan_per_bin=32,
    out_dir=None,
    save_pngs=3,          # dump truth-vs-recon PNGs for the N wettest cases
):
    if not os.path.isfile(ckpt):
        sys.exit(f"checkpoint not found: {ckpt}")
    cfg = OmegaConf.load(config)
    if cfg.get("radar_root"):
        os.environ["DGMR_RADAR_ROOT"] = cfg.radar_root
    out_dir = out_dir or os.path.join(
        os.path.dirname(os.path.abspath(ckpt)), "autoenc_ceiling"
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"checkpoint:       {ckpt}")
    print(f"sample_posterior: {sample_posterior}  "
          f"({'posterior sample (matches train loss)' if sample_posterior else 'posterior mean (best-case ceiling)'})")
    print(f"output:           {out_dir}/\n")

    print("Loading data...")
    dm = RustRadarDataModule(
        index_path=cfg.index_path,
        mode="diffusion",
        past_steps=cfg.past_steps,
        future_steps=cfg.future_steps,
        height=cfg.height,
        width=cfg.width,
        batch_size=cfg.genforecast_batch_size,
        num_workers=0,
        valid_frac=0.1,
        seed=42,
        use_weighted_sampler=cfg.use_weighted_sampler,
    )
    dm.setup("fit")

    print("Building model (loads frozen autoencoder)...")
    ldm, _ = setup_model(
        num_timesteps=cfg.future_steps // 4,
        autoenc_weights_fn=os.path.join(cfg.autoenc_dir, "state_dict.pt"),
        use_obs=True,
        use_nwp=False,
        model_dir=cfg.genforecast_dir,
        lr=cfg.genforecast_lr,
        precision=cfg.precision,
        optimizer_8bit=cfg.optimizer_8bit,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        scale_factor=1.0,
        gradient_clip_val=1.0,
        sample_every_n_epochs=1,
        max_hours=None,
        early_stopping_patience=0,
        accumulate_grad_batches=1,
        save_top_k=0,
    )
    print(f"Loading weights from {ckpt} (uses the exact frozen AE the val/csi numbers came through)...")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    ldm.load_state_dict(sd, strict=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = ldm.autoencoder.to(dev).eval()

    # Identical case set to the training monitor.
    monitor = SamplePredictionLogger(
        sample_hw=cfg.height, per_bin_cases=per_bin_cases, scan_per_bin=scan_per_bin,
    )
    cases = monitor._fixed_cases(_MockTrainer(dm))
    if not cases:
        sys.exit("no cases selected")

    thresholds = (0.1, 1.0, 5.0)
    num_bins = max(b for b, _, _, _ in cases) + 1
    counts = {(b, thr): [0, 0, 0] for b in range(num_bins) for thr in thresholds}
    lt_counts = defaultdict(lambda: [0, 0, 0])
    abs_err = [0.0, 0]      # (sum|truth-recon|, npx) in mm/h
    sq_err = 0.0
    peaks = []              # (truth_p999, recon_p999, truth_max, recon_max) per case
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if dev.type == "cuda" else nullcontext())

    print(f"Running AE round-trip on {len(cases)} cases "
          f"({per_bin_cases}/bin x {num_bins} bins)...\n")
    case_scores = []
    for (bin_idx, case_in_bin, past_b, future_b) in cases:
        future_b = future_b.to(dev)
        with torch.no_grad(), amp:
            y_recon = ae(future_b, sample_posterior=sample_posterior)[0]
        truth_mmhr = reverse_transform_R(future_b[0, 0].float().cpu().numpy())  # (T,H,W)
        recon_mmhr = reverse_transform_R(y_recon[0, 0].float().cpu().numpy())

        for thr in thresholds:
            p, o = recon_mmhr >= thr, truth_mmhr >= thr
            hits_t = (p & o).sum(axis=(1, 2))
            miss_t = (~p & o).sum(axis=(1, 2))
            fa_t = (p & ~o).sum(axis=(1, 2))
            counts[(bin_idx, thr)][0] += int(hits_t.sum())
            counts[(bin_idx, thr)][1] += int(miss_t.sum())
            counts[(bin_idx, thr)][2] += int(fa_t.sum())
            for t in range(recon_mmhr.shape[0]):
                lt = lt_counts[(t, thr)]
                lt[0] += int(hits_t[t]); lt[1] += int(miss_t[t]); lt[2] += int(fa_t[t])

        d = recon_mmhr - truth_mmhr
        abs_err[0] += float(np.abs(d).sum()); abs_err[1] += d.size
        sq_err += float((d ** 2).sum())
        peaks.append((np.percentile(truth_mmhr, 99.9), np.percentile(recon_mmhr, 99.9),
                      float(truth_mmhr.max()), float(recon_mmhr.max())))
        case_scores.append((float(truth_mmhr.mean()), bin_idx, case_in_bin,
                            truth_mmhr, recon_mmhr))

    # --- table: per-bin + overall CSI/POD/FAR ---
    print("=== AE RECONSTRUCTION-OF-TRUTH skill (the pipeline ceiling) ===\n")
    print("          |       0.1 mm/h         |        1.0 mm/h        |        5.0 mm/h        ")
    print("    bin   |   CSI     POD     FAR  |   CSI     POD     FAR  |   CSI     POD     FAR  ")
    print("    ----  |  -----   -----  -----  |  -----   -----  -----  |  -----   -----  -----  ")

    def fmt(v):
        return "  --  " if v is None else f" {v:.3f}"

    def row_for(getter):
        cells = []
        for thr in thresholds:
            h, m, f = getter(thr)
            csi, pod, far = _csi_pod_far(h, m, f)
            cells += [fmt(csi), fmt(pod), fmt(far), " | "]
        return "".join(cells)

    for b in range(num_bins):
        print(f"    bin{b:02d}  | " + row_for(lambda thr, b=b: counts[(b, thr)]))
    print("    ----  | ----------------------- | ----------------------- | ----------------------")
    def overall(thr):
        return [sum(counts[(b, thr)][i] for b in range(num_bins)) for i in range(3)]
    print("    OVER  | " + row_for(overall))

    # --- per-lead csi (does the round-trip degrade across the T axis?) ---
    leads = sorted({t for (t, _) in lt_counts})
    print("\n  per-lead-time recon CSI (10 min/step):")
    print("    lead   " + "  ".join(f"+{(t+1)*10:>3}m" for t in leads))
    for thr in thresholds:
        vals = []
        for t in leads:
            h, m, f = lt_counts[(t, thr)]
            csi = h / (h + m + f) if (h + m + f) else None
            vals.append(" --  " if csi is None else f"{csi:.3f}")
        print(f"    csi_{thr}".ljust(11) + "  ".join(vals))

    # --- peak preservation (diagnoses smoothing/clipping of heavy rain) ---
    peaks = np.array(peaks)  # (N, 4)
    print("\n  peak preservation (mean over cases, mm/h):")
    print(f"    99.9th pct:  truth={peaks[:,0].mean():6.2f}   recon={peaks[:,1].mean():6.2f}"
          f"   recon/truth={peaks[:,1].mean()/max(peaks[:,0].mean(),1e-9):.2f}")
    print(f"    max:         truth={peaks[:,2].mean():6.2f}   recon={peaks[:,3].mean():6.2f}"
          f"   recon/truth={peaks[:,3].mean()/max(peaks[:,2].mean(),1e-9):.2f}")
    print(f"\n  reconstruction error (mm/h):  "
          f"MAE={abs_err[0]/abs_err[1]:.4f}   RMSE={(sq_err/abs_err[1])**0.5:.4f}")

    # --- verdict helper ---
    h, m, f = overall(5.0)
    csi5, _, _ = _csi_pod_far(h, m, f)
    h, m, f = overall(1.0)
    csi1, _, _ = _csi_pod_far(h, m, f)
    print("\n  --- ceiling vs current diffusion pipeline (val/csi_*mm/overall) ---")
    print(f"    recon csi_1.0 = {csi1:.3f}   (diffusion ~0.10)")
    print(f"    recon csi_5.0 = {csi5:.4f}  (diffusion ~0.011)")

    # --- artifacts: truth-vs-recon PNGs for the wettest cases ---
    if save_pngs:
        case_scores.sort(key=lambda x: -x[0])
        for k, (_, bin_idx, case_in_bin, truth, recon) in enumerate(case_scores[:save_pngs]):
            try:
                _save_pair_png(
                    os.path.join(out_dir, f"recon_bin{bin_idx:02d}_case{case_in_bin}.png"),
                    truth, recon)
            except Exception as e:
                print(f"  [png skip] {type(e).__name__}: {e}")
        print(f"\n  truth-vs-recon PNGs: {out_dir}/recon_bin*.png")


def _save_pair_png(path, truth, recon):
    from matplotlib import pyplot as plt
    T = truth.shape[0]
    fig, axs = plt.subplots(2, T, figsize=(1.5 * T, 4.2), squeeze=False,
                            constrained_layout=True)
    im = None
    for t in range(T):
        im = plot_precip_image(axs[0, t], truth[t].copy())
        plot_precip_image(axs[1, t], recon[t].copy())
        axs[0, t].set_title(f"+{(t+1)*10}min", fontsize=8)
    axs[0, 0].set_ylabel("truth", fontsize=10)
    axs[1, 0].set_ylabel("AE recon", fontsize=10)
    if im is not None:
        fig.colorbar(im, ax=axs, shrink=0.8, label="mm/h")
    fig.savefig(path, dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    Fire(run)
