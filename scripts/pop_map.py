"""Probability-of-precipitation (PoP) map + decision strip from an LDCast ckpt.

Runs an N-member diffusion ensemble for one case and collapses it into the
product that actually answers "will I get wet on my run":

  1. PoP map  -- per pixel x lead time, the fraction of ensemble members with
     rain >= threshold. A single 0-100% map per lead time (white = dry-confident,
     dark = rain-confident). This is the "where will it rain, and how sure" view.
  2. Decision strip -- for one location (default: center pixel), rain probability
     vs lead time. The literal "should I wait 30 min before going out" readout.

Radar cadence is 10 min/step, so lead i -> +(i+1)*10 min.

Usage (from scripts/):
    uv run python pop_map.py                         # bin 8, case 0, 16 members
    uv run python pop_map.py --bin=6 --members=16
    uv run python pop_map.py --point=64,64 --threshold=0.1

Needs a free GPU (the trainer holds all 16 GB while running). Output goes to
<genforecast_dir>/pop_maps/.
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch
from fire import Fire
from omegaconf import OmegaConf

from ldcast.features.rust_data import RustRadarDataModule
from ldcast.models.genforecast.monitor import SamplePredictionLogger, _SAMPLERS
from ldcast.visualization.plots import plot_precip_image, reverse_transform_R

from train_genforecast import setup_model

MIN_PER_STEP = 10  # dgmr-rs radar cadence


def run(
    config="../config/train_rust.yaml",
    ckpt="../models/genforecast_rust/last.ckpt",
    bin=8,
    case=0,
    members=24,
    threshold=0.1,
    point=None,                 # "x,y" pixel for the decision strip; default center
    num_diffusion_iters=20,
    sampler="dpmpp",
    eval_seed=1234,
    use_ema=True,               # sample the EMA weights (the deployable model);
                                # False = raw/live weights, for an A/B comparison
    out_dir=None,
):
    if not os.path.isfile(ckpt):
        sys.exit(f"checkpoint not found: {ckpt}")
    cfg = OmegaConf.load(config)
    if cfg.get("radar_root"):
        os.environ["DGMR_RADAR_ROOT"] = cfg.radar_root
    out_dir = out_dir or os.path.join(cfg.genforecast_dir, "pop_maps")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    dm = RustRadarDataModule(
        index_path=cfg.index_path, mode="diffusion",
        past_steps=cfg.past_steps, future_steps=cfg.future_steps,
        height=cfg.height, width=cfg.width,
        batch_size=cfg.genforecast_batch_size, num_workers=0,
        valid_frac=0.1, seed=42, use_weighted_sampler=cfg.use_weighted_sampler,
    )
    dm.setup("fit")

    print("Building model...")
    ldm, _ = setup_model(
        num_timesteps=cfg.future_steps // 4,
        autoenc_weights_fn=os.path.join(cfg.autoenc_dir, "state_dict.pt"),
        use_obs=True, use_nwp=False, model_dir=cfg.genforecast_dir,
        lr=cfg.genforecast_lr, precision=cfg.precision,
        optimizer_8bit=cfg.optimizer_8bit, max_epochs=1,
        limit_train_batches=1, limit_val_batches=1, scale_factor=1.0,
        gradient_clip_val=1.0, sample_every_n_epochs=1, max_hours=None,
        early_stopping_patience=0, accumulate_grad_batches=1, save_top_k=0,
    )
    print(f"Loading weights from {ckpt}...")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    ldm.load_state_dict(sd, strict=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ldm = ldm.to(dev).eval()
    # Sample the EMA shadow weights -- the deployable model and the same weights
    # the training monitor's eval images/CSI/best-ckpt use. Copy the EMA shadow
    # into the live model once, then leave use_ema=False so apply_model's
    # per-step ema_scope doesn't swap again. (Checkpoints carry model_ema.*.)
    if use_ema and getattr(ldm, "model_ema", None) is not None:
        ldm.model_ema.copy_to(ldm.model)
        print("Using EMA weights.")
    else:
        print("Using raw/live weights (EMA disabled or absent).")
    ldm.use_ema = False

    # Reuse the monitor's bin-stratified case selection (scans by frac_gt_1mm).
    picker = SamplePredictionLogger(
        sample_hw=cfg.height, per_bin_cases=case + 1, ensemble_size=1,
    )

    class _T:  # minimal trainer shim for _fixed_cases
        datamodule = dm
    cases = picker._fixed_cases(_T())
    match = [c for c in cases if c[0] == bin and c[1] == case]
    if not match:
        sys.exit(f"no case for bin={bin} case={case}; "
                 f"available bins: {sorted({c[0] for c in cases})}")
    _, _, past_b, future_b = match[0]
    past_b, future_b = past_b.to(dev), future_b.to(dev)

    T = future_b.shape[2]
    sampler_obj = _SAMPLERS[sampler](ldm)
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if dev.type == "cuda" else torch.autocast("cpu", enabled=False))
    t_past = torch.arange(-cfg.past_steps + 1, 1, dtype=torch.float32,
                          device=dev).unsqueeze(0)
    x = [[past_b, t_past]]
    gen_shape = tuple(ldm.autoencoder.encode(future_b)[0].shape[1:])
    truth = reverse_transform_R(future_b[0, 0].float().cpu().numpy())  # (T,H,W)

    print(f"Sampling {members} members ({sampler}/{num_diffusion_iters}) "
          f"for bin{bin:02d} case{case}...")
    preds = []
    with torch.no_grad():
        for m in range(members):
            torch.manual_seed(eval_seed + m)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(eval_seed + m)
            ldm._cond_cache = None
            with amp:
                s, _ = sampler_obj.sample(num_diffusion_iters, 1, gen_shape, x,
                                          progbar=False, verbose=False)
                y_pred = ldm.autoencoder.decode(s / ldm.scale_factor)
            preds.append(reverse_transform_R(y_pred[0, 0].float().cpu().numpy()))
            print(f"  member {m + 1}/{members}", end="\r")
    print()
    preds = np.stack(preds, axis=0)                  # (M, T, H, W)
    pop = (preds >= threshold).mean(axis=0)          # (T, H, W) in [0,1]
    pm = SamplePredictionLogger._pm_mean(preds)      # (T, H, W) mm/h

    leads = [(t + 1) * MIN_PER_STEP for t in range(T)]
    tag = f"bin{bin:02d}_case{case}_{threshold}mm"

    # --- 1. PoP map: truth + PM-mean + PoP rows, one column per lead time ---
    fig, axs = plt.subplots(3, T, figsize=(2.2 * T, 7.2),
                            squeeze=False, constrained_layout=True)
    im_truth = None
    for t in range(T):
        im_truth = plot_precip_image(axs[0, t], truth[t].copy())
        axs[0, t].set_title(f"+{leads[t]}min", fontsize=9)
        im_truth = plot_precip_image(axs[1, t], pm[t].copy())
        axs[1, t].set_xticks([]); axs[1, t].set_yticks([])
        im_pop = axs[2, t].imshow(pop[t], vmin=0, vmax=1, cmap="viridis")
        axs[2, t].set_xticks([]); axs[2, t].set_yticks([])
    axs[0, 0].set_ylabel("truth (mm/h)", fontsize=10)
    axs[1, 0].set_ylabel("PM mean (mm/h)", fontsize=10)
    axs[2, 0].set_ylabel(f"P(rain≥{threshold})", fontsize=10)
    fig.colorbar(im_truth, ax=axs[:2, :].ravel().tolist(), shrink=0.6, label="mm/h")
    fig.colorbar(im_pop, ax=axs[2, :], shrink=0.7, label="probability")
    fig.suptitle(f"PoP map — {tag} — {members} members", fontsize=12)
    pop_path = os.path.join(out_dir, f"popmap_{tag}.png")
    fig.savefig(pop_path, dpi=90)
    plt.close(fig)

    # --- 2. Decision strip: P(rain) vs lead time at one location ---
    H, W = pop.shape[1], pop.shape[2]
    if point is None:
        px, py = W // 2, H // 2
    else:
        px, py = (int(v) for v in str(point).split(","))
    # probabilities at multiple thresholds for the chosen point
    n_members = preds.shape[0]
    fig2, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for thr in (0.1, 1.0, 5.0):
        cnt = (preds[:, :, py, px] >= thr).sum(axis=0)    # (T,) members hitting
        ax.plot(leads, cnt / n_members * 100, marker="o", label=f"≥{thr} mm/h")
        if thr == threshold:  # label the "any rain" line with member counts
            for x, c in zip(leads, cnt):
                ax.annotate(f"{int(c)}/{n_members}",
                            (x, c / n_members * 100),
                            textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=7)
    ax.set_xlabel("lead time (min)")
    ax.set_ylabel("probability of rain (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Decision strip @ pixel ({px},{py}) — "
                 f"{tag.rsplit('_', 1)[0]} — n={n_members} members")
    ax.grid(True, alpha=0.3)
    ax.legend()
    strip_path = os.path.join(out_dir, f"strip_{tag.rsplit('_', 1)[0]}_pt{px}-{py}.png")
    fig2.savefig(strip_path, dpi=90)
    plt.close(fig2)

    # text summary of the decision strip at the "any rain" threshold
    cnt_any = (preds[:, :, py, px] >= threshold).sum(axis=0)
    print(f"\nP(rain≥{threshold}) at pixel ({px},{py}) [n={n_members} members]:")
    for t in range(T):
        frac = cnt_any[t] / n_members
        bar = "#" * int(round(frac * 20))
        print(f"  +{leads[t]:>3}min  {int(cnt_any[t]):2d}/{n_members}  "
              f"{frac * 100:5.0f}%  {bar}")
    print(f"\nWrote:\n  {pop_path}\n  {strip_path}")


if __name__ == "__main__":
    Fire(run)
