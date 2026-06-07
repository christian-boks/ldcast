"""Visual comparison of the A/B forecasters on the wettest Aalborg test cases.

Regenerates, on the SAME wettest-400 Aalborg held-out test cases used by
eval_aalborg_ab.py, each forecaster's field and renders them side by side:
  Truth | Eulerian | Lagrangian | Diffusion PM-mean | Diffusion PoP(>=1mm) |
  Prob-nowcaster P(>=1mm)
Rain-rate panels use the radar colormap (log scale); probability panels use a
linear 0-1 map with the truth's 1 mm contour overlaid (cyan) so you can see
whether the probability mass landed where it actually rained.

Produces two PNGs in <prob_dir>/ab_eval/:
  - compare_overview.png   : a spread of cases at the +40 min lead, all methods.
  - compare_evolution.png  : the strongest-initiation case, all leads, all methods
                             (the obj-4 growth showcase).

Usage (from scripts/):
    uv run python plot_aalborg_ab.py
"""
import io
import os
from contextlib import nullcontext, redirect_stdout
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.ndimage import shift as nd_shift

from ldcast.features.rust_data import RustRadarDataModule
from ldcast.models.diffusion import dpm_solver
from ldcast.models.autoenc import autoenc, encoder
from ldcast.models.nowcast.prob_nowcast import ProbNowcastNet, ProbNowcaster
from ldcast.models.genforecast.monitor import SamplePredictionLogger
from ldcast.visualization.plots import reverse_transform_R, plot_precip_image

from train_genforecast import setup_model
from eval_persistence_baseline import estimate_motion
from scipy.ndimage import distance_transform_edt

THR = 1.0          # headline rain/no-rain threshold for the probability panels
INTERVAL_MIN = 5   # 5 min/step -> 8 leads = +40 min (matches the obj-6 use case)
SAFETY_FACTOR = 1.5      # matches eval_aalborg_ab.py genesis definition
GENESIS_MARGIN_PX = 8


def genesis_events(past_mmhr, truth, thr=THR):
    """Genesis = advection predicts dry AND > reach px from any t0 echo, and it
    rained. Returns (Tf,H,W) bool of true-formation pixels (no advection-in)."""
    t0 = past_mmhr[-1]
    dy, dx = estimate_motion(past_mmhr)
    speed = float(np.hypot(dy, dx))
    dist0 = distance_transform_edt(~(t0 >= thr)).astype(np.float32)
    Tf = truth.shape[0]
    gen = np.zeros((Tf,) + t0.shape, dtype=bool)
    for k in range(Tf):
        reach = max(GENESIS_MARGIN_PX, SAFETY_FACTOR * speed * (k + 1))
        lagk = nd_shift(t0, (dy * (k + 1), dx * (k + 1)),
                        order=1, mode="constant", cval=0.0)
        gen[k] = (lagk < thr) & (dist0 > reach) & (truth[k] >= thr)
    return gen


def _rain_panel(ax, R_mmhr):
    return plot_precip_image(ax, R_mmhr.copy())   # copy: plot mutates (<thr -> nan)


def _prob_panel(ax, P, truth_lead):
    im = ax.imshow(P, cmap="magma", vmin=0, vmax=1)
    ax.contour(truth_lead >= THR, levels=[0.5], colors="cyan", linewidths=0.6)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def run(
    config="../config/train_rust.yaml",
    prob_ckpt="../models/prob_nowcaster_aalborg/last.ckpt",
    diff_ckpt="../models/genforecast_rust/last.ckpt",
    n_cases=400,
    scan_rows=5000,
    n_show=6,
    ensemble_size=32,
    region_center=(685, 852),
    region_radius=64,
    test_frac=0.1,
    valid_frac=0.1,
    num_diffusion_iters=20,
    eval_seed=1234,
    out_dir=None,
):
    cfg = OmegaConf.load(config)
    if cfg.get("radar_root"):
        os.environ["DGMR_RADAR_ROOT"] = cfg.radar_root
    out_dir = Path(out_dir or os.path.join(
        os.path.dirname(os.path.abspath(prob_ckpt)), "ab_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tp, Tf = int(cfg.past_steps), int(cfg.future_steps)

    # ---- data: same Aalborg temporal TEST split + wettest-400 selection ----
    print("Loading Aalborg test split...")
    dm = RustRadarDataModule(
        index_path=cfg.index_path, mode="diffusion",
        past_steps=Tp, future_steps=Tf, height=cfg.height, width=cfg.width,
        batch_size=cfg.genforecast_batch_size, num_workers=0,
        valid_frac=valid_frac, test_frac=test_frac, split_mode="temporal",
        region_center=region_center, region_radius=region_radius,
        dedup_bin0=False, seed=42, use_weighted_sampler=False,
    )
    dm.setup()
    test_ds = dm.test_ds
    n_test = len(test_ds)
    stride = max(1, n_test // scan_rows)
    scan_idx = list(range(0, n_test, stride))
    scored = []
    for ridx in scan_idx:
        past_t, future_t = test_ds[ridx]
        truth = reverse_transform_R(future_t[0].float().numpy())     # (Tf,H,W)
        past_mmhr = reverse_transform_R(past_t[0].float().numpy())   # (Tp,H,W)
        wet = float((truth >= THR).mean())
        scored.append((wet, ridx, past_t, truth, past_mmhr))
    scored.sort(key=lambda x: -x[0])
    cases = scored[:n_cases]
    print(f"  {n_test} test rows; {len(cases)} wettest cases selected")

    # genesis-event count per case (advection-residual) = the obj-4 showcase selector
    gen_cnts = [int(genesis_events(c[4], c[3]).sum()) for c in cases]
    show_ranks = np.linspace(0, len(cases) - 1, n_show).round().astype(int)
    overview = [cases[r] for r in show_ranks]
    evo_idx = int(np.argmax(gen_cnts))
    evo = cases[evo_idx]
    print(f"  overview ranks {show_ranks.tolist()}; "
          f"evolution = rank {evo_idx} (genesis px {gen_cnts[evo_idx]})")

    sel = overview + [evo]          # all cases we must run the diffusion model on

    # ---- models ----
    print("Building prob-nowcaster + diffusion model (EMA)...")
    ae_p = autoenc.AutoencoderKL(encoder.SimpleConvEncoder(), encoder.SimpleConvDecoder())
    net = ProbNowcastNet(ae_p, thresholds=(0.1, 1.0, 5.0), embed_dim=128,
                         analysis_depth=4, forecast_depth=4, output_patches=Tf // 4)
    pmodel = ProbNowcaster(net, thresholds=(0.1, 1.0, 5.0))
    pmodel.load_state_dict(
        torch.load(prob_ckpt, map_location="cpu", weights_only=False)["state_dict"],
        strict=True)
    pnet = pmodel.net.to(dev).eval()

    ldm, _ = setup_model(
        num_timesteps=Tf // 4,
        autoenc_weights_fn=os.path.join(cfg.autoenc_dir, "state_dict.pt"),
        use_obs=True, use_nwp=False, model_dir=cfg.genforecast_dir,
        lr=cfg.genforecast_lr, precision=cfg.precision,
        optimizer_8bit=cfg.optimizer_8bit, max_epochs=1, limit_train_batches=1,
        limit_val_batches=1, scale_factor=1.0, gradient_clip_val=1.0,
        sample_every_n_epochs=1, max_hours=None, early_stopping_patience=0,
        accumulate_grad_batches=1, save_top_k=0,
    )
    sd = torch.load(diff_ckpt, map_location="cpu", weights_only=False)["state_dict"]
    ldm.load_state_dict(sd, strict=True)
    if getattr(ldm, "model_ema", None) is not None:
        ldm.model_ema.copy_to(ldm.model)
    ldm.use_ema = False
    ldm = ldm.to(dev).eval()
    sampler = dpm_solver.DPMSolverSampler(ldm)
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if dev.type == "cuda" else nullcontext())
    with torch.no_grad(), amp:
        probe = torch.zeros(1, 1, Tf, cfg.height, cfg.width, device=dev)
        gen_shape = tuple(ldm.autoencoder.encode(probe)[0].shape[1:])
        del probe
    t_rel = torch.arange(-Tp + 1, 1, dtype=torch.float32, device=dev).unsqueeze(0)

    # ---- forecast each selected case ----
    print(f"Forecasting {len(sel)} cases ({ensemble_size} diffusion members each)...")
    fc = []  # per case dict of all fields
    for ci, (wet, ridx, past_t, truth, past_mmhr0) in enumerate(sel):
        past = past_t.unsqueeze(0).to(dev)                 # (1,1,Tp,H,W)
        with torch.no_grad(), amp:
            P = torch.sigmoid(pnet([[past, t_rel]]))[0].float().cpu().numpy()  # (3,Tf,H,W)
        # diffusion ensemble
        members = []
        for m0 in range(0, ensemble_size, cfg.genforecast_batch_size):
            mb = min(cfg.genforecast_batch_size, ensemble_size - m0)
            pb = past.repeat(mb, 1, 1, 1, 1)
            tb = t_rel.repeat(mb, 1)
            torch.manual_seed(eval_seed + ci * 1000 + m0)
            if dev.type == "cuda":
                torch.cuda.manual_seed_all(eval_seed + ci * 1000 + m0)
            ldm._cond_cache = None
            with torch.no_grad(), amp, redirect_stdout(io.StringIO()):
                s, _ = sampler.sample(num_diffusion_iters, mb, gen_shape,
                                      [[pb, tb]], progbar=False, verbose=False)
                y = ldm.autoencoder.decode(s / ldm.scale_factor)
            for j in range(mb):
                members.append(reverse_transform_R(y[j, 0].float().cpu().numpy()))
        stack = np.stack(members, axis=0)                  # (M,Tf,H,W)
        past_mmhr = reverse_transform_R(past_t[0].float().numpy())
        t0 = past_mmhr[-1]
        dy, dx = estimate_motion(past_mmhr)
        eul = np.repeat(t0[None], Tf, axis=0)
        lag = np.stack([nd_shift(t0, (dy * (k + 1), dx * (k + 1)),
                                 order=1, mode="constant", cval=0.0) for k in range(Tf)])
        gen_ev = genesis_events(past_mmhr, truth)               # (Tf,H,W) bool
        fc.append({
            "wet": wet, "gen_cnt": int(gen_ev.sum()), "gen_ev": gen_ev,
            "t0": t0, "truth": truth, "eul": eul, "lag": lag,
            "pm": SamplePredictionLogger._pm_mean(stack),       # (Tf,H,W) mm/h
            "pop": (stack >= THR).mean(axis=0),                 # (Tf,H,W) [0,1]
            "prob": P[1],                                       # P(>=1mm) (Tf,H,W)
        })
        print(f"  case {ci+1}/{len(sel)} done")

    _plot_overview(out_dir, fc[:n_show], Tf)
    _plot_evolution(out_dir, fc[-1], Tf)
    print(f"\nSaved: {out_dir}/compare_overview.png, compare_evolution.png")


def _plot_overview(out_dir, fc, Tf):
    k = Tf - 1                       # last lead (+40 min)
    lead_min = (k + 1) * INTERVAL_MIN
    cols = ["t0 (now)", f"Truth +{lead_min}m", "Eulerian", "Lagrangian",
            "Diff PM-mean", f"Diff PoP≥{THR:g}", f"Prob P≥{THR:g}"]
    nr, nc = len(fc), len(cols)
    fig, axes = plt.subplots(nr, nc, figsize=(2.1 * nc, 2.1 * nr))
    axes = np.atleast_2d(axes)
    im_rain = im_prob = None
    for r, c in enumerate(fc):
        rain = [c["t0"], c["truth"][k], c["eul"][k], c["lag"][k], c["pm"][k]]
        for j, R in enumerate(rain):
            im_rain = _rain_panel(axes[r, j], R)
        im_prob = _prob_panel(axes[r, 5], c["pop"][k], c["truth"][k])
        _prob_panel(axes[r, 6], c["prob"][k], c["truth"][k])
        axes[r, 0].set_ylabel(f"wet {c['wet']:.2f}\ngen {c['gen_cnt']}px",
                              fontsize=8, rotation=0, ha="right", va="center")
    for j, name in enumerate(cols):
        axes[0, j].set_title(name, fontsize=10)
    fig.suptitle(f"Aalborg held-out test — forecasters at +{lead_min} min "
                 f"(cyan = truth ≥{THR:g} mm/h)", fontsize=12)
    fig.tight_layout(rect=(0.02, 0.04, 1, 0.97))
    cbar_rain = fig.add_axes([0.30, 0.015, 0.25, 0.012])
    fig.colorbar(im_rain, cax=cbar_rain, orientation="horizontal").set_label(
        "rain rate [mm/h]", fontsize=8)
    cbar_prob = fig.add_axes([0.66, 0.015, 0.25, 0.012])
    fig.colorbar(im_prob, cax=cbar_prob, orientation="horizontal").set_label(
        f"P(rain ≥{THR:g} mm/h)", fontsize=8)
    fig.savefig(out_dir / "compare_overview.png", dpi=110)
    plt.close(fig)


def _plot_evolution(out_dir, c, Tf):
    leads = list(range(1, Tf, 2))   # +10, +20, +30, +40 min
    rows = [("Truth", "rain", c["truth"]), ("Eulerian", "rain", c["eul"]),
            ("Lagrangian", "rain", c["lag"]), ("Diff PM-mean", "rain", c["pm"]),
            (f"Diff PoP≥{THR:g}", "prob", c["pop"]),
            (f"Prob P≥{THR:g}", "prob", c["prob"])]
    nr, nc = len(rows), len(leads)
    fig, axes = plt.subplots(nr, nc, figsize=(2.1 * nc, 2.1 * nr))
    im_rain = im_prob = None
    for r, (name, kind, arr) in enumerate(rows):
        for j, k in enumerate(leads):
            ax = axes[r, j]
            if kind == "rain":
                im_rain = _rain_panel(ax, arr[k])
            else:
                im_prob = _prob_panel(ax, arr[k], c["truth"][k])
            ys, xs = np.where(c["gen_ev"][k])       # genesis pixels = rain from nowhere
            ax.plot(xs, ys, ".", ms=2.0, color="lime", alpha=0.9)
            if r == 0:
                ax.set_title(f"+{(k + 1) * INTERVAL_MIN} min", fontsize=10)
        axes[r, 0].set_ylabel(name, fontsize=9)
    fig.suptitle(f"Strongest-genesis case ({c['gen_cnt']} genesis px) — lime = rain "
                 f"forming where advection can't reach;  cyan = truth ≥{THR:g}", fontsize=11)
    fig.tight_layout(rect=(0.02, 0.04, 1, 0.96))
    cbar_rain = fig.add_axes([0.30, 0.015, 0.25, 0.012])
    fig.colorbar(im_rain, cax=cbar_rain, orientation="horizontal").set_label(
        "rain rate [mm/h]", fontsize=8)
    cbar_prob = fig.add_axes([0.66, 0.015, 0.25, 0.012])
    fig.colorbar(im_prob, cax=cbar_prob, orientation="horizontal").set_label(
        f"P(rain ≥{THR:g} mm/h)", fontsize=8)
    fig.savefig(out_dir / "compare_evolution.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    Fire(run)
