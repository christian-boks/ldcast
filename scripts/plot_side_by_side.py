"""Focused side-by-side: prob-nowcaster vs diffusion model on ONE Aalborg case.

Cheap (one case, one prob forward + `members` diffusion samples). Rows, all 8
lead steps as columns:
  Truth (rain rate) | Nowcaster P(>=1mm) | Diffusion PoP(>=1mm) | Diffusion PM-mean
so you can compare the two PROBABILITY fields (rows 2 vs 3) and the two RAIN-RATE
fields (rows 1 vs 4). Cyan = truth >=1 mm/h.

Usage (from scripts/):  uv run python plot_side_by_side.py --rank=80
"""
import io, os
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
import numpy as np
import torch
from fire import Fire
from omegaconf import OmegaConf
import matplotlib; matplotlib.use("Agg")
from matplotlib import pyplot as plt

from ldcast.features.rust_data import RustRadarDataModule
from ldcast.models.diffusion import dpm_solver
from ldcast.models.autoenc import autoenc, encoder
from ldcast.models.nowcast.prob_nowcast import ProbNowcastNet, ProbNowcaster
from ldcast.models.genforecast.monitor import SamplePredictionLogger
from ldcast.visualization.plots import reverse_transform_R, plot_precip_image

from train_genforecast import setup_model
from scipy.ndimage import label as ndi_label

INTERVAL_MIN = 10   # DMI radar cadence is 10 min/step -> 8 steps = +10..+80 min


def best_shift_xcorr(a, b, maxshift=50):
    """Best-shift normalised cross-correlation: high value at a LARGE shift means
    b is a translated copy of a (coherent advection)."""
    a = a - a.mean(); b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0, 0.0
    xc = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real)
    xc /= (na * nb)
    c = np.array(xc.shape) // 2
    win = xc[c[0] - maxshift:c[0] + maxshift + 1, c[1] - maxshift:c[1] + maxshift + 1]
    iy, ix = np.unravel_index(np.argmax(win), win.shape)
    return float(win[iy, ix]), float(np.hypot(iy - maxshift, ix - maxshift))


def run(config="../config/train_rust.yaml",
        prob_ckpt="../models/prob_nowcaster_aalborg/last.ckpt",
        diff_ckpt="../models/genforecast_rust/last.ckpt",
        thr=1.0, select="wet", rank=0, scan_rows=4000, members=32,
        num_diffusion_iters=20, wet_lo=0.05, wet_hi=0.40, min_shift=15.0,
        min_corr=0.40, max_blob=0.12, min_blob_px=15, min_components=2,
        speed_lo=3.0, speed_hi=10.0, area_lo=0.6, area_hi=1.6, dry_run=False,
        region_center=(685, 852), region_radius=64, out_dir=None):
    THR = float(thr)
    thr_idx = (0.1, 1.0, 5.0).index(THR)
    cfg = OmegaConf.load(config)
    if cfg.get("radar_root"):
        os.environ["DGMR_RADAR_ROOT"] = cfg.radar_root
    out_dir = Path(out_dir or os.path.join(
        os.path.dirname(os.path.abspath(prob_ckpt)), "ab_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tp, Tf = int(cfg.past_steps), int(cfg.future_steps)

    dm = RustRadarDataModule(
        index_path=cfg.index_path, mode="diffusion", past_steps=Tp, future_steps=Tf,
        height=cfg.height, width=cfg.width, batch_size=cfg.genforecast_batch_size,
        num_workers=0, valid_frac=0.1, test_frac=0.1, split_mode="temporal",
        region_center=region_center, region_radius=region_radius,
        dedup_bin0=False, seed=42, use_weighted_sampler=False)
    dm.setup()
    test_ds = dm.test_ds
    n_test = len(test_ds)
    stride = max(1, n_test // scan_rows)
    cand = []
    for ridx in range(0, n_test, stride):
        past_t, future_t = test_ds[ridx]
        truth = reverse_transform_R(future_t[0].float().numpy())
        past_mmhr = reverse_transform_R(past_t[0].float().numpy())
        t0 = past_mmhr[-1]
        wet1 = float((truth >= 1.0).mean())
        if select == "advect":
            corr, shiftmag = best_shift_xcorr(np.log10(t0 + 0.01),
                                              np.log10(truth[-1] + 0.01))
            if wet_lo <= wet1 <= wet_hi and shiftmag >= min_shift and corr >= min_corr:
                cand.append({"key": corr, "wet": wet1, "past": past_t, "truth": truth,
                             "info": f"wet {wet1:.2f} shift {shiftmag:.0f}px corr {corr:.2f}"})
        elif select == "bands":
            mask0, maskN = t0 >= THR, truth[-1] >= THR        # features at display thr
            if not (mask0.any() and maskN.any()):
                continue
            cov = float(mask0.mean())
            sizes = np.bincount(ndi_label(mask0)[0].ravel())[1:]
            largest = float(sizes.max()) / mask0.size          # dominant-sheet fraction
            ncomp = int((sizes >= min_blob_px).sum())          # # of real blobs
            step = best_shift_xcorr(np.log10(past_mmhr[-2] + 0.01),
                                    np.log10(t0 + 0.01))[1]     # per-step motion (px)
            area_ratio = (maskN.mean()) / max(cov, 1e-6)        # growth/inflow factor
            # isolated + moves but stays in frame (bounded speed) + advection not growth
            if (wet_lo <= cov <= wet_hi and largest <= max_blob
                    and ncomp >= min_components and speed_lo <= step <= speed_hi
                    and area_lo <= area_ratio <= area_hi):
                cand.append({"key": ncomp, "wet": cov, "past": past_t, "truth": truth,
                             "info": f"cov {cov:.2f} blobs {ncomp} largest {largest:.2f} "
                                     f"step {step:.0f}px/10min area× {area_ratio:.2f}"})
        else:  # "wet"
            cand.append({"key": wet1, "wet": wet1, "past": past_t, "truth": truth,
                         "info": f"wet {wet1:.2f}"})
    if not cand:
        raise SystemExit("no case matched the filter — loosen the criteria")
    cand.sort(key=lambda d: -d["key"])
    if dry_run:
        print(f"  {len(cand)} candidates (select={select}); top 10:")
        for i, d in enumerate(cand[:10]):
            print(f"    [{i}] {d['info']}")
        return
    sel = cand[rank]
    past_t, truth, wet = sel["past"], sel["truth"], sel["wet"]
    print(f"  select={select} rank {rank}/{len(cand)} ({sel['info']})")

    ae = autoenc.AutoencoderKL(encoder.SimpleConvEncoder(), encoder.SimpleConvDecoder())
    net = ProbNowcastNet(ae, thresholds=(0.1, 1.0, 5.0), embed_dim=128,
                         analysis_depth=4, forecast_depth=4, output_patches=Tf // 4)
    ProbNowcaster(net, thresholds=(0.1, 1.0, 5.0)).load_state_dict(
        torch.load(prob_ckpt, map_location="cpu", weights_only=False)["state_dict"])
    pnet = net.to(dev).eval()

    ldm, _ = setup_model(
        num_timesteps=Tf // 4, autoenc_weights_fn=os.path.join(cfg.autoenc_dir, "state_dict.pt"),
        use_obs=True, use_nwp=False, model_dir=cfg.genforecast_dir, lr=cfg.genforecast_lr,
        precision=cfg.precision, optimizer_8bit=cfg.optimizer_8bit, max_epochs=1,
        limit_train_batches=1, limit_val_batches=1, scale_factor=1.0, gradient_clip_val=1.0,
        sample_every_n_epochs=1, max_hours=None, early_stopping_patience=0,
        accumulate_grad_batches=1, save_top_k=0)
    ldm.load_state_dict(torch.load(diff_ckpt, map_location="cpu",
                                   weights_only=False)["state_dict"], strict=True)
    if getattr(ldm, "model_ema", None) is not None:
        ldm.model_ema.copy_to(ldm.model)
    ldm.use_ema = False
    ldm = ldm.to(dev).eval()
    sampler = dpm_solver.DPMSolverSampler(ldm)
    amp = (torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else nullcontext())
    with torch.no_grad(), amp:
        probe = torch.zeros(1, 1, Tf, cfg.height, cfg.width, device=dev)
        gen_shape = tuple(ldm.autoencoder.encode(probe)[0].shape[1:]); del probe

    past = past_t.unsqueeze(0).to(dev)
    t_rel = torch.arange(-Tp + 1, 1, dtype=torch.float32, device=dev).unsqueeze(0)
    print("Forecasting...")
    with torch.no_grad(), amp:
        P = torch.sigmoid(pnet([[past, t_rel]]))[0].float().cpu().numpy()[thr_idx]  # (Tf,H,W)
    mem = []
    for m0 in range(0, members, cfg.genforecast_batch_size):
        mb = min(cfg.genforecast_batch_size, members - m0)
        torch.manual_seed(1234 + m0)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(1234 + m0)
        ldm._cond_cache = None
        with torch.no_grad(), amp, redirect_stdout(io.StringIO()):
            s, _ = sampler.sample(num_diffusion_iters, mb, gen_shape,
                                  [[past.repeat(mb, 1, 1, 1, 1), t_rel.repeat(mb, 1)]],
                                  progbar=False, verbose=False)
            y = ldm.autoencoder.decode(s / ldm.scale_factor)
        for j in range(mb):
            mem.append(reverse_transform_R(y[j, 0].float().cpu().numpy()))
    stack = np.stack(mem, axis=0)
    pop = (stack >= THR).mean(axis=0)
    pm = SamplePredictionLogger._pm_mean(stack)

    # dump fields so panel_with_steps.py (3.12 venv) can run STEPS on the SAME
    # case and render the full panel with STEPS rows added.
    past_mmhr_sel = reverse_transform_R(past_t[0].float().numpy())
    np.savez(out_dir / "panel_fields.npz", past_mmhr=past_mmhr_sel, truth=truth,
             P=P, pop=pop, pm=pm, thr=THR, wet=float(wet),
             interval=INTERVAL_MIN, info=sel["info"])
    print(f"Dumped fields for STEPS overlay -> {out_dir}/panel_fields.npz")

    rows = [("Truth", "rain", truth), (f"Nowcaster P≥{THR:g}", "prob", P),
            (f"Diffusion PoP≥{THR:g}", "prob", pop), ("Diffusion PM-mean", "rain", pm)]
    fig, axes = plt.subplots(len(rows), Tf, figsize=(2.0 * Tf, 2.0 * len(rows)))
    im_rain = im_prob = None
    for r, (name, kind, arr) in enumerate(rows):
        for k in range(Tf):
            ax = axes[r, k]
            if kind == "rain":
                im_rain = plot_precip_image(ax, arr[k].copy())
            else:
                im_prob = ax.imshow(arr[k], cmap="magma", vmin=0, vmax=1)
                ax.contour(truth[k] >= THR, levels=[0.5], colors="cyan", linewidths=0.6)
                ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"+{(k + 1) * INTERVAL_MIN} min", fontsize=10)
        axes[r, 0].set_ylabel(name, fontsize=10)
    fig.suptitle(f"Nowcaster vs diffusion — one Aalborg test case (wet {wet:.2f}); "
                 f"cyan = truth ≥{THR:g} mm/h", fontsize=12)
    fig.tight_layout(rect=(0.01, 0.04, 1, 0.97))
    fig.colorbar(im_rain, cax=fig.add_axes([0.30, 0.015, 0.22, 0.012]),
                 orientation="horizontal").set_label("rain rate [mm/h]", fontsize=8)
    fig.colorbar(im_prob, cax=fig.add_axes([0.62, 0.015, 0.22, 0.012]),
                 orientation="horizontal").set_label(f"P(rain ≥{THR:g})", fontsize=8)
    fn = f"side_by_side_{THR:g}mm_{select}.png"
    fig.savefig(out_dir / fn, dpi=110)
    print(f"\nSaved: {out_dir}/{fn}")


if __name__ == "__main__":
    Fire(run)
