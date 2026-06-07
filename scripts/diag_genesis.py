"""Is the 'genesis' showcase actually a front? Diagnose the top-genesis cases.

For each of the highest-genesis-count Aalborg test cases, quantify whether the
scene is advection (a coherent feature translating, incl. from outside the 128px
crop) or genuine in-place formation:
  - motion       : |estimate_motion| speed and implied +40 min displacement (px)
  - t0->+40 xcorr: best-shift normalised cross-correlation of t0 vs the +40 frame.
                   A FRONT -> high corr at a large shift (it's a shifted copy).
                   GENESIS -> low corr at every shift (the field genuinely changed).
  - edge inflow  : fraction of 'genesis' pixels within 12 px of the crop boundary
                   (rain entering from outside the crop = advection we can't see).
  - area growth  : rain-area(+40) / rain-area(t0).
"""
import os
import numpy as np
from fire import Fire
from omegaconf import OmegaConf
from scipy.ndimage import distance_transform_edt, shift as nd_shift

from ldcast.features.rust_data import RustRadarDataModule
from ldcast.visualization.plots import reverse_transform_R
from eval_persistence_baseline import estimate_motion

THR = 1.0
SAFETY, MARGIN = 1.5, 8


def genesis_mask(past_mmhr, truth, thr=THR):
    t0 = past_mmhr[-1]
    dy, dx = estimate_motion(past_mmhr)
    speed = float(np.hypot(dy, dx))
    dist0 = distance_transform_edt(~(t0 >= thr)).astype(np.float32)
    Tf = truth.shape[0]
    gen = np.zeros((Tf,) + t0.shape, dtype=bool)
    for k in range(Tf):
        reach = max(MARGIN, SAFETY * speed * (k + 1))
        lagk = nd_shift(t0, (dy * (k + 1), dx * (k + 1)), order=1,
                        mode="constant", cval=0.0)
        gen[k] = (lagk < thr) & (dist0 > reach) & (truth[k] >= thr)
    return gen, (dy, dx), speed


def best_shift_xcorr(a, b, maxshift=50):
    """Best-shift normalised cross-correlation of fields a,b over integer shifts."""
    a = a - a.mean(); b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0, (0, 0)
    A = np.fft.fft2(a); B = np.fft.fft2(b)
    xc = np.fft.ifft2(A * np.conj(B)).real / (na * nb)
    xc = np.fft.fftshift(xc)
    c = np.array(xc.shape) // 2
    win = xc[c[0] - maxshift:c[0] + maxshift + 1, c[1] - maxshift:c[1] + maxshift + 1]
    idx = np.unravel_index(np.argmax(win), win.shape)
    return float(win[idx]), (idx[0] - maxshift, idx[1] - maxshift)


def run(config="../config/train_rust.yaml", n_cases=400, scan_rows=5000,
        top=6, region_center=(685, 852), region_radius=64):
    cfg = OmegaConf.load(config)
    if cfg.get("radar_root"):
        os.environ["DGMR_RADAR_ROOT"] = cfg.radar_root
    dm = RustRadarDataModule(
        index_path=cfg.index_path, mode="diffusion",
        past_steps=cfg.past_steps, future_steps=cfg.future_steps,
        height=cfg.height, width=cfg.width, batch_size=cfg.genforecast_batch_size,
        num_workers=0, valid_frac=0.1, test_frac=0.1, split_mode="temporal",
        region_center=region_center, region_radius=region_radius,
        dedup_bin0=False, seed=42, use_weighted_sampler=False)
    dm.setup()
    test_ds = dm.test_ds
    n_test = len(test_ds)
    stride = max(1, n_test // scan_rows)
    scored = []
    for ridx in range(0, n_test, stride):
        past_t, future_t = test_ds[ridx]
        truth = reverse_transform_R(future_t[0].float().numpy())
        scored.append((float((truth >= THR).mean()), ridx, past_t, truth))
    scored.sort(key=lambda x: -x[0])
    cases = scored[:n_cases]

    rows = []
    for wet, ridx, past_t, truth in cases:
        past_mmhr = reverse_transform_R(past_t[0].float().numpy())
        gen, (dy, dx), speed = genesis_mask(past_mmhr, truth)
        rows.append((int(gen.sum()), wet, past_mmhr, truth, gen, dy, dx, speed))
    rows.sort(key=lambda r: -r[0])

    print(f"\n  Top {top} genesis-count cases (of {len(cases)} wettest):\n")
    print(f"  {'gen_px':>7} {'wet':>5} {'speed':>6} {'disp+40':>8} "
          f"{'xcorr':>6} {'shift':>10} {'edge%':>6} {'area+40/t0':>10}")
    H = cases[0][3].shape[-1]
    for gpx, wet, past_mmhr, truth, gen, dy, dx, speed in rows[:top]:
        t0 = past_mmhr[-1]
        corr, (sy, sx) = best_shift_xcorr(np.log10(t0 + 0.01),
                                          np.log10(truth[-1] + 0.01))
        # edge fraction of genesis pixels (within 12px of any crop boundary)
        ys, xs = np.where(gen.any(axis=0))
        edge = np.mean((ys < 12) | (ys > H - 12) | (xs < 12) | (xs > H - 12)) \
            if ys.size else float("nan")
        area_ratio = (truth[-1] >= THR).mean() / max((t0 >= THR).mean(), 1e-6)
        print(f"  {gpx:>7} {wet:>5.2f} {speed:>6.2f} {speed*8:>8.1f} "
              f"{corr:>6.2f} {str((sy,sx)):>10} {edge*100:>5.0f}% {area_ratio:>10.2f}")
    print("\n  Reading: high xcorr (>~0.5) at a LARGE shift => the +40 frame is a "
          "shifted\n  copy of t0 = a FRONT. High edge% => rain entered from outside "
          "the crop\n  (advection we can't see). area+40/t0>>1 with low xcorr => real "
          "growth.")


if __name__ == "__main__":
    Fire(run)
