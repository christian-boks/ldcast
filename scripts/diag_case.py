"""Integrity check on the selected side-by-side case: merged crop? reversed?

Reproduces the bands rank-0 selection, then inspects the FULL 12-frame sequence
(4 past + 8 future) for:
  - per-frame rain area + max (a merge/no-data frame shows an area discontinuity)
  - consecutive frame-to-frame motion (best-shift xcorr): a coherent sequence has
    a steady direction; a MERGE shows a low-correlation jump at the past->future
    boundary (frame 3->4); a REVERSED future shows the shift direction flip there.
"""
import os
import numpy as np
from fire import Fire
from omegaconf import OmegaConf
from scipy.ndimage import label as ndi_label

from ldcast.features.rust_data import RustRadarDataModule
from ldcast.visualization.plots import reverse_transform_R

THR = 0.1


def bxc(a, b, ms=40):
    a = a - a.mean(); b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0, 0, 0
    xc = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real) / (na * nb)
    c = np.array(xc.shape) // 2
    win = xc[c[0]-ms:c[0]+ms+1, c[1]-ms:c[1]+ms+1]
    iy, ix = np.unravel_index(np.argmax(win), win.shape)
    return float(win[iy, ix]), int(iy - ms), int(ix - ms)


def run(config="../config/train_rust.yaml", scan_rows=4000, rank=0,
        wet_lo=0.03, wet_hi=0.20, max_blob=0.08, min_shift=20.0, min_components=4,
        min_blob_px=15, region_center=(685, 852), region_radius=64):
    cfg = OmegaConf.load(config)
    if cfg.get("radar_root"):
        os.environ["DGMR_RADAR_ROOT"] = cfg.radar_root
    dm = RustRadarDataModule(
        index_path=cfg.index_path, mode="diffusion", past_steps=cfg.past_steps,
        future_steps=cfg.future_steps, height=cfg.height, width=cfg.width,
        batch_size=cfg.genforecast_batch_size, num_workers=0, valid_frac=0.1,
        test_frac=0.1, split_mode="temporal", region_center=region_center,
        region_radius=region_radius, dedup_bin0=False, seed=42, use_weighted_sampler=False)
    dm.setup()
    ds = dm.test_ds
    n = len(ds)
    stride = max(1, n // scan_rows)
    cand = []
    for ridx in range(0, n, stride):
        past_t, future_t = ds[ridx]
        truth = reverse_transform_R(future_t[0].float().numpy())
        past = reverse_transform_R(past_t[0].float().numpy())
        t0 = past[-1]
        m0, mN = t0 >= THR, truth[-1] >= THR
        if not (m0.any() and mN.any()):
            continue
        cov = float(m0.mean())
        sizes = np.bincount(ndi_label(m0)[0].ravel())[1:]
        largest = float(sizes.max()) / m0.size
        ncomp = int((sizes >= min_blob_px).sum())
        y0, x0 = np.where(m0); yN, xN = np.where(mN)
        disp = float(np.hypot(yN.mean()-y0.mean(), xN.mean()-x0.mean()))
        if wet_lo <= cov <= wet_hi and largest <= max_blob and ncomp >= min_components and disp >= min_shift:
            cand.append((ncomp, ridx, past, truth))
    cand.sort(key=lambda c: -c[0])
    _, ridx, past, truth = cand[rank]
    print(f"selected ridx={ridx} (rank {rank} of {len(cand)})\n")

    seq = np.concatenate([past, truth], axis=0)   # 4 past + 8 future = 12 frames
    npast = past.shape[0]
    print("  frame  role     area>=0.1   max(mm/h)")
    for i, f in enumerate(seq):
        role = "past" if i < npast else "FUT "
        tag = "  <-- past/future boundary" if i == npast else ""
        print(f"   {i:2d}    {role}   {(f>=THR).mean():8.3f}   {f.max():8.2f}{tag}")

    print("\n  transition   corr   shift(dy,dx)   |shift|")
    for i in range(len(seq) - 1):
        corr, dy, dx = bxc(np.log10(seq[i]+0.01), np.log10(seq[i+1]+0.01))
        b = "  <== boundary" if i == npast - 1 else ""
        print(f"   {i:2d}->{i+1:2d}     {corr:5.2f}   ({dy:+3d},{dx:+3d})    {np.hypot(dy,dx):5.1f}{b}")

    # reversal test: does future connect better forwards or backwards to the past?
    f_fwd = bxc(np.log10(past[-1]+0.01), np.log10(truth[0]+0.01))[0]
    f_rev = bxc(np.log10(past[-1]+0.01), np.log10(truth[-1]+0.01))[0]
    print(f"\n  corr(past[-1], future[0]) = {f_fwd:.2f}   "
          f"corr(past[-1], future[-1]) = {f_rev:.2f}")
    print("  (forward should be HIGHER; if backward is higher, future may be reversed)")


if __name__ == "__main__":
    Fire(run)
