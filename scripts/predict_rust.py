"""Full-frame LDCast inference against dgmr-rs radar data.

Loads past frames from the rust loader (full-frame, not 128x128 crops),
runs the rust-trained autoencoder + diffusion model, and writes each
predicted future frame as a PNG using the dgmr-rs Marshall-Palmer ->
dBZ -> 256-entry RGB palette.

Memory strategy (16 GB GPU, full 1440x1856 frame, all on GPU):

  * The diffusion UNet runs on the *full* latent in a single forward —
    spatial tiling the diffusion produces visible seams at tile
    boundaries (AFNO blocks couple spatial dims globally via FFT). To
    make the full pass fit, the UNet weights are converted to bfloat16
    (saves 1.35 GB), and inference runs under torch.autocast(bf16). AFNO
    blocks self-cast to fp32 around the FFT so FFT precision is
    preserved (see ldcast/models/blocks/afno.py:165-166,208-209).
  * The autoencoder encode runs on GPU under autocast(bf16). The
    autoencoder decode is the memory bottleneck — at full frame its
    final ResBlock holds a (1, 64, T_future, 1440, 1856) intermediate
    that PyTorch's F.group_norm wants a ~5 GB workspace for. We patch
    the decoder to use a manual in-place GroupNorm (~2x input peak
    instead of ~4x) and an in-place SiLU, then move the UNet weights to
    CPU just for the decode call (the UNet isn't needed after sampling).
    This brings the decode peak down to ~7-8 GB, leaving headroom on
    16 GB. Pass `--ae_on_cpu=True` to offload the decode to CPU instead
    if you ever want to free those slots for something else.

Usage (from this directory):

    DGMR_RADAR_ROOT=/path/to/radar_data \\
    DGMR_RADAR_INDEX=/path/to/radar_data/index_128.txt \\
        uv run python predict_rust.py \\
            --timestamp=2024-07-15T14:30:00Z \\
            --ldm_weights_fn=../models/genforecast_rust/epoch=N-val_loss_ema=X.XXXX.ckpt

`ldm_weights_fn` accepts either a Lightning `.ckpt` (the `state_dict`
key is stripped on the fly) or an already-extracted raw `.pt`. Same for
`autoenc_weights_fn`.
"""
from __future__ import annotations

import gc
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from PIL import Image

import dgmr_py

from ldcast.forecast import Forecast
from ldcast.visualization.dgmr_colors import precip_to_rgb


R_ZERO_VALUE = 0.02  # mm/hr below-threshold fill, matches Forecast default

_PRECISION_DTYPES = {
    "bf16-mixed": torch.bfloat16,
    "fp16-mixed": torch.float16,
    "fp32": None,
}


def _normalize_timestamp(ts: str) -> str:
    """Strip optional trailing Z; IndexEntry.timestamp is an ISO string without Z."""
    return datetime.fromisoformat(ts.rstrip("Z")).isoformat()


def _find_entry(entries: list, ts: str):
    for e in entries:
        if e.timestamp == ts:
            return e
    raise SystemExit(
        f"No index entry with timestamp {ts} in the index. "
        "Pick a timestamp that appears in DGMR_RADAR_INDEX."
    )


def _materialize_state_dict(path: str) -> str:
    """Return a path to a raw state_dict .pt; extracts from Lightning .ckpt if needed."""
    p = Path(path)
    if p.suffix == ".ckpt":
        out = p.with_name(p.stem + ".state_dict.pt")
        if not out.exists():
            print(f"Extracting state_dict from {p.name} -> {out.name}")
            sd = torch.load(p, map_location="cpu")["state_dict"]
            torch.save(sd, out)
        return str(out)
    return str(p)


def _pad_to_multiple_of_32(R: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad (T, H, W) on the bottom/right with R_ZERO_VALUE; return (padded, (H, W))."""
    T, H, W = R.shape
    Hp = int(math.ceil(H / 32)) * 32
    Wp = int(math.ceil(W / 32)) * 32
    if (Hp, Wp) == (H, W):
        return R, (H, W)
    out = np.full((T, Hp, Wp), R_ZERO_VALUE, dtype=R.dtype)
    out[:, :H, :W] = R
    return out, (H, W)


# --- Memory-efficient GroupNorm replacement ---------------------------------
# F.group_norm allocates a ~5 GB workspace for the (1, 64, 8, 1440, 1856)
# bf16 tensor at the last decoder ResBlock — about 4x the input. The manual
# implementation below keeps the peak at ~2x input (input + normalised
# output), which is what makes full-frame decode fit on 16 GB after we also
# offload the UNet weights for the decode call.

class _MemoryEfficientGroupNorm(torch.nn.Module):
    def __init__(self, num_groups, num_channels, eps, weight, bias):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        # Re-use the trained parameters; don't copy.
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        B, C = x.shape[:2]
        spatial = x.shape[2:]
        G = self.num_groups
        dtype = x.dtype
        x_g = x.view(B, G, C // G, *spatial)
        reduce_dims = list(range(2, x_g.dim()))
        # Compute stats in the input's dtype (PyTorch reductions use an
        # fp32 accumulator internally on bf16/fp16 input). Casting the
        # whole tensor to fp32 here would double its size and OOM at
        # full frame, defeating the purpose.
        mean = x_g.mean(dim=reduce_dims, keepdim=True)
        var = x_g.var(dim=reduce_dims, keepdim=True, unbiased=False)
        rstd = (var + self.eps).rsqrt()
        # (x_g - mean) allocates one new tensor the size of x; then we
        # do everything else in-place on that tensor.
        x_out = (x_g - mean).mul_(rstd).view(B, C, *spatial)
        if self.weight is not None:
            shape = (1, C) + (1,) * len(spatial)
            w = self.weight.to(dtype).view(shape)
            b = self.bias.to(dtype).view(shape)
            x_out.mul_(w).add_(b)
        return x_out


def _patch_groupnorm(module: torch.nn.Module) -> int:
    """Recursively replace nn.GroupNorm modules with the memory-efficient version.

    Returns the count of replacements made.
    """
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.GroupNorm):
            setattr(module, name, _MemoryEfficientGroupNorm(
                num_groups=child.num_groups,
                num_channels=child.num_channels,
                eps=child.eps,
                weight=child.weight,
                bias=child.bias,
            ))
            n += 1
        else:
            n += _patch_groupnorm(child)
    return n


def _patch_silu_inplace(module: torch.nn.Module) -> int:
    """Set inplace=True on every nn.SiLU. Safe in inference (no_grad)."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.SiLU):
            setattr(module, name, torch.nn.SiLU(inplace=True))
            n += 1
        else:
            n += _patch_silu_inplace(child)
    return n


# --- CPU-offload helpers for the autoencoder ---------------------------------

def _make_cpu_encode(orig_encode, target_device):
    """Wrap autoencoder.encode to run on CPU and return GPU tensors.

    Bit-exact-equivalent to a single-pass GPU encode: same weights, same
    inputs, same code path — just executed on CPU because the activations
    don't fit in 16 GB at full frame and GroupNorm(num_groups=1) makes
    tiling produce visible seams.
    """
    def cpu_encode(x):
        x_cpu = x.detach().cpu()
        mean, log_var = orig_encode(x_cpu)
        mean = mean.to(target_device)
        log_var = log_var.to(target_device) if log_var is not None else None
        return mean, log_var
    return cpu_encode


def _make_cpu_decode(orig_decode, target_device):
    """Wrap autoencoder.decode to run on CPU and return a GPU tensor."""
    def cpu_decode(z):
        z_cpu = z.detach().cpu().float()  # ensure fp32 for the CPU autoencoder
        dec = orig_decode(z_cpu)
        return dec.to(target_device)
    return cpu_decode


# --- Main --------------------------------------------------------------------

class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def predict(
    timestamp: str,
    ldm_weights_fn: str,
    autoenc_weights_fn: str = "../models/autoenc_rust/state_dict.pt",
    out_dir: str = "../predictions",
    index_path: str | None = None,
    past_steps: int = 4,
    future_steps: int = 8,
    num_diffusion_iters: int = 50,
    frame_height: int = 1440,
    frame_width: int = 1856,
    precision: str = "bf16-mixed",        # "bf16-mixed" | "fp16-mixed" | "fp32"
    crop_h: int | None = None,            # centre-crop height (smoke testing on small GPUs)
    crop_w: int | None = None,            # centre-crop width
    ae_on_cpu: bool = False,              # True: offload autoenc.decode to CPU; False: decode on GPU via patched in-place norm/silu
    # Optimisation knobs that DIDN'T pay off on this model (kept as flags
    # for future experimentation, all default False):
    #   cudnn_benchmark: +21% slower (algo-search cost dominates)
    #   tf32: no effect (most ops are bf16 already)
    #   channels_last: no effect (input arrives contiguous; PyTorch transposes silently)
    #   compile: +3% (Python control flow in AFNO breaks Inductor fusion)
    cudnn_benchmark: bool = False,
    tf32: bool = False,
    channels_last: bool = False,
    compile: bool = False,
):
    if index_path is None:
        index_path = os.environ["DGMR_RADAR_INDEX"]
    assert past_steps % 4 == 0 or (past_steps + future_steps) % 4 == 0, (
        "past+future must be divisible by the autoencoder time ratio (4)"
    )
    if precision not in _PRECISION_DTYPES:
        raise SystemExit(f"--precision must be one of {list(_PRECISION_DTYPES)}")
    autocast_dtype = _PRECISION_DTYPES[precision]

    # Free perf knobs (no quality impact).
    if cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark mode ON")
    if tf32:
        torch.set_float32_matmul_precision("high")
        print("TF32 matmul precision ON")

    ts = _normalize_timestamp(timestamp)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Parsing index {index_path}...")
    entries = dgmr_py.parse_index(index_path)
    entry = _find_entry(entries, ts)
    print(f"Found entry: timestamp={entry.timestamp}")

    print(f"Loading past frames (full_frame=True) for {past_steps} steps before {ts}...")
    cache = dgmr_py.FrameCache(64)
    past, _future = dgmr_py.load_sample(
        entry, cache, past_steps, future_steps,
        frame_height, frame_width, True,
    )
    # past: (1, past_steps, H, W) float32 mm/hr
    R_past = past[0].astype(np.float32, copy=False)
    print(f"R_past shape: {R_past.shape}, range [{R_past.min():.3f}, {R_past.max():.3f}] mm/hr")

    if crop_h is not None and crop_w is not None:
        H_full, W_full = R_past.shape[1], R_past.shape[2]
        y0, x0 = (H_full - crop_h) // 2, (W_full - crop_w) // 2
        R_past = R_past[:, y0:y0 + crop_h, x0:x0 + crop_w]
        print(f"Cropped to {R_past.shape[1]}x{R_past.shape[2]} (centred)")

    R_past_padded, (H, W) = _pad_to_multiple_of_32(R_past)
    if R_past_padded.shape != R_past.shape:
        print(f"Padded {H}x{W} -> {R_past_padded.shape[1]}x{R_past_padded.shape[2]} (multiples of 32)")

    ldm_sd = _materialize_state_dict(ldm_weights_fn)
    autoenc_sd = _materialize_state_dict(autoenc_weights_fn)

    print("Building Forecast...")
    fc = Forecast(
        ldm_weights_fn=ldm_sd,
        autoenc_weights_fn=autoenc_sd,
        past_timesteps=past_steps,
        future_timesteps=future_steps,
    )

    # Cache the analysis-cascade context across PLMS steps.
    # Default apply_model recomputes context_encoder(cond) every step
    # (diffusion.py:107-111). The past-frame conditioning is identical
    # across all PLMS steps, so encode once and reuse — and free it
    # before the decode (it's ~160 MB at full frame).
    import types
    _cached_ctx: list = [None]
    def cached_apply_model(self, x_noisy, t, cond=None, return_ids=False):
        if self.conditional:
            if _cached_ctx[0] is None:
                _cached_ctx[0] = self.context_encoder(cond)
            cond_encoded = _cached_ctx[0]
        else:
            cond_encoded = None
        with self.ema_scope():
            return self.model(x_noisy, t, context=cond_encoded)
    fc.ldm.apply_model = types.MethodType(cached_apply_model, fc.ldm)

    # autoenc.DECODE is the memory bottleneck at full frame: F.group_norm
    # on the widest intermediate (1, 64, 8, 1440, 1856) bf16 asks for
    # ~5 GB workspace. Two changes let it fit on 16 GB without CPU:
    #   1) Replace decoder GroupNorms with a manual implementation whose
    #      peak is ~2x the input tensor (saves ~3 GB).
    #   2) Free everything we no longer need at decode time — UNet
    #      weights (1.35 GB) and the cached analysis context (160 MB) —
    #      then empty_cache to defrag the allocator pool.
    ae = fc.ldm.autoencoder
    target_device = next(fc.ldm.model.parameters()).device
    n_norm = _patch_groupnorm(ae.decoder)
    n_silu = _patch_silu_inplace(ae.decoder)
    print(f"Patched {n_norm} GroupNorm and {n_silu} SiLU modules in decoder for low-memory decode")

    orig_dec = ae.decode
    if ae_on_cpu:
        print("Offloading autoencoder.decode to CPU (seamless, ~15-30s on host)")
        def decode_fn(z):
            ae.to("cpu")
            try:
                out = orig_dec(z.detach().cpu().float())
            finally:
                ae.to(target_device)
            return out.to(target_device)
    else:
        def decode_fn(z):
            # Stage out anything we don't need for decode.
            fc.ldm.model.to("cpu")
            _cached_ctx[0] = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                out = orig_dec(z)
            finally:
                # Restore so a subsequent fc(...) call would still work.
                fc.ldm.model.to(target_device)
            return out
    ae.decode = decode_fn

    # Shrink the diffusion UNet to bf16 to make the full-latent forward
    # fit on 16 GB. Only the UNet — the autoencoder (CPU, fp32) and the
    # analysis cascade (GPU, fp32) stay at full precision.
    if autocast_dtype is torch.bfloat16:
        print("Converting diffusion UNet weights to bfloat16 (~1.35 GB saved)")
        fc.ldm.model = fc.ldm.model.to(torch.bfloat16)

    if channels_last:
        # channels_last_3d packs (B, T, H, W, C) physically. Faster on
        # tensor-core convs. Apply only to Conv3d/ConvTranspose3d weights
        # — the model has non-rank-5 weights (norms, linears) too.
        n_cl = 0
        for m in fc.ldm.model.modules():
            if isinstance(m, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
                m.weight.data = m.weight.data.to(memory_format=torch.channels_last_3d)
                n_cl += 1
        print(f"channels_last_3d applied to {n_cl} 3D-conv weights")

    if compile:
        # torch.compile wraps the model; LitEma's name-based weight copy
        # breaks against the wrapped module. Bake the EMA weights into the
        # main model once and short-circuit ema_scope so apply_model never
        # tries the swap.
        if fc.ldm.use_ema:
            fc.ldm.model_ema.copy_to(fc.ldm.model)
            fc.ldm.use_ema = False
            print("EMA weights baked into main UNet; ema_scope disabled")
        print("torch.compile() on UNet (first call pays ~30s)")
        fc.ldm.model = torch.compile(fc.ldm.model, dynamic=False)

    print(f"Sampling ({num_diffusion_iters} PLMS steps, precision={precision})...")
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None and torch.cuda.is_available()
        else _nullcontext()
    )
    import time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), autocast_ctx:
        R_pred = fc(R_past_padded, num_diffusion_iters=num_diffusion_iters)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"Sampler+decode elapsed: {time.perf_counter() - t0:.2f} s")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Crop back to native frame
    R_pred = R_pred[:, :H, :W]
    print(f"R_pred shape: {R_pred.shape}, range [{R_pred.min():.3f}, {R_pred.max():.3f}] mm/hr")

    print(f"Writing {R_pred.shape[0]} PNGs to {out_path}...")
    for t in range(R_pred.shape[0]):
        rgb = precip_to_rgb(R_pred[t])
        Image.fromarray(rgb).save(out_path / f"frame_{t:02d}.png")

    print("Done.")


if __name__ == "__main__":
    Fire(predict)
