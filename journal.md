# LDCast Engineering Journal

## 2026-05-20 — Inference speed-ups + dgmr-rs full-frame pipeline

Hardware: RTX 5080 (Blackwell, 16 GB). Goal: make LDCast inference faster on a 16 GB GPU,
then exercise the rust full-frame path end-to-end (train → predict) and measure its timing.

### Changes made

#### 1. BF16 mixed-precision inference — `ldcast/forecast.py`, `scripts/eval_genforecast.py`
- Added a `precision` knob to `Forecast`/`ForecastDistributed` (`"bf16"` default; `"fp16"`/`"fp32"`),
  resolved to a `torch.autocast` dtype and auto-disabled on CPU (`_AMP_DTYPES`).
- `Forecast.__call__` wraps the PLMS sampler + autoencoder decode in `torch.autocast`, then casts
  the decoder output back to FP32 before the rain-rate inversion (`torch.pow(10, ·)` needs it).
- Same autocast wrap added to the `eval_genforecast.py` ensemble loop.
- Safe because AFNO already self-casts its FFT/spectral section to FP32 (`ldcast/models/blocks/afno.py`).
- **Result:** ~1.6× faster on the demo. **Caveat:** BF16 autocast does *not* cut peak VRAM (keeps FP32
  master weights + adds BF16 casts — slightly higher). It's a speed lever, not a memory lever.

#### 2. Output-exact inference optimizations — `ldcast/models/diffusion/diffusion.py`
- **Conditioning cache:** `apply_model` was re-running the whole AFNO analysis cascade every diffusion
  step though the conditioning (past frames) is constant. Now encoded once per sampling run and reused
  (keyed on input identity). Exact because `AutoencoderKL.encode` returns the deterministic posterior mean.
- **EMA weight hoist:** `apply_model`'s `ema_scope` stored/copied/restored all UNet params every step;
  `Forecast._init_model` now installs EMA weights once (`model_ema.copy_to`, `use_ema=False`).
- **Result:** combined with BF16, ~2.3× total on the demo (18.15 s → 7.99 s), verified output-exact
  (FP32 mean identical; BF16-vs-FP32 correlation 0.998 unchanged). The conditioning cache helps
  `eval_genforecast.py` most (conditioning reused across 50 steps × N ensemble members).

#### 3. Diffusion training fixes — `ldcast/models/genforecast/training.py`
- `save_top_k` 3 → 1 (a 670 M-param checkpoint is ~6.7 GB; only ~16 GB free disk).
- `gradient_clip_val=1.0` — rust diffusion training NaN'd from step ~0 without it. The forward was
  finite in both fp32 and bf16; the explosion was in the gradients, so clipping fixed it (loss 1.0 → 0.057).

#### 4. Default lead time `future_steps` 12 → 8
Changed in `predict_rust.py`, `train_rust.py`, `train_genforecast_rust.py`, `train_autoenc_rust.py`,
`ldcast/features/rust_data.py`. `future_steps` sets the UNet temporal dim (`num_timesteps = future_steps//4`),
which is baked into the weights — train and infer must use the same value; valid values satisfy
`(past + future) % 4 == 0` (→ 4, 8, 12).

#### 5. Rust models trained + full-frame inference run
- `models/autoenc_rust/` — ~11 min, val_rec_loss 0.17 (convolutional in time, reusable across `future_steps`).
- `models/genforecast_rust/` — minimal 1-epoch, **timing-only** (not quality-trained).
- Full-frame (1440×1856) `predict_rust.py` on real 2026-03-14 07:10 radar → 8 PNGs in `predictions/rust-fullframe/`.

### Measurements (RTX 5080, 16 GB, 50 PLMS steps, bf16)
| run | sampler+decode | total wall | notes |
|---|---|---|---|
| Demo 352×448, FP32, no opts | 18.15 s | — | peak 10.85 GB |
| Demo 352×448, BF16, no opts | 11.36 s | — | peak 12.18 GB (≈1.6×) |
| Demo 352×448, BF16 + exact opts | 7.99 s | — | ≈2.3×, output-exact |
| Full-frame 1440×1856, `future_steps=12` | 91.8 s | 103 s | |
| Full-frame 1440×1856, `future_steps=8` | 57.5 s | 71 s | |

(`future_steps=20` OOMs even in bf16. Timing is independent of training quality.)

### Future improvements (prioritized)

1. **Faster sampler (DPM-Solver++ / UniPC).** Biggest remaining inference win. The bottleneck is AFNO's
   fp32 FFT × 50 PLMS steps; a higher-order solver typically matches 50-step PLMS quality at ~15–25 steps
   (~2× fewer network evals). Plain DDIM is *not* worth it — same per-step cost as PLMS, and PLMS holds
   quality at fewer steps (established this session).
2. **Latent scale factor.** `LatentDiffusion` has no latent normalization (cf. Stable Diffusion's 0.18215).
   A configurable `scale_factor` would improve training stability — likely the proper fix that removes the
   need for the `gradient_clip_val` band-aid and allows a higher LR.
3. **NaN-safe training.** `EarlyStopping(check_finite=False)` + a val metric that can be NaN means a diverged
   run wastes hours. Set `check_finite=True` (and/or guard NaN batches) so training stops on NaN.
4. **True memory reduction.** BF16 autocast didn't cut VRAM. To fit larger domains / `future_steps=20` /
   more ensemble members on 16 GB, convert the UNet to bf16 *weights* in the `Forecast` path (predict_rust
   already does this for full-frame) and generalize its memory-efficient decode.
5. **Lighter inference checkpoints.** The 6.7 GB `.ckpt` carries optimizer + EMA + model; inference needs
   only EMA weights. Auto-save an inference-only state_dict — also eases the near-full disk.
6. **eval_genforecast.py EMA hoist.** It still swaps EMA weights per step; apply the same hoist (handle the
   per-batch multi-weights path).
7. **Faster index lookup.** `predict_rust.py` parses the full 93 MB / 2.3 M-line index just to find one
   timestamp. A sorted/indexed lookup or a small inference index would cut startup.
8. **Script CWD-independence.** Scripts require CWD = `scripts/` (relative `../models`, `../data`, sibling
   imports) — a recurring footgun. Make paths package-relative or absolute.
9. **Proper rust model training.** The current `genforecast_rust` is timing-only. Real forecasts need a
   multi-hour/overnight diffusion run (and a better autoencoder) plus disk headroom — the root fs is one
   457 GB disk at ~97 % full.
10. **Disk hygiene.** Automate `.ckpt` → state_dict extraction in `train_rust.py` and prune large
    checkpoints; the near-full disk repeatedly constrained training (`save_top_k`) and inference.
