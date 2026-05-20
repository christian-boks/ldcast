# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LDCast is a latent diffusion model (LDM) for precipitation nowcasting — same family of models as Stable Diffusion, but conditioned on past radar frames rather than text. The repo contains both the library (`ldcast/`) for using the model and the scripts (`scripts/`) used to train and evaluate it for the LDCast paper (https://arxiv.org/abs/2304.12891).

## Install (uv)

The project is set up for uv. From the repo root:

```bash
uv sync                          # creates .venv/ and uv.lock, installs runtime deps
uv sync --extra benchmarks       # also installs tensorflow + pysteps for the DGMR/PySTEPS baselines
```

Run anything through `uv run <cmd>` (e.g. `uv run python forecast_demo.py`), or activate once with `source .venv/bin/activate`. Dependencies and `[project.optional-dependencies]` live in `pyproject.toml`; there is no `setup.py`.

There is no test suite and no linter config.

### Known dependency footguns

The current lockfile resolves to recent versions of two libraries that have API breaks the code predates:

- **NumPy 2.x** — aliases `np.float`/`np.int`/`np.bool` were removed; if a script trips on one, it's the cause.
- **PyTorch Lightning 2.x** — `train_autoenc.py` and `train_genforecast.py` pass `strategy='dp'` to `pl.Trainer`, which Lightning removed. Multi-GPU training will need `strategy='ddp'` (or simply `None` for single-GPU) before it runs.

## Running scripts

**All scripts in `scripts/` use relative paths like `../data/...` and `../models/...` and expect the CWD to be `scripts/`.** Always `cd scripts/` before running them.

Scripts use `fire` for the CLI plus `omegaconf` for optional YAML config files. The pattern is `uv run python <script>.py [--config=<yaml>] [--key=value ...]`, where CLI kwargs override the YAML.

Common commands (all from `scripts/`):

- Demo forecast: `uv run python forecast_demo.py [--ensemble-members=N]`
- Train autoencoder: `uv run python train_autoenc.py --model_dir="../models/autoenc_train"` (resume from a NaN crash with `--ckpt_path=<latest_ckpt>`)
- Train diffusion model: `uv run python train_genforecast.py --model_dir="../models/genforecast_train" [--config=../config/genforecast-radaronly-256x256-20step.yaml]`
- Evaluate: `eval_genforecast.py` / `eval_dgmr.py` / `eval_pysteps.py` produce ensemble NetCDFs; then `metrics.py` and `plots_genforecast.py` consume them.

Pretrained weights and training/demo data come from the Zenodo repo (https://doi.org/10.5281/zenodo.7780914): unzip `ldcast-models.zip` into `models/`, `ldcast-datasets.zip` into `data/`, and `ldcast-demo-20210622.zip` into `data/` for the demo.

## Architecture

The end-to-end pipeline assembled inside `ldcast/forecast.Forecast` is:

```
past radar frames ──► autoenc encoder ──► analysis cascade ──► UNet denoiser ──► PLMS sampler ──► autoenc decoder ──► predicted frames
   (shape [4,H,W])    (latent /4 spatial,        (multi-scale       (3D denoiser conditioned     (50-step latent       (back to rain rates)
                       /4 temporal)               context tensors)   on context cascade)          diffusion)
```

The three composable pieces live under `ldcast/models/`:

- **`autoenc/`** — a KL-regularized 3D VAE (`AutoencoderKL` in `autoenc.py`) with `SimpleConvEncoder`/`SimpleConvDecoder` (`encoder.py`). Compresses radar frames into a latent space (spatial ÷4, time ÷4, `hidden_width=32` channels). The autoencoder is trained alone first; the diffusion stage freezes it (`requires_grad_(False)` inside `LatentDiffusion`).
- **`genforecast/`** — the generative forecasting pieces: `unet.UNetModel` is the 3D denoiser; `analysis.AFNONowcastNetCascade` extends the AFNO nowcaster to emit a multi-resolution cascade of feature maps that conditions the UNet at each scale. `training.py` wires both into a `LatentDiffusion` for Lightning training.
- **`diffusion/`** — `diffusion.LatentDiffusion` is the LightningModule that runs diffusion in latent space (DDPM-style noise schedule, EMA via `ema.LitEma`, PLMS sampler in `plms.py`). Adapted from CompVis latent-diffusion.

Shared building blocks are in `ldcast/models/blocks/`: `afno.py` (Adaptive Fourier Neural Operator blocks plus `PatchEmbed3d`/`PatchExpand3d`), `attention.py` (positional encoding + temporal transformer), `resnet.py`. `ldcast/models/nowcast/nowcast.py` is the deterministic AFNO nowcaster baseline that `genforecast/analysis.py` builds on.

`ldcast/forecast.py` is the user-facing wrapper. `Forecast` runs on a single GPU; `ForecastDistributed` shards cases × ensemble members across all visible GPUs via `torch.multiprocessing.spawn`. Both wire up models with hyperparameters matched to the released checkpoints (`past_timesteps=4`, `future_timesteps=20`, `autoenc_time_ratio=4`, `autoenc_hidden_dim=32`). Input `R_past` shape is `(4, H, W)` for `Forecast` and `(cases, 4, H, W)` for `ForecastDistributed`; **H and W must be divisible by 32**.

## Data pipeline

`ldcast/features/`:

- `patches.py` — loads patches from per-variable HDF5 stores (one directory per variable under `data/<VAR>/`).
- `split.py` — `get_chunks` slices time into chunks for train/valid/test split (saved to `data/split_chunks.pkl.gz`); `DataModule` is the Lightning `LightningDataModule`.
- `sampling.py`, `batch.py` — bin-weighted samplers and batch assembly. Sampler indices are cached as pickles under `cache/` (e.g. `sampler_autoenc_train.pkl`); regenerated on first run if missing.
- `transform.py` — log-rain-rate normalization (`default_rainrate_transform`) and the `Antialiasing` filter used at inference; NWP-variable transforms live in `train_nowcaster.setup_data`.
- `io.py` — `save_batch` writes evaluation ensembles to NetCDF.

`train_nowcaster.setup_data` is the single source of truth for the radar+NWP data module; `train_genforecast.py` imports it directly.

## Conventions worth knowing

- Multi-GPU training: PyTorch Lightning trainers auto-pick a strategy when `torch.cuda.device_count() > 1` (see `models/autoenc/training.py` and `models/genforecast/training.py`). The hardcoded `strategy='dp'` is stale (see footguns above).
- Checkpointing: both trainers keep the top-3 checkpoints by monitored metric (`val_rec_loss` for the autoencoder, `val_loss_ema` for the diffusion model) and early-stop with patience 6.
- Config files in `config/` are sparse OmegaConf YAMLs that only set overrides; the 128×128 config is intentionally empty (defaults), the 256×256 config bumps `sample_shape`, `batch_size`, `lr`, and points `initial_weights` at the 128×128 checkpoint for staged training.
- Autoencoder training has a known instability: loss can go to NaN. Resume from the latest checkpoint with `--ckpt_path=...` (README notes this).
