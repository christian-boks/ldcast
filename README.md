LDCast is a precipitation nowcasting model based on a latent diffusion model (LDM, used by e.g. [Stable Diffusion](https://github.com/CompVis/stable-diffusion)).

This repository contains the code for using LDCast to make predictions and the code used to generate the analysis in the LDCast paper (a preprint is available at https://arxiv.org/abs/2304.12891).

A GPU is recommended for both using and training LDCast, although you may be able to generate some samples with a CPU and enough patience.

# Installation

It is recommended you install the code in its own virtual environment (created with e.g. pyenv or conda).

Clone the repository, then, in the main directory, run
```bash
$ pip install -e .
```
This should automatically install the required packages (which might take some minutes). In the paper, we used PyTorch 11.2 but are not aware of any problems with newer versions.

If you don't want the requirements to be installed (e.g. if you installed them manually with conda), use:
```bash
$ pip install --no-dependencies -e .
```

# Using LDCast

## Pretrained models

The pretrained models are available at the Zenodo repository https://doi.org/10.5281/zenodo.7780914. Unzip the file `ldcast-models.zip`. The default is to unzip it to the `models` directory, but you can also use another location.

## Producing predictions

The easiest way to produce predictions is to use the `ldcast.forecast.Forecast` class, which will set up all models and data transformations and is callable with a past precipitation array.
```python
from ldcast import forecast

fc = forecast.Forecast(
    ldm_weights_fn=ldm_weights_fn, autoenc_weights_fn=autoenc_weights_fn
)
R_pred = fc(R_past)
```
Here, `ldm_weights_fn` is the path to the LDM weights and `autoenc_weights_fn` is the path to the autoencoder weights. `R_past` is a NumPy array of precipitation rates with shape `(timesteps, height, width)` where `timesteps` must be 4 and `height` and `width` must be divisible by 32.

### Ensemble predictions

If want to process multiple cases at once and/or generate several ensemble members, there is the `ldcast.forecast.ForecastDistributed` class. The usage is similar to the `Forecast` class, for example:
```python
from ldcast import forecast

fc = forecast.ForecastDistributed(
    ldm_weights_fn=ldm_weights_fn, autoenc_weights_fn=autoenc_weights_fn
)
R_pred = fc(R_past, ensemble_members=32)
```
Here, `R_past` should be of shape `(cases, timesteps, height, width)` where `cases` is the number of cases you want to process. For each case, `ensemble_members` predictions are produced (this is the last axis of `R_pred`). `ForecastDistributed` automatically distributes the workload to multiple GPUs if you have them.

## Demo

For a practical example, you can run the demo in the `scripts` directory. First download the `ldcast-demo-20210622.zip` file from the [Zenodo repository](https://doi.org/10.5281/zenodo.7780914), then unzip it in the `data` directory. Then run
```bash
$ python forecast_demo.py
```
A sample output can be found in the file `ldcast-demo-video-20210622.zip` in the data repository. See the function `forecast_demo` in `forecast_demo.py` see how the `Forecast` class works. To run an ensemble mean of 8 members using the `ForecastDistributed` class, you can use:
```bash
$ python forecast_demo.py --ensemble-members=8
```

The demo for a single ensemble member runs in a couple of minutes on our system using one V100 GPU; with a CPU around 10 minutes or more would be expected. A progress bar will show the status of the generation.

# Training 

## Training data

The preprocessed training data, needed to rerun the LDCast training, can be found at the [Zenodo repository](https://doi.org/10.5281/zenodo.7780914). Unzip the `ldcast-datasets.zip` file to the `data` directory.

## Training the autoencoder

In the `scripts` directory, run
```bash
$ python train_autoenc.py --model_dir="../models/autoenc_train"
```
to run the training of the autoencoder with the default parameters. The training checkpoints will be saved in the `../models/autoenc_train` directory (feel free to change this).

It has been reported that this training may encounter a condition where the loss goes to `nan`. If this happens, try restarting from the latest checkpoint:
```bash
$ python train_autoenc.py --model_dir="../models/autoenc_train" --ckpt_path="../models/autoenc_train/<checkpoint_file>"
```
where `<checkpoint_file>` should be the latest checkpoint in the `../models/autoenc_train/` directory.

## Training the diffusion model

In the `scripts` directory, run
```bash
$ python train_genforecast.py --model_dir="../models/genforecast_train"
```
to run the training of the diffusion model with the default parameters, or
```bash
$ python train_genforecast.py --model_dir="../models/genforecast_train" --config=<path_to_config_file>
```
to run the training with different parameters. Some config files can be found in the `config` directory. The training checkpoints will be saved in the `../models/genforecast_train` directory (again, this can be changed freely).

## Training on dgmr-rs radar data (via the dgmr-py bridge)

As an alternative to the NetCDF patches from Zenodo, both stages can be trained directly on the raw `.img` radar files used by the [dgmr-rs](https://github.com/christian-boks/dgmr-rs) trainer. A small PyO3 bridge crate (`dgmr-py`, sibling of `dgmr-rs`) exposes the Rust loader to Python; a new `RustRadarDataModule` in `ldcast/features/rust_data.py` bypasses the NetCDF/`PatchIndex` pipeline entirely.

### Build and install the bridge

`dgmr-py` is a standalone Rust crate at `../dgmr-py/` (sibling to this repo and to `dgmr-rs`). It is built with [maturin](https://www.maturin.rs/) and installed directly into LDCast's `.venv`:

```bash
uv pip install 'maturin>=1.7,<2'
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release \
    --manifest-path ../dgmr-py/Cargo.toml
.venv/bin/python -c "import dgmr_py; print(dir(dgmr_py))"
```

The bridge needs an SSH key authorised against the `christian-boks/hdf5_*` repos to fetch its git dependencies on the first build.

### Required environment

Set the same two env vars `dgmr-rs` uses:

```bash
export DGMR_RADAR_ROOT=/path/to/radar_data        # YYYY/MM/DD/HH/MM_00Z_all.img layout
export DGMR_RADAR_INDEX=/path/to/radar_data/index_128.txt   # CSV: timestamp,x,y[,weight]
```

### Cadence and time-axis constraint

The Rust loader serves frames at the dgmr-rs native 10-minute cadence. LDCast's autoencoder applies a temporal compression of 4, so `past_steps + future_steps` must be divisible by 4. The defaults are `past_steps=4, future_steps=12` (= 120 min lead time, latent time = 4); the only other clean options are 4 or 8 future frames.

### Train both stages

One command runs the autoencoder, extracts its best `state_dict`, and trains the diffusion model on top:

```bash
cd scripts
uv run python train_rust.py
```

The index file passed in `DGMR_RADAR_INDEX` must list crops sized for the `--height`/`--width` you choose: `index_128.txt` for 128×128 (the verified smaller-GPU recipe below), `index_256.txt` for the 256×256 defaults. The two-column `index.txt` produced by dgmr-rs for full-frame training is not compatible — `parse_index` requires `timestamp,x,y[,weight]`.

Common overrides:

```bash
uv run python train_rust.py --height=128 --width=128   # smaller crops (needs index_128.txt)
uv run python train_rust.py --skip_autoenc=True        # only stage 2 (uses best existing ckpt)
uv run python train_rust.py --force_autoenc=True       # always retrain stage 1
```

If `--autoenc_dir` (default `../models/autoenc_rust`) already contains checkpoints you'll be prompted whether to re-train stage 1; pressing Enter skips it and goes straight to diffusion.

GPU memory: the defaults (256×256, `--autoenc_batch_size=16`, `--genforecast_batch_size=8`) assume a ~24 GB GPU. Both stages default to `--precision=bf16-mixed`. Stage 2's 670 M-param UNet plus an EMA shadow copy and fp32 AdamW state alone is ~13.5 GB, so on a 16 GB card you also need 8-bit AdamW to fit. Install the optional bitsandbytes extra and pass `--optimizer_8bit=True`:

```bash
uv sync --extra low-vram                              # installs bitsandbytes
uv run python train_rust.py \
    --height=128 --width=128 \
    --autoenc_batch_size=16 \
    --genforecast_batch_size=2 \
    --optimizer_8bit=True                             # 16 GB recipe (verified on RTX 5080)
```

Use `--autoenc_batch_size=N` / `--genforecast_batch_size=N` to scale to your hardware; pass `--precision=32` to opt out of mixed precision.

The two underlying scripts (`train_autoenc_rust.py` and `train_genforecast_rust.py`) can still be invoked directly when you need finer control — `train_rust.py` is just a thin orchestrator. All scripts split the index file 90/10 deterministically (seed 42, matching dgmr-rs) into train and validation, and use the `weight` column of the index as a `WeightedRandomSampler` on the training loader (disable with `--use_weighted_sampler=False`).

### Full-frame inference (16 GB GPU, no seams)

`scripts/predict_rust.py` runs a trained rust checkpoint against a full radar frame (1440×1856) loaded via `dgmr_py.load_sample(..., full_frame=True)` and writes each predicted future frame as a PNG using the dgmr-rs Marshall–Palmer mm/hr → dBZ → 256-entry RGB palette (ported verbatim from `radar_img_to_gif::colors` into `ldcast/visualization/dgmr_colors.py`).

```bash
cd scripts
DGMR_RADAR_ROOT=/path/to/radar_data DGMR_RADAR_INDEX=/path/to/radar_data/index_128.txt \
  uv run python predict_rust.py \
    --timestamp=2025-10-04T03:40:00Z \
    --ldm_weights_fn=../models/genforecast_rust/<ckpt>.ckpt \
    --autoenc_weights_fn=../models/autoenc_rust/state_dict.pt \
    --out_dir=../predictions/<run-name> \
    --future_steps=8
```

`--ldm_weights_fn` accepts either a Lightning `.ckpt` (the `state_dict` key is stripped on the fly to a sibling `.state_dict.pt`) or an already-extracted raw `.pt`. Timestamps must be in the index file and the past+future window must be loadable from the `.img` archive (historical timestamps only — live "predict from now" needs a different code path because the loader validates the full 16-frame window).

The script also accepts `--crop_h` / `--crop_w` to centre-crop the loaded frame (useful for smoke testing) and `--num_diffusion_iters=N` to override the 50-step PLMS default.

#### Why this fits on 16 GB without spatial tiling

Spatial tiling of the diffusion UNet produces visible seams because AFNO blocks couple spatial dimensions globally via FFT. Tiling the autoencoder instead produces a smaller seam because its `GroupNorm(num_groups=1)` is non-local — different tiles see different statistics. The whole pipeline therefore has to run at full resolution, which `predict_rust.py` makes fit through three tricks (all GPU, no CPU):

1. **bf16 UNet weights** — `fc.ldm.model.to(torch.bfloat16)` after building Forecast. Saves 1.35 GB on the 670 M-param UNet. AFNO self-casts to fp32 around its FFT (see `ldcast/models/blocks/afno.py:165-166,208-209`) so FFT precision is preserved.
2. **Memory-efficient decoder norm/activation** — `predict_rust.py` swaps the autoencoder decoder's `nn.GroupNorm` for an in-place manual implementation (peak ~1× input vs `F.group_norm`'s ~4×) and sets `inplace=True` on its `nn.SiLU`. Numerically equivalent.
3. **Stage UNet out before decode** — the wrapped `decode_fn` moves the UNet weights to CPU and clears the cached analysis-cascade context just before the autoencoder decode runs, then `empty_cache()` to defrag the allocator.

These three together drop the decode peak from ~16 GB+ (OOM) to ~7-8 GB. Add `--ae_on_cpu=True` to fall back to a CPU autoencoder decode if you ever need those GPU slots for something else (~15-30 s extra wall time).

#### Numbers (RTX 5080, 16 GB, full 1440×1856)

| `--future_steps` | latent T | PLMS steps | sampler+decode | total wall |
|---|---|---|---|---|
| 8 (60 min lead) | 2 | 50 | 57 s | ~75 s |
| 12 (120 min lead) | 3 | 50 | ~92 s | ~110 s |
| 20 (200 min lead) | 5 | 50 | OOM (needs >16 GB even in bf16) | — |

Total wall includes Python/uv startup + Forecast init + ckpt load + decode + PNG writes. Sampler+decode is the bit that scales with model work.

#### Optimisation flags that didn't help

`predict_rust.py` exposes `--cudnn_benchmark`, `--tf32`, `--channels_last`, and `--compile` (all default `False`). On this model they ranged from no-op to ~21 % slower:

* `--cudnn_benchmark --tf32`: +21 % (cuDNN's algo-search cost dominates; TF32 only helps fp32 matmul and most ops are bf16).
* `--channels_last`: 0 % (only the weights move; the input arrives contiguous so PyTorch transposes silently).
* `--compile`: +3 % (Python control flow inside AFNO breaks Inductor fusion; `mode="reduce-overhead"` would use CUDA Graphs but PLMS's tensor reuse triggers aliasing errors).

The bottleneck is AFNO's fp32 FFT (~12 calls per UNet forward × 50 PLMS steps). For real speed-ups you need fewer sampler steps (DPM-Solver++ at 20-25 typically matches 50-step PLMS) or a different model architecture.

# Evaluation

You can find scripts for evaluating models in the `scripts` directory:
* `eval_genforecast.py` to evaluate LDCast
* `eval_dgmr.py` to evaluate DGMR (requires tensorflow installation and the DGMR model from https://github.com/deepmind/deepmind-research/tree/master/nowcasting placed in the `models/dgmr` directory)
* `eval_pysteps.py` to evaluate PySTEPS (requires pysteps installation)
* `metrics.py` to produce metrics from the evaluation results produced with the functions in scripts above
* `plot_genforecast.py` to make plots from the results generated
