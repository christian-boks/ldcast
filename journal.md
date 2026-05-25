# LDCast Engineering Journal

## 2026-05-25 — Got rust training running again on the Python 3.14 venv (3 blockers)

**What:** Brought the dgmr-rs rust training pipeline back up after the venv moved to Python
3.14. Three separate blockers, fixed in order:

1. **maturin/cargo SSH fetch (env only).** `maturin develop` failed fetching the private
   `hdf5_*` deps (`no authentication methods succeeded` from cargo's built-in libgit2). The git
   CLI authenticates fine (`git ls-remote` returns the exact rev cargo called "not found"). Fixed
   by building with `CARGO_NET_GIT_FETCH_WITH_CLI=true`. Note `dgmr-py/.cargo/config` already sets
   `git-fetch-with-cli = true` but cargo reads `.cargo/config` from the CWD, not the manifest dir,
   so running maturin from `ldcast/` ignored it — the env var works regardless of CWD.
2. **bitsandbytes missing (env only).** `optimizer_8bit: true` → `configure_optimizers` does
   `import bitsandbytes`, which wasn't in the 3.14 venv (it's the `low-vram` optional extra, not a
   core dep). Installed with `uv sync --extra low-vram --inexact`. `--inexact` is essential: a
   plain sync would uninstall the editable `dgmr-py` and `maturin` (dry-run confirmed). Got
   `bitsandbytes==0.49.2`; verified a real `AdamW8bit` step runs on the RTX 5080 (sm_120 / CUDA 13).
3. **CODE CHANGE — Python 3.14 forkserver pickling.** `ldcast/features/rust_data.py`. Py3.14
   switched the default multiprocessing start method on Linux from `fork` to `forkserver`, which
   pickles the dataset to hand it to each DataLoader worker. `RustRadarDataset` holds dgmr_py
   `IndexEntry` objects (PyO3, not picklable) and was written to build its per-worker cache *after
   a fork* → `TypeError: cannot pickle 'builtins.IndexEntry'` during the sanity check. Forced
   `multiprocessing_context="fork"` on the train+val DataLoaders (only when num_workers>0). Workers
   are CPU-only, so forking after CUDA init in the parent is safe — this is just the pre-3.14 default.

Also refreshed the `_check_bridge()` build hint in `train_rust.py` (it had a stale `/home/christian`
path and omitted the `CARGO_NET_GIT_FETCH_WITH_CLI=true` that the build actually needs).

**Why:** The pipeline last trained on Python ≤3.13 (epoch=40 genforecast_rust checkpoints exist);
the 3.14 venv reintroduced these because dgmr-py/bitsandbytes aren't reinstalled by a venv rebuild
and the fork→forkserver default is a 3.14 behaviour change.

**Outcome:** Working. `train_rust.py` (stages=diffusion, resume=true) resumed from
`epoch=40-step=170500.ckpt` and is training epoch 41 at ~5 it/s, GPU 15.4/16.3 GB, 100% util —
matching the config's ~15.2 GB batch-4 estimate. No NaNs.

## 2026-05-24 — Forecast-quality CSI wired into per-epoch validation

**What:** Added CSI @ {0.1, 1, 5} mm/h (+ ens-mean MAE/RMSE) to the `SamplePredictionLogger`
callback (`ldcast/models/genforecast/monitor.py`). It scores the 4 wettest val cases (scanned
once, cached, reused → comparable trend) under a fixed seed (CPU+CUDA RNG restored so training
is unperturbed) and logs them as TensorBoard scalars (`val/csi_*`, `val/mae_mmhr`,
`val/rmse_mmhr`) next to `val_loss_ema`. Reuses the forecast sampling that already runs each
epoch on the GPU — no separate job.

**Why:** `val_loss_ema` (eps-MSE) doesn't track sample quality and previews need a human. First
built a standalone CPU ensemble-eval skill, but CPU diffusion sampling is too slow
(~2-4 s/UNet-step → a trustworthy run was 30-70 min) and a 70-min comparison was disruptive.
Pivoted (user's call) to riding the existing validation, and **removed** the standalone skill
(`.claude/skills/eval-forecast/`) and its `eval_quality.csv`.

**Takes effect on the next training restart** — the running process already imported
`monitor.py`. With `resume: true`, stop & re-run `train_rust.py` to pick it up.

**Outcome (2026-05-25):** Confirmed live. After the Python-3.14 restart (run `version_8`, resumed
at epoch 41) the metrics log correctly. Exact tags carry an `mm` suffix — `val/csi_0.1mm`,
`val/csi_1.0mm`, `val/csi_5.0mm`, `val/mae_mmhr`, `val/rmse_mmhr` — and TensorBoard groups them under
a `val/` card (separate from `val_loss_ema`, which has no slash; easy to miss). Epoch-41 baseline (4
wettest val cases, single point): CSI **0.306 / 0.0225 / 0.0001** @ {0.1, 1, 5} mm/h; MAE 0.864,
RMSE 1.749 mm/h. The collapse toward zero at higher thresholds matches the documented
under-forecasting (2026-05-23 entry) — CSI now gives a real per-epoch quality signal to trend,
while `val_loss_ema` sits at its usual plateau (0.0981). Watch whether `csi_1.0mm`/`csi_5.0mm` climb
with more training; if they stay flat it points to a recipe issue, not undertraining.

## 2026-05-24 — Gradient accumulation to match the authors' effective batch (genforecast)

**What:** Exposed two diffusion-stage knobs in `config/train_rust.yaml` and threaded them
through `train_rust.py -> train_genforecast_rust.py -> setup_model -> setup_genforecast_training`
into `pl.Trainer`:
- `genforecast_accumulate_grad_batches: 16` (new) -> effective batch = 4 micro x 16 = 64
- `genforecast_lr: 1.0e-4` (lr already existed in the lower functions; it just wasn't
  configurable from the orchestrator)

EMA left unchanged (it updates per batch, so its wall-clock smoothing window is unaffected
by accumulation).

**Why:** The `genforecast_rust` run plateaued — `val_loss_ema` best 0.098 @ epoch 6, flat
~0.108 for the 13 epochs since; `train_loss` still drifting down; no NaNs. The run is
memory-bound at micro-batch 4 (batch 8 OOMs at 128^2 on the 16 GB RTX 5080). The original
LDCast authors trained the 128^2 diffusion model at batch 64 / lr 1e-4 with NO accumulation
(git 7eec9d0, "Changes for paper revision"). Accumulation reproduces their effective batch of
64 at zero extra VRAM, which (a) denoises the diffusion gradient (4 random timesteps/step ->
64) and (b) re-matches lr=1e-4 (which the authors tuned for batch 64) to the batch, instead of
running it against an effective batch of only 4.

**Watch:** Judge from the TensorBoard forecast previews / offline metrics, NOT val_loss_ema
(eps-MSE — saturates early and doesn't track sample quality; see comment in diffusion.py).
`global_step` now advances ~16x slower per epoch (8000 batches -> 500 optimizer steps);
wall-clock/epoch is ~unchanged. VRAM should be unchanged (~15.2 GB).

**Outcome (interim, 2026-05-24, epoch 35 / +15 ep since resume):** Change confirmed live —
`global_step` advances ~500/epoch (8000 batches / 16) vs 8000/epoch before, so accumulate=16 is
active. Stable, no NaN. `val_loss_ema` still flat: best 0.0918 (first epoch after resume) vs prior
0.098, bouncing 0.092-0.108, 14 epochs since best. So effective-batch-64 did NOT break the plateau
on this proxy metric — consistent with the plateau being metric-saturation / data-limited rather
than gradient-noise-limited. Preview check (user, 2026-05-24): forecast previews look a little
better than ~epoch 20 — a modest *real* quality gain despite the flat val_loss_ema, confirming the
metric/quality disconnect. Net: matching the authors' effective batch (64) helped slightly. Since a
clean optimization fix bought only a little, the remaining headroom looks data/generalization-bound,
not optimization-bound.

## 2026-05-23 — Diffusion under-forecasts; stop letting eps-MSE (`val_loss_ema`) control training

Trained the rust autoencoder to convergence and ran a first real overnight diffusion run. The
`val_loss_ema` curve looked converged (best 0.0955 @ epoch 44, ran to ~epoch 65), but an ensemble
read showed the forecasts are badly under-covered — because the eps-MSE was driving early-stopping,
the LR schedule, and checkpoint selection, none of which it's fit for.

### The finding: forecasts dry out; the loss hid it
A 16-member ensemble from the best ckpt (epoch 44) on a *persistent widespread-rain* val case (past
61→73 % wet, truth 77–84 %) collapses coverage to ~20 % (+1) → ~2 % (+8) — it erases existing rain
and is **worse than naive persistence** (the input alone is ~73 %). Confirmed real, not an artifact:
- The ensemble *mean* is just as sparse (not a single-sample fluke).
- The training-time `val/forecast` monitor (separate code path) shows the same dry-out.
- Not the autoencoder — the rust AE reconstructs heavy rain faithfully (val_rec_loss ~0.061; verified
  by rendering an input-vs-recon grid on a 63 %-wet case).
- The case is fair (persistent rain, not unforecastable initiation).
Root cause: undertrained diffusion (~130 K steps) **plus** `val_loss_ema` (an eps-prediction MSE)
being treated as a quality/convergence signal. It plateaus early and barely tracks sample quality
(already noted 2026-05-21), yet it controlled three things it shouldn't.

### Stop using `val_loss_ema` as a control signal
- **Constant LR** (`diffusion.py` `configure_optimizers`): removed `ReduceLROnPlateau(monitor=
  "val_loss_ema")`. On the eps-loss plateau it had decayed the LR to **6.1e-9** (verified in
  `last.ckpt`) while forecasts were still poor — so "more training" learned nothing. Now constant
  `lr`; the linear warmup in `optimizer_step` is preserved.
- **No early-stop** (`config/train_rust.yaml`): `early_stopping_patience` 20 → 0. The eps-loss must
  not stop training; judge progress by the `val/forecast` images and offline ensemble metrics.
- **No "best" checkpoint** (`genforecast/training.py`): `save_top_k` 1 → 0, keep only `last.ckpt`.
  The eps-MSE "best" is meaningless for diffusion, and each ckpt is 6.7 GB on a near-full disk.
  Removed the now-dead `monitor`/`filename`. Inference / resume use `last.ckpt`.
- `limit_train_batches` 2000 → 8000 (now only sets val/preview/checkpoint cadence).
- Deleted the two dead 6.7 GB epoch-44 checkpoints (18 → 31 GB free). Checkpointing had silently
  frozen at epoch 44 (`last.ckpt` stuck at step 90000 while TB ran to 130000) — disk too full to
  write a new 6.7 GB ckpt, which likely also ended the run.

### Autoencoder NaN no longer poisons stage 2
A NaN'd autoencoder early-stops *gracefully* (exit 0), and `train_rust.py` only checks the subprocess
exit code — so it was building the diffusion stage on a broken/2-epoch autoencoder.
- `autoenc/training.py`: added `gradient_clip_val=1.0` to the Trainer (plain AdamW, so Lightning's
  built-in clip works; mirrors the diffusion stage). Tames the bf16 KL-VAE blow-up.
- `train_autoenc_rust.py`: after `fit`, `sys.exit(...)` if the final `val_rec_loss` is non-finite, so
  the orchestrator's exit-code check aborts before stage 2. Verified: a clean short run exits 0 (a
  finite metric doesn't false-trip).

### TensorBoard preview was always blank (both stages)
The recon/forecast loggers cached the **first** val case, which is bone-dry, and `plot_precip_image`
masks <0.1 mm/h to white → an all-white grid every epoch.
- `autoenc/monitor.py`: pick the **wettest crop in the batch**.
- `genforecast/monitor.py`: scan up to 8 batches, keep the **wettest-future** case (batch 4 + dry
  leading val entries → one batch isn't enough).

### `resume`: continue-or-fresh (resolves the 2026-05-21 footgun)
`train_rust.py` `_last_ckpt` (errored when missing) → `_resume_ckpt` (path or `None`). `resume: true`
now **continues from `last.ckpt` if present, else starts fresh**; `resume: false` always restarts.
One set-and-forget `resume: true` survives the stop/restart cycles of power-window training. Updated
docstring, config comment, README. (Implemented as a redefinition of `true` rather than a third
`auto` value.)

### Misc
- **README**: the `tensorboard --logdir ../models/...` command now shows `cd scripts` first — that
  relative path only resolves from `scripts/`, so launching elsewhere gave "No dashboards are active".
- Silenced torch's pytree `LeafSpec` deprecation in both training modules (third-party, benign; was
  spamming every run).
- New **`/analyze`** Claude Code skill (`.claude/skills/analyze/`): summarizes the active TB run —
  stability/NaN, primary-metric whole-run + recent-slope trend + epochs-since-best, train-loss bins,
  checkpoints. For diffusion it flags `val_loss_ema` as a divergence detector, not a quality verdict.

### Next
- Re-run diffusion fresh (`stages: diffusion`, `resume: true`, no ckpt yet → fresh) and watch
  `val/forecast` coverage at +4/+8, not the loss. If coverage stays low after much more training,
  it's a recipe issue (conditioning strength / rain-weighted loss), not just undertraining.

## 2026-05-21 — Rust diffusion OOMs at batch 8 on 16 GB; config-driven stages/resume

### batch 8 doesn't fit (corrected the config default)
The `genforecast_batch_size: 8` default was wrong for 16 GB: it came from a *throughput* benchmark
that had deleted the EMA store to fit, and was never run as a real training step — so it OOM'd at
runtime. Verified on a clean GPU: diffusion at 128² with 8-bit AdamW **OOMs at the first optimizer
step at batch 8** (peak 15837 / 16303 MiB, in the stock AFNO einsum) and **fits at batch 4** (peak
15239 MiB, full train+val+sample loop). The OOM is inherent to the 670 M UNet + grads + 8-bit Adam +
the fp32 EMA copy (~2.7 GB, present since the initial commit) — not from any recent change (`afno.py`
/`unet.py` untouched in 3 years; the only uncommitted file was the config). Default is now batch 4.
Measured batch-4 throughput ~20 samples/s (~4.9 it/s) → ~29 h full epoch; throughput entry updated.

### stages + resume replace the skip/force/ckpt_path flags
The earlier interface (`skip_autoenc`/`force_autoenc` + `--autoenc_ckpt_path`/`--genforecast_ckpt_path`
+ an interactive "re-train?" prompt) was clunky and asymmetric — stage-1 resume needed the
`force_autoenc` dance plus an explicit ckpt path that was *silently ignored* whenever stage 1 was
skipped. Replaced with two config knobs in `train_rust.py`:
- `stages: both | autoenc | diffusion` — which stage(s) to run (diffusion-only = the old skip_autoenc;
  autoenc-only is new: extend the autoencoder without running diffusion).
- `resume: false | true` — restart from scratch vs continue each run stage from its `<dir>/last.ckpt`.
No prompt; the config declares intent. Verified all combos build the right subprocess commands and that
missing-`last.ckpt` / invalid-`stages` exit cleanly. Supersedes the flag list in the entry below.

### train_rust.py loads the config by default
`main()` falls back to `config/train_rust.yaml` (resolved from the script dir) when no `--config` is
given, so the whole command is `uv run python train_rust.py`. `--config=<other.yaml>` overrides; CLI
kwargs still win over the file. Verified: no-arg run loads the shipped config, CLI overrides apply,
explicit `--config` still works.

### Open / next (NOT implemented — pick up here)
- **`resume` is a footgun.** Currently a manual bool: `false` = restart, `true` = continue (errors if
  no `last.ckpt`). But for the diffusion stage you almost always want to continue — you'd only *not*
  resume on the first run (no ckpt yet), after retraining the autoencoder / changing `future_steps`/dims
  (stale or shape-incompatible ckpt), or to scrap a diverged run. So the default `false` means you must
  remember to flip it to `true` after run #1 or you silently restart from scratch. Proposed: make
  `resume: auto` the default → continue if `last.ckpt` exists else start fresh; `false` forces a restart;
  `true` requires a ckpt. **Awaiting go-ahead — not built yet.**
- **To kick off the real diffusion run** (autoencoder trained, diffusion never checkpointed):
  `uv run python train_rust.py --stages=diffusion` (resume stays false; nothing to resume yet).

## 2026-05-21 — Config-driven, resumable, time-boxed rust training

_Superseded in parts by the entry above: batch 4 (not 8); `stages`/`resume` replace the
`skip_autoenc`/`force_autoenc`/`*_ckpt_path` flags; the config loads by default._

Makes the two-stage rust training drivable from a YAML and survivable across runs (needed because a
full epoch over the 2.3M-crop 128 index is ~17 h — see the throughput entry).

### Changes
- **`save_last=True`** on both stages' `ModelCheckpoint` → a `last.ckpt` always reflects the most
  recent epoch (the best-named ckpt is still kept for deployment). The right target for `--ckpt_path`.
- **`max_hours`** (both stages) → `Trainer(max_time=timedelta(hours=...))` for time-boxed chunks: run
  N hours, stop, re-run with `--ckpt_path=.../last.ckpt` to continue. Lightning resume restores
  optimizer/LR/EMA/epoch — a true continue, not a weights-only restart.
- **`early_stopping_patience`** (both stages) exposed; `0` disables EarlyStopping. Default stays 6, but
  the config sets 20 because the recommended short epochs shrink the patience window (it counts epochs).
- **`train_rust.py` now takes `--config=<yaml>`** — it was the only training script without it (the
  per-stage scripts already had the OmegaConf `main()` pattern). CLI args still override the file. It
  also forwards `--autoenc_ckpt_path` / `--genforecast_ckpt_path`, so resume works through the
  orchestrator + same config (e.g. `--skip_autoenc=True --genforecast_ckpt_path=.../last.ckpt`).
- **`config/train_rust.yaml`** — sane 16 GB / 128² defaults: batch 16/8, `optimizer_8bit`, bf16-mixed,
  `num_workers` 8, `limit_train_batches` 2000 + `limit_val_batches` 50 (≈8 min epochs so
  val/preview/checkpoint/early-stop fire at a usable cadence vs the ~17 h full epoch), patience 20.
- **Data location in the config** (`radar_root` / `index_path`): `index_path` is forwarded to the
  stage scripts as `--index_path` (no env round-trip); `radar_root` is exported as `DGMR_RADAR_ROOT`
  because the dgmr-py Rust loader reads the archive root from the env (`std::env::var`) and exposes
  no way to pass it. Env vars are needed only when the config omits a value.

### Why cap limit_val_batches too
With `limit_train_batches` making ~8 min epochs, the full val set (~10 % ≈ 230K crops ≈ 38 min) would
dominate. Capping to 50 batches keeps validation ~12 s — a small consistent subset, fine for progress
tracking and checkpoint selection.

Verified: config → correct subprocess commands (CLI override respected); `save_last` / Timer /
EarlyStopping toggles behave as expected.

## 2026-05-21 — Training observability: TensorBoard sample + reconstruction logging

Added a dgmr-rs-style "watch the forecast during training" view. Before: scalar-only (Lightning's
default CSVLogger; `train_loss` / `val_loss` / `val_loss_ema`), no sample images — useless for a
diffusion model whose eps-MSE barely moves.

### Changes
- **`ldcast/models/genforecast/monitor.py`** — `SamplePredictionLogger` (`pl.Callback`). Every
  `sample_every_n_epochs`, samples one fixed val case through the PLMS sampler and logs a
  ground-truth-vs-prediction precip image grid (`val/forecast`) to TensorBoard.
- **`training.py`** — Trainer now uses `TensorBoardLogger(save_dir=model_dir, name="tb")` (so the
  existing scalars stream live too) and adds the callback when `sample_every_n_epochs > 0`. Threaded
  `sample_every_n_epochs` (default 1) through `setup_model` + both train scripts. Added `tensorboard` dep.
- **`ldcast/models/autoenc/monitor.py`** — `ReconstructionLogger` (stage-1 symmetry): every
  `sample_every_n_epochs`, logs an input-vs-`decode(encode(x))` grid (`val/reconstruction`) + the
  autoenc scalar curves. Simpler than the forecast logger (no sampler / EMA / crop; the autoenc is
  tiny, no memory pressure). Threaded through `train_autoenc_rust.py` / `train_autoenc.py`. Note:
  reconstruction is the autoencoder *alone* — it can't forecast (that needs stage 2) — but its
  fidelity is the ceiling on stage 2, so it's the right stage-1 progress signal.
- Watch with `tensorboard --logdir <model_dir>`.

### Memory (16 GB): the callback must not OOM a tight run
Training already holds model+EMA+optimizer (~10.8 GB), so a full-res EMA sample OOMs (peak 16.1 GB).
Two defaults keep the preview at ~12.2 GB (fits at any training resolution):
- **live weights** (`use_ema=False`): skips the +2.7 GB EMA-store backup; the current model's
  forecast is at least as informative as the lagging EMA early on.
- **center-crop to 128²** (`sample_hw=128`): bounds activations; a no-op when training ≤128².

Wrapped in try/except so a sampling OOM is logged and skipped, never crashing training. Verified
end-to-end (`trainer.fit`): `val/forecast` images logged once per epoch + scalar curves.

## 2026-05-21 — Diffusion training speed: profiled, fused AdamW (+16–27%)

Profiled rust diffusion training (670M UNet) on the 16 GB 5080 to find what to optimize.

### Bottleneck: compute-bound, memory-constrained
The dataloader delivers ~95–103 samples/s (4 workers); the GPU processes only 7–22 samples/s — so
data loading is **not** the limit, compute is. Per-step breakdown (128², batch 4, bf16, 180 ms
baseline): fwd 53 ms (encode 11 + context-encoder 8 + UNet 34) + backward ~67 + **optimizer 60**.

Memory is the other wall: 256²/batch 2 peaks at 15.8 GB *with EMA freed*; real training (EMA on,
+2.7 GB) barely fits batch 2. EMA shadow (2.7 GB) + fp32 AdamW state (5.4 GB) dominate.

### Change: fused AdamW (`diffusion.py` `configure_optimizers`, default on CUDA)
The optimizer step was a third of the step time — default `AdamW` (foreach) over 671M params.
`fused=True` cuts it **60 → 23 ms** (single fused kernel, numerically identical):

| config | baseline | fused AdamW | speedup |
|---|---|---|---|
| 128², batch 4 | 180 ms/step (22 sps) | 142 ms/step (28 sps) | **+27%** |
| 256², batch 2 | 290 ms/step (6.9 sps) | 251 ms/step (8.0 sps) | **+16%** |

Optimizer cost is fixed (param-count bound), so the relative win is larger at the small batches
forced by 16 GB. No quality impact. (Autoenc training left alone — tiny params, already ~11 min.)

**Caveat (fixed 2026-05-21):** the fused optimizer is incompatible with Lightning's *automatic*
gradient clipping ("does not allow ... performs unscaling internally") — and since
`gradient_clip_val=1.0` is the default, this crashed `trainer.fit` at step 0 (only caught once a
real fit was run end-to-end, not the manual-loop timing). Fix: clip manually in
`LatentDiffusion.on_before_optimizer_step` (`clip_grad_norm_`, runs after backward / before step)
and removed `gradient_clip_val` from the Trainer. Same clip value, fused preserved.

### Tested, didn't help / deferred
- **cudnn.benchmark**: ≈ noise — not conv-bound.
- **TF32** (`set_float32_matmul_precision("high")`): no change — confirms the inference-side finding
  (ops already bf16; the fp32 AFNO FFT isn't a TF32 matmul).
- **8-bit AdamW + larger batch** (existing `--optimizer_8bit`): frees ~4 GB → fits batch 4–6 at 256²,
  7.5 → 9.1 → 9.7 sps (per-sample 133 → 103 ms; GPU underutilized at small batch). +14–29%, but
  changes numerics and EMA still caps the batch. Path to more speed: 8-bit Adam + CPU/periodic EMA
  (free the 2.7 GB shadow) → larger batch. Not done (numerics/quality tradeoff).

## 2026-05-21 — Diffusion training: scale_factor tested (doesn't help), two monitor fixes

Investigated the training items flagged below (futures #2, #3) plus a validation-loop bug found
while reading the code.

### Fixes (kept)
- **`val_loss_ema` logged the wrong tensor** (`diffusion.py` `validation_step`): it computed
  `loss_ema` under EMA weights, then logged the *non-EMA* `loss` into `val_loss_ema`. The checkpoint
  monitor, early-stopping, and the LR scheduler all key on `val_loss_ema` → all three were tracking
  the raw training weights, not the EMA weights used at inference, and the EMA validation forward was
  computed and thrown away. One-word fix (`loss` → `loss_ema`).
- **`check_finite=False` → `True`** (`training.py` EarlyStopping, future #3): a NaN-diverged run now
  stops instead of wasting hours. Extra-justified by the spike data below.
- **`gradient_clip_val` made configurable** (`training.py` + train scripts; default 1.0 = unchanged),
  so the clip can be tuned/disabled for experiments.

### scale_factor (future #2): implemented, measured, A/B-tested — does NOT help
Wired a `scale_factor` knob through `LatentDiffusion` (encode ×s) and every latent-decode site
(`forecast.py`, `eval_genforecast.py`, ÷s) plus the train/predict CLIs. Default 1.0 = exact no-op;
not stored in the ckpt, so train and inference must pass the same value. **Kept default-off** as
infrastructure.

Measured the rust-autoenc diffusion latent: **std ≈ 0.46, mean ≈ -0.04** — already in the benign
[-1,1]-image range (~0.5) that plain DDPM trains on un-rescaled. So unit-variance means *amplifying*
by 1/0.46 ≈ 2.18.

Controlled A/B (128×128, batch 4, EMA freed, **no clip**, 300 identical fresh batches, seed-matched):

| scale_factor | NaN | max grad_norm | spikes(>5) | loss[last 50] |
|---|---|---|---|---|
| 1.00 (off)     | none | 43       | 101 | 0.576 |
| 2.18 (unit var)| none | **3818** | 56  | 0.708 |

Amplifying to unit variance made the worst gradient spike ~88× larger (loss hit 37 at step 54, a
hair from divergence) and **slowed convergence**. The latent is already well-scaled; scale_factor=2.18
is counterproductive. The hypothesis that scale_factor "removes the need for the gradient_clip
band-aid" is **not supported** — keep the clip: it is load-bearing (both runs throw frequent large
unclipped spikes that would eventually NaN over a full run). Left `scale_factor=1.0` everywhere.

(Caveat: bf16, 128×128, batch 4, 300 steps, single seed — directional, not a full-quality verdict.
256×256/batch 8 OOMs a manual training loop on 16 GB; EMA shadow weights + fp32 AdamW state dominate.)

## 2026-05-20 — DPM-Solver++ / UniPC fast samplers

Implements future-improvement #1 below. Adds two training-free ODE samplers as drop-in
alternatives to PLMS, selectable via a `sampler=` knob, to cut inference time by taking fewer
diffusion steps — the dominant cost, since each step is one 670M-param UNet forward including an
FP32 AFNO FFT (`unet.py` → `afno.py`) that BF16 cannot accelerate. The conditioning cascade is
already cached (one-time), so step count is the only big remaining lever.

### Changes
- **New `ldcast/models/diffusion/dpm_solver.py`** — DPM-Solver / DPM-Solver++ (Lu et al.) vendored
  from the UniPC repo's Stable-Diffusion integration, plus a `DPMSolverSampler` adapter mirroring
  `PLMSSampler.sample`.
- **New `ldcast/models/diffusion/uni_pc.py`** — UniPC (Zhao et al.) class + `UniPCSampler` adapter;
  shares `NoiseScheduleVP`/`model_wrapper`/`expand_dims` with dpm_solver.py (imported, not
  duplicated). One fix vs upstream: the hardcoded `bkchw` einsums in `multistep_uni_pc_bh_update`
  generalized to `bk...` ellipsis for LDCast's 5D (B,C,T,H,W) video latents (upstream assumed 4D
  images; the unreachable `vary_update` path is left as-is).
- **`sampler=` knob** threaded through `forecast.Forecast`/`ForecastDistributed`,
  `eval_genforecast.py`, `forecast_demo.py`, `predict_rust.py` — `"plms"` | `"dpmpp"` | `"unipc"`.
  Default stays `"plms"` until visually confirmed.
- Why it's a drop-in: LDCast is eps-parameterized on a linear discrete schedule (`alphas_cumprod`),
  with no latent scale_factor and no classifier-free guidance — exactly what these solvers expect.
  BF16 autocast still wraps the sampler call, so the speedups compound. **Not output-exact** (changes
  the sample for a given seed) — validated by A/B, not bit-equality.

### Verification (released demo model, 352×448, bf16; same seed; vs plms-50)
| run | corr | mean-abs | time | speedup |
|---|---|---|---|---|
| plms-50 (baseline) | 1.0000 | 0.0000 | 7.97 s | 1.00× |
| dpmpp-50 (sanity) | 0.9954 | 0.0245 | 8.04 s | 0.99× |
| unipc-50 (sanity) | 0.9956 | 0.0237 | 8.04 s | 0.99× |
| dpmpp-20 | 0.9919 | 0.0401 | 3.35 s | 2.38× |
| unipc-15 | 0.9829 | 0.0343 | 2.59 s | 3.08× |

dpmpp/unipc at 50 steps reproduce plms-50 (corr ≈0.995) → the schedule + continuous→discrete
timestep conversion is correct. At reduced steps quality holds (corr ≥0.98) for 2.4–3.1×.
Full-frame (1440×1856, rust fs=8, bf16): plms-50 57.5 s → **dpmpp-20 23.9 s (2.40×)**.
`ForecastDistributed` (ensemble) path also verified end-to-end.

### Recommended use
`--sampler=dpmpp --num_diffusion_iters=20` (robust ~2.4×) or `--sampler=unipc
--num_diffusion_iters=15` (~3×). Possible follow-up: A/B CRPS/FSS via `eval_genforecast.py` +
`metrics.py`, then flip the default sampler once happy with the demo PNGs.

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

1. **DONE — Faster sampler (DPM-Solver++ / UniPC).** Implemented; see the 2026-05-20 entry above.
   ~2.4× (dpmpp-20) / ~3× (unipc-15) at corr ≥0.98 vs plms-50, no retraining. Plain DDIM was *not*
   worth it — same per-step cost as PLMS. Follow-up: CRPS/FSS A/B and flipping the default sampler.
2. **TESTED, doesn't help — Latent scale factor.** Implemented as a configurable `scale_factor` (default
   1.0, off). Measured latent std ≈0.46 (already benign); scale_factor=2.18 (unit variance) *worsened*
   the max gradient spike (43→3818) and slowed convergence vs no scaling. The `gradient_clip` is
   load-bearing and not removable. Knob kept default-off. See the 2026-05-21 entry.
3. **DONE — NaN-safe training.** `EarlyStopping(check_finite=True)` set, so a NaN-diverged run stops
   instead of wasting hours. See the 2026-05-21 entry.
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
