---
name: analyze
description: Analyze the current/most-recent LDCast training run and report how it's going. Reads the TensorBoard scalar logs under models/<dir>/tb and gives a short, honest verdict — healthy / watch / broken — covering the validation metric, the (noisy) training-loss trend, stability (NaN / KL), and what to watch. Use when asked "how is training going", "analyze the training", "check training progress", "is the run healthy", or similar.
version: 0.1.0
---

# Analyze training

Turn the raw TensorBoard logs of the current LDCast run into a short, honest verdict.

## How to run

From the repo root (the script resolves paths relative to the repo, so CWD doesn't matter):

```bash
uv run python .claude/skills/analyze/analyze_training.py
```

- Analyzes the **most recently written** run under `models/*/tb/version_*` — that's the "current" training. It auto-detects the stage (autoencoder vs diffusion).
- Target a specific run with `--run models/autoenc_rust/tb/version_1`.
- Read-only: it never touches checkpoints or training.

## How to read the output and what to tell the user

Lead with a one-line verdict (**healthy / watch / problem**), then back it with the numbers from the report. Use this framing:

1. **Stability first.** If STABILITY reports NaN/Inf, the run diverged — say so plainly. This is the known autoencoder instability; the fix is to resume from the last good checkpoint (gradient clipping in `ldcast/models/autoenc/training.py` should prevent recurrence). No NaN = good.

2. **The PRIMARY metric is the real signal for the autoencoder** — `val_rec_loss` is what stage-1 checkpoints/early-stopping track. It's computed on a small (~50-batch) validation subset, so it's noisy epoch-to-epoch — judge the trend, not single points. The output gives two trend reads: a `whole-run` label and, more importantly for "is it *still* improving?", a `recent` slope over the last ~8 epochs (PLATEAU / still improving / worsening) plus `epochs since best`. Lead with the recent slope when the user asks whether it's still going down; the whole-run label stays IMPROVING long after the recent trend has flattened. **For diffusion (stage 2), `val_loss_ema` is the checkpoint target but NOT a quality signal** — it's eps-MSE and plateaus very early (~0.08-0.10) whether the model improves or not. Treat it as a divergence detector only; the QUALITY METRICS section below is the real verdict.

3. **QUALITY METRICS (diffusion only) — `val/csi_*mm`, `val/mae|rmse_mmhr`.** Logged once per validation_epoch_end by the genforecast monitor on a small fixed set of the wettest val cases, so single epochs are noisy — read the trend (rising / plateau / falling) and `epochs since best`. CSI is higher = better; MAE/RMSE lower = better. What to lean on per threshold:
   - **`csi_5.0mm`** — heavy rain, the hardest case. This is the LDCast-sampler hypothesis test (prior uniform-sampler ceiling was ~0.024). If this is rising over many epochs the sampler is paying off; if it stays flat near the prior ceiling once the model is past warmup, the hypothesis is in trouble.
   - **`csi_1.0mm`** — moderate rain, the most reliable progress signal (denominator stays populated; less noisy than csi_5).
   - **`csi_0.1mm`** — light rain / "is it raining at all". Saturates fastest; mostly a sanity check that the model isn't drying out.
   When stage 2 is fresh, expect the CSI panel to be empty for ~1 epoch (first val hasn't fired) — say so plainly instead of speculating.

4. **TRAIN LOSS is per-step and noisy — read the binned trend, not the raw spikes.** The training loader uses a weighted sampler that oversamples heavy-rain crops, so per-step `train_loss` has high variance and stretches of harder batches. A temporary bump that recovers is normal, **not** divergence. Only a sustained climb across many bins (especially with the val metric also rising) is a real concern. When the user says "loss went up," check whether it's just a transient in the bins.

5. **Cross-checks that explain the dynamics:**
   - A step-down in the val metric *together with* `train_loss` calming usually means `ReduceLROnPlateau` (patience 3, factor 0.25) dropped the LR — healthy.
   - `val_kl_loss` should stay roughly stable (~1.x for the autoencoder); a runaway KL is a red flag.
   - The CHECKPOINTS section shows the best val actually saved — sanity-check it against the curve.

6. **Close with status + what to watch:** current epoch, whether the run is ACTIVE, and the early-stopping budget (`early_stopping_patience`; if 0 / disabled, mention training won't stop on its own). For autoencoder: suggest watching the *smoothed* `train_loss` in TensorBoard and using `val_rec_loss` as the progress signal. For diffusion: point them at `val/csi_*mm` and the `val/forecast` image, NOT `val_loss_ema`.

Keep it concise and scannable — numbers + verdict, not a wall of text.
