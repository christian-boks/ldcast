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

2. **The PRIMARY metric is the real signal** — `val_rec_loss` (autoencoder) or `val_loss_ema` (diffusion); it's what checkpoints and early-stopping track. It's computed on a small (~50-batch) validation subset, so it's noisy epoch-to-epoch — judge the trend, not single points. The output gives two trend reads: a `whole-run` label and, more importantly for "is it *still* improving?", a `recent` slope over the last ~8 epochs (PLATEAU / still improving / worsening) plus `epochs since best`. Lead with the recent slope when the user asks whether it's still going down; the whole-run label stays IMPROVING long after the recent trend has flattened.

3. **TRAIN LOSS is per-step and noisy — read the binned trend, not the raw spikes.** The training loader uses a `WeightedRandomSampler` that oversamples heavy-rain crops, so per-step `train_loss` has high variance and stretches of harder batches. A temporary bump that recovers is normal, **not** divergence. Only a sustained climb across many bins (especially with the val metric also rising) is a real concern. When the user says "loss went up," check whether it's just a transient in the bins.

4. **Cross-checks that explain the dynamics:**
   - A step-down in the val metric *together with* `train_loss` calming usually means `ReduceLROnPlateau` (patience 3, factor 0.25) dropped the LR — healthy.
   - `val_kl_loss` should stay roughly stable (~1.x for the autoencoder); a runaway KL is a red flag.
   - The CHECKPOINTS section shows the best val actually saved — sanity-check it against the curve.

5. **Close with status + what to watch:** current epoch, whether the run is ACTIVE, and the early-stopping budget (`early_stopping_patience`, default 20 — training continues while the primary metric improves within that many epochs). Suggest watching the *smoothed* `train_loss` in TensorBoard (smoothing slider) and using the primary val metric as the progress signal.

Keep it concise and scannable — numbers + verdict, not a wall of text.
