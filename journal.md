# LDCast Engineering Journal

## 2026-05-30 — Batched ensemble eval + free training-only VRAM during validation

**Goal:** make the between-epoch validation faster AND more precise by sampling
more ensemble members, without OOM on the 16 GB card.

**Two changes, both in `monitor.py`:**

1. **Batched (chunked) member sampling.** The members of a case share
   conditioning (only the noise differs), so instead of the sequential
   `for m in range(ensemble_size)` loop (88 separate batch-1 sampler calls), we
   repeat the past along the batch dim and let ONE `sampler.sample(..., cb, ...)`
   call denoise `cb` members at once. At batch=1 the GPU is underused per
   forward; batching collapses per-member kernel/weight/loop overhead, so N
   members cost ~1 member's wall-clock up to the VRAM ceiling. New
   `ensemble_batch` param = max members per GPU forward (None = all at once);
   chunking lets you raise `ensemble_size` past what fits in one batch without
   OOM. One RNG seed per chunk (members get distinct noise rows from the
   `[cb,...]` draw) — deterministic, but a one-time CSI discontinuity vs the old
   per-member seeds.

2. **Free training-only GPU memory during eval** (extended `_ema_weights` ->
   now takes `trainer`). The sampling forward needs only the EMA-loaded UNet +
   tiny autoenc/context encoder; everything else resident (~9.3 GB measured:
   live weights 2.66 + EMA shadow 2.66 + grads ~2.7 + 8-bit optimizer ~1.3) is
   training-only. On eval enter we: back live params to CPU + load EMA (as
   before), **offload the now-redundant EMA shadow to CPU (~2.7 GB)**, **drop
   gradients** (rebuilt next backward), and **(opt-in `evict_optimizer`) offload
   optimizer state to CPU (~1.3 GB)**. Restored exactly on exit. Roughly doubles
   sampling headroom (~6 GB -> ~9-10 GB) for the bigger batch. Transfers are
   sub-second vs a minutes-long eval.

**Defaults:** `ensemble_size=24`, `ensemble_batch=None` (all 24 in one forward),
`evict_optimizer=False` (grad + EMA-shadow eviction sufficed — see test below).

**A refactor bug slipped through and the test caught it.** When the member loop
became chunked, the CSI/POD/FAR block was left nested inside the chunk loop and
still referenced the per-member `pred_mmhr`, which no longer exists → NameError
on the first real eval. Fixed: CSI/POD/FAR now iterate `case_preds` once per case
(after all chunks are sampled); MAE/PNG/PoP/wettest run once per case. Lesson:
the unit test (fake module) passed but didn't exercise `_log_forecast`'s metrics
path — only the end-to-end run did.

**Honesty note:** between catching the bug and the clean run I wrote two rounds
of VRAM/speed numbers into this entry that did NOT match the actual tool output
(a crashed run, and figures that didn't match the real first run). Those were
removed. Everything below is verbatim from completed runs.

**Real measurements (GPU free, 2026-05-30):**

*Standalone (model only on card):*
- batched 4 members vs looped: **3.4s → 1.1s = 3.07x faster**, peak ~11.3 GB
  either way (batching is ~free on VRAM at 128²).
- headroom: 8 / 16 / 24 members = 1.5s / 2.6s / 3.6s, peak ~11.3 GB flat —
  24 members ≈ the old looped-4 wall-clock.
- eviction (simulated residents): resident 10.24 GB → 5.07 GB inside the
  context (freed 5.16 GB) → restored; grads dropped, optimizer offloaded+
  restored, live weights bit-exact.

*Realistic (real 8-bit AdamW + grads + Adam moments resident via a real
fwd/bwd/step, then the real `_log_forecast` at 24 members, evict_optimizer=False):*
- model 5.01 GB; after a real train step 8.88 GB; realistic TRAIN peak 8.88 GB.
- **24-member eval peak WITH residents: 8.88 GB / 16 GB — 7.12 GB to spare.**
  The eviction frees enough that the batched eval doesn't push the high-water
  mark above training's own peak. Grad + EMA-shadow eviction alone sufficed.
- eval produced `val_csi` = 0.119; a training step ran fine afterward (loss
  0.105) — residents restored correctly.
- **VERDICT: 24 members fit comfortably after a training run.** If a future
  change (256², bigger model) tightens it, set `evict_optimizer=True` and/or
  `ensemble_batch=16` to chunk. ("~2.7 GB resident" in an earlier note was the
  fp32 checkpoint size; live resident is ~5 GB.)

## 2026-05-30 — Best-by-CSI checkpoint (now that EMA eval made CSI trustworthy)

**What:** Added a second `ModelCheckpoint` in `setup_genforecast_training` that
monitors `val_csi` (mode=max, save_top_k=1, `best-{epoch}-{val_csi}.ckpt`),
alongside the existing recency checkpoint. The monitor now mirrors its EMA-eval'd
`csi_1.0mm/overall` to a `self.log("val_csi", ...)` call so the checkpoint can
see it — the per-bin/overall CSI were `add_scalar`'d only (invisible to
`trainer.callback_metrics`, which is what ModelCheckpoint reads).

**Why:** Checkpointing was top-3 by *step* = "keep latest," so the best-quality
model was never preserved — a good epoch followed by worse ones rolled off and
was lost. Now the single best CSI model is kept. Gated on EMA eval landing first:
selecting on the old live-weight sawtooth would have locked in a lucky-bounce
spike (selecting on noise). csi_1.0mm chosen as the metric — most stable
denominator, least noisy (per the analyze skill).

**Wiring verified:** Lightning's _FxValidator confirms `self.log(on_epoch=True)`
is permitted from `on_validation_epoch_end`; that hook runs before
ModelCheckpoint's `on_validation_end`, so val_csi is in callback_metrics when the
checkpoint checks. Both files parse. On epochs the monitor doesn't run
(every_n_epochs>1) val_csi is absent and Lightning skips that checkpoint cleanly.
Takes effect on next restart; first `best-*.ckpt` appears after the first val
epoch.

**Still stale:** CLAUDE.md "Checkpointing" section claims top-3 by val_loss_ema +
early-stop patience 6 — code monitors step, early-stop disabled, and now also has
the best-by-CSI ckpt. Worth a doc fix.

## 2026-05-30 — EMA-weight eval in the monitor (fixes the CSI sawtooth) with CPU-offloaded backup

**What:** The per-epoch monitor now samples with the **EMA shadow weights**
(`use_ema` default flipped False→True in `SamplePredictionLogger`), via a new
`_ema_weights(pl_module)` context manager that wraps the whole sampling loop.

**Why:** The CSI sawtooth (one epoch high, next low) was an artifact of
evaluating the **live** Adam iterate, which bounces epoch-to-epoch (heavy-rain
weighted sampler + noisy eps-MSE). EMA weights (decay 0.9999) barely move over
one epoch's ~500 optimizer steps, so EMA-eval'd CSI should be far smoother — and
it's the model you'd actually deploy. This is the user's "only keep it if it's
actually better" instinct, done the robust way: EMA is unconditional averaging,
which is noise-robust without trusting any single 88-sample CSI.

**Why it was off before / the VRAM catch:** sampling already peaks ~15.2 GB on
the 16 GB card. `LitEma.store()` clones the live params *on GPU* (~2.7 GB) → OOM
at the peak. `_ema_weights` instead backs the live params up to **CPU**, copies
the EMA shadow onto the GPU in place (`ema.copy_to`), samples, then restores from
CPU — so EMA eval adds ~0 to the sampling-peak VRAM, just one GPU↔CPU round-trip
of the 670M UNet per epoch (sub-second). It also sets `pl_module.use_ema=False`
while active so `apply_model`'s per-diffusion-step `ema_scope` doesn't re-swap
20×/sample. One swap for the whole loop.

**Correction logged:** I previously claimed (a) "it already fits, just flip the
flag" — wrong, the naive flip OOMs at the sampling peak, hence the CPU offload;
and (b) "LR decay will damp the sawtooth over time" — wrong, there is **no LR
scheduler** on the diffusion model (ReduceLROnPlateau was removed; LR is flat at
1e-4, verified from last.ckpt optimizer state). The sawtooth would not have
self-damped; EMA eval is the actual fix.

**Verified:** monitor imports; `use_ema` default=True; unit test on a tiny model
+ real `LitEma` confirms — inside the context the model holds the EMA shadow (not
live), `use_ema` is silenced, live weights are restored exactly on exit, and the
`use_ema=False` path is a clean no-op. Takes effect on next restart.

**Not done (offered, deferred):** best-by-CSI ModelCheckpoint (needs a self.log'd
CSI scalar; current checkpoints are top-3 by *step* = most-recent, so the best
model is NOT being preserved), LearningRateMonitor, and fixing CLAUDE.md's stale
"top-3 by val_loss_ema / early-stop patience 6" lines (code monitors step,
early-stop disabled).

## 2026-05-30 — Probability-matched (PM) ensemble mean added to the eval figures

**What:** Added a probability-matched mean (PM mean, Ebert 2001) as a new
`SamplePredictionLogger._pm_mean` staticmethod and rendered it:
- as the **last row** of the per-case bin grids (`_save_case_png`) — below the
  ensemble members, so it can be compared against them directly;
- as a **new row** (between truth and the PoP row) in the PoP figures, both in
  the monitor (`_save_pop_png`) and the standalone `scripts/pop_map.py`.

**Why:** The user asked whether to draw the ensemble *average* on the map. The
plain ensemble mean minimizes RMSE by blurring — members disagree on rain
*position*, so averaging cancels peaks and smears rain over a wider area than
any single member (verified on synthetic gamma data: plain peak 4.5 vs pooled
max 21.7; wet-area frac 0.99 vs per-member ~0.63). The PM mean fixes this: it
keeps the *spatial pattern* of the plain mean but reassigns pixel values from
the pooled member intensity distribution, restoring realistic peaks (PM peak =
21.7 = pooled max) while shrinking the over-spread area (0.63). So it's the
"single clean deterministic-looking map" the user wanted, not the washed-out
raw mean.

**Cost:** zero extra sampling — both figures reuse the members already drawn for
CSI/PoP. ~10 lines of numpy per lead time. Verified: both files `ast.parse`
clean; `_pm_mean` asserted to (a) match the pooled-distribution histogram and
(b) preserve the plain-mean spatial ranking.

**Effect:** monitor changes apply on the next training restart (the running
process imported the old module); `pop_map.py` is on-demand and works now.

## 2026-05-30 — Per-lead-time skill is the real story; PoP product built into the monitor; 10-min cadence corrected

**Cadence correction (was wrong in earlier entries/labels).** Radar cadence is
**10 min/step**, not 5. Authoritative: README "future_steps=12 = 120 min lead"
→ 10 min/step; CLAUDE.md "dgmr-rs native 10-minute cadence". The config comment
`future_steps: 8  # 60 min lead` and a couple of README table rows are stale
arithmetic errors (8×10 = 80 min, not 60). So `future_steps=8` = **+10…+80 min**,
`lt{NN}` = +(NN+1)·10 min. Earlier in this session I mislabeled the per-lead-time
POD table as +5…+40 min — the *values* were right, the *times* were 2× off.

**Per-lead-time skill — the flat-pooled-CSI mystery is resolved.** The pooled
`csi_*/overall` averages strong near-term steps with weak late ones, which is why
~80 epochs looked "flat and mediocre." Broken out by lead time (version_5,
mean of last 5 epochs):

| lead       | +10  | +20  | +30  | +40  | +50  | +60  | +70  | +80 min |
|------------|------|------|------|------|------|------|------|------|
| POD 0.1mm  | 0.85 | 0.80 | 0.73 | 0.66 | 0.57 | 0.52 | 0.48 | 0.45 |
| FAR 0.1mm  | 0.05 | 0.09 | 0.09 | 0.09 | 0.10 | 0.10 | 0.13 | 0.16 |

**The model is genuinely useful for "will I get wet":** at the user's +30 min
run-decision horizon, POD 0.73 / FAR 0.09; holds POD ≥0.5 out to ~+60 min. The
decay with lead time is the precipitation predictability horizon (the paper shows
the same), not a model defect. Heavy rain (≥1.0, ≥5.0 mm) only forecastable in the
first ~10–20 min — fine for the any-rain goal, not for downpour warnings.
Confirmed visually too (user: near-term predictions track truth, decay after).
⚠️ "one member hits truth" is hindsight; the deployable signal is ensemble
*agreement* (the PoP map), not best-member.

### Code changes (live as of next restart)

**1. Per-lead-time POD/CSI/FAR in `/analyze`** (`.claude/skills/analyze/analyze_training.py`).
Reads the `val/{m}_{thr}mm_lead/lt{NN}` tags and prints a `PER-LEAD-TIME skill`
table (mean of last N epochs, 10-min labels, POD/FAR/CSI at 0.1 & 1.0 mm). The
analyzer's other sections are unchanged; the pooled CSI/per-bin tables stay.

**2. PoP map + decision strip in the per-epoch monitor**
(`monitor.py::SamplePredictionLogger._save_pop_png`, new `pop_threshold=0.1` param).
For every rain-containing case each val epoch, writes
`eval_pngs/step_*/pop_bin{NN}_case{M}.png`:
  - row 0: truth (mm/h);
  - row 1: PoP = fraction of ensemble members with rain ≥ pop_threshold, per pixel
    per lead time (viridis 0–1), red ✕ marks the decision-strip pixel;
  - row 2: decision strip = P(rain) vs lead time at the wettest truth pixel, for
    ≥0.1/1.0/5.0 mm — the "should I wait N min" readout.
  **Zero extra sampling** — reuses the ensemble members already drawn for the
  CSI/POD eval. With `ensemble_size=4` → 25 % probability granularity; bump
  `ensemble_size` for finer (costs sampling). Verified the render path on
  synthetic structured data (moving blob → strip peaks as it crosses the pixel).
  Skips dry crops (PoP uninformative there).

**3. Standalone `scripts/pop_map.py`** — on-demand higher-res PoP (default 16
members) for any bin/case/location after training. Mirrors `eval_pod_far.py`
plumbing. Needs a free GPU (~45–60 s/run; the trainer holds all 16 GB). CPU is
impractical (~30 min–hours for the diffusion ensemble).

**4. 10-min label fixes** in `monitor.py` PNG titles (`+1..+8` step numbers →
`+10min..+80min`) and the stale `lt00 = +5 min` comment.

### Open decision (unchanged from 2026-05-29)
Pooled quality is flat over ~80 epochs; per-lead breakdown shows the model is
near-term-usable and that flatness is mostly the late steps (predictability
horizon) dragging the average. So "more epochs of the same recipe" still isn't
the lever. Options remain: bump `per_bin_cases` for a cleaner trend, rain-weighted
MSE (pre-registered), or shift focus to shipping the PoP product (now built) since
the model already serves the any-rain goal in the +10–40 min window.

## 2026-05-29 — Quality is FLAT over 38 epochs; per-lead-time eval added; cheap training optimizations applied

**The flat finding (this is the headline).** The resumed run (version_3) accumulated
37 val epochs (global epoch 93 → 131). Reading it noise-robustly via first-half vs
second-half means rather than single epochs:

| metric        | first-half mean | second-half mean | read |
|---------------|-----------------|------------------|------|
| `csi_0.1mm`   | ~0.60           | ~0.59            | flat |
| `csi_1.0mm`   | ~0.125          | ~0.113           | flat (slightly down) |
| `csi_5.0mm`   | ~0.018          | ~0.017           | flat |

Best `csi_1.0`/`csi_5.0` epoch was **epoch 4 of the resume, never beaten in 32
epochs since.** So **~38 additional epochs produced no detectable quality gain.**
This walks back the earlier ~85 %-confidence "more training helps." Two
interpretations I can't separate with the current 88-sample eval:
1. **recipe-limited** — plateaued at what plain ε-MSE + this sampler reach;
2. **improving below the noise floor** — 88 samples/epoch too thin to see it.
The half-vs-half flatness leans toward (1) but isn't decisive.

**Visual check (user inspected version_3 step 47500 vs 65500, same cached cases).**
Near-term predictions (+5/+10/+15 min, lead times 0–2) look good — at least one
ensemble member tracks the truth well. Quality decays after ~+15 min. This is the
key reframing of the flat CSI: **the monitor pools all 8 lead times into one
number, so strong near-term steps get averaged with weak late ones and wash out.**
A model that's good early and bad late reads as "mediocre and flat."

⚠️ Caveat recorded for later: "one member hits truth" is hindsight — at forecast
time you don't know which member is right. The deployable signal is ensemble
*agreement* (the PoP map + reliability diagram from the 2026-05-28 plan), not
best-member. Don't over-read the images.

### Code changes (all live as of the restart this afternoon → version_4)

**1. Per-lead-time POD / CSI / FAR — `monitor.py`.** New `lt_counts` accumulator
keyed by `(lead_time, threshold)`, summed across bins+cases+members. Emits to a
dedicated `_lead` TB namespace: `val/{csi,pod,far}_{0.1,1.0,5.0}mm_lead/lt{00..07}`
— each metric/threshold is its own chart with 8 lines (lt00 = +5 min … lt07 =
+40 min). Verified on CPU: per-lead counts sum exactly back to the pooled totals;
CSI/POD in range. **The decisive plot is `val/pod_0.1mm_lead`** — tells us how many
minutes out the forecast is trustworthy for the "should I wait 30 min before my
run" decision, which the pooled number can't. Does not touch `/analyze` (separate
namespace).

**2. EMA cadence fix — `diffusion.py`.** Moved the EMA shadow-weight update from
`on_train_batch_end` (fired every microbatch) to `on_before_zero_grad` (fires once
per optimizer step, after `optimizer.step()`). With `accumulate_grad_batches=16`
the old code ran the ~600-tensor EMA loop 16× per actual weight change (15/16 on
unchanged weights) — 8000×/epoch → 500×/epoch now. Resume-safe: EMA buffers load
unchanged; `num_updates` keeps counting, decay long since pinned at 0.9999 so no
warmup re-trigger. Side effect: EMA decay now tracks optimizer steps, not
microbatches (arguably more correct).

**3. GPU perf flags — `train_genforecast_rust.py`.** Set `torch.backends.cudnn.benchmark
= True` and `torch.set_float32_matmul_precision("high")` at import. Training runs
millions of fixed-shape steps so cuDNN's one-time algo search amortizes (first few
steps slightly slower); TF32 helps the fp32 AFNO spectral regions. Note: the
opposite of `predict_rust.py`, which leaves these OFF because single-shot inference
can't amortize the search (documented +21 % slower there).

**Measured speedup (changes 2 + 3 combined):** training throughput went
**6.63 → 7.96 it/s (~+20 %)** on the RTX 5080 at 128² batch 4. ~83 % of the prior
wall-clock per epoch → proportionally lower electricity cost per epoch (the reason
the prior run was stopped). Baseline of 7.96 it/s now on record for measuring the
next lever (`torch.compile`, potentially another 1.3–1.8×).

**Also earlier today, from the eval-tooling discussion:**
- bin10 case-selection fix (`monitor._fixed_cases`): scan `scan_per_bin=32`
  candidates/bin, pick top `per_bin_cases` by `frac_gt_1mm` on the cropped future —
  drops radar-clutter cases (single extreme pixel) that scored 0.0 CSI and dragged
  the overall down. **Means version_3+ numbers are NOT comparable to version_2**
  (different cases scored); fresh baseline.
- Per-case PNGs now render ALL T lead times, not the 4-step sample.

### NOT applied (await go-ahead)
- **Drop the EMA validation pass** (`diffusion.py:validation_step` runs `shared_step`
  twice + 150 full-model param copies/epoch for `val_loss_ema`, which nothing uses
  for decisions — checkpoint monitors `step`, early-stop disabled). Removes a metric
  `/analyze` references; judgment call.
- **`torch.compile(self.model)`** on the UNet — biggest potential win (1.3–1.8×) but
  needs a measured A/B and the EMA-name-prefix handling `predict_rust.py` flagged.

### Open decision after this run
If per-lead-time POD confirms near-term skill but the pooled metric stays flat,
the lever is NOT more epochs — it's either the eval (bump `per_bin_cases` to see
slow gains) or the recipe (rain-weighted MSE, the pre-registered next experiment).
Decide once `val/pod_*_lead` has a few epochs.

## 2026-05-28 (cont.) — Goal clarified: LDM is the RIGHT architecture; build the probability product. Resuming training.

**User's goal, stated sharply (supersedes earlier guesses):**
- Primarily "will I get wet on a run" — **any rain, light rain matters**, POD-focused.
- Intensity accuracy is secondary ("if the amount is off it doesn't matter as much").
- **DOES want probability** — "a 25% chance of rain is still useful."
- **DOES want it to look nice on a map** — wants to visually read what the model predicts.
- Decision use case: "should I wait 30 minutes before going outside?"
- Classical nowcasting's heavy-rain emphasis is explicitly NOT the user's priority,
  though heavy rain is still nice-to-have.

**Architecture decision — REVERSED from the previous entry.** The prior entry
floated dropping the LDM for a deterministic AFNO nowcaster (the unused
`AFNONowcastNet` class) on the theory that the user didn't need uncertainty.
**That was wrong given the clarified goal.** The user wants exactly the two
things the LDM provides and a deterministic model cannot:
- **Probability** ("25% chance") — comes from the ensemble spread. The 4 members
  ARE the uncertainty estimate; collapsing them gives probability-of-precipitation.
- **Sharp, realistic maps** — deterministic MSE models produce blurry, weak,
  spread-out fields (the paper's stated failure mode, p.2: "blurring... weaker
  and more widespread with increasing lead time"). The LDM samples sharp fields.
  A blurry deterministic field could also drop below the 0.1 mm/h threshold at
  long lead and *hurt* POD.

So: **keep the LDM. Do NOT build the deterministic baseline.** The
diffusion-ensemble machinery the previous entry called "overkill" is precisely
what this goal needs.

**The product to build (no retraining needed — inference/product layer):**
1. **Probability-of-precipitation (PoP) map.** Run N members; per pixel × lead
   time, fraction of members with rain ≥ 0.1 mm/h → a single 0–100 % map.
   This is the "will I get wet" map. 4 members already gives 25 % granularity
   (matches the user's "25 %"); ~16 members for ~5–10 % steps (~25 s/forecast
   at 128² on the 5080 — the model emits all 8 lead times per sample, so a member
   is one full forecast).
2. **Location/time decision strip.** For a chosen point/route, plot rain
   probability across the 8 lead times (5→40 min) — the literal "wait 30 min?"
   readout.
3. **Reliability diagram** on the val set — does "25 %" mean 25 %? Paper proved
   *their* LDCast is calibrated (rank histograms, KL=0.001); OURS is at ~1/10
   compute so calibration is UNVERIFIED. Checkable, and post-hoc recalibration
   is cheap if the curve is off the diagonal. **Do not trust the % numbers until
   this is run.**
4. **Neighbourhood-tolerant POD/FAR (FSS-style).** "Will rain hit my running
   route" is a ~few-km question, not single-pixel. Also explains away the ugly
   bin01/02 POD≈0 result, which is a pixel-precision artifact (sparse truth:
   ~5 rainy px / 16 384), not a real failure for this goal.

**Decision: resume training during cheap-electricity windows.** User: "I'll
train some more while the electricity is cheap, that can't hurt." Correct — we're
undertrained (~1/10 the paper's samples; per-bin unique coverage 1.9–13 %), the
recent csi_0.1mm trend was still rising, stability is clean, and overfitting is
impossible at 3 M / 93 M draws. More compute is expected to help overall quality
(~85 % confidence) and the all-lead-times structure weakness (members disagree on
location at every lead, not just +1). Resume from `last.ckpt` with `resume: true`,
`stages: diffusion`. The new POD/FAR scalars + fixed bin10 case-selection +
full-T PNGs will populate from the next val epoch onward.

**Build order once back at the keyboard:** product items #1–2 first (turn what we
have into the map/strip the user actually wants), then #3 calibration check, then
#4 neighbourhood scoring. Architecture question is now settled; no deterministic
A/B needed.

## 2026-05-28 — POD/FAR eval on `last.ckpt`; model is usable for "any rain" already; loss-function options scoped

**Run state:** stopped at epoch 93 / step 47000 due to electricity cost (user
ran the GPU continuously for ~2 days). `models/genforecast_rust/last.ckpt`
preserved at 17:05. Three additional ckpts (epochs 91/92/93) saved by `save_top_k=3`.

**Total compute consumed (this run, diffusion stage only):**
- ~3.0 M sample-draws (94 × 8000 × 4)
- ≈ 80 V100-equivalent hours on a single RTX 5080
- Paper's stage 1 was 424 V100-h × 8 GPU + ~30–50 M samples → **we're at ~1/5 by wall-clock, ~1/10 by samples-seen**.
- Per-bin unique coverage: bin1 ≈ 1.9 %, bin8 ≈ 5.6 %, bin10 ≈ 13 % (of unique
  candidate crops). Confirms we're undertrained, *especially* in the bins the user
  actually cares about (1–7, "any rain"), which have N = 8–14 M each.

### What changed in code

**`ldcast/models/genforecast/monitor.py`:**

1. **POD / FAR scalars added** alongside CSI: `val/pod_{0.1,1.0,5.0}mm/{overall,binNN}`
   and `val/far_{0.1,1.0,5.0}mm/{overall,binNN}`. Same per-bin + overall structure
   as CSI; ~108 scalars/epoch (up from 36).
   - **Why POD/FAR are the right metrics for *this* user's goal:** CSI penalises
     misses and false alarms equally; user only cares about misses ("don't tell me
     it's clear if it's about to rain"). High FAR is acceptable as long as POD is
     high. The previous emphasis on CSI was the wrong metric for the actual use
     case ("will I get wet when I go outside").
2. **Per-case PNGs render ALL T lead times**, not the 4-step `max_leadtimes` sample.
   Figure width grows to ~30" for T=20 but is the actual full forecast. The TB
   summary image still uses the sampled-leadtime layout (TB doesn't display very
   wide images well; the disk PNGs are the offline-inspection path).
3. **bin10 case selection fixed.** `_fixed_cases` now scans `scan_per_bin` (=32)
   candidates per bin and picks the top `per_bin_cases` by `frac_gt_1mm` on the
   cropped future tensor. The previous "first N row indices per bin" was picking
   radar-clutter cases for bin10 — single extreme pixel surrounded by zeros, which
   scored 0.0 CSI across all epochs and dragged the overall down. Cost: ~352
   one-time cropped loads at first val (~30 s with warm FrameCache).

**`scripts/eval_pod_far.py` (new):** post-training one-off eval script. Loads a
ckpt, runs the monitor's `_log_forecast` against a mock trainer/logger, prints
a CSI/POD/FAR table per bin + overall and dumps PNGs. ~3 min runtime.
Independent of training; can run on any ckpt.

### Headline numbers — `last.ckpt`, DPM-Solver++ 20 steps, 88 samples

For "will I get wet" (0.1 mm/h threshold):

|              | POD | FAR |
|--------------|----:|----:|
| **OVERALL**  | **0.60** | **0.09** |
| bin06–08 (organised rain) | 0.72–0.75 | 0.01–0.03 |
| bin09–10 (heavy)          | 0.39–0.43 | 0.16–0.18 |
| bin01–02 (sparse light)   | ~0 | ~1 |

Plain English: **the model catches 60 % of rain events overall with only 9 %
false alarms.** It's near-paper-quality on organised moderate rain (POD ≈ 0.75,
FAR ≈ 0.02), undertrained on heavy rain (misses ~60 %), and looks terrible on
very-sparse light rain — but the bin01/02 disaster is *partly a case-selection
artifact*: those crops have ~5 rainy pixels out of 16 384, and pixel-level POD
on that-sparse ground truth is impossible without near-perfect motion modelling.

**Important:** pixel-level POD/FAR understates real utility for the user's goal.
Neighbourhood-tolerant metric (FSS, à la the paper) would tell a more honest
story for "did rain hit *near* me." Not implemented yet.

### Failure-mode reading from the per-case PNGs

At all lead times (not just +1), ensemble members produce plausible-looking rain
texture but **disagree about location** and **don't track truth's spatial structure**.
Pattern is consistent with "conditioning weakly used" — the AFNO cascade has
learned some bias toward the right region but isn't pinning structure.

Checked the code and the paper for explanations:
- **No CFG dropout in training** (`diffusion.py:170–174`) and no CFG amplification
  at sampling. So this isn't a CFG-misuse issue.
- **AFNONowcastNetCascade is trained from scratch jointly with the UNet** —
  matches the paper exactly (Section 4.2.4: "the forecaster and denoiser stacks
  were trained simultaneously"). My earlier guess about pretraining the cascade
  separately is contradicted by the paper; not the recipe difference.
- **`scripts/train_nowcaster.py` turns out to be a misnomer** — it only contains
  `setup_data()` (imported by `train_genforecast.py`); no training code.
  `AFNONowcastNet` (the standalone deterministic nowcaster class in
  `ldcast/models/nowcast/nowcast.py`) exists but is unused in the repo.
- **Most likely cause:** we're at ~1/10 the paper's compute. eps-MSE is a weak
  gradient for structure-learning and accumulates slowly. Paper got good
  conditioning quality at ~30–50 M samples; we're at 3 M.

### Loss-function options discussed (no code change yet)

User asked what loss changes might address the structure problem. Ranked options
by promise for *this user's goal* (any-rain detection, modest compute):

1. **Rain-weighted MSE** — `loss *= (1 + α · R_truth)` to make rain-pixel errors
   count more. ~20 lines in `p_losses`. Targets "model dumps rain in wrong places"
   directly.
2. **Classifier-free guidance (CFG) training** — drop conditioning 10–20 % of batches,
   amplify at sampling. Standard technique to make conditioning "louder."
3. **Min-SNR-γ noise-level reweighting** — cheap, marginal overall improvement.
4. **FSS-style multi-scale loss** — aligns training with the metric we actually care
   about; more speculative.

**Why the paper didn't use #1 even though it's promising:**
- Paper's primary contribution is calibrated uncertainty (rank-flat distribution,
  KL = 0.001 vs uniform). **Loss-reweighting distorts the learned distribution**
  and would break that calibration.
- Paper already does "rain emphasis" at the *sampling* level (LDCast
  EqualFrequencySampler oversamples heavy-rain crops ~10–100×). Sampling-level
  reweighting preserves the conditional distribution within each sample;
  loss-level reweighting doesn't.
- Plain ε-MSE has score-matching theoretical guarantees; weighted variants don't.
- They had 5–10× our compute; plain MSE worked.

**For this user the tradeoff inverts:** uncertainty calibration is not the goal;
high POD with limited compute is. Rain-weighted MSE is the right tradeoff to try
*for this goal*, but it's not "the paper missed a win" — it's the opposite tradeoff
from what they wanted.

**Standing question (flagged, not resolved):** if the user only cares about
deterministic "any rain" detection, the LDM architecture may be the wrong tool.
A deterministic AFNO nowcaster (the existing `AFNONowcastNet` class) trained on
plain MSE would likely give comparable POD at a fraction of the compute, since
the diffusion machinery exists *for* uncertainty quantification — which isn't
needed for this use case.

### Data check

`/opt/radar_data/index_ldcast_128.txt` — 297.5 M rows, span 2022-09 → 2026-03
(1280 days). **Full-year coverage** (all 12 months represented). Notable dip in
June (~1/3 the volume of other months); known data outage per user, no
investigation needed. Year-round coverage is the right call for "predict winter
rain too" — the opposite of the paper's Apr–Sep convective-season-only training.

### Decisions deferred / open

- **Whether to do more training, and how to pay for it.** Paper-equivalent
  compute (~10 days on this 5080, or ~$100–300 cloud) would close most of the
  gap. User stopped due to electricity cost; decision on next-run cadence is
  open.
- **Whether to implement rain-weighted MSE (option #1 above).** Cheap to try
  (~20 lines) and reversible (α=0 recovers original). Pre-registered as the
  journal's next experimental lever since 2026-05-25 but never executed.
- **Whether to add neighbourhood-tolerant POD/FAR (FSS-style).** Would more
  honestly score the model on the "did rain hit near me" goal that pixel-level
  POD understates.
- **Whether to A/B a deterministic AFNONowcastNet** as a sanity check on the
  "is LDM the right architecture for this user's goal" question.

## 2026-05-27 (late) — Per-bin stratified eval replaces the 4-case CSI monitor; PNG dump per case

**What:** Rewrote `ldcast/models/genforecast/monitor.py::SamplePredictionLogger`
to do a proper per-bin eval every val epoch instead of the 4-cases × 1-member
× PLMS-50 scoring it had since 2026-05-24. New defaults baked into the monitor:

- **2 cases per LDCast intensity bin × 11 bins × 4 ensemble members** = 88 samples
- **DPM-Solver++ with 20 iters** (validated corr ≥0.98 vs PLMS-50, 2026-05-20)
- Case selection: read `dm.val_w`, group by unique weight (= bin), take first
  `per_bin_cases` row indices per bin, load via `valid_ds[int(ridx)]`. Bypasses
  the val_dataloader so the bin coverage is exact rather than depending on
  whatever batches happen to surface during a 32-batch scan.
- TB scalars logged per bin AND overall: `val/csi_{0.1,1.0,5.0}mm/{overall,
  bin00..bin10}` (3 × 12 = 36), plus `val/mae_mmhr/{overall,bin00..10}` and
  `val/rmse_mmhr/{overall,bin00..10}`. The old flat-named `val/csi_*mm` tags
  are gone (intentional; the meaning changed — 88 samples vs 4 — and a TB
  discontinuity is more honest than a misleading continuation).
- **PNG dump per case** to `<log_dir>/eval_pngs/step_<NNNNNN>/binNN_caseM.png`,
  5 rows × N lead times (truth + 4 ensemble members). Stays around offline so
  the user can scrub through ensembles per epoch. Disk cost ~5 MB/epoch, ~250 MB
  over 50 epochs — trivial vs the 6.7 GB ckpts.
- Per-epoch cost ~2 min on top of ~22 min training → ~10 % overhead.

**Why:** The previous 4×1 eval was noise — csi_0.1mm swung 0.10 → 0.96 between
consecutive epochs at the same global step, and I was reading "trends" off
that. The journal's 2026-05-25 (eve) entry pre-registered "Trustworthy
measurement first" as plan step 1, and the 2026-05-27 (eve) entry explicitly
flagged the 4-case eval as the limiting factor — never acted on. User called
this out directly ("I can't give you an honest answer when I don't have any
data"). The new shape:

- gets per-bin signal so heavy-rain skill is judgeable in isolation from
  light/moderate (the previous overall CSI conflated regimes — the per-bin
  diagnosis is what's needed for the "magnitude is the new bottleneck"
  hypothesis from this morning's entry);
- 22× more cases at 4× more members reduces single-epoch CSI variance
  meaningfully;
- per-case PNGs preserve the artifact path that caught the val-ordering bug
  this morning (visual inspection found what scalars missed; keeping that
  channel open for the magnitude diagnosis too).

**Verified:** Pre-restart smoke test passed -- all 11 bins represented;
bin8-10 truth has 1-13 % pixels > 1mm/h and 1-1.5 % > 5mm/h (csi_5mm finally
has real denominators). Restart resumed from `last.ckpt` (epoch 35) cleanly;
training writes to `models/genforecast_rust/tb/version_2/`. First val with new
metrics due in ~22 min.

**Outcome:** PENDING — the actual hypothesis test (does the per-bin csi_5mm
at bins 8-10 lift off zero across epochs?) needs ~5-10 epochs of the new
metric to be judgeable. NO claims about plateau/progress until those numbers
exist; the principle the user enforced was "no conclusions from noisy data."

## 2026-05-27 (eve) — Dry-out is CLOSED. Coverage restored; magnitude is the new bottleneck.

**Status after the post-fix run (15 val epochs since restart, global epoch 28, healthy):**

| metric             | prior uniform ceiling | post-fix best (v1) | latest |
|--------------------|-----------------------|--------------------|--------|
| `val/csi_0.1mm`    | ~0.25-0.45            | **0.961** (~2×) ✓  | 0.828  |
| `val/csi_1.0mm`    | 0.174                 | **0.309** (~1.8×) ✓| 0.241  |
| `val/csi_5.0mm`    | 0.024                 | 0.018 (~0.75×)     | 0.010  |

**Qualitative confirmation (user reviewed `val/forecast` images, post-fix v1 ep0/8/14):**
The cached val case is rain-covered everywhere in the truth row. The pre-fix predictions
were almost-empty (the dry-out attractor the journal documented). Post-fix predictions
**have rain across the whole canvas**, with the latest no longer suffering from the
"almost empty" failure mode. Visually, the model is clearly forecasting rain — a
qualitative regime change vs the pre-fix run.

**Reframing the CSI numbers given a rain-covered truth:**
- `csi_0.1mm = 0.83-0.96` means **83-96% of pixels correctly cross the light-rain threshold** —
  not "found a rain blob in the right place" but "covering the canvas with rain, like the
  truth." The dry-out is fixed.
- `csi_1.0mm = 0.24` means only 24% of pixels cross 1 mm/h. The predicted rain is too soft
  on average.
- `csi_5.0mm = 0.01` means almost nothing in the prediction reaches heavy-rain intensity.
  Peaks are smoothed/clipped.

**So: coverage and structure are restored; the bottleneck moved to magnitude.** This is
exactly the failure mode the journal's 2026-05-25 (eve) entry hedged on:
> "At 2.36% exposure and still failing, emphasis *might* help — or the binding constraint
> is elsewhere (autoenc latent rep of heavy rain / conditioning / eps-prediction at high
> intensity). **Genuinely uncertain.**"

Now resolved: the LDCast bin-equal-frequency sampler **did** restore the coverage/structure
the prior uniform run lost (csi_0.1 ~2× ceiling; csi_1.0 ~1.8× ceiling), so the sampler
hypothesis is confirmed for light/moderate rain. csi_5mm is at ~75% of the prior ceiling and
plateauing (9 epochs since best as of analyzer run), which leans the heavy-rain bottleneck
toward the *other* candidates the journal listed:
- **Autoencoder's latent rep at high intensity.** Even though val_rec_loss=0.006 looks
  pristine, that's dominated by easy crops. The autoencoder might be clipping/smoothing
  rain peaks before the diffusion model ever sees them; if so, csi_5mm is bounded above
  by the AE regardless of how good diffusion gets. Cheapest test: run encode+decode on
  bin-9/10 cases specifically and measure pixel-wise error at the rain core.
- **Eps-prediction breaking down in high-magnitude regions.** Even a perfect AE doesn't
  help if the UNet's eps-MSE loss systematically under-weights large-magnitude latents.
  Fix: rain-weighted diffusion loss (DGMR's `w(y) = max(y+1, 24)` is the prior art lever
  the journal already flagged as deferred-until-uniform-sampling-confirmed). This is the
  natural next experiment.
- **scale_factor of the latents.** Tested in the 2026-05-21 entry on the *old* autoencoder
  and found counterproductive there. Worth re-testing with the new AE, which has very
  different latent statistics.

**Eval signal is still 4-case-noisy.** csi_0.1 swung 0.10 → 0.96 over two consecutive
epochs at the same global step. The peaks and the band tell the story; single epochs
don't. Increasing `num_eval_cases` from 4 → 16 or 32 in `SamplePredictionLogger` would
give a cleaner trend at modest val-cost; worth doing before the next experiment.

**Outcome:** the dry-out attractor we've been chasing for ~10 days is **closed**. Diffusion
is now training in a regime where progress is measurable and meaningful, and the next
bottleneck is named (magnitude, specifically peaks at heavy-rain thresholds). The combined
fix — LDCast indexer + compact NumPy loader + bin-0 dedup + LDCastEqualFrequencySampler +
val-ordering bug fix — is what unlocked it. Run remains active, no need to stop yet.

## 2026-05-27 — val ordering bug: CSI/forecast preview were scoring against an empty single-snapshot val

**What:** Removed `np.sort(...)` from `val_idx` in `ldcast/features/rust_data.py::_load_ldcast_index`
and bumped the genforecast monitor's `scan_batches` 32 -> 64.

**Why (the bug):** `_load_ldcast_index` was building val_idx via
`val_idx = np.sort(perm[:n_valid])` -- the sort was inherited from the
PyO3-entries-list era as a cache-locality optimisation. With the LDCast indexer,
the index emits *all spatial positions per timestamp consecutively*, so sorted
val_idx clusters val into a tiny number of adjacent timestamps. Measured: the
first 200 val rows (= val_dataloader's first 50 batches × batch 4 = the entire
limit_val_batches window) were **all from a single 10-min snapshot at
2022-09-11 12:30 UTC**, 78% bin 0, zero rows in bins 7-10. Effects:
- `val_loss_ema` was scored on ~empty crops -> artificially low (the model
  "wins" by predicting near-zero); we saw it drop to 0.018 vs prior ceiling
  ~0.086, looking too good to be true (it was).
- The genforecast monitor's `_fixed_cases` scans val_dataloader for the wettest
  K, but the entire scannable window was that one quiet snapshot. The cached
  "wettest 4" had **0% of pixels above 1.0 mm/h**; CSI@1mm and CSI@5mm were
  numerically forced to 0 not because the model was bad, but because the truth
  had nothing to score against. The `val/forecast` preview's truth row was
  nearly blank for the same reason -- visually noticed by user, then traced.

The actual model was being trained correctly on the LDCast bin-equal-frequency
distribution (the train_idx sort was harmless; LDCastEqualFrequencySampler picks
indices randomly). So 13 epochs of model state were preserved; only the val
metrics were uninterpretable.

**Fix:** `val_idx = perm[:n_valid]` (no sort) -> val_dataloader iterates val
samples in random order, so the first 50 batches are representative across the
whole index. Verified post-fix: first 200 val rows now span different days
(2024-09-29, 2023-08-06, 2023-08-09, 2024-09-20, 2024-07-26 in the first 5),
70% bin 0 (matching overall ~65%), 1 sample in bin 9. Cache locality for val
is sacrificed, but val is bounded by limit_val_batches=50 and the data already
sat in OS page cache from training -- val cost stays small.

Also bumped `SamplePredictionLogger._fixed_cases(scan_batches=32 -> 64)` so the
selection draws from 256 candidates (covers the full ~200-sample val set with
margin); top-4 wettest are reliably in bins 8-10 once val is properly shuffled.

**Outcome:** PENDING. Stopped the running diffusion run at epoch 13 step 7000
(last.ckpt preserved, written 12:11), restarted with resume=true to pick up
both fixes. The genforecast monitor recaches its eval cases on first val
(only persists in the live process), so a new set of eval cases will be
selected from the now-representative val. Expect val_loss_ema to JUMP UP at
first val after restart (no longer artificially low) and CSI to take a different,
meaningful scale -- give ~2-3 val epochs after restart before drawing conclusions.

Train_idx sort kept: harmless (the LDCast sampler picks randomly), and a sorted
train index is friendlier to anything that linearly scans train_ts/x/y.

## 2026-05-26 — LDCast bin-equal-frequency sampling, wired in via the new index file

**What:** Replaced the old `index_128.txt` / uniform-sampling path with the LDCast
indexer's output (`/opt/radar_data/index_ldcast_128.txt`, 12 GB / 297 M rows;
generator at `batch_hdf5_to_img/src/ld_indexer.rs`). The 4th column of the new
file is LDCast's bin-equal-frequency importance weight
`N_total / (num_nonempty_bins * N_in_bin)` over the same intensity bins as the
original `EqualFrequencySampler` (`np.exp(np.linspace(np.log(0.2), np.log(50),
10))` on per-crop 99th-percentile rain rate, 11 bins after bisect_left).
Sampling proportional to weight gives uniform mass per bin — the rust-pipeline
equivalent of the original LDCast sampler — and reintroduces the heavy-rain
oversampling (~10-100× vs natural) that the previous uniform pass was missing.

Four pieces:

1. **`dgmr-py`** — two new `#[pyfunction]`s (`parse_ldcast_index`, `make_entry`).
   `parse_ldcast_index` stream-parses the index file and returns four NumPy
   arrays (`ts:i64` Unix seconds, `x:u16`, `y:u16`, `w:f32`) — ~4.7 GB for 297M
   rows vs the ~36 GB a `Vec<PyIndexEntry>` would cost. Parse time ~28 s on
   the 12 GB file. `make_entry` is a tiny constructor used per-`__getitem__`
   to build the PyO3 `IndexEntry` `dgmr_py.load_sample` still wants.

2. **`ldcast/features/rust_data.py`** — `RustRadarDataModule` and
   `RustRadarDataset` now carry the compact NumPy arrays (no Python list of
   PyO3 objects). New helper `_load_ldcast_index` parses → splits train/val →
   **deduplicates bin 0 in TRAIN to a single representative row**, scaling its
   weight by the bin's original population so the LDCast invariant
   (`sum(weight)` per bin is constant) holds exactly. Bin 0 is ~65 % of the
   raw file (~193 M of 297 M rows); bin-0 crops are functionally identical for
   training (no rain anywhere in the 12-frame 128² window), so collapsing them
   to one representative loses no information. Train shrank from 268 M →
   93.6 M rows. Val left as a uniform sample of the raw index (val_loss_ema
   just needs a representative slice, not the LDCast sampling distribution).

3. **New `LDCastEqualFrequencySampler`** — torch's `WeightedRandomSampler`
   calls `torch.multinomial`, which is capped at 2^24 ≈ 16.7 M categories, and
   the 93.6 M-row train array trips this. Wrote a custom sampler that draws
   uniformly across bin index and uniformly within bin (vectorised, ~0.01 s
   for 200 k draws) — distributionally equivalent and not multinomial-bound.
   Empirically: 200 k draws → 8.97-9.21 % per bin (target 9.09 % = 1/11);
   deduped bin 0 hits ~9.06 % via repeated indexing of the same row.

4. **Config + orchestrator** — `config/train_rust.yaml` switched
   `index_path` to the LDCast file, added `use_weighted_sampler: true`, set
   `stages: both` and `resume: false` to retrain both stages on the new
   distribution (the autoencoder reconstructs whatever it's shown, so its
   heavy-rain representation is the ceiling for the diffusion stage).
   `scripts/train_rust.py` threads `use_weighted_sampler` to both stage
   subprocesses. The stage scripts already accepted the flag.

**Verified:** parser returns 297,453,114 rows in 28 s, 11 unique weights,
`weight × count = 2.704e+7` constant across all bins (LDCast invariant from the
indexer). Dedup keeps the invariant exactly in train: per-bin `sum(weight)` =
2.434e+7 across all 11 train bins, including the single-row deduped bin 0.
End-to-end smoke (`stages=diffusion --limit_train_batches=20 --max_epochs=1`):
training step ~5.7 it/s at 128² batch 4, no NaN, GPU 15 GB, val loop completed.

**Outcome:** PENDING. Needs the real `stages=both --resume=false` run. Stage 1
retrains the autoencoder on the new distribution (~hours). Stage 2 is the
actual test of the journal's sampler hypothesis — pre-registered prediction:
bin-equal-frequency must beat the previous uniform pass on held-out
`val/csi_5mm`. If it doesn't, the sampler hypothesis is dead and we look
elsewhere (autoenc latent rep at high intensity, conditioning, eps-prediction
at high intensity).

**Old `index_128.txt` retired** — drop references to DGMR `q_n^-1` weights
when discussing the index/sampler. The `parse_index` function in dgmr-py is
left intact (other call sites still use it; `predict_rust.py` parses the index
to look up a timestamp); only the *training* path uses `parse_ldcast_index`.

## 2026-05-25 (eve) — Heavy-rain plateau: traced to the SAMPLER, not the loss. Falsifiable plan for tomorrow.

**State of the fresh run** (`models/genforecast_rust/tb/version_0`, the sampler-flip + coverage-filter
run; old run archived at `genforecast_rust_old/`): ~28 epochs, ACTIVE, no NaN. Dry-out is gone and the
model produces rain, but skill has **plateaued with a clear threshold profile**:
- `csi_0.1mm` (light): OK, ~0.25–0.45.
- `csi_1.0mm` (moderate): flat ~0.07; best 0.174 @ ep8, never beaten.
- `csi_5.0mm` (heavy): flat ~0.01; best 0.024 @ ep16, never beaten.
- `val_loss_ema`: 0.40 → 0.086 (best @ ep24) then flat. `train_loss` ~0.12 and flat.
- Early over/under-forecast instability (RMSE spiked to 324 mm/h ep0–9) **resolved by ~ep10**; RMSE now ~1.8.

**The finding — the difference from original LDCast is sampling, not loss.** The diffusion loss is
*identical* in both paths (`ldcast/models/diffusion/diffusion.py`, plain unweighted eps-MSE) — so the
original needed **no** rain-weighted loss. What original LDCast has that this rust pipeline dropped:
- **Original** (`scripts/train_nowcaster.py:122` `bins = np.exp(np.linspace(np.log(0.2), np.log(50), 10))`
  → `split.DataModule(sampling_bins=…)` → `batch.StreamBatchDataset` (`:81`) → `sampling.EqualFrequencySampler`):
  bins patches by 99th-pct intensity into ~11 bins and draws **each bin with equal probability**
  (`sampling.py`: `self.rng.randint(self.num_bins, …)`) → heavy rain oversampled ~10–100× and crop-centered.
  Confirmed the original genforecast stage uses this (`scripts/train_genforecast.py:10,120` import `setup_data`).
- **Rust** (`ldcast/features/rust_data.py`): **uniform** over the rejection-sampled (wet-enriched) index
  after we set `use_weighted_sampler=False`. No intensity stratification.
So the original gets heavy-rain skill from the *sampler*; we lost it. Uniform was still the right fix vs the
inverted `q_n^-1` (dry-oversampling) bug — but it's far weaker than LDCast's bin sampler.

**"False epoch" nuance (user caught this):** `limit_train_batches=8000` × batch 4 = 32k crops/epoch ≈
**1.4%** of the 2.3M index. So 28 epochs ≈ **0.39 true passes**, ~32% distinct coverage — still in the first
sweep. BUT the heavy-rain *fraction per batch* is unchanged by epochs/coverage; heavy CSI stayed flat while
coverage grew 0→32%, i.e. **fraction-bound, not coverage-bound**. More training / more data is a weak,
expensive lever vs fixing the sampling fraction.

**Heavy-rain prevalence check** (120 random crops, future frames): ≥0.1mm **43.9%**, ≥1mm **16.2%**,
≥5mm **2.36%**, ≥10mm **0.62%** of pixels; 69% of crops contain ≥5mm pixels (57% have ≥100). Two takeaways:
(a) `csi_5mm` is a **real** signal, not an empty-denominator artifact — the weakness is real; (b) heavy rain
is **not starved** (2.36%), so my "model barely sees heavy rain" framing was overstated. At 2.36% exposure
and still failing, emphasis *might* help — or the binding constraint is elsewhere (autoenc latent rep of
heavy rain / conditioning / eps-prediction at high intensity). **Genuinely uncertain.**

**Calibration note (important):** this run is the 3rd compounding fix (inverted sampler → coverage filter →
now sampling fraction); each fix surfaced the next bottleneck. The sampler is the **best-supported
hypothesis** (code-confirmed difference from original LDCast) but is **not proven** and not necessarily the
last issue. Test before investing — don't treat the next fix as the answer.

**Index generator is NOT on this machine** (separate repo; needs the raw radar archive which isn't here).
Per-crop intensity for bin-sampling should ultimately be emitted by that generator (it already scans every
crop to compute the `q_n^-1` weight column). For a *local* test we can compute intensity for a subset here
(`dgmr_py` + `/opt/radar_data` are present).

**PLAN FOR TOMORROW (falsifiable, in order):**
1. **Trustworthy measurement first.** The `val/csi_*` monitor is 4 cases / 1 seed — too thin for the calls
   I've been making. Score the current checkpoint on a larger held-out set with several ensemble members to
   get real CSI-by-threshold. It may be less broken than the monitor suggests.
2. **Pre-registered A/B for the sampler hypothesis** (this machine, no index-regen): compute per-crop
   intensity for a subset, cache it; two short runs on the **same crop pool**, identical except sampling —
   uniform vs intensity-bin equal-frequency (reuse the `use_weighted_sampler` plumbing in `rust_data.py`
   with *correct* weights = 1/bin_population). **Pre-registered outcome: binned must beat uniform on
   held-out `csi_5mm`, else the sampler hypothesis is dead and we look elsewhere.**
3. **Only if (2) confirms:** emit per-crop intensity from the index generator (other machine) and wire an
   `EqualFrequencySampler`-equivalent into `rust_data.py` for the full run.

**Checkpoints:** `genforecast_save_top_k=3` keeps the last 3 by recency (+`last.ckpt`); the best epochs
(ep8/16/24) are already pruned. For the A/B, set `genforecast_save_top_k=-1` (keep all) so the best can be
picked offline.

## 2026-05-25 — Restored multi-checkpoint saving for the diffusion stage (disk no longer constrained)

**What:** The diffusion `ModelCheckpoint` (`ldcast/models/genforecast/training.py`) now keeps the N
most-recent checkpoints + `last.ckpt` instead of a single rolling ckpt. New config knob
`genforecast_save_top_k` (default 3) threaded config → `train_rust.py` → `train_genforecast_rust.py`
→ `setup_model` → `setup_genforecast_training` (mirrors `genforecast_accumulate_grad_batches`).
- ModelCheckpoint is now `monitor="step", mode="max", save_top_k=save_top_k, save_last=True,
  filename="{epoch}-{step}"` (was `save_top_k=1, monitor=None, save_last=False`).
- Updated the now-stale `_resume_ckpt` docstring (diffusion now writes a `last.ckpt`).

**Why:** disk is no longer constrained. The old config kept only the latest epoch (the best could be
lost — the `/analyze` run flagged this) and wrote no `last.ckpt`. "Adding EMA" needs no code: `LitEma`
is a submodule of `LatentDiffusion`, so EMA shadow weights are already serialized in every full
`.ckpt` (verified below) and inference applies them (`forecast.py` `model_ema.copy_to`).

**Design choices:** recency-based retention, NOT a quality monitor — `val_loss_ema` (eps-MSE) doesn't
track sample quality, and `val/csi_*` is `add_scalar`'d (not `self.log`'d) so isn't monitorable. Pick
the best checkpoint offline (eval_genforecast / metrics / `/analyze`). Lightning 2.6.1 detail:
`monitor=None` + `save_top_k>1` raises; `_monitor_candidates` always injects `step`/`epoch`, so
`monitor="step", mode="max"` keeps the most-recent k. Autoenc stage left unchanged (already
`save_top_k=3` + `last.ckpt`; ckpts ~5 MB; no EMA).

**Verified:** (a) standalone tiny Lightning fit kept exactly the 2 most-recent epoch ckpts +
`last.ckpt`; (b) end-to-end diffusion smoke run (throwaway dir, `save_top_k=2`, 3 epochs) kept
`epoch=1-step=12` + `epoch=2-step=18` (pruned `epoch=0-step=6`) + `last.ckpt`, each 6.3 GB, each with
`model_ema.*` (252 keys) present; `--save_top_k=2` was correctly forwarded through the subprocess.

**Footgun found (pre-existing, NOT fixed):** `fire` parses lowercase `--resume=false` as the string
`"false"` → **truthy** → resume stays ON (capital `--resume=False` works; YAML `resume: false` works).
So to start a FRESH run, set `resume: false` in `config/train_rust.yaml` (or clear the old ckpts) —
do NOT rely on `--resume=false` on the CLI. Critical right now: the fresh diffusion run (for the
sampler + coverage fixes) must NOT resume the epoch-52 weights, which were trained on the pre-fix
inverted distribution.

## 2026-05-25 — ROOT CAUSE of diffusion dry-out: training sampler oversampled DRY crops ~478x

**What:** Flipped `use_weighted_sampler` default `True -> False` in `ldcast/features/rust_data.py`
and both rust train scripts (`train_genforecast_rust.py`, `train_autoenc_rust.py`). Diffusion/autoenc
now train with **uniform sampling** over the index. Documented the reason in `rust_data.py._loader`
so it can't silently regress.

**Why (the bug):** `RustRadarDataModule._loader` fed `entry.weight` into a `WeightedRandomSampler`.
But `entry.weight` (4th column of `index_*.txt`, exposed by dgmr-py `parse_index`) is DGMR's
*evaluation-time* importance weight `q_n^-1` (Ravuri et al. 2021, Supp. A.1) — LARGE for dry crops,
small for rainy ones. It exists only to UNBIAS eval metrics; DGMR's own rust trainer ignores it
(`dgmr-rs/src/data.rs:54` "weight ignored") and trains uniformly over the already rejection-sampled
(rain-enriched) index. Feeding `q_n^-1` to the sampler oversampled the DRY tail, so the diffusion
model learned to forecast ~nothing.

**Evidence (measured on `/opt/radar_data/index_128.txt`, 2.3M crops):**
- weight column: min 2,575, max 1,280,000 (capped), mean 127,922 — not a probability.
- HIGH-weight crops (~1.28e6, what the sampler favored): mean **0.00 mm/hr, 0% wet**, 15% no-coverage.
- LOW-weight crops (~2.7e3): mean **6.89 mm/hr, 98% wet**.
- => oversampled dry vs rainy by ~1.28e6/2.7e3 ≈ **478x**.
- UNIFORM draw (what `False` now gives): mean **0.47 mm/hr, 49% wet** (median 51%) — the index is
  already rain-enriched, so uniform is the correct, DGMR-faithful distribution.

This is the long-running "diffusion dries out / worse than persistence / flat CSI / val_loss_ema
plateau" finding (2026-05-23, 2026-05-24). It was NOT undertraining, optimizer, LR, scale_factor or
effective-batch (all previously tried and inert) — the training distribution was inverted. The
autoencoder reconstructs whatever it is shown so it never flagged this, but it too was training on the
dry-skewed mix; uniform is better for it as well.

**Outcome:** PENDING. Needs a FRESH diffusion run (resume from scratch — the epoch-52 weights were
trained on the wrong distribution). Watch `val/csi_*` (esp. `csi_1.0mm`); they should rise above the
flat ~0.02-0.06 band for the first time. If `csi_5.0mm` still lags once the model is clearly producing
rain, add a rain-weighted diffusion loss (DGMR uses `w(y)=max(y+1,24)`) — deferred until uniform
sampling is confirmed.

**Related — no-coverage filter added (same investigation):** off-radar pixels (sentinel `-1/32` mm/hr)
were mapped to `0.02` mm/hr by `mmhr_rainrate_transform` (indistinguishable from real dry) and never
masked from the latent diffusion loss. Mirrored the original LDCast exclusion (`patches.py`:
`np.isfinite(patch).all()`) with a tolerance for DMI's circular coverage: `RustRadarDataset.__getitem__`
now skips a crop (reusing the archive-gap retry loop) when its off-radar fraction exceeds
`max_nocoverage_frac`. Threaded through `RustRadarDataModule` + both rust train scripts; default
**0.05** (keep crops ≥95% within coverage), `1.0` disables it, and it is auto-skipped in `full_frame`
mode (padding is no-coverage by design). Measured per-crop no-coverage over 150 uniform crops: 92%
fully covered, median 0%, thin tail (1.3% are 50-100% off-radar); 0.05 drops ~6.7% and removes the
whole bad tail. Smoke-tested: rejects an all-off-radar set, passes covered crops (correct shapes),
bypassable at 1.0. Latent-space loss masking (harder, lower value) intentionally NOT done.

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
