# LDCast Retrain Plan

Working plan for a from-scratch retrain that bakes in the findings from the
2026-05-31 investigation. **Living document — edit as decisions land.** The
underlying evidence is in `journal.md` (2026-05-31 entries); the eval tooling is
`scripts/eval_autoenc_ceiling.py` and `scripts/eval_val_large.py`.

## Objectives
- Rain or no rain is the most important objective.
- Getting the amount of rain correct is a nice to have feature.
- The prediction should still look like rain.
- Capturing whether rain is growing out of nowhere, or collapsing into nothing is important.
- It has to be better than Lagrangian persistence.
- Main use case is determining what is the probablility it will rain or not in the next 30-40 minutes at a given location or path.


---

## Why retrain at all

The current model is **recipe-limited, not data-limited**: flat over ~40 clean
(EMA, 24-member) epochs at only ~8% of one pass through the data, and the
autoencoder ceiling test proved the latent space can represent heavy rain
(reconstruction csi_5 ≈ 0.9). So the bottleneck is the diffusion model's
*generation* — plain ε-MSE under-produces intensity — not the data or the codec.

A from-scratch run is the right vehicle for the three things you **can't** get by
continuing the current checkpoint:
- a **clean test number** (the current model trained on a leaky random split that
  overlaps every day, so it has no honest held-out data);
- recipe changes **learned from the start** rather than nudging a converged model;
- **new inputs** (temperature) that require retraining anyway.

---

## Honest scope — what a retrain can and can't do

The model already serves the **primary goal** well: short-range, year-round
light-rain "will I get wet," with probability. (POD dialable to ~0.95; useful to
~+40–60 min; the PM-mean gives a clean deterministic map.)

A retrain realistically **buys**:
- ✅ trustworthy generalization numbers + a real held-out test set;
- ✅ winter/snow capability (temperature);
- ✅ calibrated probabilities (recalibration, or better-conditioned training);
- ◻️ *maybe* modest moderate-rain gains.

It will **not**:
- ❌ make heavy rain good — heavy rain is weak at every metric (point csi_5 =
  0.014, FSS ≈ 0.06 at 51 px, gone after +10 min), capped by predictability and
  our ~1/10 compute, not by anything a retrain fixes.

**Scope the run to what you actually want, not to "fix everything."**

---

## Findings that ground the plan (2026-05-31, large held-out eval: 660 cases × 32 members)

| finding | evidence | implication |
|---|---|---|
| recipe-limited, not data-limited | flat 40 clean epochs at 8% of one pass | lever is the loss, not more data/epochs |
| AE is **not** the bottleneck | recon csi_5 ≈ 0.9 vs model 0.014 | rain-weighted loss has real headroom; don't retrain the AE |
| diffusion under-produces intensity | 66 / 10 / 1 % of ceiling at 0.1 / 1 / 5 mm | rain-weighted loss targets this directly |
| heavy rain weak **everywhere** | csi_5 0.014 [0.011–0.018]; FSS 0.06 @51 px | don't expect a retrain to fix it |
| no usable regime | per-case dist a single spike at ~0 (345/660); p95 0.23; 2 % >0.3 | no "ship the good subset" option |
| season = climatology | light identical warm=cold (0.50); moderate 0.148 vs 0.092 | all-year is fine; heavy rain is intrinsically summer |
| PoP miscalibrated | forecast "20 %" verifies at ~50–70 % | recalibrate (cheap, no retrain) or condition better |
| serves primary goal | POD→0.95; useful +40–60 min | retrain is for extension/trust, not rescue |

---

## The plan, by objective

### Tier 0 — Trustworthy measurement (do it; low cost)
- **Train on the temporal split** — already built and verified (`split_mode: temporal`,
  `test_frac: 0.1` in `config/train_rust.yaml`; whole-UTC-day holdout; confirmed no
  day leaks across train/val/test). Gives a clean val *during* training and a test
  set evaluated **once** at the end.
- **Keep the current autoencoder.** Ceiling test proved it's excellent (recon
  csi_5 ≈ 0.9); it's radar-only so temperature doesn't touch it; its split-leakage
  is negligible for the *forecast* question. Retraining only the diffusion stage
  saves a whole stage-1 run. (Retrain the AE on temporal only for maximal purity —
  low value.)
- **Fold FSS + reliability into the per-epoch monitor** (logic exists in
  `eval_val_large.py`) so they're tracked live.

### Tier 1 — Recipe changes for skill (VALIDATE CHEAP FIRST, then bake in)
Prove these on short **continue**-runs against the current best ckpt *before*
committing days of from-scratch compute. Both trade against the primary goal —
guardrail: watch light-rain POD (`csi_0.1` / `pod_0.1mm`).
- **Rain-weighted diffusion loss** — `loss *= (1 + α·R_truth)` on a downsampled,
  max-pooled rain map applied in `p_losses` (latent space; renormalised; α=0 =
  exact baseline, reversible). De-risked by the AE ceiling test. Targets the
  under-production directly. Risks: distorts calibration (recalibrate after); can
  hurt light rain if α too high.
- **CFG (classifier-free guidance)** — drop conditioning 10–20 % in training,
  amplify at sampling. Targets the *misplacement/discrimination* problem (POD and
  FAR both poor; members disagree on location) that recalibration can't touch. Also
  the cheaper fix for "conditioning under-used," which matters if we lean on
  temperature conditioning.

### Tier 2 — New capability: temperature conditioning (gated on DATA)
- **Architecturally already supported** — the genforecast `use_nwp` path feeds the
  analysis cascade / context encoder. Current training uses `use_nwp=False`.
- **The gate is historical temperature data** (the reason this was shelved). Likely
  unblock: **ERA5 reanalysis** (free, hourly 2 m temperature, ~0.25°, covers
  2022–2026 via Copernicus CDS), regridded/aligned to the radar crops.
  **Decision needed: ERA5, a DMI NWP/station archive, or skip.**
- **Buys:** rain/snow discrimination (on-goal — snow is still "you'll get
  wet/cold") and possibly regime conditioning (convective vs frontal). **Does not**
  fix heavy-rain under-production.
- Biggest new-work item (data pipeline + transforms + retrain). Only worth it if
  winter/snow is a real objective.

### Considered and ranked lower: season-specific models
Brainstorm: train separate per-season models, each on one season's data.
- **Strongest form:** a *summer-only* model (a winter specialist adds ~nothing —
  winter's job is light rain, already maxed and season-independent).
- **Merits:** focuses capacity on the convective regime; matches the paper's
  Apr–Sep setup; heavy bins well-fed.
- **Why below the loss change:** our diagnosis is *recipe*-limited, but
  season-splitting attacks *capacity/heterogeneity* — the wrong joint; the same
  timid loss under-produces in a summer-only model too. The warm-vs-all-year gap is
  small (csi_1 0.148 vs 0.128), so the upside is modest while the cost is real
  (halves data/compute per model, two-model ops + a season router, shoulder-season
  boundaries). It's *additive* to the loss fix, not a substitute.
- **Largely dominated by temperature conditioning** (Tier 2): soft regime-awareness
  in one model, all the data, continuous shoulder-season handling. Hard-splitting's
  only edge is *guaranteeing* specialization our weak conditioning might not learn —
  which CFG addresses more cheaply.
- **To settle it empirically:** train one summer-only model, compare to all-year on
  warm-season *test* cases. One bounded run — recommended over committing to a
  two-model architecture on spec.

---

## Recommended sequencing

1. **Cheap de-risking on the current model (hours each, not days):**
   rain-weighted-loss A/B + CFG A/B (continue-runs), and the **PoP recalibration**
   (isotonic, fit on one split / validate on another). Learn which levers help.
2. **In parallel, stand up temperature data** (ERA5 ingest aligned to the radar
   grid/times) — if winter/snow is in scope.
3. **One from-scratch diffusion run** on the temporal split with the proven recipe
   + temperature, trained longer than before. (Keep the AE.)
4. **One final eval on the withheld test set** + recalibrate the PoP on clean data.

---

## Open decisions (need input before building)

1. **Objective** — what is the retrain *for*? trustworthy numbers / winter-snow /
   heavy-rain nice-to-have / calibrated probabilities / a subset?
2. **Temperature data** — ERA5, another source (DMI NWP / station archive), or skip?
3. **Compute budget** — from-scratch diffusion is several days on the RTX 5080.
   Spend it, or time-box (`max_hours`)?
4. **Risk appetite** — chase heavy/moderate rain with rain-weighted loss even though
   it risks the working light-rain goal?

---

## Status

- **2026-05-31:** plan drafted from the investigation. Temporal split + held-out
  test set **built and verified**. AE ceiling test + large held-out eval **done**.
  **Awaiting the four decisions above before building.** Cheapest first action
  regardless of those: **PoP recalibration** + the two recipe A/Bs.
