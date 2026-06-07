# LDCast Retrain Plan
claude --resume 17bdf0f1-1954-4b5c-9d6e-15b7e7bc0882

Working plan for the next training effort, **driven by the objectives below** and
the 2026-05-31 investigation. **Living document — edit as decisions land.**
Evidence is in `journal.md` (2026-05-31 entries); eval tooling in
`scripts/eval_autoenc_ceiling.py` and `scripts/eval_val_large.py`.

## Objectives
- Rain or no rain is the most important objective.
- Getting the amount of rain correct is a nice to have feature.
- The prediction should still look like rain.
- Capturing whether rain is growing out of nowhere, or collapsing into nothing is important.
- It has to be better than Lagrangian persistence.
- Main use case is determining what is the probablility it will rain or not in the next 30-40 minutes at a given location or path.
- Doesn't have to use data from all of Denmark, we could just use data seen around Aalborg. This should allow a model to see the same data multiple times. 
---

## What the objectives imply

**North-star metric (replaces pooled CSI):** a *calibrated probability of rain /
no-rain at a location or path over +30–40 min* that (1) **beats Lagrangian
persistence** and (2) earns credit specifically for **growth (initiation) and decay
(dissipation)** — the part advection physically cannot do. Intensity ("amount") and
"looks like rain" are explicitly secondary.

**Focus shift vs the original plan:**
- ⬇️ **Down:** heavy rain / intensity / rain-weighted loss (obj 2 = nice-to-have).
- ⬆️ **Up:** growth-decay skill (obj 4), beating persistence (obj 5), and calibrated
  point/path probability (obj 1, 6).
- Success is no longer "raise CSI" but "**beat persistence, especially on
  growth/decay, with honest probabilities.**"

---

## Step 0 — Measure the baseline (DO FIRST; gates everything)

Objectives 4 & 5 are **unmeasured today**, and they are the model's entire reason to
exist over advection. Before any retrain or architecture choice:

1. **Lagrangian persistence (+ PySTEPS) skill** on our held-out cases — same cases,
   same CSI/POD/FAR/Brier code as the model eval, per threshold and per lead.
   (`scripts/eval_persistence.py` / `eval_pysteps.py` exist for the old DWD data
   path; wire to the rust/DMI val cases used by `eval_val_large.py`.)
2. **Growth/decay metric:** per lead, split pixels into **initiation** (dry@t0 →
   rain@t+Δ) and **dissipation** (rain@t0 → dry@t+Δ); score the model **and**
   persistence on each. Persistence ≈ 0 on initiation by construction — any model
   skill there is its real edge.

**Decisive question:** *does the current model beat persistence at all, and is its
edge in growth/decay?* It is entirely possible it does **not** beat persistence on
pooled metrics (persistence is strong at short lead) — in which case the whole value
proposition rests on objective 4, and the retrain must optimise for that explicitly.

---

## The architecture question — is the LDM the right core?

The objectives reopen this, and tip it. The **#1 objective and the main use case are
both calibrated probability**, which is the home turf of a *directly-optimised
probabilistic nowcaster*, not a generative ensemble:

- Train a deterministic nowcaster to output **P(rain ≥ thr) per pixel/lead** with a
  proper scoring loss (BCE/Brier). **Calibrated by construction** (no post-hoc
  recalibration), **cheap** (one forward, not 32 samples), can learn growth/decay.
- This corrects the 2026-05-28 premise *"wants probability → needs an ensemble."*
  Probability is better & cheaper from a classifier; the miscalibration we measured
  ("20 %" → ~55 %) is partly an artifact of using a generative model for a
  classifier's job.
- **In-repo vehicle:** `ldcast/models/nowcast/nowcast.py::AFNONowcastNet`
  (deterministic AFNO nowcaster, currently unused) + a probabilistic head.
  (MetNet-2/3 is the SOTA reference for "P(precip) at a location, short-range, beats
  persistence" — but a large build; the AFNO nowcaster is the practical option.)
- **The LDM's unique value is objective 3** (fields that *look like rain* + coherent
  scenarios). Likely resolution: **probabilistic model as the core** (obj 1, 4, 5, 6),
  **LDM kept only as an optional realistic-field visualisation** (obj 3). The
  diffusion work isn't wasted — it stops being the probability engine.
- **Caveats:** deterministic models blur; not *proven* to capture growth/decay
  better; switching is a real pivot. **Decision gated on Step 0.**

**Orthogonal lever for growth (obj 4): environmental / NWP conditioning.** Radar-only
models can't see rain that doesn't exist yet, so **convective initiation is the known
weak spot** — and temperature / instability / moisture fields are the standard lever
for it. So NWP conditioning (incl. temperature) is now justified by **obj 4**, not
just winter snow — it moves **up**, and applies to *either* architecture.

---

## Why retrain / honest scope

The current model is **recipe-limited, not data-limited** (flat 40 clean epochs at
8 % of one pass; AE ceiling test proves the codec is fine — recon csi_5 ≈ 0.9, so the
bottleneck is generation, not data or AE). It *appears* to serve the primary goal
(POD dialable to ~0.95, useful +40–60 min) — **but Step 0 has not yet confirmed it
beats Lagrangian persistence**, which is now a hard requirement (obj 5).

A retrain/rebuild realistically **buys**: a clean test number; a probability product
that's calibrated *by construction* (if we switch architecture) or recalibrated (if
not); NWP-conditioned initiation skill; and training optimised for the north-star.
It will **not** make heavy rain good — deprioritised anyway (obj 2).

---

## Findings (2026-05-31 large held-out eval: 660 cases × 32 members)

| finding | evidence | implication (under the objectives) |
|---|---|---|
| recipe-limited, not data-limited | flat 40 clean epochs at 8 % of one pass | lever is the loss/architecture, not more epochs |
| AE is **not** the bottleneck | recon csi_5 ≈ 0.9 vs model 0.014 | don't retrain the AE; reuse the codec/features |
| under-produces intensity | 66 / 10 / 1 % of ceiling at 0.1/1/5 mm | intensity is nice-to-have (obj 2) — deprioritise |
| heavy rain weak everywhere | csi_5 0.014 [0.011–0.018]; FSS 0.06 @51 px | drop heavy-rain chasing |
| no usable regime | per-case spike at ~0 (345/660); 2 % > 0.3 | no "ship the good subset" option |
| season = climatology | light identical warm=cold (0.50); moderate 0.148 vs 0.092 | all-year is fine; season-split low value |
| PoP miscalibrated | forecast "20 %" verifies at ~50–70 % | calibrated-by-construction model, or recalibrate |
| **persistence skill** | **UNKNOWN — Step 0** | **the actual bar to clear (obj 5)** |
| **growth/decay skill** | **UNKNOWN — Step 0** | **the actual edge over advection (obj 4)** |

---

## The plan (reprioritised)

### Tier 0 — Measurement & trustworthy setup (do first)
- **Step 0 baseline** — persistence + growth/decay (above).
- **Temporal split** — built & verified (`split_mode: temporal`, `test_frac: 0.1`;
  whole-UTC-day holdout; no day leaks). Clean val + a test set scored **once** at the
  end.
- **Keep the autoencoder** — excellent (recon csi_5 ≈ 0.9), radar-only, leakage
  negligible for the forecast question. Reuse it whatever the architecture.
- **Monitor upgrades** so training optimises the north-star, not pooled CSI: add a
  **persistence-relative skill score**, the **initiation/dissipation** breakdown,
  FSS, and reliability to the per-epoch monitor.

### Tier 1 — Architecture decision (gated on Step 0)
- Deterministic **probabilistic nowcaster** (`AFNONowcastNet` + BCE/Brier head) as
  the core, vs **keep the LDM**, vs **hybrid** (probabilistic core + LDM for the
  realistic-field view). Pick after Step 0.

### Tier 2 — NWP / temperature conditioning (now justified by obj 4)
- For **initiation** (radar can't see un-formed rain) and rain/snow. Architecturally
  supported (`use_nwp`). Gate: data — **ERA5** (free, hourly 2 m temp, ~0.25°,
  2022–2026) vs a DMI NWP/station archive vs skip. Applies to either architecture.

### Optional / deprioritised
- **Rain-weighted diffusion loss** — only if we keep the LDM *and* decide intensity
  matters after all. Off the critical path now (obj 2).
- **CFG** — still useful for discrimination if we keep the LDM.
- **Season-specific models** — ranked lower: our diagnosis is *recipe*-limited but a
  season split attacks *capacity/heterogeneity* (wrong joint); warm-vs-all-year gap
  is small (csi_1 0.148 vs 0.128); halves data/compute; dominated by NWP
  conditioning. Settle via one summer-only-vs-all-year A/B on warm-season test cases
  if curious.

---

## Recommended sequencing
1. **Step 0** — persistence + growth/decay baseline (cheap; gates everything).
2. **Architecture decision** from Step 0 + objectives.
3. **Monitor upgrades** (persistence-relative + initiation/dissipation) so the run
   optimises the right thing.
4. **NWP/temperature data** stood up if in scope (the initiation lever).
5. **Train the chosen model** on the temporal split.
6. **Final eval on the withheld test set**, scored against the north-star.

---

## Open decisions
1. ~~**Architecture**~~ — **RESOLVED 2026-06-01: probabilistic nowcaster is the
   core.** It matched/beat the diffusion model on Brier + initiation on the Aalborg
   held-out test (decisively at 1.0 mm), with raw probabilities (no recalibration),
   ~400× cheaper, despite the diffusion model's leakage advantage. LDM kept only for
   obj 3 (realistic fields). See Status + journal 2026-06-01.
2. **NWP conditioning** — ERA5, DMI archive, or skip? (now justified by obj 4.)
3. **Compute budget** — several days on the RTX 5080, or time-box (`max_hours`)?
4. ~~Heavy-rain risk appetite~~ — resolved: deprioritised (obj 2).

---

## Status
- **2026-05-31:** objectives added; plan refocused around them. Temporal split +
  held-out test set built & verified; AE ceiling + large held-out eval done.
- **Step 0 (persistence baseline) DONE — and it's a red flag.** On the same 660
  held-out cases, the model **loses to even Eulerian persistence** (hold the last
  frame): CSI@1.0 mm 0.15 vs 0.30, @5.0 mm 0.02 vs 0.08; @0.1 mm a wash (0.53 vs
  0.56). So **objective 5 is currently failed.** Consistent with the under-production
  diagnosis (timid model → too-dry field → loses to "keep the rain").
  `scripts/eval_persistence_baseline.py`, `val_large_eval/persistence_baseline.json`.
- **Step 0b DONE — it resolves the contradiction.** The model loses on *bulk
  detection* (CSI) but **wins on the two axes the objectives name**:
  - **Probability (Brier):** beats persistence at every threshold (BSS vs Eulerian
    +0.36 / +0.25 / +0.39 at 0.1/1.0/5.0 mm). Caveat: vs a weak 0/1 baseline; would
    *widen* after PoP recalibration.
  - **Light-rain initiation (obj 4):** model init POD 0.257 vs Lagrangian 0.213 at
    0.1 mm — a real edge where persistence is blind. (Loses at moderate/heavy
    initiation — under-production; its high dissipation POD is a timidity artifact.)
  So the diffusion model **has a defensible niche** exactly on-objective (probabilistic
  + light-rain growth) — it is not worthless, just mis-evaluated by pooled CSI.
- **Architecture decision is now fair and sharp:** can a *cheaper* deterministic
  probabilistic nowcaster (`AFNONowcastNet` + BCE/Brier head; one forward, native
  calibration) **match the diffusion model's Brier-skill-vs-persistence + initiation
  POD**? Decision metrics = those two, NOT pooled CSI. If yes → switch (cheaper,
  calibrated). If no → the diffusion approach earns its cost.
- **Cheap parallel win regardless:** recalibrate the diffusion PoP — it already beats
  persistence on Brier *while miscalibrated*; recalibration widens that and makes the
  product shippable. **DONE** (`scripts/recalibrate_pop.py`, `pop_calibration.npz`):
  held-out reliability gap 0.54 → 0.04 at 1.0 mm.
- **2026-06-01: ARCHITECTURE A/B DONE — probabilistic nowcaster WINS.**
  `scripts/eval_aalborg_ab.py`, 400 wettest Aalborg held-out **test** cases × 32
  members, BSS vs Eulerian (95% CI over cases):
  - **Brier overall:** prob (raw) +0.508 / **+0.482** / +0.476 at 0.1/1.0/5.0 mm vs
    diff_cal +0.507 / +0.351 / +0.425. Tie at 0.1 mm; **prob wins at 1.0 & 5.0 mm
    (disjoint CIs).** Raw diffusion (diff_raw) is far worse at 1.0 mm (+0.071) —
    badly miscalibrated; the prob model needs no fix.
  - **Brier on initiation pixels (obj 4 growth):** prob +0.578 / **+0.377** /
    **+0.103** vs diff_cal +0.580 / +0.270 / +0.031. Both beat persistence
    everywhere; prob clearly wins at 1.0 & 5.0 mm.
  - **init POD@0.5:** prob 0.504 / **0.304** / 0.038 vs diffusion 0.254 / 0.009 /
    0.000 vs Lagrangian 0.250 / 0.195 / 0.067. The diffusion model is too timid to
    fire at 1/5 mm (diss POD ≈ 1.0, init ≈ 0 = dry-out); prob discriminates.
  - **Robust:** the diffusion model trained on these test days (random-split
    leakage = unfair advantage) and still lost. ~400× fewer params, one forward.
  → **Adopt the probabilistic nowcaster as the core** (obj 1, 4, 5, 6). Next:
  Tier-2 NWP/temperature conditioning (open decision #2) for initiation, and a
  longer prob-nowcaster train (the Aalborg run converged in ~30 epochs; scale
  back up to all-Denmark or add NWP).
- **2026-06-01 CORRECTION — the "initiation" numbers above were inflated by an
  advection confound.** The `dry@t0 → rain` set counts a front *moving in* as
  "initiation," and Eulerian's 0 there is an artifact (it can't move). Replaced with
  **GENESIS = advection predicts dry AND > 1.5·|motion|·lead px from any t0 echo**
  (true formation only). On that honest set (BSS vs persistence, 95% CI):
  - **Genesis Brier:** prob +0.120 / **+0.154** / **+0.062** at 0.1/1.0/5.0 mm vs
    diff_cal +0.169 / +0.116 / **−0.018**. Modest (not +0.58); prob ≥ diff, and the
    only one positive at 5 mm.
  - **Genesis POD@0.5:** prob 0.055 / 0.070 / 0.011 vs diff ~0 — prob ≫ diff but
    **absolute genesis POD is LOW**: radar-only models barely forecast rain from
    nowhere. *Most of the old "growth skill" was advection.*
  - **Decay (the other half of obj 4):** prob clearly beats persistence — BSS vs
    Lagrangian +0.58 / +0.88 at 1/5 mm; it anticipates clearing.
  → **Architecture verdict unchanged** (overall Brier untouched). But genesis is the
  real weak spot — **the strongest quantitative case for NWP/temperature
  conditioning** (open decision #2): convective initiation is ~unforecastable from
  radar alone.
