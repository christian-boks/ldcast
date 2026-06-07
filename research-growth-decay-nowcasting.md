# Research: Why DL precipitation nowcasters can't predict growth/decay — and what the literature proposes

**Date:** 2026-06-07
**Produced by:** deep-research workflow (5 search angles → 22 sources fetched → 108 claims extracted → 25 put through 3-vote adversarial verification → **22 confirmed, 3 killed** → 8 synthesized findings).
**Raw output (ephemeral, `/tmp`, will be cleared):** `…/tasks/wbyzaiwty.output`

### Research question
In DL radar nowcasting, models tend to only advect + blur/decay existing rain (regression to the mean under MSE/pixelwise losses) and fail to predict convective **growth, decay, and initiation** beyond ~30–60 min. Find papers that (1) characterize/diagnose this and the predictability ceiling, and (2) propose solutions — noting for each **what it actually fixes (sharpness/realism vs. genuine growth-decay skill)** and its evaluated limitations.

---

## Bottom line

**No surveyed method demonstrably solves convective growth/decay/initiation from radar alone.** The best results either improve *calibration/realism* (generative models) or *claw back* skill by keeping an explicit deterministic component (motion/residual decomposition) — all still bounded by the radar-extrapolation predictability ceiling.

---

## 1. The predictability ceiling (strongest-evidenced; fully peer-reviewed)

- **Germann & Zawadzki 2002**, *Mon. Wea. Rev.* 130:2859. Defines predictability as the decorrelation lifetime (1/e cross-correlation) of radar patterns; scale decomposition shows small features lose predictability fastest; **Lagrangian advection ≈ doubles useful lead time vs Eulerian** (follow-on Germann et al. 2006: 5.1 h vs 2.9 h). Advection is the high-value easy part — and its limit. *(verified 3-0)*
- **Seed 2003**, *J. Appl. Meteor.* 42:381 (basis of S-PROG/STEPS/pysteps). Formalizes "dynamic scaling": large features evolve slower than small ones, so the cascade model **deliberately smooths toward larger scales to minimize RMS error.** This is *why* such models blur — by design. *(verified 3-0)*
- **Radhakrishna, Zawadzki & Fabry 2012**, *J. Atmos. Sci.* 69:3336 (the "Growth and Decay" paper of the McGill predictability series). Adding growth/decay on top of advection helps, **but growth/decay is predictable only for scales >100–250 km and lead times ~1–2 h.** ⇒ **convective-scale growth/decay/initiation is essentially unpredictable from radar history alone.** This is the quantified ceiling. *(verified 3-0)*
- **pysteps** (Pulkkinen et al. 2019, *GMD*) implements the FFT/Gaussian bandpass cascade + per-level AR(2) under Lagrangian advection.

## 2. "Sharpness ≠ skill" — direct evidence (the headline)

- **Bonte et al. 2026**, arXiv:2601.19298 (under review, *Weather & Climate Dynamics*). **STEPS and LDCast ensembles are statistically informative but carry *no dynamical* information**: phase-randomized surrogates (MAAFT/SPEC) that merely reproduce the rain-intensity distribution and power spectrum achieve the **same FSS and spatial-error scores** as the real ensembles — the Fourier phases that would encode *where* are effectively random. *(verified 3-0)*
  **⚠️ Caveat that matters for this project:** LDCast was run **pre-trained on Switzerland, NOT fine-tuned** on the Belgian RADCLIM test data; small sample (10 stratiform + 10 convective events, ≤90 min, 320×320). Strong evidence for the *phenomenon*, **not** a definitive verdict on a properly region-fine-tuned LDCast. Whether the "statistics-not-dynamics" result survives retraining is an explicit open question — and it's the one this project's Aalborg fine-tune directly bears on.
- **Ritvanen et al. 2025**, *GMD* 18:1851. A cell-tracking diagnostic (T-DaTing) built *specifically because* "realistic-looking long-lead output ≠ realistic convective development." Tracks cell volume/area/mean rain rate across inputs/targets/forecasts. This is the principled way to measure whether growth/decay is real. *(verified 3-0)*

## 3. The blurring diagnosis

- **Ravuri et al. 2021** (DGMR), *Nature* s41586-021-03854-z: constraint-free DL "lack of constraints produces blurry nowcasts at longer lead times, yielding poor performance on more rare medium-to-heavy rain events." *(verified 3-0)*
- **DiffCast** (Yu et al., CVPR 2024, arXiv:2312.06734) — finer diagnosis: **deterministic** models blur and badly underestimate high-value (heavy-rain) echoes (overlook local stochastics); **whole-system stochastic** models (GAN/diffusion) produce realistic detail but spatially inaccurate, position-mismatched forecasts ("freedom of generation too high"). *(verified 3-0)*

## 4. Solutions — what each actually buys

| Approach | Mechanism | Fixes | Doesn't fix |
|---|---|---|---|
| **DGMR** (Ravuri 2021, *Nature*) | Conditional GAN ensemble | Reduces blur; ranked #1 by 56 Met Office forecasters in **88%** of cases | Heavy-rain CSI "mixed"; authors note T+90 intensity decay *(88% fig. verified 2-1)* |
| **LDCast** (Leinonen 2023, arXiv:2304.12891, *Phil. Trans. R. Soc. A*) | Latent-diffusion ensemble | **Best calibration/diversity** (clean rank histograms) | Threshold-exceedance skill "mixed"; advantage is UQ, *not* heavy-rain skill *(verified 3-0)* |
| **DiffCast** (Yu 2024, CVPR) | **Deterministic motion μ + stochastic residual** (ŷ = μ + r̂) | **Best skill support** (see ablation below) | Residual is *local stochastic detail*, not life-cycle skill; no growth/decay eval |
| **L-CNN** (arXiv:2402.10747, *GMD* 2025) | Advection + explicit **source/sink** term | Names the source/sink (growth/decay) as "the biggest challenge of nowcasting" | — |
| **FACL** (Wong 2024, NeurIPS, arXiv:2410.23159) | Fourier amplitude + correlation loss | Shifts metrics toward skill scores | Gains **entangled with sharpness** (pooled CSI/FSS reward it); **no** growth/decay/timing gain *(framing verified 2-1; numbers 3-0)* |

**The DiffCast ablation is the trustworthy signal** (mechanism-level, robust to metric games): CSI for α=1 pure-stochastic **0.243** < α=0 pure-deterministic **0.258** < α=0.5 full **0.308**. Modeling everything as stochastic (≈ what whole-field diffusion does) *harms* CSI — keeping a deterministic trend genuinely helps. *(verified 3-0)*

> **Metric caveat threading through everything:** pooled CSI and FSS — the metrics most generative/loss papers cite for "skill" — themselves reward sharpness/intensity-distribution match (DiffCast and FACL both admit this). Headline CSI/pooled-CSI/FVD gains are partly the very realism the question warns about. Unpooled, location/timing-sensitive metrics and cell-tracking life-cycle diagnostics are the honest tests.

## 5. Claims that were REFUTED in verification

1. **"DGMR is sharp *and* genuinely skilful (both at once)"** → refuted **0-3**. The literature does *not* support that generative sharpness equals skill.
2. **"L2/MSE is *the* mechanistic cause of blurring"** → refuted **1-2**. Association, not proven sole cause (blurring also arises from the predictability ceiling and architecture). The math that the MSE/BCE optimum under uncertainty *is* the blurry conditional mean still holds as a major contributor — for a BCE classifier (e.g. this project's prob-nowcaster) it's exact — but don't pin all blurring on the loss.
3. **"Growth/decay = the advection/rotation residual"** → refuted **1-2**. A deterministic-residual split helps, but growth/decay is *not* simply "what's left after advection."

## 6. Caveats & gaps

- **Source strength:** only Germann/Zawadzki/Seed/Radhakrishna (ceiling) and Ravuri/DGMR (*Nature*) are fully peer-reviewed top-tier. LDCast (preprint→*Phil. Trans.*), DiffCast (CVPR), FACL (NeurIPS), Ritvanen (*GMD*), and the decisive Bonte (Jan-2026 preprint under review) are weaker on that axis.
- **Notable gap — named in the question but NO surviving verified claims (unresolved, *not* refuted):** **NowcastNet** (Zhang 2023, *Nature* s41586-023-06184-4) and its evolution operator, **PreDiff**, **CasCast**, **PhyDNet** (arXiv:2003.01460), and **MetNet-1/2/3** (Nat Commun s41467-022-32483-x; Google blog). Their specific growth/decay evidence was not established here. NWP-blending and conditioning on environmental/thermodynamic fields (CAPE, moisture, convergence) likewise has no surviving claim.

## 7. Open questions
1. Do NowcastNet's evolution operator / explicit advection+intensity-residual nets show location/timing-resolved growth/decay skill, or only pooled-CSI/visual realism? *(unresolved)*
2. Does larger spatial context (MetNet family) measurably improve growth/decay beyond the ~1–2 h, >100–250 km ceiling, or just extend large-scale skill? *(unaddressed)*
3. Would a **region-fine-tuned** LDCast still show "statistics-not-dynamics" ensembles, or does that artifact disappear after retraining? *(directly relevant to the Aalborg fine-tune)*
4. Can environmental/NWP conditioning add convective-initiation/decay skill separable from added realism — i.e. beat the Radhakrishna ceiling? *(no verified evidence either way)*

## 8. Implications for this project (ties to the prob-nowcaster / LDCast / DGMR discussion)
- The ceiling we reasoned to is **confirmed and quantified** (Radhakrishna): radar-only growth/decay is an *information* problem past short lead, not architecture.
- The worry that LDCast's realistic samples may not track observed dynamics is **directly supported** (Bonte) — with the live caveat that the test used an out-of-domain LDCast, and our Aalborg fine-tune is exactly the untested case.
- **DiffCast** is the most promising architecture *for this setup*: pixel-space deterministic backbone (escapes the AE time-collapse bottleneck the prob-nowcaster + LDCast share), residual-over-deterministic keeps positional accuracy (answers Bonte's "spread doesn't localize"), and **μ can be the existing Lagrangian advection**. But it fixes blur + position-accuracy + calibration — **not** growth/decay past the ceiling.

## 9. Full source list (22 fetched)

**Behind verified findings:**
- Germann & Zawadzki 2002 — https://journals.ametsoc.org/view/journals/mwre/130/12/1520-0493_2002_130_2859_sdotpo_2.0.co_2.xml
- Seed 2003 (S-PROG) — https://journals.ametsoc.org/view/journals/apme/42/3/1520-0450_2003_042_0381_adassa_2.0.co_2.xml
- Radhakrishna et al. 2012 — https://journals.ametsoc.org/view/journals/atsc/69/11/jas-d-12-029.1.xml
- Bonte et al. 2026 (sharpness≠skill; names LDCast) — https://arxiv.org/abs/2601.19298
- Ritvanen et al. 2025 (T-DaTing cell-tracking) — https://gmd.copernicus.org/articles/18/1851/2025/
- Ravuri et al. 2021 (DGMR) — https://www.nature.com/articles/s41586-021-03854-z
- Leinonen et al. 2023 (LDCast) — https://arxiv.org/abs/2304.12891
- Yu et al. 2024 (DiffCast) — https://arxiv.org/abs/2312.06734
- Wong et al. 2024 (FACL) — https://arxiv.org/abs/2410.23159
- L-CNN (advection + source/sink) — https://arxiv.org/abs/2402.10747

**Fetched, but no surviving 3-vote claim (named-but-unresolved + others):**
- NowcastNet (Zhang 2023, *Nature*) — https://www.nature.com/articles/s41586-023-06184-4
- PhyDNet (Le Guen & Thome 2020) — https://arxiv.org/abs/2003.01460
- MetNet-2 (Espeholt 2022, *Nat Commun*) — https://www.nature.com/articles/s41467-022-32483-x
- MetNet-3 (Google blog) — https://research.google/blog/metnet-3-a-state-of-the-art-neural-weather-model-available-in-google-products/
- JTECH-D-21-0013 (motion/intensity decomposition) — https://journals.ametsoc.org/view/journals/atot/38/12/JTECH-D-21-0013.1.xml
- AIES-D-23-0098 (large context / blending) — https://journals.ametsoc.org/view/journals/aies/3/3/AIES-D-23-0098.1.xml
- https://arxiv.org/abs/2301.11707 · https://arxiv.org/html/2512.08974v1 · https://arxiv.org/html/2409.10367v1 · https://arxiv.org/abs/2511.04659 · https://www.sciencedirect.com/science/article/pii/S2590197425000783 · https://arxiv.org/pdf/2406.10108

*(Titles for the last six were not established by surviving claims and are left as URLs to avoid mis-attribution.)*
