# Preregistration — the selection channel

Gate-first plan. Written before running any of the analyses below (the only prior
information is the legacy `skipping_verify_report.json`, whose interaction terms are
treated here as an unvalidated pilot, not as evidence).

## Motivation

The 2×2 landmark tests word→eye coupling in **fixation duration, conditional on the word
being fixated**. The mind-wandering literature (Mézière et al., 2025, *Memory & Cognition*
meta-analysis; Reichle, Reineberg & Schooler, 2010; the cascade model of inattention)
localises mindless-reading decoupling in two other channels:

1. **Selection** — reduced influence of word frequency/length on *skipping*.
2. **Repair** — reduced use of *corrective regressions* to under-processed words.

Neither has been tested in ROAMM. Because fixation is itself a lexically-driven decision
that changes markedly between states (skip rate on-task .439 → MW .648), the existing
duration contrast is also conditioned on a collider. This plan tests the two missing
channels, and adds three identification/robustness analyses that the current headline needs.

## Gates

### G0 — Skip-measurement audit (mandatory pass before G1)
`skipped` = no first-pass fixation mapped to that word inside the subject-run span. During
MW, gaze may leave the text or fail to map, manufacturing false skips.

- Quantify the MW skip-rate increase and its spatial structure (run-length of consecutive
  skips, local mapped-fixation density).
- Build **bracketed skip**: word retained only if a mapped fixation exists within ±3 reading
  positions on *both* sides, i.e. the reader demonstrably traversed the local region.
- **PASS** if ≥60% of MW words survive bracketing and the bracketed MW skip-rate increase
  remains positive and significant. Primary analyses are reported on the bracketed set with
  the raw set as sensitivity.
- *A priori note:* a blackout/tracking-loss mixture `y = 1 w.p. π + (1−π)p(x)` **attenuates**
  the fitted lexical slope. Tracking loss therefore biases G1 toward the null / toward the
  literature's predicted direction. It cannot manufacture an amplification.

### G1 — PRIMARY: lexical control of skipping, MW vs on-task
Scale-free by construction, because logistic coefficients are not comparable across
conditions with different base rates.

- Per subject, per state: **Somers' D** (= 2·AUC − 1) of `zipf` → skip, and of `length` → skip.
- Paired subject-level test of D(MW) − D(on-task); subject bootstrap CI; sign count.
- Secondary: per-subject logistic slopes; base-rate-matched case-control resampling.
- **Directional predictions:** literature ⇒ |D| decreases under MW. Autopilot account ⇒ |D|
  increases. Either signed outcome is informative; a null is reported as such.

### G2 — Lexical vs visual dissociation
`zipf`/`surprisal` index lexical control of skipping; `length` indexes visual/oculomotor
control. Test ΔD_lexical vs ΔD_visual (paired, per subject). A selective change in one
channel is the interpretable result; a uniform change indicates global rescaling.

### G3 — Controls for G1/G2 (all must be run and reported)
- **Pseudo-MW**: matched on-task spans (same per-run position distribution and span length)
  substituted for MW; the effect must not reproduce.
- **Position/page**: repeat within page-position tertiles.
- **Composition**: word-property distributions per state; repeat on the common support.
- **Deep MW** (`mw_frac` = 1) subset.
- **Per-story** leave-one-story-out.

### G4 — Corrective regressions to skipped material
For each forward inter-fixation step that skips ≥1 word, code whether any of the next 5
fixations lands inside the skipped region (**corrective regression**). Model
`P(corrective) ~ zipf + length + surprisal + MW + MW×{...}` per subject.
Meta-analytic prediction: corrective regressions decrease under MW, especially for long words.
Also report overall regression rate and regression-to-difficulty × MW.

### G5 — Within-token-instance identification of the duration result
The 111% "preserved/enhanced" duration coupling is estimated across words. `word_key` is a
unique corpus token instance read by ~44 readers, so MW varies *within* token instance.
Two-way fixed-effects model:
`log(fix_dur) ~ word_key FE + subject FE + is_mw + is_mw×z(zipf) + is_mw×z(surprisal)`.
This holds the exact token, its context, and its page position fixed. **PASS** if the MW×zipf
term keeps the sign and rough magnitude of the across-word estimate.

### G6 — Measurement-scale audit
Recompute retention on the raw millisecond scale alongside log, for ROAMM MW and ZuCo
NR→TSR. Report the direction of the scale-induced bias implied by each state's intercept shift.

### G7 — ZuCo session control
ZuCo 1.0 task order is fixed and unbalanced: session 1 = NR then SR-half-1; session 2 = TSR
then SR-half-2. Split SR by `sent_idx` median into its two session halves.
- SR-half-1 vs SR-half-2 ⇒ session/day effect on coupling.
- SR-half-2 vs TSR ⇒ **within-session-2** task contrast.
**PASS** if SR-half-2 retention ≈ 100% and TSR remains ≈ 62%, isolating task from session.

## Inference
Subject is the unit throughout. Subject-bootstrap 95% CIs (10,000 resamples), paired tests,
sign counts reported for every effect. Holm correction within each gate family. No gate is
opened before the preceding mandatory gate reports.

## Status
Exploratory/post-hoc with respect to ROAMM (the dataset has been analysed extensively).
G1/G4 directions are *predicted a priori by published literature*, which is the strongest
available framing short of new data. Any positive result requires preregistered replication.
