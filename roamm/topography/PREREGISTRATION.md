# Preregistration — multiscale EEG mechanisms of natural reading

Frozen: 2026-07-25, before inspecting any new test statistic produced by these
analyses. Existing ROAMM results and artifacts are known and are treated as prior
work, not independent confirmation.

**Scope note.** This plan declares three questions. Only question 2, the topographic
rescaling test, is reported in the manuscript (Supplementary S1), and only its script
and result are included here. The plan is reproduced in full so that the reported test
can be read against everything that was declared alongside it.

## Scope and motivation

Earlier work on this dataset has tested tonic spectral power, aperiodic slope, spectral
entropy, wPLI, fixation-related potentials (FRPs), overlap-corrected regression ERPs
(rERPs), lexical and discourse coupling, inter-subject correlation, fixation-onset
phase concentration at one slow band, EEG-to-text decoding, and item-level RSA. Three
mechanistic questions remain materially open:

1. Does natural reading have fixation-locked **induced oscillatory** organization
   beyond the already-tested evoked FRP and onset-phase scalars, and does mind
   wandering (MW) alter it?
2. Is the established additive MW rERP effect a change in response magnitude with a
   stable scalp configuration, or a change in the configuration of the scalp field?
3. Is there an overlap-corrected EEG signature of deciding to revisit text and of
   processing a repeated word?

These analyses test those questions. They do not optimize an MW classifier.

## Data and inferential unit

- 44 readers, five shared Wikipedia stories, 64-channel EEG sampled at 256 Hz.
- Continuous EEG was already ICA-cleaned, common-average referenced, bad-channel
  interpolated, and band-pass filtered 0.5–50 Hz in the released dataset. Any added
  filtering will be stated exactly.
- The reader is the inferential unit. Event-level regressions are fit within reader;
  group inference is over the 44 reader coefficients.
- MW is the corrected direct span label. The report/annotation interval and the final
  two seconds preceding the report remain excluded by the upstream event tables.
- All tests are two-sided unless an explicitly directional signed replication is
  declared.

## Feasibility facts fixed before testing

- 404,557 mapped first-pass fixations.
- 18,628 reader × sentence traversals across 487 sentences.
- Entire sentence traversal state: 16,474 on-task, 1,594 mixed, 560 entirely MW;
  40 readers contribute at least one entirely-MW sentence.
- Sentence-terminal fixation state: 17,703 on-task, 925 MW; 43 readers contribute an
  MW terminal event, but only 23 readers contribute at least 15.
- Rereading table: 402,082 mapped transitions, including 78,417 regressions and
  69,445 same-word refixations.

The all-MW sentence contrast is therefore secondary. The primary oscillatory state
test is fixation-level, where every reader has substantially more support.

## Family A — fixation- and sentence-related oscillatory dynamics

### Estimation

For posterior, centroparietal, central, and frontocentral sensor groups:

- Compute analytic signals for theta (4–7 Hz), alpha (8–12 Hz), lower beta
  (13–18 Hz), and upper beta (19–30 Hz), using zero-phase filters for descriptive
  event-related analyses.
- Extract fixation epochs from −400 to +800 ms. Log power is baseline-centered using
  −300 to −100 ms. Phase consistency is calculated from unit analytic phase and does
  not use amplitude.
- Primary induced-power estimates use a continuous time-expanded regression on the
  log-power envelope so that overlapping fixation responses are separated. Nuisance
  regressors include incoming saccade amplitude/direction, log fixation duration,
  word length, Zipf frequency, local surprisal, fixation order, page position,
  blink proximity where available, and main effects of state.
- Phase-consistency contrasts use within-reader count matching and 1,000
  episode-preserving circular shifts within reader × run. The time-frequency search
  is controlled by a max-cluster statistic.

### Frozen hypotheses

**A1 (constructive on-task effect).** Fixations evoke a posterior oscillatory response
that is not exhausted by the mean FRP: a change in induced lower-beta power in
0–400 ms and/or theta phase consistency in 0–300 ms. This is a feasibility/construct
gate, not the MW claim. Continue if at least one of the two predeclared measures has a
subject-level 95% CI excluding zero and survives its within-family correction.

**A2 (primary MW test).** MW reduces the temporal fidelity or induced organization of
the fixation response: on-task minus MW theta phase consistency in posterior sensors,
0–300 ms. A positive claim requires both subject-level p < .05 and an
episode-preserving permutation p < .05. Lower-beta induced-power state modulation is
co-primary within Family A; Holm correction is applied across A2 phase and power.

**A3 (sentence-scale secondary).** Sentence entry/termination produces a slow
delta/theta organization beyond matched within-sentence pseudo-boundaries, and its
strength decreases with the fraction of the sentence traversed during MW. Binary
all-MW versus all-on-task estimates will be reported with a minimum detectable effect
and cannot become the headline unless at least 30 readers contribute 10 events per
state and the episode-preserving permutation passes.

### Falsification and specificity

- Pre-fixation pseudo-onsets shifted by a within-run circular offset must not reproduce
  the event-locked effect.
- Effects must survive exclusion of blink-adjacent events and matching on fixation
  duration and incoming saccade amplitude.
- A state effect confined to frontal pole sensors, mirrored in horizontal/vertical
  gaze covariates, or eliminated by saccade controls is classified as ocular.
- Results that exist only in total power but not after subtracting the evoked response
  are described as evoked, not induced.

## Family B — full-scalp rERP field structure

This family reuses the already-frozen eight-predictor overlap-corrected rERP. No new
model is fit before the field tests.

### Frozen hypotheses and estimands

**B1 (positive-control topology).** Frequency (150–290 ms) and surprisal
(300–450 ms) kernels have stable, nonzero global field power (GFP) and distinct
normalized scalp maps. This validates that the field analysis detects known lexical
and semantic effects.

**B2 (primary state-field test).** The MW additive kernel at 150–290 ms is tested for:

1. nonzero GFP relative to a subject-wise sign-flip null; and
2. topographic dissimilarity from the frequency kernel and from zero-centered
   on-task fixation response maps using reference-invariant global map
   dissimilarity/TANOVA.

Interpretation is frozen:

- nonzero GFP with no corrected topographic difference: gain/offset-like change with
  stable field configuration;
- corrected topographic difference: altered scalp-field configuration;
- neither: the earlier ROI effect does not generalize to the full field.

**B3 (MW coupling interactions).** Frequency × MW and surprisal × MW undergo the same
GFP and time-resolved TANOVA tests. These are confirmatory null audits; a positive
claim requires cluster-FWER p < .05.

### Multiplicity

Time-resolved field tests use one max-cluster sign-flip null across the full
0–500-ms epoch and all tested kernels. A-priori scalar windows use Holm correction
within Family B.

## Family C — neural control of rereading

### Estimation

A new continuous deconvolution model separates fixation-onset responses from outgoing
saccade-onset responses.

- Saccade-locked window: −400 to +200 ms relative to the estimated end of the launch
  fixation.
- Landing/fixation window: −100 to +500 ms.
- Primary sensor groups: frontocentral for the presaccadic decision signal and
  occipitotemporal plus centroparietal for repeated-word processing.
- Event classes: forward, same-word refixation, and regression.
- Controls: log launch-fixation duration, incoming and outgoing saccade amplitude,
  direction/line change, launch and target word frequency, length and surprisal,
  page location, prior exposure, and reader-specific underprocessing/DEFICIT where
  available.

### Frozen hypotheses

**C1 (primary presaccadic effect).** Regression- or refixation-launching fixations show
a frontocentral presaccadic field difference in −200 to 0 ms relative to matched
forward launches. The gate is an overlap-corrected absolute mean difference of at
least 0.10 microvolts with p < .05 and at least 28/44 readers in the same direction.

**C2 (repetition effect).** Landing on a previously viewed word changes the
occipitotemporal 100–250-ms and centroparietal 300–450-ms response relative to matched
first visits, with a graded dependence on repetition lag/prior exposure. Holm
correction is applied across the two windows.

**C3 (MW interaction, conditional).** Test MW modulation only if C1 or C2 passes its
construct gate. A state claim additionally must pass an episode-preserving label
permutation and a within-launch-location control. Failure of either kills the state
interpretation.

## General robustness and reporting rules

- Report effect estimates, reader bootstrap 95% CIs, sign counts, p values, and
  MDE80, including for nulls.
- Known-effect gates are evaluated before state interactions.
- No post-hoc channel/window may replace a failed primary. Exploratory maps are
  labeled exploratory and use cluster correction.
- Results are checked in odd/even readers and leave-one-story-out. A headline effect
  must have the same direction in both reader halves and every leave-one-story-out
  estimate.
- This is discovery on a repeatedly analyzed dataset. Even a result passing all gates
  requires independent replication.
