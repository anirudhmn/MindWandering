# Results — the selection and repair channels

Executed against `PREREGISTRATION.md`. All inference is subject-level (44 ROAMM readers,
12 ZuCo readers) with 10,000-sample subject bootstrap CIs, paired tests, and sign counts.
Machine-readable results in `results/`.

---

## Headline

**Mind-wandering does not decouple the eyes from the words at any measured level of
oculomotor control. It makes reading uniformly more effortful.** The single most-cited
behavioural marker of mindless reading — increased skipping — is, in ROAMM, a measurement
artefact; corrected, the effect reverses.

---

## G0 / G0b — the skip variable was contaminated (`results/g0_skip_audit.json`, `g0b_traversal.json`)

`reading_words.parquet` scored a word as skipped if no first-pass fixation mapped to it
anywhere inside the subject-run span. On that definition MW raises skipping from .439 to
.648 (Δ = +.168 [.131, .206], p = 7.4e-11, 41/44 readers) — apparently a textbook
replication of the meta-analytic marker.

Requiring that the reader's scan path demonstrably **stepped over** the word in a single
forward saccade of ≤ 4 intervening words (the standard operationalisation) removes the
effect and reverses it:

| gap threshold G | on-task | MW | Δ (MW − on-task) | p | readers |
|---|---:|---:|---:|---:|---:|
| 1 | .159 | .146 | −.026 [−.035, −.018] | 6.2e-7 | 9/44 pos |
| 2 | .228 | .209 | −.040 [−.052, −.029] | 3.6e-8 | 5/44 |
| **4 (primary)** | **.251** | **.226** | **−.042 [−.057, −.028]** | **1.8e-6** | **6/44** |
| 10 | .263 | .243 | −.034 [−.050, −.018] | 1.8e-4 | 9/44 |

**Readers skip ~17% *fewer* words during mind-wandering**, not more. A bracketing check
(`G0`) reaches the same conclusion: once the local region must be demonstrably traversed,
108% of the raw MW skip excess disappears.

### What the excess actually was (`results/blackout_anatomy.json`)
Not off-text gaze. Within a page the off-word time budget is identical across states
(16.6% vs 16.9%, p = .49), and off-word intervals per gap bin are indistinguishable
(medians 26–43 ms in every bin and state). The raw excess is structural: of MW's
stepped-over words, 74.3% sit inside large same-page forward jumps versus 48.3% on-task,
while 29.4% of on-task stepped-over words come from page transitions versus 0.6% for MW.
One genuine MW-specific event does emerge: the fixation step at a reported MW **onset**
carries an off-text interval > 500 ms 6.8% of the time versus 0.6% within-state — an
11-fold enrichment, though it accounts for a minority of onsets.

---

## G1 / G2 — PRIMARY: lexical control of skipping is fully preserved (`results/g1_selection.json`)

Somers' D (rank-based, base-rate free) of each property predicting whether a word is skipped,
per reader, per state; 332,016 words, 36 readers meeting the ≥150-words-per-state criterion.

| property | D on-task | D during MW | retention | Δ [95% CI] | p | p (Holm) |
|---|---:|---:|---:|---|---:|---:|
| zipf | +.5031 | +.5055 | **100.5%** | +.0025 [−.0167, +.0214] | .80 | .89 |
| length | −.5557 | −.5657 | **101.8%** | −.0100 [−.0283, +.0099] | .32 | .89 |
| surprisal | −.3818 | −.3936 | **103.1%** | −.0118 [−.0326, +.0099] | .30 | .89 |

Expressed as a fraction of the on-task effect the zipf CI is **[−3.3%, +4.2%]**.
Formal equivalence (`results/neural_equivalence.json`): all three properties are
**statistically equivalent within ±10%** (TOST p = 1.1e-5 / 2.6e-5 / .011), and within ±20%
at p ≤ 6.5e-7. Bayes factors BF01 = 5.4 / 3.5 / 3.3.

**G2 (lexical vs visual):** no dissociation. The zipf-minus-length change is
−.0075 [−.0271, +.0133], paired t = −0.72, p = .48. Lexical and visual control of skipping
change together, i.e. not at all.

**G3 controls, all consistent with the null:** stable across gap thresholds 1–10; unchanged
for deep MW (`mw_frac` = 1); no consistent pattern across run-position tertiles; stable
leave-one-story-out (only "drop serena_williams" gives a nominal surprisal effect, p = .019,
uncorrected); and a 200-iteration **pseudo-MW relocation** control reproduces the observed
values (p_perm = .81 / .28 / .31), i.e. even the small numerical trends are not MW-specific.

The legacy `skipping_verify_report.json` interaction (`zipf:mw` = +0.79, p = 2.5e-8) does
**not** survive: it was produced by the contaminated skip variable. On the corrected variable
the corresponding logistic term is +0.072 [+0.012, +0.133], p = .027 — and logistic
coefficients are not comparable across conditions with different base rates, which is why the
rank-based measure is primary.

---

## G4 — the repair channel: rates rise, selectivity is preserved (`results/g4_repair.json`)

76,643 single-skip events (one word stepped over), 44 readers.

| measure | on-task | MW | change | p |
|---|---:|---:|---:|---:|
| regression rate | .1929 | .2105 | **+18.1%** | 1.3e-4 (39/44) |
| immediate refixation rate | .1704 | .1896 | **+13.9%** | .0057 (34/44) |
| corrective return to a skipped word | .1714 | .1684 | +12.6% | .13 (30/44) |
| gaze duration | — | — | **+14.6%** | 1.7e-6 |
| skipping | .251 | .226 | **−18.5%** | 1.8e-6 |

Selectivity of the same behaviours is unchanged:

- corrective-return selectivity: zipf Δ = +.023 (p = .42), length Δ = −.006 (p = .80),
  surprisal Δ = +.005 (p = .87) — on-task selectivity is strong and real
  (D = −.251 / +.247 / +.208, all p < 1e-10), so the instrument is working.
- refixation selectivity: retention 98.9% / 97.3% / 96.0%, all p > .5.
- regression-to-difficulty: retention 74% (zipf, p = .25) and 73% (length, p = .056) — the
  only place in the whole battery with a hint of attenuation, and it does not survive
  correction.

The literature's specific prediction — fewer corrective regressions during MW, especially
for long words — is **not supported**; the rate moves in the opposite direction.

---

## G5 — within-token identification of the duration result (`results/g5_g6_duration.json`)

`word_key` is a unique corpus token instance; 7,264 of 10,183 tokens (71.3%) were read both
on-task and during MW by different readers, so MW varies *within* token instance. Two-way
(token × subject) fixed effects, subject-clustered SE, n = 234,800 word observations:

| term | β (log GD) | SE | p |
|---|---:|---:|---:|
| MW (additive) | **+.1030** | .0114 | 1.6e-11 |
| MW × zipf | −.0109 | .0070 | .129 |
| MW × surprisal | −.0084 | .0061 | .181 |
| MW × length | +.0087 | .0077 | .268 |

Compare the across-word per-subject signed retention on the same rows: zipf **114.6%**
(paired p = .00098), length 111.6% (p = .0066), surprisal 105.2% (p = .18).

**This corrects the "MW enhances coupling" reading.** Two things follow. (i) The apparent
enhancement is not lexically specific — *length*, a visual property, is enhanced as much as
frequency, which is the signature of a global rescaling of log-slopes rather than sharpened
lexical control. (ii) Holding the exact token instance fixed roughly halves the estimate
(≈ 7% for zipf) and it is no longer significant. The defensible claim is **preservation**,
not enhancement. Equivalence is correspondingly not established at ±20% for duration-zipf
(BF01 = 0.03, i.e. evidence for a small difference in the enhancing direction), while
duration-surprisal *is* equivalent within ±20% (p = 2.0e-4, BF01 = 2.4).

Neural coupling under the same design (`results/neural_equivalence.json`): MW × zipf on the
occipital FRP = −.047 µV (p = .34) and MW × surprisal on the centroparietal N400 = +.059 µV
(p = .072), with a significant additive occipital offset (+.088 µV, p = .0074) — the same
additive-shift-with-preserved-slope pattern. Per-subject neural slopes are underpowered
(n = 23 readers meeting the criterion; BF01 = 4.3 / 3.0), confirming that the neural row must
stay a group-level claim.

---

## G6 — measurement-scale audit (`results/g5_g6_duration.json`)

If a state changes duration additively, the log-scale slope is divided by the mean, so the
reported retention is biased by the intercept shift — in opposite directions for MW (slower)
and skimming (faster).

| contrast | mean-duration ratio | log retention if coupling were **unchanged** | log retention (as reported) | raw-ms retention |
|---|---:|---:|---:|---:|
| ROAMM MW, zipf | 1.160 | 86.2% | 115.2% | **133.8%** |
| ROAMM MW, surprisal | 1.160 | 86.2% | 108.1% | **125.3%** |
| ZuCo skim, zipf | 0.893 | 112.0% | 62.3% | **53.4%** |
| ZuCo skim, surprisal | 0.886 | 112.8% | 62.4% | **49.8%** |

**The log scale understates both effects.** The ZuCo goal-driven decoupling is ~47–50% on the
raw scale, not 38%. Both scales should be reported; the qualitative conclusion is unchanged.

---

## G7 — the ZuCo goal effect is partly a session effect (`results/g7_zuco_session.json`)

ZuCo 1.0 task order is fixed and identical for every reader (Hollenstein et al. 2018,
Methods): session 1 = NR then SR-half-1; session 2 = TSR then SR-half-2. The published
NR→TSR contrast is therefore also a between-session contrast. Splitting SR at its sentence
midpoint isolates the two:

| contrast | what it varies | zipf retention | p | surprisal retention | p |
|---|---|---:|---:|---:|---:|
| SR-h1 → SR-h2 | **session only** | 84.9% | .020 | 83.4% | .019 |
| SR-h2 → TSR | **task, within session 2** | **66.4%** | .0018 | **56.0%** | 1.3e-4 |
| NR → TSR (headline) | task + session | 62.3% | 1.7e-4 | 62.7% | 7.4e-5 |
| NR → SR-h1 | **materials, within session 1** | 110.5% | .054 | 134.2% | 2.0e-4 |

Three consequences for the existing coupling result:

1. **There is a real session effect** — coupling drops ~15–17% from session 1 to session 2
   with task and materials held constant. `GATE_G7` therefore records a FAIL on its
   "no session effect" clause.
2. **The goal effect nevertheless survives** a same-day, same-session, deep→shallow contrast
   at 66% (zipf) / 56% (surprisal). The core claim holds; the effect size shrinks from ~38%
   to ~34% for frequency.
3. **The published "materials control" was an artefact of averaging.** The earlier
   NR ≈ SR (102%, p = .70) result pooled SR across both sessions, mixing a *materials* effect
   of +10% (zipf) / +34% (surprisal) with a *session* effect of −15%, which cancelled. The
   claim "materials/genre do not matter" is not supported: movie-review sentences produce
   **stronger** coupling than Wikipedia sentences. This must be corrected wherever it appears.

---

## What this changes

**Strengthened.** The MW-preservation claim is now far stronger than the duration-only
version it replaces. It covers the two channels the field actually localises decoupling in
(selection, repair), it is supported by ±10% equivalence rather than ±20%, it survives a
pseudo-MW relocation control, and it holds in the channel — skipping — where a 2025
meta-analysis reports the most consistent MW marker.

**Corrected.** Three claims in the current manuscript plan need revision: the "MW enhances
coupling" reading (does not survive token-level control and is not lexically specific); the
ZuCo materials control (mixed two opposing effects); and the reported effect sizes (the log
scale understates both arms).

**New and positive.** MW is a *more-effortful*, not a more-cursory, reading state: longer
gaze, more regressions, more refixations, fewer skips, with every selectivity measure intact.
That is the opposite of the standard "skimming/mindless" picture and it is the natural
behavioural partner to the existing additive-state and ISC results.

## Limits

- ROAMM MW labels are self-caught and span-level; the reader marks the onset word after
  catching the lapse, so onset timing carries some imprecision, and the
  onset-enrichment result is descriptive.
- Exploratory with respect to ROAMM. G1 and G4 test directions predicted a priori by
  published literature, which is the strongest available framing short of new data.
- The corrected-skipping analysis conditions on the reader having scanned the local region;
  this is the standard definition but it is a state-dependent selection, mitigated by the
  matched word-property composition (zipf 5.445 vs 5.438; length 4.82 vs 4.86).
- ZuCo n = 12, and the SR half-split assumes the documented fixed task order.
- Neural per-subject estimates remain underpowered; the neural claim stays group-level.

---

## Mechanism tests (scripts 10-12; `results/neural_power.json`, `mechanism_tests.json`, `mechanism_controls.json`)

Added to discriminate between accounts of what mind-wandering does, after the neural row was
challenged.

### Neural power audit (script 10)
Working on the deconvolved betas directly (44 x 8 x 155 x 64) rather than ROI summaries:

| channel | change (% of on-task) | 95% CI | attenuation excluded | detectable at 80% |
|---|---:|---|---:|---:|
| frequency, occipitotemporal | +16.0% | [−12.5, +45.9] | >8% | 43% |
| surprisal, centroparietal N400 | −13.4% | [−112.2, +81.6] | >97% | **144%** |

**The N400 contrast cannot detect complete abolition.** It is uninformative, not supportive.
The frequency contrast is genuinely informative. Time-resolved cluster tests p=1.000 for both.
Non-deconvolved single-trial estimates disagree and are overlap-contaminated; item-level
averaging (2046 tokens) is no better. ZuCo TSR neural: +32.2% [−30.6, +88.4], excludes
attenuations >20% against a behavioural attenuation of 34–44%.

### A. Spillover: inconclusive
Lag-1 surprisal effect on-task is +0.003, essentially zero, so there is nothing to attenuate and
the test has no leverage. Lag-1 length changes −36% (p=0.16). 24 readers.

### B. Targeting: the extra effort is uniform
Splitting words into within-reader surprisal quartiles, the mind-wandering increase does not vary
with difficulty: fixation duration trend p=0.25, regressions p=0.33, refixations p=0.54. The one
direct test of extended-context integration (discourse gain x MW) is also null,
behaviourally p=0.37 and for the N400 p=0.28. **Readers are not repairing hard words.**

### C. Time course: regressions accumulate, and it is mind-wandering specific
982 episodes, median 23 fixations / 10.6 s.

| measure | 0-2 s | 2-5 s | 5-10 s | >10 s | trend | vs matched control |
|---|---:|---:|---:|---:|---|---:|
| regression rate | +.016 | +.024 | +.040 | +.080 | **+.0216 [.0107,.0341] p=.0012, 26/37** | **p_perm < 0.005** |
| fixation duration | flat | | | | +.0039 p=.63 | p_perm=.52 |
| refixation | flat | | | | −.0033 p=.49 | p_perm=.43 |

300 matched on-task stretches (lengths sampled from the observed episode-length distribution)
never reproduce the regression trend. Survives removing each reader's position through the run
(+.0224, p=.0007), so it is not time on task.

### D. Shared response: the alignment loss is confined to the residual
Template = other readers' **on-task** gaze durations for the same word (>=5 contributors),
on-task words subsampled to match the mind-wandering count.

| response | on-task | MW | retention | p |
|---|---:|---:|---:|---:|
| raw log gaze duration | .360 | .349 | 97.1% | .30 |
| z-scored within state (effort removed) | .361 | .351 | 97.0% | .29 |
| **residual after word properties** | .201 | .175 | **86.7%** | **.037** |

Shuffling mind-wandering labels within reader at matched count: +0.0004 +/- 0.0104,
**p_perm = 0.0067**. So the drop is specific to reported spans and to the non-lexical component.

### Synthesis
Not sensory gating (coupling preserved everywhere measurable). Not word repair (effort is
undirected; extended-context integration unchanged). What the data show is an immediate, uniform
step change in local effort, plus something that accumulates across the episode and expresses
itself as undirected backtracking, plus a loss of inter-reader alignment confined to the part of
gaze behaviour that word properties do not explain. The natural reading is that the reader loses
the thread rather than the words. The two measures that would test that directly, the N400 and
the contribution of extended context, are respectively unmeasurable here and null.
