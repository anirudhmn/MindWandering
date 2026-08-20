# Preregistration — semantic importance tracking during reading: gaze, EEG, and mind-wandering

Frozen 2026-07-25 before any outcome model was fitted. Annotations (importance, evidence spans)
were written and saved before the analysis scripts existed.

## 1. The question

ROAMM readers answer one comprehension question per page, and the question appears **only after
the page is gone**. That design detail is the whole opportunity. McCrudden & Schraw (2007)
separate two constructs that the literature routinely conflates:

* **text-based importance** — how much a segment is needed to understand the text
* **task-oriented relevance** — how much a segment is needed for a specific task

Every eye-tracking demonstration of selective processing has given readers the task first
(relevance instructions, question-before-reading, reading-to-summarise; Hyönä & Lorch 2004;
McCrudden & Schraw 2007; the jemr summary study). ROAMM cannot: a reader here has no way to know
which sentence the question will ask about. So if reading is still selectively allocated, it must
be allocated by **intrinsic importance**, computed on the fly from the text — endogenous
prioritisation, not instructed prioritisation. That is measurable here for the first time with
gaze, EEG, self-reported mind-wandering, and the item outcome on the same trials.

The three questions, in priority order:

1. **Does free reading track semantic importance at all** — in the eyes, and in the brain?
2. **What does mind-wandering do to importance tracking?** The selection-and-repair analysis established that MW in
   ROAMM leaves *lexical* coupling intact (frequency/length/surprisal → gaze selectivity
   equivalent within ±10%) while readers spend more time, regress more and skip less, and that
   the ISC drop lives entirely in the non-lexical residual. If MW is "losing the thread, not the
   words", then a discourse-level property — importance — is exactly what should decouple while
   word-level properties stay coupled. That is a **directional, pre-derived dissociation**, and
   it inverts the cascade model of inattention (Smallwood 2011), which puts the lexical link
   first in the causal chain.
3. **Does importance-weighted encoding explain who answers the question?** MW already predicts
   failing that page's item (claim C0, −5.9 points). With the answer span localised, the
   question becomes whether the failure is *informationally specific*: MW while the eyes were
   on the answer, versus MW elsewhere on the same page.

## 2. Measures of importance (three, deliberately)

The LLM-annotation literature's core warning is that a single model with a single prompt is not
a measurement instrument: results move when either changes. So importance is measured three ways
and all three are reported.

| measure | source | judgement? | script |
|---|---|---|---|
| `importance_llm` (primary) | Claude Opus 5, in-context, 1–5 rubric, question-blind | yes | `01` |
| `importance_qwen` | Qwen2.5-1.5B-Instruct, same rubric, digit logits, deterministic | yes, independent | `03` |
| `centrality_lm` | GPT-2 leave-one-sentence-out ΔNLL of the next 150 words | no | `02` |

`in_summary_llm` (would the sentence appear in a 2–3 sentence summary) is recorded as the
standard NLP salience operationalisation.

**Primary is `importance_llm`** — it is the construct the analysis is about. The other two are
convergent validity, and the pre-declared rule is: *a claim about importance stands only if the
sign replicates on `importance_qwen` and does not reverse on `centrality_lm`.* Disagreement
between judged importance and LM predictive centrality is reported as a finding about the
constructs, not smoothed over.

Unit of annotation is the **sentence**, not the word, and that is a deliberate identification
choice: because importance is assigned at the sentence level, the *same word type* occurs in
high- and low-importance sentences, which licenses the within-word-type contrast that removes
every lexical confound (the identification trick that worked in the ZuCo relevance analysis).

## 3. Disclosed unblinding

Before the importance ratings were written, this session had incidentally seen 5 Pluto question
stems and one-line answer descriptions for 7 items (encoded in a prior script's OVERRIDE dict) —
11 items in total, flagged `exposed_to_annotator`. Ratings were assigned from page text alone.
Questions 1 and 2 never involve the questions, so exposure cannot act on them. For question 3 the
predeclared sensitivity analysis re-fits on the 39 unexposed items.

`Prisoners Dilemma_p8` is **mis-keyed** (the page states the opposite of the keyed option and
states a different option verbatim); flagged and dropped in sensitivity analysis.

## 4. Feasibility gates — declared before fitting, KILL if failed

* **G0 variance.** Every page must have within-page SD of `importance_llm` ≥ 0.4.
  → PASSED at annotation time (median 0.823, min 0.515, 0/50 below 0.5).
* **G1 convergent validity.** Spearman ρ(`importance_llm`, `importance_qwen`) ≥ 0.30 within page.
  If it fails, the construct is not measurable by LLM and this is a methods null.
* **G2 not-a-lexical-confound.** |r(importance, mean sentence zipf/length/surprisal)| < 0.7, and
  ≥ 20,000 word tokens whose type occurs in both a high- and a low-importance sentence.
* **G3 MW overlap.** ≥ 300 word tokens read in both an on-task and an MW state with importance
  variation (the selection-and-repair analysis found 234,800 such rows, so this is a formality — verify, don't assume).
* **G4 neural power.** Report the minimum detectable effect for the importance FRP contrast
  *before* interpreting it. The lesson from the selection-and-repair analysis: the N400×MW contrast had MDE = 144% of its own
  base effect, so a null there was uninformative. Any neural null here is reported
  with a one-sided attenuation bound and an MDE, or not reported as a null at all.

## 5. Hypotheses and tests

Behaviour uses the existing validated tables (`reading_fixations.parquet`,
`words_traversal.parquet` with the corrected scan-path skip variable). Subject-level bootstrap
(10k) and two-way (token × subject) fixed-effects absorbers with subject-clustered SE, as in
the selection-and-repair analysis. Holm correction within each family.

**H1 — importance tracking (eyes).** Gaze duration, refixation, regression-in and skipping vary
with sentence importance, controlling zipf/length/surprisal, word position in sentence, sentence
position on page, sentence length and line position.
*Identification:* within-word-type contrast (same token type, different-importance sentences)
must survive, otherwise the effect is a lexical artefact and is reported as such.

**H2 — the dissociation (primary claim).** Importance→gaze coupling is attenuated during MW by
more than lexical→gaze coupling is, on the same rows, in the same two-way FE design.
Estimand: ratio of MW to on-task importance slope, versus the same ratio for zipf.
Pre-derived direction: importance attenuates, zipf does not. A null must come with the one-sided
bound and MDE (G4 discipline applies to behaviour too).

**H3 — reader strategies.** Per-reader importance-tracking slopes, with split-half reliability
reported first: if slopes are not reliable (split-half r < 0.3) the typology is abandoned rather
than clustered on noise. If reliable: relate slope to comprehension accuracy and MW rate.

**H4 — neural.** A priori windows from the two closest precedents, not chosen post hoc:
successful encoding in natural reading shows FRP effects at 100–210 ms (N1–P2) and 380–480 ms
(frontal P3) plus theta (Nakano/eNeuro 2018); discourse-level information gain shows a positive
shift (Sci Rep 2020). Existing ROI features: `frp_occ_N1`, `frp_occ_P2`, `frp_cp_N400`,
`frp_front_late`. Screen on the per-fixation table with overlap caveats, then confirm with an
overlap-corrected rERP refit that adds importance and importance×MW as predictors — per
the selection-and-repair analysis, deconvolved betas are the estimate of record, single-trial ones are not.

**H5 — outcome.** Evidence-region encoding predicts answering that item, against a matched
equal-size control region on the same page and a 1000-iteration random-region permutation that
refits the estimator with the evidence span replaced by a random equal-size sentence set.
Localisation test for MW: MW while on the evidence span vs MW elsewhere on the same page.
Item-type control: `single_fact` items should depend on evidence dwell, `negated` items on broad
coverage; failure of that dissociation weakens the whole localisation.

## 6. What would make this a null

Stated now so it cannot be renegotiated later:

* G1 fails → importance is not reliably annotatable; methods null, stop.
* H1 fails within-word-type → importance effects are lexical confounds; report as a cautionary
  null against the LLM-importance-predicts-fixations claim.
* H2 shows equivalent attenuation of importance and lexical channels → MW is a uniform,
  non-selective state change; the "loses the thread" account loses its main positive prediction.
* H5 evidence region indistinguishable from matched control, or observed statistic inside the
  random-region null → the localisation carries no information and no outcome claim is made.

---

## Amendment, logged at gate-check time (before any outcome model was fitted)

**G2 was misspecified and cannot pass as written.** It required "≥ 20,000 word tokens whose type
occurs in both a high- and a low-importance sentence". There are only 10,839 stimulus words in the
whole corpus, so that threshold was impossible by construction — it was written against the wrong
unit. The quantity the analysis actually depends on is the number of *reading observations* in the
within-lemma models. Corrected criterion, ≥ 20,000 observations: **satisfied, 248,379**
(8,758 stimulus tokens × 44 readers). The confound half of G2 passed as written (max |r| = 0.294,
sentence length). The gate is recorded as PASSED ON THE CORRECTED UNIT, with the original wording
left above unedited.
