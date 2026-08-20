# Results — semantic importance, mind-wandering, and comprehension

**Headline.** Mind-wandering's cost to reading comprehension is **informationally localised**.
A lapse anywhere on a page costs 4 accuracy points; a lapse that happens while the eyes are on
the words that answer the question costs 27. The effect beats all 1000 random equal-size regions
on the same page and scales continuously with how much of the answer the lapse covered. Yet
*reading* the answer span predicts nothing beyond reading the page at all — its coverage sits at
the 38th percentile of that same random-region null. **Where the eyes went is unremarkable; where
the mind was, at which words, is decisive.**

Secondary: readers do track intrinsic semantic importance during free reading, but only in
**refixation** and about an order of magnitude more weakly than the uncontrolled literature
implies; the evoked response to importance is a well-bounded null; and the pre-derived
importance-×-MW dissociation is **not testable in this dataset** — reported with its MDE rather
than as a null.

Preregistration (with its one logged amendment):
`PREREGISTRATION.md`. All numbers below are in `results/*.json`.

---

## 1. Measurement: three importance annotations, honestly compared

529 sentences across 50 pages, rated 1–5 for text-based importance (McCrudden & Schraw's
construct) **question-blind**, plus a question-anchored evidence span per item annotated
afterwards.

| pair | Spearman ρ | within-page ρ |
|---|---|---|
| Claude Opus 5 × Qwen2.5-1.5B-Instruct | +0.336 | **+0.334** (p=2.9e-15) |
| Claude Opus 5 × GPT-2 information centrality | +0.265 | +0.245 (p=1.6e-8) |
| Qwen2.5 × GPT-2 centrality | +0.064 | +0.081 (p=0.065, n.s.) |
| Claude Opus 5 × "would be in a 2–3 sentence summary" | +0.775 | +0.703 |

Quadratic-weighted κ between the two LLM raters = **0.290**. So importance is annotatable but
only moderately reproducible across models — and the two judgement-based raters agree with each
other far better than either agrees with GPT-2's leave-one-sentence-out predictive centrality
(which is close to independent of the smaller model's judgements). *Judged importance and
LM information gain are not the same construct.* G1 passed on the predeclared threshold (≥0.30).

**Evidence spans.** 50 items localised to a median 20.9% of their page (a 4.8× targeting gain
over page averaging). Against the previous session's automatic embedding+IDF localisation: mean
Jaccard **0.825**, any-overlap 98%, exact match 64% — one complete disagreement. 34 items are
`single_fact`, 12 `negated`, 2 `inferential`, 2 `integrative`. One item
(`Prisoners Dilemma_p8`) is **mis-keyed** — the page states the opposite of the keyed option and
states a different option verbatim.

**Importance is not a lexical variable in disguise** (G2). Word-level Spearman ρ with importance:
zipf +0.013, length −0.009, surprisal −0.054. The only real correlate is sentence length
(ρ=+0.295). 80.8% of tokens are of a type that also occurs in a differently-rated sentence, giving
248,379 within-lemma reading observations — so every effect below can be estimated holding word
identity fixed. (G2's token-count threshold was misspecified against the wrong unit and is
re-recorded in the amendment; the confound half passed as written.)

---

## 2. H1 — readers do prioritise important sentences, in re-inspection only

Per-reader slopes, page + **lemma** fixed effects, controlling zipf/length/surprisal, word
position in sentence, sentence position on page, sentence length, line position and line-edge
status; 10k bootstrap over 44 readers.

| measure | effect per SD importance | reader p | stimulus-permutation p |
|---|---|---|---|
| **P(refixation)** | **+0.0052** [+0.0019,+0.0087], 30/44 | **0.005** | **0.018** (98.3rd pct) |
| log first-pass gaze duration | +0.0051 [+0.0011,+0.0092], 30/44 | 0.017 | 0.104 (89.7th pct) |
| first fixation duration | +0.0001 [−0.0023,+0.0025] | 0.964 | — |
| P(regression out) | −0.0003 [−0.0028,+0.0020] | 0.793 | — |

The stimulus-side permutation reassigns the ratings among the sentences of the *same page* (1000
refits), preserving each page's rating multiset, sentence lengths, positions, layout and word
identities. **It matters a great deal:** gaze duration, which looked significant at reader level,
falls to p=0.10 — importance varies over only 529 sentences, and reader-level bootstrapping treats
readers as the unit, so it overstates the evidence for any stimulus-level predictor. Only
refixation survives both.

Convergent measures on gaze duration all point the same way (pooled within-lemma): Qwen
β=+0.0067 p=0.005, GPT-2 centrality β=+0.0107 p=7.5e-13, summary-membership β=+0.0100 p=0.008.

**Two cautions that matter for the literature.** (i) The *raw* contrast is nil — 342.5 ms on
importance-5 words vs 344.9 ms on importance-≤3 words (−0.7%). The positive effect appears only
after control, because important sentences are much longer (30.5 vs 18.5 words) and sit earlier and
higher on the page, and those factors shorten fixations. (ii) The published uncontrolled result
(ZuCo, +0.291 fixations on LLM-labelled "important" nodes) is therefore not a safe estimate of
endogenous importance tracking; here the controlled effect is roughly an order of magnitude
smaller and confined to refixation. **Importance is registered after lexical access, not in
first-glance processing** — it changes whether the eyes come back, not how long they first stay.

---

## 3. H2 — the pre-derived dissociation is UNTESTABLE here (not a null)

The prediction from the selection-and-repair analysis was sharp: if MW makes readers "lose the thread, not the words",
the discourse-level channel should decouple while the lexical channels do not. The answer is that
this dataset cannot adjudicate it, and the reason is worth recording.

Pooled word-instance × subject FE, subject-clustered SE, expressed as a percentage of each
channel's own on-task base effect:

| channel | duration: MW × channel | MDE at 80% power | selection (skipping) | MDE |
|---|---|---|---|---|
| **importance** | **−164% of base** [−412,+83] | **354%** | −239% of base | **1103%** |
| zipf | +5% [+23,−13] | 26% | −9% | 39% |
| length | +19% [+4,+34], p=0.013 | 21% | +7% | 17% |
| surprisal | +3% [−20,+25] | 33% | +63%, p=0.015 | 72% |

The importance point estimate is in the predicted direction and large, but **the design cannot
detect even complete abolition of the importance channel** (MDE 354% and 1103% of base). Per the
preregistration's G4 discipline this is reported as uninformative, not as support and not as a
null. The lexical channels, by contrast, are well powered and **preserved or slightly enhanced**,
replicating the selection-and-repair analysis (Somers' D retention: zipf 136%, length 135%, surprisal 143%; additive MW
gaze shift +0.134, p=1.2e-25, cf. +0.103 there).

**A trap worth publishing.** The unadjusted Somers' D analysis *did* produce the predicted
dissociation — importance→skip selectivity D −0.022 on-task → +0.012 during MW, Δ=+0.033,
p=0.006, 22/31 readers, an apparent reversal — and it is an artefact:

| specification | Δ D (MW − on-task) | p |
|---|---|---|
| raw importance | +0.0333 | 0.006 |
| importance residualised on lexical + layout covariates | **+0.0063** | **0.64** |
| line-interior steps only (raw) | −0.0082 | 0.37 |
| line-interior + residualised | −0.0208 | 0.043 *(opposite sign)* |

Importance correlates with sentence length and line position, and MW changes the *structural*
composition of skipping (steps of >4 words, i.e. return sweeps, fall from 27.7% to 17.8%). Once
importance is orthogonalised the effect is gone, and the one surviving specification has the
opposite sign. Any "MW decouples semantics" claim built on unadjusted selectivity measures should
be treated as unproven.

---

## 4. H4 — no evoked signature of importance, with a bound that makes that informative

397,238 artifact-clean fixation-related epochs, 44 readers. A-priori ROIs and windows fixed from
the two closest precedents (FRP subsequent-memory at N1–P2 and frontal-late; discourse information
gain as a positive shift).

| ROI | µV per SD importance (pooled, within-lemma) | p | MDE at 80% |
|---|---|---|---|
| frontal late | +0.0121 | 0.071 | 0.0188 |
| occipital N1 | −0.0067 | 0.389 | 0.0217 |
| occipital P2 | +0.0026 | 0.795 | 0.0279 |
| central N400 | −0.0017 | 0.830 | 0.0224 |

Per-reader Holm-adjusted p = 1.0 for all four; none of Qwen, centrality or summary-membership does
better. The bound is the point: **no importance-evoked effect larger than ~0.02 µV/SD**, which is
smaller than half the N400 surprisal effect (0.038 µV/SD) measured in these same epochs. This is a
genuinely informative null, unlike the importance×MW interaction (MDE 508–2326% of base —
uninformative, including a nominal occipital-P2 result that must not be read as an effect).

One hint, explicitly not a result: on the answer span, later-correct trials show a more negative
early occipital response than later-wrong ones (occ_N1 −0.067 µV, p=0.029, 27/44 readers negative;
occ_P2 −0.057 µV, p=0.063). It is below its own MDE (0.083 µV), uncorrected across four ROIs, and
would need confirmation. **Caveat on all of §4:** these are window means, not overlap-corrected.
Because importance is a sentence-level property, adjacent-fixation contamination is *correlated
with the predictor*. The deconvolved rERP refit was pre-declared as the confirmation step and was
not run, because nothing survived the screen to confirm.

**Follow-up.** The overlap-corrected page-wide lead dies
(centroparietal 220–300 ms implied correct-minus-wrong −0.011 µV, p=.436).
The answer-span estimate itself reproduces almost exactly (occipital 150–220 ms
−0.068 µV, p=.045), but the preregistered evidence-minus-matched-control
contrast does not pass (−0.073 µV, p=.102; Holm p=.203) and is underpowered at
that scale. It remains a suggestive lead, not a neural result.

---

## 5. H5 — the localised-MW result

### 5a. What fails: localising *reading*

| test | result |
|---|---|
| T1 evidence coverage → correct | +0.033, p=0.016 |
| T1 matched control region coverage → correct | +0.025, p=0.047 |
| T1 difference | +0.008, p=0.65 |
| **T2 random-region permutation** | observed +0.033 vs null +0.0365 ± 0.0119 → **p=0.62, 38th pct** |
| T4 item-type dissociation (`single_fact` vs `negated`) | interaction +0.0007, p=0.98 |
| T6 importance-weighted dwell → correct | −0.011, p=0.57 |

By the preregistration's own kill criterion, **the reading-localisation claim is dead**: the
evidence span is no better than a matched control region, no better than random equal-size regions,
and the item-type dissociation that would have validated the spans is absent. (T3's "lesion" model
flips sign because conditioning on page coverage makes it a collider; not interpretable.) What
predicts comprehension in reading terms remains what the selection-and-repair analysis found: how much of the page you
read at all.

### 5b. What holds: localising *attention*

The identical span, with MW instead of coverage as the predictor — subject and item fixed effects,
MW-elsewhere-on-the-page and page coverage partialled out, reader-clustered SE:

* **MW on the evidence span: −0.073** [−0.092,−0.053], p=1.7e-13
* MW elsewhere on the same page: −0.035 [−0.073,+0.002], p=0.064
* **Random-region control: observed −0.0613 vs null −0.0212 ± 0.0118 — more extreme than all
  1000 permutations (0th percentile, p=0.001).** The predeclared matched control region gives
  −0.0017, i.e. nothing.

Every stress test holds:

| test | result |
|---|---|
| **R1** restricted to trials that already contain MW somewhere (n=617) | MW-on-answer **−0.098**, p=3.3e-5; MW-elsewhere −0.059, p=0.020 |
| accuracy gradient | no MW anywhere **.680** (n=1187) → MW on page but not on the answer **.640** (n=311) → MW on the answer **.405** (n=306) |
| **R2** adding evidence coverage, fixation count and dwell | −0.053, p=4.6e-6 — not "they didn't read it" |
| **R3** overlap gradient across the 1000 random regions | slope **−0.072** per unit overlap, r=−0.24, **p=1.5e-14**; at zero overlap the statistic is −0.004 |
| **R4** what it does to the response | correctness −0.061 (p=4.3e-8); "I am not sure" +0.021 (p=0.099); among *answered* trials −0.063 (p=1.5e-6) |
| **R6** by item type | `single_fact` −0.056 (p=7.9e-5); `negated` −0.076 (p=0.014) |
| sensitivity: 39 annotator-unexposed, correctly-keyed items | −0.0745, p=1.5e-9 |

R3 is the strongest single piece of evidence: because every random region is the *same size* as
the true evidence span, region size and hence measurement noise are held fixed, and the predictive
strength still rises linearly with informational overlap with the answer. R4 says the reader is
mostly left with a *wrong* answer rather than a known gap — the lapse damages the representation
without reliably flagging it.

**Identification.** This is a within-reader, within-item contrast with the reader's overall MW
level partialled out, so the remaining variance is *where on the page the lapse fell* — close to
arbitrary given that the reader lapsed at all, which is what supports reading it causally. It is
not experimentally randomised, and lapses are slightly more frequent on evidence spans than
elsewhere (0.086 vs 0.077), so "quasi-experimental", not "experimental".

### 5b-bis. Independent convergence, and what this says about the answerability floor

Two cross-checks against a parallel line of work on the same item bank, both of which
land in the same place:

* **The reading-localisation null replicates independently.** That work localised all 50 items
  with its own validated spans and found the true evidence region at the **39th percentile** of
  1000 random equal-size same-page regions (p=0.61). The independent annotation here gives
  the **38th percentile, p=0.62**. Two separate span annotations, the same null — the failure is a
  fact about reading, not about either annotation. Its withdrawal of the "never fixated the answer"
  lesion contrast as unidentifiable also matches T3's collider behaviour here.
* **This analysis supplies the missing human floor.** That work's headline caution is an
  answerability floor: a no-passage Qwen2.5-1.5B scores **.558** against humans at .618, implying
  only ~6–10 points of reading-dependent headroom. The MW-on-answer condition is the human analogue
  of "did not encode the relevant passage", and it sits at **.405** — far below the model's .558 and
  only 15 points above chance. So the *machine* floor substantially overstates the *human* floor,
  and the human reading-dependent range is roughly .41 → .68, about **27 points, not 6–10**. The
  LLM exploits option plausibility that readers do not. (Caveat: these readers did read the rest of
  the page and may have partially encoded the answer, so .405 is an estimate of "read the page but
  lapsed over the answer", not literally "no passage".) Either way the localised effect itself is
  identified *within item*, so item-level answerability is absorbed and cannot drive it.

R4 sharpens this: at .405 readers do worse than an uninformed guesser would with good option
heuristics, and they mostly commit to a wrong option rather than "I am not sure". A lapse over the
answer does not just fail to write the memory — it leaves something wrong behind, unflagged.

### 5c. Reader level (H3): a trial-level phenomenon, not a trait

Per-reader importance-tracking slopes are **not reliable** (split-half by story: refixation
r=−0.18, gaze r=−0.32; Spearman-Brown negative), so the reader typology was abandoned as
preregistered. For calibration, the *lexical* slopes are barely reliable either (zipf→refixation
SB = 0.31), so this dataset does not support individual differences in coupling slopes at all.

* Within-reader cost of a lapse landing on the answer: **+0.239** accuracy points
  [+0.171,+0.304], p=7.8e-8, 28/34 readers.
* Across readers: MW rate → accuracy r=−0.32 (p=0.033), but *where* the lapses land does **not**
  predict a reader's accuracy (r=−0.08, p=0.60), and in a joint model neither term survives.

Which is what should be expected — whether a lapse coincides with the answer sentence is close to
chance for a given reader, so it is a per-trial event, not a stable characteristic. "Who mind-wanders
on the important parts" is not a person, it is a trial.

---

## 6. What this changes

1. **It converts claim C0 from a correlation into a content-specific loss.** MW predicting item
   failure (−5.9 points) decomposes into a small global component (−4 points for lapsing anywhere)
   and a large localised one (−27 points when the lapse covers the answer). This is the
   informational lesion that a **surprisal-based dependency gate could not test** — that approach
   fails because ROAMM's encyclopedic texts have too few tight local dependencies. The comprehension items supply the readout that language-model surprisal could not.
2. **It explains why page-level analyses keep coming up empty.** Page-averaged physiology
   finds nothing predicting comprehension because the signal is
   *informationally localised*: the same measure over the right ~31 words is decisive while over
   217 words it vanishes. It also predicts which future analyses will fail — anything page-averaged.
3. **It dissociates the eye from the mind on the outcome side.** Fixating the answer is worth
   nothing beyond general coverage; being on-task at that moment is worth 27 points. Eye-movement
   markers of comprehension should therefore be expected to be weak, and attention-state markers
   strong, which is the opposite of the usual methodological bet.
4. **Cautionary methodology:** (a) reader-level bootstrap
   overstates evidence for stimulus-level predictors — the stimulus-side permutation moved gaze
   duration from p=0.017 to p=0.10, and any stimulus-level claim should be re-checked this way; (b) unadjusted rank-selectivity measures manufactured a clean-looking MW ×
   semantics dissociation that vanished on residualisation; (c) uncontrolled LLM-importance →
   fixation differences are largely sentence-length and layout confounds.

## 7. What confirmation would require

The localised-MW result is strong internally but rests on one dataset, retrospective self-report,
and one item per page. Confirmation needs: independent data (ZuCo has no comparable
per-page items — this needs a new collection); probe-caught rather than retrospective MW so lapse
timing is not reconstructed; several items per page so the same lapse can be scored against hit and
missed content within a trial; and ideally gaze-contingent manipulation of *when* content is
presented relative to a detected lapse, which would turn the quasi-experiment into an experiment.
The neural side needs either many more trials or a stronger importance manipulation — the bound
here (0.02 µV/SD) says the effect, if any, is below half an N400 surprisal effect.
