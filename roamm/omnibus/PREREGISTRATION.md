# Omnibus model-based coupling test — analysis plan

Frozen before any model in this directory was fitted.

## Why

The selection, repair and duration tests each fix a decision and a word property in advance.
That leaves open the objection that the measures which do not move are the ones we chose. This
stage replaces the choice with a model: a network is given the whole text and is free to use any
of it, and the question becomes whether *anything* it can extract is used differently while the
mind is wandering.

## Design

Unit is one fixation-to-fixation transition with both words mapped inside a page's genuine
reading interval. The candidate set for each transition is the words at page positions
pos-20 .. pos+20.

Two heads:

- **target** scores every candidate, trained with a softmax over the set
- **duration** predicts log fixation duration

Each head receives a **geometry** block and, in the compared model, a **text** block.
Geometry carries page progress, line, position in line, distance to the line end, gaze position
and its offset from the word centre, incoming saccade amplitude, previous fixation duration,
first-pass status, ordinal position in the run, and the number of times this reader has already
fixated the candidate on this page. Text carries Zipf frequency, length, within-sentence
surprisal and the two extended-context terms.

Everything except the text block is in the baseline. A model of this kind will find page
geometry before it finds anything cognitive, so geometry is never in the ablated block.

Both heads are trained on **on-task transitions only** and evaluated on five article folds
crossed with four reader groups, so each of the 20 fitted models is tested on transitions from
an article and from readers it never saw.

## Read-out

Per transition, paired across the two models on the same row:

    target    log2 p_text(true) - log2 p_no text(true)                       [bits]
    duration  r^2 - (r - text prediction)^2, r the geometry residual         [log-ms^2]

## Gates and tests

- **G1, gate.** The text block must buy more than zero on held-out on-task transitions, tested
  across readers. If it does not, there is no instrument and nothing downstream is interpreted.
- **G2, negative control.** Refit with word features permuted within page, which preserves
  layout, event timing, the reader's own movements and the marginal distribution of every
  predictor while destroying the binding between a word and its position. The read-out must
  collapse. If it does not, the measure is geometry under another name.
- **T1.** Retention, D(mind-wandering) / D(on-task), with a reader bootstrap, a one-sided
  attenuation bound and the smallest change detectable at 80% power. Reporting the bound and
  the detectable change is required, not the p value alone.
- **T2, ladder.** The state coefficient with reader, launch-line, word and reader-by-page fixed
  effects absorbed in turn, reader-clustered standard errors. Launch location is the **line**:
  (article, page, line, position in line) is a bijection with word identity, so using it would
  absorb exactly what the word fixed effect absorbs and the two rungs would not be distinct.
- **T3, permutation.** Each reader's mind-wandering labels are replaced by another reader's,
  transferred by article, page and word position. This preserves episode length, contiguity and
  where in the text episodes fall, and breaks only the link between a reader's own lapses and
  their own eye movements.
- **Placebo.** The known additive lengthening of fixations must come through the identical
  ladder. A null from a pipeline that cannot recover an effect known to be real is not
  interpretable.

A state claim must clear **both** T2 and T3. Either alone is insufficient: elsewhere in this
programme one candidate effect passed the permutation and was removed by a location absorber,
and another passed the location ladder and was removed by the permutation.

## Robustness

The per-transition read-out is heavy-tailed and retention is a ratio of means, so retention is
re-derived under trimming, winsorising, a median estimator, saccade-length and
fixation-duration restrictions, and episode interiors. Retention is also decomposed by kind of
movement, with the within-reader difference of differences as the test, because a difference in
ratios between movement kinds can be a difference in their denominators.
