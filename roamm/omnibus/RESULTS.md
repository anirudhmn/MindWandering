# Omnibus model-based coupling test — results

All numbers below are in `results/`. `state_tests_real.json` and `state_tests_shuf1.json` come
from `02_state_tests.py`; `robustness.json` from `03_robustness.py`.

392,766 transitions, 44 readers, 20 doubly held-out folds. 0.87% of transitions have their
target outside the +-20-word candidate window and are dropped; within the window the true target
is present at its index on every retained transition.

## G1, the gate

| read-out | D on-task | 95% CI | t(43) | p | readers > 0 |
|---|---|---|---|---|---|
| target choice | **+0.1434 bits** | [+0.1308, +0.1557] | 22.57 | 1.8e-25 | **44/44** |
| duration, ridge text stage | +0.00095 | [+0.00073, +0.00118] | 8.20 | 2.5e-10 | 41/44 |
| duration, network text stage | +0.00031 | [+0.00002, +0.00060] | 2.11 | .041 | 24/44 |

The geometry-only target model sits at 2.70 bits of the 5.36 available from a 41-way candidate
set, so the text block removes about 5% of the remaining uncertainty, or equivalently raises the
probability the model places on the word actually fixated next by about a tenth.

The network text stage on the duration residual is three times worse than the ridge one and
positive in barely half the readers, so the text-to-duration mapping is treated as linear.

## G2, the negative control

Word features permuted within page (`--shuffle 1`), everything else untouched:

| read-out | real | permuted | t | readers > 0 |
|---|---|---|---|---|
| target choice | +0.1379 | **-0.0127** | -9.17 | **4/44** (against 44/44) |
| duration | +0.00096 | -0.000026 | -2.97 | 14/44 (against 41/44) |
| duration, network | +0.00031 | -0.00096 | -14.34 | 1/44 (against 24/44) |

Every read-out goes to zero or reliably below it. Adding scrambled word features makes held-out
prediction worse than not adding them, which is what an ablation on noise has to do.

## T1, retention

| read-out | retention | 95% CI | one-sided lower | detectable at 80% |
|---|---|---|---|---|
| target choice | **0.911** | [0.822, 0.995] | 0.835 | **12%** |
| duration | 1.071 | [0.728, 1.424] | 0.787 | 49% |

## T2, the ladder, and T3, the permutation

Target choice, reader-clustered standard errors:

| rung | beta | SE | p |
|---|---|---|---|
| reader | -0.01506 | 0.0043 | 4.6e-4 |
| + launch line | -0.01884 | 0.0047 | 5.7e-5 |
| + word | -0.02019 | 0.0044 | 4.7e-6 |
| + reader by page | -0.01345 | 0.0057 | .019 |

Duration is null at every rung (p = .24 to .82).

Cross-reader label swap, 2,000 draws:

| read-out | observed | null | z | p |
|---|---|---|---|---|
| target choice | -0.01282 | +0.00171 +- 0.00791 | -1.84 | **.067** |
| duration | +0.00007 | +0.00005 +- 0.00030 | +0.04 | .965 |

**The target-choice shortfall clears the ladder and does not clear the permutation.** Under the
rule that a state claim must clear both, it is a lead and not a result. Note the two ways a
permutation can fail: here the null sits at essentially zero, so the shortfall is not being
reproduced by the null, it is simply not far enough out given the spread of a 44-reader,
7%-prevalence design. That is an argument about power, not about artefact.

## Placebo

The known additive lengthening of fixations through the identical ladder: +4.19%, +4.08%,
+3.86%, +2.86%, all p < 1e-9. It attenuates at the reader-by-page rung in the same proportion as
the effect of interest, which is what identifies that rung as over-control rather than as a
confound being removed.

## The extra dwell is not aimed at anything in the text

Taking the residual left after the geometry model and the on-task text model, and asking whether
any word property predicts the additional dwell during mind-wandering, cross-validated by
reader: held-out R^2 = **-0.0048** on mind-wandering transitions (n = 28,450) and -0.0030 on
on-task transitions (n = 364,316). Negative, so nothing.

## Robustness

Retention never leaves 0.88 to 0.97 across eleven exclusions and two robust estimators; the most
outlier-resistant estimator, the per-reader median, gives the smallest shortfall at 0.942
[0.878, 1.005]. See `results/robustness.json` for the table.

Retention looks higher for forward saccades (0.965) than for regressions and refixations
(0.899), but that is a difference in denominators. In bits the shortfall is -0.0121 for forward
saccades and -0.0110 for the rest, and the within-reader difference of differences is +0.0012
[-0.026, +0.029], t = 0.08, p = .935, with 20 of 40 readers on each side. The shortfall is
uniform across kinds of movement.
