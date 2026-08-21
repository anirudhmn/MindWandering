# Pooled deconvolutional encoding model — results

All numbers below are in `results/`.

## The solver is exact

`00_validate_solver.py` refits the frozen eight-predictor model through the Toeplitz path for
three readers: correlation 1.0000 with `rerp_betas.npy` on all eight kernels, maximum absolute
difference 0 uV. The fast path is what makes 220 held-out folds affordable; it is not a
different model.

## The model works

`state_tests.json`, at the lambda that maximises on-task D:

| channels | D on-task (uV^2) | t(43) | p | readers > 0 |
|---|---|---|---|---|
| all 64 | +0.00694 | 3.67 | 6.6e-4 | 34/44 |
| occipitotemporal | **+0.01361** | 6.25 | **1.6e-7** | 36/44 |
| centroparietal | **+0.00942** | 5.84 | **6.3e-7** | 37/44 |

A text kernel learned from other readers on other articles predicts this reader's recording on
this article, and does so most strongly at the sensors that carry the frequency response and the
N400.

Negative control, word features permuted within page: D on-task = -0.0014 (13 of 44 readers) and
-0.0019 (5 of 44) across two draws.

Positive control: the nuisance model's mind-wandering kernel reproduces the additive
occipitotemporal offset at **+0.070 uV, t = 3.62, p = 7.8e-4**, against the +0.087 uV the main
text reports.

## It does not buy power, which is the finding

| channel | this model | single-window contrast |
|---|---|---|
| frequency, occipitotemporal | detectable at 80%: **82%** | **43%** |
| surprisal, centroparietal | detectable at 80%: **127%** | **144%** |

Aggregating over 64 sensors and 129 latencies did not rescue the surprisal channel. A held-out
prediction gain is a noisier statistic than a directly estimated coefficient, and that loss
outweighs what aggregation gains. The limit is the signal, not the model.

The state contrast itself is null: paired within-reader difference +28% [-27, +88] of the
on-task effect occipitotemporally (p = .35) and -49% [-140, +38] centroparietally (p = .29).

## The gain is not purely cortical

Refitting with the text block restricted (`--text`):

| text block | all 64 | occipitotemporal | centroparietal | lateral frontal | top sensors |
|---|---|---|---|---|---|
| all five | +0.0076 | +0.0143 | +0.0099 | +0.0132 | F8, O1, PO7, FT8, AF8, Oz |
| **length only** | +0.0060 | +0.0075 | +0.0070 | **+0.0175** | **F8, FT8, AF8**, Pz, O1 |
| frequency and surprisal | +0.0059 | **+0.0112** | +0.0069 | +0.0112 | F8, **O1, PO7**, AF8, Oz |

Word length predicts better at right lateral frontal sensors than occipitotemporally, which is
the signature of residual ocular contamination: length drives saccade amplitude, and an
amplitude nuisance kernel of fixed shape does not remove all of it. Removing length restores the
occipitotemporal sensors to the top. Any predictor correlated with saccade amplitude carries
this risk.

## Language-model representations do not beat three word properties

`lm_layers.json`, 16 components per layer, states residualised on within-article token position:

| layer | components carry | D on-task | readers > 0 | detectable at 80% | gate |
|---|---|---|---|---|---|
| 0 | 56.5% | +0.00578 | 30/44 | 639% | pass |
| 2 | 44.7% | +0.00388 | 30/44 | 6886% | pass |
| **4** | 33.6% | **+0.00675** | 31/44 | **204%** | pass |
| 6 | 28.4% | +0.00485 | 29/44 | 1939% | pass |
| 8 | 26.2% | +0.00367 | 27/44 | 9701% | pass |
| 10 | 26.5% | +0.00243 | 28/44 | 40102% | fail |
| 12 | 98.3% | **-0.00219** | 20/44 | 60931% | fail |

The word-property reference is +0.00694 and the permutation control is -0.00139. The best layer
does not reach the reference, prediction falls with depth, and the final layer falls below the
control. Every layer's detectable change is at least twice its own on-task effect, so no layer
can address retention; the apparent depth trend is carried entirely by the two layers that fail
the gate.

Layer 12's 16 components carrying 98.3% of the variance is the known anisotropy of a final
transformer layer, so its basis is not comparable with the others. Recorded, not repaired.

## Nonlinearity buys nothing

Both arms share a learned rank-8 basis and differ only in the map from word properties to
component amplitudes:

| arm | D on-task (uV^2) | t |
|---|---|---|
| linear | +0.01061 | 5.6 |
| nonlinear | +0.01118 | 4.5 |

Difference +0.00056 uV^2, t = 0.72, **p = .47**, better in 24 of 44 readers. This agrees with the
behavioural result in `roamm/omnibus/`, where a network text stage was three times worse than a
ridge one. At this signal-to-noise the mapping from words to response is linear.
