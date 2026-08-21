# Pooled deconvolutional encoding model — analysis plan

Frozen before any model in this directory was fitted.

## Why

The single-window contrast reported in the main text cannot resolve the centroparietal surprisal
channel: the smallest change it could detect at 80% power is larger than the on-task effect
itself, so it cannot distinguish a preserved response from an abolished one. This stage asks
whether that is a limit of the analysis or of the data, by aggregating the evidence over all 64
sensors and the whole 0 to 500 ms window instead of one region and one window.

## Design

The continuous recording is modelled as a sum of overlapping fixation kernels. Nuisance kernels
(intercept, log fixation duration, fixation order, incoming and outgoing saccade amplitude, page
progress, mind-wandering, and mind-wandering by log duration) are fitted **per reader** on that
reader's other runs, so the additive state change is absorbed before any word property is
examined. The text kernel (Zipf frequency, length, within-sentence surprisal, and the two
extended-context terms) is fitted **pooled across readers, on on-task fixations only**, and
evaluated on a reader-by-article cell that contributed to neither axis of the fit.

Per-reader text kernels were the original plan and were abandoned on evidence: on a three-reader
check the held-out improvement was -0.194, -0.110 and +0.005 uV^2, the kernels making prediction
worse rather than better. Pooling is the more conservative design, since no evaluated cell
contributes to its own kernel on either axis.

## Read-out

Per held-out fixation and channel,

    D = mean over 0 to 500 ms of  resid^2 - (resid - text prediction)^2      [uV^2]

## Gates and tests

- **Gate.** D must exceed zero on held-out on-task fixations, tested across readers.
- **Negative control.** Word features permuted within page must remove it.
- **Positive control.** The nuisance model's mind-wandering kernel must reproduce the additive
  occipitotemporal offset that the main text reports. A null from a machine that cannot recover
  a known effect is not interpretable.
- **State.** The paired within-reader difference in D between states, with the smallest change
  detectable at 80% power, stated **beside the single-window contrast it is meant to improve on**.
  The comparison, not the p value, is the point of this stage.
- **Specificity.** The text block is refitted restricted to word length alone and to the
  remaining properties. Any predictor correlated with saccade amplitude can produce apparent
  coupling that survives amplitude nuisance regressors, and length is such a predictor.
- **Depth.** The text block is replaced by principal components of GPT-2 hidden states at
  successive layers. A layer whose on-task gain does not clear the word-permutation control is
  not interpreted for retention.
- **Nonlinearity.** Both arms share a learned rank-8 spatiotemporal basis and differ only in
  whether the map from word properties to component amplitudes is linear, so any difference
  isolates nonlinearity rather than kernel capacity.

The retention ratio is not the primary statistic here. Several readers have a non-positive
on-task value, which makes a ratio of means unstable; the paired difference is used instead.
