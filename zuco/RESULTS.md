# Control-sample results — coupling under an instructed shallow reading goal (ZuCo 1.0)

12 subjects, NR (deep, 300 sent) vs TSR (shallow relation-search, 407 sent), 47 shared sentences.
Deconvolved rERP (overlap-corrected, unfold-style) + behavioral coupling. Gate-first; the plan is in the manuscript repository alongside the ROAMM preregistrations.

## GATE A0 — coupling capability / cross-dataset replication
- **A0a Frequency FRP: PASS (replicated).** Deconvolved zipf kernel, occipital, 150-290 ms:
  +0.139 uV/SD, t(11)=3.11 p=0.010; **mass-univariate sign-flip cluster p=0.040** (peak E50 178 ms).
  Clean occipital topography. This is a genuine cross-dataset/independent-lab (EGI-128 vs BioSemi-64,
  Zurich vs US) replication of ROAMM's frequency-FRP capability result.
- **A0b Surprisal N400: neural SNR-FLOORED (informative partial).** A-priori centroparietal 300-450 ms
  null (+0.004, p=0.82); mass-univariate cluster p=0.22 (focal t=6.1 hint at E41/400 ms but no reliable
  cluster). Converges with the ROAMM item-level RSA result (reliable FRP item geometry is lexical and
  visual rather than deep-semantic) and is expected: ZuCo reads ISOLATED sentences (weaker discourse-driven semantic prediction
  than ROAMM's connected stories) with 12 vs 44 subjects. The naive fixation-locked N400 is unrecoverable
  due to overlap (median next-fix 206 ms; surprising words fixated longer -> overlap covaries w/ surprisal)
  -> rERP required, and even then no reliable N400.
- **A0c Behavior: PASS.** log-reading-time tracks zipf(-) and surprisal(+); bivariate all p<1e-3,
  strongest in TRT (surprisal +0.176 t=13.5 p=3e-8; zipf -0.200 t=-12.5 p=8e-8). Readers clearly process
  both frequency and surprise.

## GATE A1 — depth manipulation is real: PASS (decisive)
TSR shallower than NR on EVERY eye metric (paired, 12 subj): FFD -10.9 ms (p=1.6e-3), GD -30.9 (1.3e-4),
TRT -79.4 ms/-20% (1.5e-3), nFix -0.27 (2.9e-3), **words-fixated/sentence -4.35/-30% (9e-5)** — readers
skim, skipping ~30% of words. Clean, robust.

## GATE A2 — the dissociation
**Predeclared hypothesis (selective SEMANTIC decoupling, lexical preserved): NOT SUPPORTED.**
The lexical-vs-semantic double dissociation appeared strong in MULTIPLE regression (zipf atten 76% vs
surprisal 25%, p=0.021) but this was a **collinearity artifact** (zipf/surprisal r=-0.62). BIVARIATE and
on the 47 SHARED sentences, both couplings attenuate ~EQUALLY (~33%), dissociation only marginal (p=0.07).
No reliable selectivity -> do not claim.

**Emergent finding (EXPLORATORY, robust, needs independent confirmation): a BEHAVIORAL-vs-NEURAL
dissociation that separates task-shallow reading from spontaneous MW.**
- **Behavioral coupling MULTIPLICATIVELY attenuated in TSR:** frequency and surprisal -> logTRT slopes
  both drop ~33% (zipf -0.200->-0.122; surprisal +0.175->+0.108; both p<1e-3), survives outcome-
  standardization (correlation 33%/32% drop) and the 47 shared-sentence matched-word control (34%/29%,
  p=0.002/0.011). NOT range-restriction (zipf SD among fixated words NR 1.47 vs TSR 1.42). PLUS an
  ADDITIVE global speedup (intercept -0.167 logTRT, p=3e-4). So shallow reading = multiplicative
  attenuation of behavioral coupling + additive speedup.
- **Neural word-processing PRESERVED in TSR:** frequency FRP (occipital 150-290 ms) unchanged, if anything
  larger (NR +0.139 -> TSR +0.193, p=0.23; 58% of subjects up). The brain still lexically registers each
  fixated word identically.

## Synthesis (cross-dataset landmark framing — exploratory)
Two forms of reduced engagement, two signatures:
- **Spontaneous MW (ROAMM):** purely ADDITIVE — uniform slowing; behavioral AND neural coupling slopes
  intact (freq×MW p=0.78, surprisal×MW p=0.28 n.s.).
- **Task-induced shallow reading (ZuCo):** MULTIPLICATIVE attenuation of the BEHAVIORAL (oculomotor)
  word->reading-time coupling (~33%, the eyes stop tracking word properties) + additive speedup, while
  the NEURAL word->brain coupling is PRESERVED.

**Invariant across both datasets and both states: the word->brain neural coupling (perceptual/lexical
registration). What changes is the eye-movement control policy** — spontaneous MW leaves it intact
(engaged disengagement), whereas goal-directed skimming reconfigures it (oculomotor decoupling from word
properties, in service of the semantic search goal). This resolves ROAMM's open additive-vs-multiplicative
question by showing BOTH regimes exist, determined by the TYPE of disengagement — not a property of
disengagement per se. Figure: figures/partA_synthesis.png.

## Honesty / limits
- 12 subjects; the MW comparison is cross-dataset (ROAMM values from prior analysis, not re-run here).
- The predeclared selective dissociation failed; the behavioral/neural dissociation is post-hoc/
  exploratory and needs a preregistered confirmation.
- Neural N400 absent -> the "semantic" side of the story rests on BEHAVIOR only; the neural claim is
  about the (lexical) frequency FRP.
