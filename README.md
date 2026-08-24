# Mind-wandering preserves word-level coupling

Version 1.2. Analysis code and derived tables for the manuscript *Mind-wandering preserves
word-level coupling but produces localized comprehension failure*. This repository holds the
code and the numeric record only; the manuscript itself is not distributed here.

Two simultaneous eye-tracking and EEG datasets are used. The primary sample is
**ROAMM** (OpenNeuro `ds007629`): 44 adults reading five Wikipedia articles page by page,
reporting each lapse as they caught it and marking the span of text it covered, and
answering one multiple-choice question per page. The control sample is **ZuCo 1.0**, which contributes an
instructed shallow-reading condition.

## Reproducing the figures

The derived tables the figures read are in the repository, so this needs no raw data:

```bash
pip install -r requirements.txt
python figures/make_figures.py
```

About five seconds; writes `fig1..fig9` to `figures/out/` as PDF and PNG and reproduces every
number printed in them. The file names still carry the draft numbering; the table below maps
them to the printed numbers.

File names carry the draft numbering, not the printed numbering:

| file | printed as | subject |
|---|---|---|
| `fig1_design` | 1 | design and datasets |
| `fig3_preserved` | 2 | selection, duration, frequency kernel |
| `fig4_measurement` | 3 | the definition-dependence of skipping |
| `fig5_states` | 4 | skimming versus mind-wandering |
| `fig9_comprehension` | 5 | answer-span localisation |
| `fig2_instrument` | S1 | baseline coupling during engaged reading |
| `fig7_index` | S2 | the continuous index of text-driven reading |
| `fig6_gain` | S3 | amplitude rescaling of the fixation response |
| `fig8_changes` | S4 | what does change during mind-wandering |

Updated 2026-08-24, when the manuscript was cut for length. `fig3_preserved` went from six
panels to three, `fig4_measurement` from six to four, and `fig5_states` from five to three; the
panels removed are the repair breakdown, the surprisal kernel, the equivalence forest plot, the
threshold sweep, the off-word-time diagnostic and the two neural paired panels, all of which now
appear as numbers in the text or in the Supplementary Information rather than as panels.
`fig2_instrument` moved from the Results to SI S1. The previous panel layouts are recoverable
from `figures/make_figures.py.pre-trim`.

## Layout

```
figures/        the figure script; writes to figures/out/
roamm/          analysis of the primary dataset
  build/            raw recordings -> per-fixation, per-word and per-page tables
  coupling/         the coupling instrument, the preserved-coupling tests, extended context
  comprehension/    page-level comprehension outcomes and inter-subject alignment
  selection_repair/ selection, repair and duration channels; the skipping audit
  localisation/     semantic importance, answer spans, and the localised MW cost
  topography/       reference-invariant scalp-field tests of the fixation response
  attention_index/  the label-free index of text-driven reading
  omnibus/          the model-based omnibus coupling test; Table 1's omnibus rows and S10
  encoding/         the pooled deconvolutional encoding model; S11
  bridge/           the alignment decomposition against Sun & Jangraw (2026); S11
  southwell/        rebuild of the Southwell et al. (2020) global gaze model; S13
  artifacts/        derived tables shared across stages
zuco/           control-dataset extraction and the instructed-goal contrast
```

Every script is run **from the repository root**, which is where its relative paths resolve:

```bash
python roamm/coupling/lmm_coupling.py
python roamm/selection_repair/scripts/02_g1_selection.py
python roamm/omnibus/scripts/02_state_tests.py --tag real
```

Each analysis stage keeps its own `results/` (machine-readable JSON and CSV, one file per test)
and `artifacts/` (intermediate tables). `PREREGISTRATION.md` and `RESULTS.md`, where present, are
the frozen plan and the full numeric record for that stage; the plans were written before the
corresponding analyses were run.

## Rerunning the analyses

Most analysis scripts run against the derived tables included here. The extraction scripts in
`roamm/build/` and `zuco/` do not: they read the raw recordings, which are not redistributed.

**ROAMM.** Download `ds007629` from OpenNeuro into `data/`. `roamm/build/extract_fixations.py`
and `extract_all_fixations.py` read the synchronised 256 Hz frame (`data/derivatives/features_df.pkl`,
about 47 GB, and needs a machine that can hold a chunk of it); `word_features.py` reads the
stimulus coordinate CSVs under `data/derivatives/stimuli/wiki_stories/`. The comprehension
scripts additionally expect the released trial table and item banks in `reading_data/`
(`trial_level_data.csv` and `*_questions.xlsx`).

**ZuCo.** `zuco/download_*.sh` fetch the MATLAB releases from OSF into `zuco/task*_matlab/`.
`zuco/scripts/parse_frp.py` re-epochs fixation-related potentials from the sentence-continuous
raw data, which is the only way to reach the N400 window; its per-subject `frp_*.npy` outputs are
large and are not included, though the `meta_*.parquet` tables they pair with are. `zuco/RESULTS.md`
is the numeric record for the control sample, including the two gate results the manuscript quotes
that the shipped tables cannot regenerate (the cross-dataset frequency replication and the N400
null, both of which need `frp_*.npy`).

**The omnibus stage** (`roamm/omnibus/`) ships the per-transition read-outs, so
`02_state_tests.py` and `03_robustness.py` reproduce every omnibus number in the manuscript with
no raw data and no refitting. `00_build_transitions.py` rebuilds the candidate sets from the
shared tables; `01_fit_policy.py` refits the twenty held-out models and wants an accelerator,
taking a few minutes per fold. Run it with `--shuffle 1 --tag shuf1` for the negative control.

**The encoding stage** (`roamm/encoding/`) is the one stage whose intermediates are too large to
redistribute: a 12 GB preprocessed recording, a 6 GB residual cache and a 400 MB array of
language-model states. Its `results/` files are included, and `01_cache_eeg.py` through
`06_nonlinearity.py` regenerate them in order from the raw dataset. `00_validate_solver.py`
should be run first: it checks the fast deconvolution against the frozen kernels in
`roamm/artifacts/coupling/rerp_betas.npy`, and everything else depends on that holding.

**The bridge stage** (`roamm/bridge/`) has the same constraint for the same reason.
`bridge_alignment.py` reads the single-trial fixation epochs written by
`roamm/build/extract_frp_epochs.py` and the language-model state array written by
`roamm/encoding/scripts/05a_extract_lm_states.py`, neither of which is redistributed. Its
`results/bridge_alignment.json` is included and holds every number quoted in S11, including all
fourteen layer-by-dimensionality specifications. A full run takes about five minutes on a
many-core machine.

Order within `roamm/build/`: `word_features` → `surprisal_features` (GPT-2) →
`extract_multiscale_surprisal` → `extract_fixations` → `extract_all_fixations` →
`extract_frp_epochs` → `extract_frp_roi` → `build_reading_table` → `build_rerp` →
`extract_dynamics`; `map_subject_ids` → `build_comprehension` → `build_answer_spans_v2` for the
comprehension side. Stage scripts are numbered in their run order.

## What is and is not included

Included: all analysis code behind the manuscript, the preregistration documents, the
machine-readable result files, and the derived tables the figures and analyses need (about
250 MB, mostly `reading_fixations.parquet`, `fixations_frp.parquet`, `all_fixations.parquet`,
`fixations.parquet`, `saccades.parquet`, `rerp_betas.npy`, `attention_index.parquet` and the two
per-transition read-outs of the omnibus stage).

Not included: the raw recordings of either dataset, single-trial EEG epochs, the preprocessed
continuous recording and residual cache of the encoding stage, the language-model state array,
and analyses that are not reported in the manuscript. The encoding and bridge stages are the two
that cannot run from a clean checkout for this reason; their `results/` files are included in
full. `roamm/southwell/` has no such constraint: it reads only `all_fixations.parquet` and
`pages_full.parquet`, both shipped, and reproduces `southwell_replication.json` byte-identically
in about five seconds.

## Changes in version 1.2

Two stages were added, both answering papers that appeared after version 1.1 and both post hoc
with respect to this dataset. Neither changes any result that was already here.

- `roamm/bridge/` decomposes the loss of embedding-to-EEG alignment that Sun & Jangraw (2026)
  report on ROAMM into a response-gain term, a stimulus term and a noise term. The gain term
  carries essentially all of it and matches the rescaling factor `roamm/topography/` estimates
  from different features, so the two papers' apparently opposite conclusions are the same
  measurement seen twice. Manuscript SI S11.
- `roamm/southwell/` rebuilds the global page-level gaze model of comprehension from Southwell
  et al. (2020) on these readers. It replicates (held-out r = 0.338 against their 0.384, 0.362
  and 0.372), which fixes the level at which the manuscript's gaze null holds: gaze predicts how
  well a reader read and not which content they lost. Manuscript Table S1 and SI S13.

`roamm/southwell/` was verified on addition in the same way as everything below: run from a clean
checkout, output compared byte-for-byte against the record here. `roamm/bridge/` cannot be, for
the reason given above, and ships its result file instead.

## What has been checked

Every result file in this repository was regenerated from a fresh checkout and compared against
the version recorded here. All 38 scripts that can run without the raw recordings succeed, and
their outputs agree to floating-point noise; the nine figures regenerate byte-identically. Every
number in the manuscript was then traced to the result file that produces it.

That check predates the 2026-08-24 length trim. The trim changed which panels
`make_figures.py` draws for three figures, so those three PDFs no longer match the ones checked
byte-for-byte then; the data each panel reads is unchanged, and the script still regenerates
reproducibly from a clean checkout.

Five values in the manuscript were stale relative to the current code and were corrected to match
it: three confidence intervals in Table 1, the equivalence-bound column of Table 1, one
significance threshold in the skipping section, and the neural cell of the instructed-goal
contrast. One analysis was wrong rather than stale: in `10_readers.py` the predictor of interest
also appeared in its own covariate list, which made the normal equations singular and the
per-reader frequency slopes unstable between runs. It is fixed, and the affected reliability
figures in `roamm/localisation/RESULTS.md` are updated. No manuscript claim depended on them.

Three analyses cannot be rerun here because they need inputs that are not redistributed:
`roamm/coupling/coupling_equivalence_and_waveforms.py` needs the single-trial epoch array,
and `roamm/selection_repair/scripts/09_reviewer_checks.py`, `roamm/localisation/scripts/08*.py`
and `roamm/comprehension/analyze_evidence_region.py` need the stimulus coordinate files from the
dataset. Their stored outputs are included.

Two notes on provenance:

- `roamm/selection_repair/results/neural_power.json` is a summary assembled from
  `neural_deepdive.json` and `neural_equivalence.json`; the assembling step was not kept as a
  script. Its inputs and the scripts that produce them are here.
- `roamm/encoding/results/lm_layers.json` and `nonlinearity.json` were produced by the run the
  reported numbers come from, rather than by a fresh execution of the scripts as they stand.
  The scripts regenerate them, but only from the intermediates of that stage, which are too
  large to redistribute. Every other result file in `roamm/encoding/` and all of
  `roamm/omnibus/` was regenerated from a clean checkout and agrees with the record here.
- Some linguistic annotations were produced by a language model, which is stated in the
  docstring of each script that did so (`roamm/localisation/scripts/01_annotate_importance.py`,
  `03_annotate_qwen.py`, `04_annotate_evidence.py`). `03_annotate_qwen.py` is the fully
  reproducible one; the others record a frozen annotation that is shipped as a table.

## Citation

See `CITATION.cff`. Please cite the two datasets alongside this repository: ROAMM, OpenNeuro
`ds007629` version 1.3.0 (doi:10.18112/openneuro.ds007629.v1.3.0), and ZuCo 1.0
(Hollenstein et al., *Scientific Data* 5:180291, 2018; https://osf.io/2urht/).

## Licence

Code is released under the MIT Licence (`LICENSE`).
