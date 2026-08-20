# Keeping the words, losing the thread

Analysis code and derived tables for the manuscript *Keeping the words, losing the thread:
word-level coupling survives mind-wandering while comprehension fails where the mind was*.

Two simultaneous eye-tracking and EEG datasets are used. The primary sample is
**ROAMM** (OpenNeuro `ds007629`): 44 adults reading five Wikipedia articles page by page,
marking afterwards the spans over which their minds had wandered, and answering one
multiple-choice question per page. The control sample is **ZuCo 1.0**, which contributes an
instructed shallow-reading condition.

The manuscript source is in `paper/`. `paper/main.pdf` is the current build.

## Reproducing the figures

The derived tables the figures read are in the repository, so this needs no raw data:

```bash
pip install -r requirements.txt
python paper/make_figures.py
```

About five seconds; writes `fig1..fig9` to `paper/figs/` as PDF and PNG and reproduces every
number printed in the figures. `paper/main.tex` builds with pdfLaTeX and BibTeX (`latexmk -pdf
main.tex` from `paper/`); it uses no non-standard packages.

File names carry the draft numbering, not the printed numbering:

| file | printed as | subject |
|---|---|---|
| `fig1_design` | 1 | design and datasets |
| `fig2_instrument` | 2 | coupling at all three levels |
| `fig3_preserved` | 3 | selection, repair, duration, brain |
| `fig4_measurement` | 4 | the skipping artefact |
| `fig5_states` | 5 | skimming versus mind-wandering |
| `fig9_comprehension` | 6 | answer-span localisation |
| `fig6_gain` | S1 | amplitude rescaling of the fixation response |
| `fig7_index` | S2 | the continuous index of text-driven reading |
| `fig8_changes` | S3 | what does change during mind-wandering |

## Layout

```
paper/          manuscript, bibliography, figures, and the figure script
roamm/          analysis of the primary dataset
  build/            raw recordings -> per-fixation, per-word and per-page tables
  coupling/         the coupling instrument and the preserved-coupling tests
  comprehension/    page-level comprehension outcomes and inter-subject alignment
  selection_repair/ selection, repair and duration channels; the skipping audit
  localisation/     semantic importance, answer spans, and the localised MW cost
  topography/       reference-invariant scalp-field tests of the fixation response
  attention_index/  the label-free index of text-driven reading
  artifacts/        derived tables shared across stages
zuco/           control-dataset extraction and the instructed-goal contrast
```

Every script is run **from the repository root**, which is where its relative paths resolve:

```bash
python roamm/coupling/lmm_coupling.py
python roamm/selection_repair/scripts/02_g1_selection.py
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
stimulus coordinate CSVs under `data/derivatives/stimuli/wiki_stories/`.

**ZuCo.** `zuco/download_*.sh` fetch the MATLAB releases from OSF into `zuco/task*_matlab/`.
`zuco/scripts/parse_frp.py` re-epochs fixation-related potentials from the sentence-continuous
raw data, which is the only way to reach the N400 window; its per-subject `frp_*.npy` outputs are
large and are not included, though the `meta_*.parquet` tables they pair with are.

Order within `roamm/build/`: `word_features` → `surprisal_features` (GPT-2) →
`extract_fixations` → `extract_frp_epochs` → `extract_frp_roi` → `build_reading_table` →
`build_rerp`; `map_subject_ids` → `build_comprehension` → `build_answer_spans_v2` for the
comprehension side. Stage scripts are numbered in their run order.

## What is and is not included

Included: all analysis code behind the manuscript, the preregistration documents, the
machine-readable result files, and the derived tables the figures need (about 100 MB, mostly
`reading_fixations.parquet`, `fixations_frp.parquet`, `all_fixations.parquet`, `rerp_betas.npy`
and `attention_index.parquet`).

Not included: the raw recordings of either dataset, single-trial EEG epochs, and analyses that
are not reported in the manuscript.

Two notes on provenance:

- `roamm/selection_repair/results/neural_power.json` is a summary assembled from
  `neural_deepdive.json` and `neural_equivalence.json`; the assembling step was not kept as a
  script. Its inputs and the scripts that produce them are here.
- Some linguistic annotations were produced by a language model, which is stated in the
  docstring of each script that did so (`roamm/localisation/scripts/01_annotate_importance.py`,
  `03_annotate_qwen.py`, `04_annotate_evidence.py`). `03_annotate_qwen.py` is the fully
  reproducible one; the others record a frozen annotation that is shipped as a table.

## Citation

The datasets are cited in `paper/refs.bib` as `roammdata` (ROAMM, OpenNeuro ds007629) and
`hollenstein2018` (ZuCo 1.0). Please cite them alongside this repository.

## Licence

Code is released under the MIT Licence (`LICENSE`). The manuscript text and figures in `paper/`
are © the authors.
