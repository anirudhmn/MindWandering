#!/usr/bin/env python3
"""Curated evidence regions for all 50 comprehension items.

v1 scored every sentence on a page against every option automatically. Reviewing all 50
showed the automatic top-1 is right for most items but fails in a specific, systematic way:
prose splits a fact across sentences (the antecedent in one, the consequence in the next),
so items whose answer needs two adjacent sentences get localised to whichever half scored
higher. Those are corrected by hand here and flagged `curated`.

Item typing matters as much as localisation and is recorded explicitly:

  single_fact  one sentence states the answer; the encoding-lesion prediction is sharp
  inferential  the answer is inferred from a small set of sentences, none of which state it
  integrative  the answer requires combining several sentences across the page
  negated      "which is NOT true / EXCEPT" -- the correct response is the option the page
               never supports, so the reader must have encoded the OTHER THREE options'
               sentences. Evidence = union of the true options' sources, and the prediction
               inverts: these items should depend on broad page coverage rather than dwell
               on any one span. That inversion is a built-in control on the whole approach:
               if span localisation were noise, single_fact and negated items could not
               dissociate.

Output: artifacts/comprehension/item_evidence.parquet -- one row per item, carrying the set
of word_keys that constitute its evidence region and a matched control region on the same
page (equal word count, drawn from the sentences no option maps to).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STIM = ROOT / "data" / "derivatives" / "stimuli" / "wiki_stories"
OUT = ROOT / "roamm" / "artifacts" / "comprehension"
RNG = np.random.default_rng(11)

# hand corrections after reading every item against its page: item -> (sentence indices, type)
OVERRIDE = {
    "Pluto_p2": ([6, 7], "single_fact"),        # widow's legal battle delayed the search
    "Pluto_p6": ([4, 5], "single_fact"),        # Neptune mass revision -> need for Planet X vanished
    "The Voynich Manuscript_p7": ([1, 2, 3], "inferential"),   # aim of the dating tests is inferred
    "History of Film_p4": ([0, 1, 2, 3, 4, 5], "integrative"),  # "both films have multiple shots"
    "Prisoners Dilemma_p4": ([0], "inferential"),   # donation-game payoff structure
    "Prisoners Dilemma_p5": ([4, 6, 7, 8], "single_fact"),      # backward induction
    "Prisoners Dilemma_p8": ([6, 7], "single_fact"),           # one-shot vs iterated optimum
}

spans = pd.read_parquet(OUT / "answer_spans.parquet")
rows = []

for item, g in spans.groupby("item", sort=False):
    stem = g["story_phys"].iloc[0]
    page = int(g["page"].iloc[0])
    coords = pd.read_csv(STIM / f"{stem}_coordinates.csv")
    pg = coords[coords["page"] == page]
    sent_ids = list(dict.fromkeys(pg["sentence_id"]))
    keys_by_sent = {s: pg.loc[pg["sentence_id"] == s, "word_key"].tolist() for s in sent_ids}
    negated = bool(g["negated_item"].iloc[0])

    if item in OVERRIDE:
        idx, itype = OVERRIDE[item]
        curated = True
    elif negated:
        # evidence = the sentences supporting the three TRUE options
        true_opts = g[~g["is_correct_option"]]
        idx = sorted({sent_ids.index(s) for s in true_opts["sentence_id"] if s in sent_ids})
        itype, curated = "negated", False
    else:
        s = g[g["is_correct_option"]].iloc[0]["sentence_id"]
        idx = [sent_ids.index(s)] if s in sent_ids else []
        itype, curated = "single_fact", False

    ev_keys = [k for i in idx for k in keys_by_sent[sent_ids[i]]]
    # matched control: sentences on the same page that no option maps to, sampled to the
    # same word count, so "did you read the evidence" is not just "did you read anything"
    used = {sent_ids.index(s) for s in g["sentence_id"] if s in sent_ids} | set(idx)
    free = [i for i in range(len(sent_ids)) if i not in used]
    RNG.shuffle(free)
    ctrl_keys: list[str] = []
    for i in free:
        if len(ctrl_keys) >= len(ev_keys):
            break
        ctrl_keys += keys_by_sent[sent_ids[i]]
    ctrl_keys = ctrl_keys[: len(ev_keys)]

    rows.append({
        "item": item, "reading": g["reading"].iloc[0], "story_phys": stem, "page": page,
        "item_type": itype, "negated": negated, "curated": curated,
        "n_sentences_page": len(sent_ids), "n_evidence_sentences": len(idx),
        "evidence_sentence_idx": idx,
        "evidence_word_keys": ev_keys, "n_evidence_words": len(ev_keys),
        "control_word_keys": ctrl_keys, "n_control_words": len(ctrl_keys),
        "n_words_page": int(len(pg)),
        "evidence_frac_of_page": len(ev_keys) / max(len(pg), 1),
        "min_margin_correct": float(g[g["is_correct_option"]]["margin"].min()),
    })

E = pd.DataFrame(rows)
E.to_parquet(OUT / "item_evidence.parquet", index=False)

print(E.groupby("item_type").agg(
    n=("item", "size"),
    med_evidence_words=("n_evidence_words", "median"),
    med_frac_of_page=("evidence_frac_of_page", "median"),
    med_control_words=("n_control_words", "median"),
).round(3))
print()
print(f"items: {len(E)}   hand-curated: {int(E['curated'].sum())}")
print(f"evidence region is {E['evidence_frac_of_page'].median():.1%} of the page (median) "
      f"-> {1/E['evidence_frac_of_page'].median():.1f}x targeting gain over page averaging")
print(f"items with an equal-size control region: {(E['n_control_words'] == E['n_evidence_words']).sum()}/{len(E)}")
print(f"wrote {OUT/'item_evidence.parquet'}")
