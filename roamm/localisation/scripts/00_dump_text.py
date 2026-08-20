#!/usr/bin/env python3
"""Dump every page of every story as a numbered sentence inventory for annotation.

Sentence indices here are the annotation key: `story|page|sent_idx`, where sent_idx is the
position of the sentence in reading order on that page (the same ordering
build_answer_spans_v2 used, `dict.fromkeys(sentence_id)`), so annotations join back to
word_keys without any text matching.
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STIM = ROOT / "data" / "derivatives" / "stimuli" / "wiki_stories"
OUT = Path(__file__).resolve().parents[1] / "artifacts"
STEMS = ["pluto", "the_voynich_manuscript", "history_of_film", "serena_williams", "prisoners_dilemma"]

inv, lines = {}, []
for stem in STEMS:
    c = pd.read_csv(STIM / f"{stem}_coordinates.csv")
    for pg, g in c.groupby("page", sort=True):
        sids = list(dict.fromkeys(g["sentence_id"]))
        rec = []
        for i, sid in enumerate(sids):
            gs = g[g["sentence_id"] == sid]
            rec.append({"sent_idx": i, "sentence_id": sid,
                        "sentence": gs["sentence"].iloc[0], "n_words": int(len(gs))})
        inv[f"{stem}|{int(pg)}"] = rec
        lines.append(f"\n===== {stem} | page {int(pg)} | {len(g)} words | {len(sids)} sentences =====")
        for r in rec:
            lines.append(f"[{r['sent_idx']}] ({r['n_words']}w) {r['sentence']}")

(OUT / "sentence_inventory.json").write_text(json.dumps(inv, indent=1) + "\n")
(OUT / "sentence_inventory.txt").write_text("\n".join(lines) + "\n")
n_s = sum(len(v) for v in inv.values())
print(f"pages {len(inv)}  sentences {n_s}  words {sum(r['n_words'] for v in inv.values() for r in v)}")
