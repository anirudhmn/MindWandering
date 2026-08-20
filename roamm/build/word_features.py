#!/usr/bin/env python3
"""Build a word_key -> linguistic-feature table from the stimulus coordinate CSVs.

Features: cleaned word, length (alpha chars), Zipf log-frequency (wordfreq), and a
content-word flag. Surprisal is added by a separate script (needs a language model).
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
from wordfreq import zipf_frequency

STIM = Path("data/derivatives/stimuli/wiki_stories")
OUT = Path("roamm/artifacts/coupling")
OUT.mkdir(parents=True, exist_ok=True)

_alpha = re.compile(r"[^a-z]")

def clean(w: str) -> str:
    return _alpha.sub("", str(w).lower())

rows = []
for csv in sorted(STIM.glob("*_coordinates.csv")):
    d = pd.read_csv(csv)
    d["story_file"] = csv.stem.replace("_coordinates", "")
    rows.append(d)
coords = pd.concat(rows, ignore_index=True)
print("coord rows:", len(coords), "unique word_key:", coords["word_key"].nunique())

# one row per word_key (word_key is the fixation join key)
coords = coords.drop_duplicates("word_key").reset_index(drop=True)
coords["clean"] = coords["words"].map(clean)
coords["length"] = coords["clean"].str.len().astype("float")
coords["zipf"] = coords["clean"].map(lambda w: zipf_frequency(w, "en") if w else 0.0)
# position of the word within its sentence (for later surprisal / spillover)
coords["sent_pos"] = coords.groupby("sentence_id").cumcount()

feat = coords[["word_key", "words", "clean", "length", "zipf",
               "sentence_id", "sent_pos", "sentence", "story_file", "page"]].copy()
feat.to_parquet(OUT / "word_features.parquet", index=False)
print("wrote word_features.parquet", feat.shape)
print(feat[["words", "clean", "length", "zipf"]].describe(include="all").to_string())
print("\nzero-zipf (OOV) fraction:", float((feat["zipf"] == 0).mean()))
print("empty-clean fraction:", float((feat["clean"].str.len() == 0).mean()))
