#!/usr/bin/env python3
"""Per-reader duration-coupling slopes, split by state.

The word-level table from 04_g5_g6_duration.py, reduced to the estimator the manuscript
plots: for each reader and each state, the bivariate OLS slope of log first-pass gaze
duration on the z-scored word property. Readers need at least 100 mind-wandering words
for a slope to be estimable, which keeps 38 of the 44.

Output: artifacts/duration_slopes_by_state.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ART, COUP

PROPS = ["zipf", "surprisal", "length"]
MIN_MW = 100

fx = pd.read_parquet(COUP / "fixations_frp.parquet",
                     columns=["subject", "word_key", "fix_dur", "fix_order",
                              "is_mw", "is_firstpass"])
wf = pd.read_parquet(COUP / "word_features.parquet")[["word_key"] + PROPS]

W = (fx[fx.is_firstpass == 1]
     .sort_values(["subject", "word_key", "fix_order"])
     .groupby(["subject", "word_key"])
     .agg(GD=("fix_dur", "sum"), is_mw=("is_mw", "first"))
     .reset_index()
     .merge(wf, on="word_key", how="left")
     .dropna(subset=PROPS))
W["logGD"] = np.log(W.GD.clip(lower=1))
for p in PROPS:
    W["z" + p] = (W[p] - W[p].mean()) / W[p].std()

rows = []
for subject, g in W.groupby("subject"):
    if (g.is_mw == 1).sum() < MIN_MW:
        continue
    r = {"subject": subject}
    for p in PROPS:
        for state, tag in ((0, "on"), (1, "mw")):
            gs = g[g.is_mw == state]
            r[f"{p}_{tag}"] = float(np.polyfit(gs["z" + p], gs.logGD, 1)[0])
    rows.append(r)

out = pd.DataFrame(rows)
out.to_csv(ART / "duration_slopes_by_state.csv", index=False)
print(f"{len(out)} readers of {W.subject.nunique()} with >= {MIN_MW} MW words")
print(out.drop(columns="subject").mean().round(4).to_string())
print("wrote", ART / "duration_slopes_by_state.csv")
