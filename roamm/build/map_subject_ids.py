#!/usr/bin/env python3
"""Recover the sub_id <-> integer subject-index mapping used by the physiology artifacts.

reading_fixations.parquet / fixations_frp.parquet index subjects by their ordinal
position in features_df.pkl (0..43); the behavioural tables use sub_id strings.
Rather than assume the pickle is in sorted sub_id order, we recover the mapping from
data and verify it: each subject has a 50-dimensional fingerprint consisting of
(a) which run each story was read in (a counterbalanced permutation) and
(b) the per-page reading duration in seconds.

We match on the fingerprint, then check the assignment is a bijection, is unique by a
wide margin, and agrees (or does not) with the naive sorted-order assumption.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "roamm" / "artifacts" / "comprehension"
COUP = ROOT / "roamm" / "artifacts" / "coupling"

pages = pd.read_parquet(OUT / "pages.parquet")

# --- behavioural fingerprint: (story, page) -> duration, and (story) -> run
beh_dur = pages.pivot_table(index="sub_id", columns=["story_phys", "page"], values="page_dur")
beh_run = pages.pivot_table(index="sub_id", columns="story_phys", values="run", aggfunc="first")

# --- physiology fingerprint from fixation timestamps
fx = pd.read_parquet(COUP / "reading_fixations.parquet", columns=["subject", "run", "story", "page", "tStart"])
span = fx.groupby(["subject", "story", "page"]).agg(
    t0=("tStart", "min"), t1=("tStart", "max"), run=("run", "first")
)
span["dur"] = span["t1"] - span["t0"]
phys_dur = span.reset_index().pivot_table(index="subject", columns=["story", "page"], values="dur")
phys_run = span.reset_index().pivot_table(index="subject", columns="story", values="run", aggfunc="first")

cols = [c for c in beh_dur.columns if c in phys_dur.columns]
B = beh_dur[cols].to_numpy(float)
P = phys_dur[cols].to_numpy(float)
subs = list(beh_dur.index)
idxs = list(phys_dur.index)

# cost = mean |page_dur difference| over pages both tables have, + big penalty for run mismatch
runcols = [c for c in beh_run.columns if c in phys_run.columns]
BR = beh_run[runcols].to_numpy(float)
PR = phys_run[runcols].to_numpy(float)

cost = np.zeros((len(subs), len(idxs)))
for i in range(len(subs)):
    d = np.abs(B[i][None, :] - P)
    cost[i] = np.nanmean(d, axis=1) + 1000.0 * (BR[i][None, :] != PR).sum(axis=1)

r, c = linear_sum_assignment(cost)
mapping = {subs[i]: int(idxs[j]) for i, j in zip(r, c)}
matched_cost = cost[r, c]

# margin: best vs 2nd-best physiology index for each subject
srt = np.sort(cost, axis=1)
margin = srt[:, 1] - srt[:, 0]
best_is_assigned = np.array([np.argmin(cost[i]) == c[list(r).index(i)] for i in range(len(subs))])

naive = {s: k for k, s in enumerate(sorted(subs))}
agree_naive = sum(mapping[s] == naive[s] for s in subs)

rep = {
    "n_subjects": len(subs),
    "n_fingerprint_pages": len(cols),
    "bijective": len(set(mapping.values())) == len(subs),
    "matched_cost_mean_s": float(matched_cost.mean()),
    "matched_cost_max_s": float(matched_cost.max()),
    "unmatched_cost_min_s": float(np.min(srt[:, 1])),
    "margin_min_s": float(margin.min()),
    "greedy_equals_hungarian": bool(best_is_assigned.all()),
    "agrees_with_sorted_order": int(agree_naive),
    "mapping": mapping,
}
(OUT / "subject_map.json").write_text(json.dumps(rep, indent=2) + "\n")
print(json.dumps({k: v for k, v in rep.items() if k != "mapping"}, indent=2))
print("worst-matched subjects:", [(subs[i], round(float(matched_cost[i]), 3)) for i in np.argsort(-matched_cost)[:5]])
print(f"wrote {OUT/'subject_map.json'}")
