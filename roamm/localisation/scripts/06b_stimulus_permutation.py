#!/usr/bin/env python3
"""The stimulus-side permutation test for H1 -- the control that matters most here.

Importance varies over only 529 sentences. Subject-level inference treats readers as the unit
and so says nothing about whether the *ratings* carry information: any sentence-level property
correlated with reading would produce the same subject-level bootstrap. The decisive test
reassigns the importance ratings among the sentences of the SAME page, which preserves each
page's rating multiset, every sentence's length, its position on the page, its line layout and
its word identities, and destroys only the mapping from rating to sentence. 1000 refits.

Statistic: the pooled within-(subject x page x lemma) coefficient, computed by Frisch-Waugh so
that only the permuted column has to be re-absorbed each iteration (the fixed-effect structure
and the residualised outcome are computed once).
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common60 import ART, RES, absorb, z, load_word_measures

NPERM = 1000
COV = ["zipf_z", "length_z", "surprisal_z", "n_words_sentence_z", "rel_pos_in_sentence_z",
       "sent_pos_on_page_z", "line_pos_z", "is_line_first", "is_line_last",
       "is_page_boundary_sentence"]


def make_codes(groups):
    out = []
    for g in groups:
        _, c = np.unique(g, return_inverse=True)
        out.append((c.astype(np.int64), int(c.max()) + 1, np.bincount(c).astype(float)))
    return out


def absorb_fast(v, codes, tol=1e-8, maxit=80):
    v = np.asarray(v, float).copy()
    for _ in range(maxit):
        prev = v.copy()
        for c, k, cnt in codes:
            v -= (np.bincount(c, weights=v, minlength=k) / cnt)[c]
        if np.max(np.abs(v - prev)) < tol:
            break
    return v


D = load_word_measures()
D["log_gaze"] = np.log(D["gaze_dur"])
D["refix"] = (D["n_refix"] > 0).astype(float)
D["page_id"] = D["story_phys"] + "_" + D["page"].astype(str)
for c in ["importance_llm", "zipf", "length", "surprisal", "n_words_sentence",
          "rel_pos_in_sentence", "sent_pos_on_page", "line_pos"]:
    D[c + "_z"] = z(D[c])
D["surprisal_z"] = D["surprisal_z"].fillna(0.0)
D = D.dropna(subset=["log_gaze", "refix", "importance_llm_z"] + COV).reset_index(drop=True)

codes = make_codes([D["subject"].to_numpy(), D["page_id"].to_numpy(), D["lemma"].to_numpy()])
Xc = np.column_stack([absorb_fast(D[c].to_numpy(float), codes) for c in COV])
Q, _ = np.linalg.qr(Xc)
resid = lambda v: v - Q @ (Q.T @ v)

Y = {o: resid(absorb_fast(D[o].to_numpy(float), codes)) for o in ["log_gaze", "refix"]}


def beta(x_raw, y):
    x = resid(absorb_fast(x_raw, codes))
    d = float(x @ x)
    return float((x @ y) / d) if d > 0 else np.nan


key = ["story_phys", "page", "sent_idx"]
S = D.drop_duplicates(key)[key + ["importance_llm"]].reset_index(drop=True)
# map each row to its position in S, so a permuted rating vector broadcasts by index
S["sid"] = np.arange(len(S))
D = D.merge(S[key + ["sid"]], on=key, how="left")
sid = D["sid"].to_numpy()
page_of_sent = (S["story_phys"] + "_" + S["page"].astype(str)).to_numpy()
imp_sent = S["importance_llm"].to_numpy(float)
page_groups = [np.flatnonzero(page_of_sent == p) for p in pd.unique(page_of_sent)]

mu, sd = imp_sent[sid].mean(), imp_sent[sid].std()
obs = {o: beta((imp_sent[sid] - mu) / sd, Y[o]) for o in Y}

rng = np.random.default_rng(600)
null = {o: np.empty(NPERM) for o in Y}
for b in range(NPERM):
    v = imp_sent.copy()
    for g in page_groups:
        v[g] = rng.permutation(v[g])
    xr = v[sid]
    xr = (xr - xr.mean()) / xr.std()
    for o in Y:
        null[o][b] = beta(xr, Y[o])
    if (b + 1) % 100 == 0:
        print(f"  {b+1}/{NPERM}", flush=True)

rep = {}
for o in Y:
    n = null[o]
    rep[o] = {"n_perm": NPERM, "observed_beta": obs[o], "null_mean": float(n.mean()),
              "null_sd": float(n.std()),
              "p_one_sided": float(((n >= obs[o]).sum() + 1) / (NPERM + 1)),
              "p_two_sided": float(((np.abs(n - n.mean()) >= abs(obs[o] - n.mean())).sum() + 1) / (NPERM + 1)),
              "percentile_of_observed": float((n < obs[o]).mean() * 100)}
    print(f"{o:10s} observed {obs[o]:+.5f}  null {n.mean():+.5f} (SD {n.std():.5f})  "
          f"p1={rep[o]['p_one_sided']:.4f}  pct={rep[o]['percentile_of_observed']:.1f}")
(RES / "h1_stimulus_permutation.json").write_text(json.dumps(rep, indent=2) + "\n")
