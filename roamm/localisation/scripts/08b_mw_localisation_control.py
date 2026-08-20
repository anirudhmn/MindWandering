#!/usr/bin/env python3
"""The control T5 needs: is MW-on-the-answer better than MW-on-a-random-region?

T5 found that the MW fraction while the eyes were on the evidence span predicts failing the item
(-0.073, p=1.7e-13) far more strongly than MW elsewhere on the page (-0.035, p=.064). But T1/T2
showed the evidence span is NOT special for READING (its coverage predicts accuracy no better
than a matched control region, and no better than a random equal-size region). So the same
sceptical test has to be applied to the MW result, and it is the one that decides whether the
cost of mind-wandering is informationally localised or just globally measured.

Two nulls, both refitting the same estimator 1000 times:
  RANDOM REGION   evidence span replaced by a random equal-size set of sentences from the same
                  page; MW fraction recomputed over the fixations landing there.
  MATCHED CONTROL the predeclared equal-size control region (a single fixed comparison).

A further confound is measurement: MW-on-evidence averages over ~28 fixations, MW-elsewhere over
~200, so the two coefficients are not on comparable footing. The random-region null holds the
region SIZE fixed, which controls exactly that.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common60 import ART, RES, IT, COUP, z, mde

RNG = np.random.default_rng(6088)
NPERM = 1000
COMP = IT.parent / "artifacts" / "comprehension"
STIM = IT.parents[1] / "data" / "derivatives" / "stimuli" / "wiki_stories"

E = pd.read_parquet(ART / "item_evidence_llm.parquet")
D = pd.read_parquet(ART / "evidence_trials_llm.parquet")
pages = pd.read_parquet(COMP / "pages_full.parquet")
subs = sorted(pages["sub_id"].unique())

fx = pd.read_parquet(COUP / "reading_fixations.parquet",
                     columns=["subject", "story", "page", "word_key", "fix_dur", "is_mw"])
fx = fx[fx["fix_dur"].between(50, 1000)].copy()
fx["sub_id"] = fx["subject"].map({i: s for i, s in enumerate(subs)})

# per (reader, page): the fixation sequence with word index on the page and MW flag
sent_map, page_words = {}, {}
for stem in E["story_phys"].unique():
    c = pd.read_csv(STIM / f"{stem}_coordinates.csv")
    for p_, g in c.groupby("page"):
        keys = g["word_key"].tolist()
        page_words[(stem, int(p_))] = keys
        sids = list(dict.fromkeys(g["sentence_id"]))
        sent_map[(stem, int(p_))] = [[keys.index(k) for k in g.loc[g["sentence_id"] == s, "word_key"]]
                                     for s in sids]
kidx = {(stem, p_): {k: i for i, k in enumerate(keys)} for (stem, p_), keys in page_words.items()}

FIXSEQ: dict = {}
for (s, stem, p_), g in fx.groupby(["sub_id", "story", "page"], observed=True):
    key = (stem, int(p_))
    if key not in kidx:
        continue
    m = kidx[key]
    j = np.array([m.get(k, -1) for k in g["word_key"]])
    ok = j >= 0
    FIXSEQ[(s, stem, int(p_))] = (j[ok], g["is_mw"].to_numpy()[ok].astype(float))

item_meta = E.set_index("item")[["story_phys", "page", "n_evidence_words",
                                 "evidence_sent_idx", "control_word_keys"]].to_dict("index")
D = D[(D["n_fix_evidence"] >= 3) & (D["n_fix_elsewhere"] >= 10)].copy()
D["mw_else_z"] = z(D["mw_frac_elsewhere"].fillna(0))
D["page_cov_z"] = z(D["coverage"])

# design without the region term: subject FE, item FE, MW-elsewhere, page coverage
X0 = pd.concat([pd.get_dummies(D["sub_id"], prefix="s", drop_first=True).astype(float).reset_index(drop=True),
                pd.get_dummies(D["item"], prefix="i", drop_first=True).astype(float).reset_index(drop=True),
                D[["mw_else_z", "page_cov_z"]].reset_index(drop=True)], axis=1)
X0 = sm.add_constant(X0, has_constant="add").to_numpy(float)
Q, _ = np.linalg.qr(X0)
res = lambda v: v - Q @ (Q.T @ v)
yv = res(D["correct"].to_numpy(float))
rows = list(zip(D["sub_id"], D["item"]))
item_rows = {it: np.flatnonzero((D["item"] == it).to_numpy()) for it in item_meta}


def mw_frac_on(region_idx_by_item):
    out = np.full(len(D), np.nan)
    for it, m in item_meta.items():
        stem, p_ = m["story_phys"], int(m["page"])
        sel = region_idx_by_item[it]
        S = set(sel.tolist())
        for r in item_rows[it]:
            seq = FIXSEQ.get((D["sub_id"].iat[r], stem, p_))
            if seq is None:
                continue
            j, mw = seq
            hit = np.isin(j, list(S))
            if hit.sum() >= 1:
                out[r] = mw[hit].mean()
    return out


def stat(v):
    v = np.where(np.isfinite(v), v, np.nanmean(v))
    x = res((v - v.mean()) / (v.std() + 1e-12))
    return float((x * yv).sum() / max((x * x).sum(), 1e-12))


# observed: the real evidence span
ev_idx = {}
for it, m in item_meta.items():
    stem, p_ = m["story_phys"], int(m["page"])
    sents = sent_map[(stem, p_)]
    ev_idx[it] = np.array([i for si in m["evidence_sent_idx"] for i in sents[si]])
obs = stat(mw_frac_on(ev_idx))

# matched control region
ctrl_idx = {}
for it, m in item_meta.items():
    stem, p_ = m["story_phys"], int(m["page"])
    mm = kidx[(stem, p_)]
    ctrl_idx[it] = np.array([mm[k] for k in m["control_word_keys"] if k in mm])
ctrl_stat = stat(mw_frac_on(ctrl_idx))

null = np.empty(NPERM)
for b in range(NPERM):
    pick = {}
    for it, m in item_meta.items():
        stem, p_ = m["story_phys"], int(m["page"])
        sents = sent_map[(stem, p_)]
        order = RNG.permutation(len(sents))
        sel, n = [], 0
        for jj in order:
            if n >= m["n_evidence_words"]:
                break
            sel += sents[jj]; n += len(sents[jj])
        pick[it] = np.array(sel[:max(m["n_evidence_words"], 1)])
    null[b] = stat(mw_frac_on(pick))
    if (b + 1) % 100 == 0:
        print(f"  {b+1}/{NPERM}", flush=True)

rep = {"n_trials": int(len(D)), "observed_evidence_stat": obs,
       "matched_control_region_stat": ctrl_stat,
       "n_perm": NPERM, "null_mean": float(null.mean()), "null_sd": float(null.std()),
       "p_one_sided_more_negative": float(((null <= obs).sum() + 1) / (NPERM + 1)),
       "percentile_of_observed": float((null < obs).mean() * 100),
       "null_q": {q: float(np.percentile(null, q)) for q in [2.5, 25, 50, 75, 97.5]}}
(RES / "mw_localisation_control.json").write_text(json.dumps(rep, indent=2) + "\n")
print(json.dumps(rep, indent=2))
