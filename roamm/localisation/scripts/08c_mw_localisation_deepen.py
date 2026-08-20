#!/usr/bin/env python3
"""Stress tests for the localised-MW result.

The finding: the MW fraction while the eyes were on the answer-bearing text predicts failing that
item (-0.061) more strongly than every one of 1000 random equal-size regions on the same page
(null mean -0.021, p=.001), while the predeclared matched control region gives -0.002. Reading
the same span carries no such specificity (T1/T2). Before that is called a result it has to
survive the things that could manufacture it:

  R1 WITHIN-MW-TRIALS. Restrict to trials that already contain MW somewhere on the page. This
     removes "did the reader mind-wander at all" from the comparison entirely, so anything left is
     purely about WHERE the lapse fell.
  R2 READING-AMOUNT. Add evidence coverage, evidence fixation count and evidence dwell. If the
     effect is really "MW meant they did not read the answer", it should collapse.
  R3 OVERLAP GRADIENT. Regress each permutation's statistic on how much that random region
     overlapped the true evidence span. A genuine informational localisation implies a gradient;
     an artefact of region size or position implies none. This uses the 1000 permutations as data
     rather than only as a null.
  R4 OUTCOME DECOMPOSITION. Does MW-on-answer push the reader to a wrong option, or to "I am not
     sure"? Different mechanisms: a missing memory vs a wrong memory.
  R5 EPISODE TIMING. MW spans are extended; does the effect require the lapse to overlap the
     answer, or merely to be nearby? Distance from the MW episode to the evidence span.
  R6 ITEM TYPE. single_fact items have a tight answer span; negated items need three statements
     verified. The localisation should be sharper for single_fact.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common60 import ART, RES, IT, COUP, z, mde, boot_ci

RNG = np.random.default_rng(60888)
NPERM = 1000
COMP = IT.parent / "artifacts" / "comprehension"
STIM = IT.parents[1] / "data" / "derivatives" / "stimuli" / "wiki_stories"

E = pd.read_parquet(ART / "item_evidence_llm.parquet")
D = pd.read_parquet(ART / "evidence_trials_llm.parquet")
pages = pd.read_parquet(COMP / "pages_full.parquet")
subs = sorted(pages["sub_id"].unique())
fx = pd.read_parquet(COUP / "reading_fixations.parquet",
                     columns=["subject", "story", "page", "word_key", "fix_dur", "is_mw", "tStart"])
fx = fx[fx["fix_dur"].between(50, 1000)].copy()
fx["sub_id"] = fx["subject"].map({i: s for i, s in enumerate(subs)})

sent_map, page_words = {}, {}
for stem in E["story_phys"].unique():
    c = pd.read_csv(STIM / f"{stem}_coordinates.csv")
    for p_, g in c.groupby("page"):
        keys = g["word_key"].tolist()
        page_words[(stem, int(p_))] = keys
        sids = list(dict.fromkeys(g["sentence_id"]))
        sent_map[(stem, int(p_))] = [[keys.index(k) for k in g.loc[g["sentence_id"] == s, "word_key"]]
                                     for s in sids]
kidx = {k: {w: i for i, w in enumerate(v)} for k, v in page_words.items()}

SEQ = {}
for (s, stem, p_), g in fx.groupby(["sub_id", "story", "page"], observed=True):
    key = (stem, int(p_))
    if key not in kidx:
        continue
    m = kidx[key]
    j = np.array([m.get(k, -1) for k in g["word_key"]])
    ok = j >= 0
    SEQ[(s, stem, int(p_))] = (j[ok], g["is_mw"].to_numpy()[ok].astype(float))

meta = E.set_index("item")[["story_phys", "page", "n_evidence_words", "evidence_sent_idx",
                            "control_word_keys", "item_type"]].to_dict("index")
ev_idx = {}
for it, m in meta.items():
    sents = sent_map[(m["story_phys"], int(m["page"]))]
    ev_idx[it] = np.array([i for si in m["evidence_sent_idx"] for i in sents[si]])

D = D[(D["n_fix_evidence"] >= 3) & (D["n_fix_elsewhere"] >= 10)].copy()
D["mw_else_z"] = z(D["mw_frac_elsewhere"].fillna(0))
D["page_cov_z"] = z(D["coverage"])
D["ev_cov_z"] = z(D["evidence_cov"])
D["ev_nfix_z"] = z(np.log1p(D["evidence_n_fix"]))
D["ev_dwell_z"] = z(np.log1p(D["evidence_dwell_ms"]))
D["mw_ev_z"] = z(D["mw_frac_evidence"].fillna(0))
D["has_mw_page"] = (D["mw_frac_page"].fillna(0) > 0).astype(int)
item_rows = {it: np.flatnonzero((D["item"] == it).to_numpy()) for it in meta}
rep = {}


def lpm(dd, outcome, terms, fe=("sub_id", "item")):
    dd = dd.dropna(subset=[outcome] + list(terms)).copy()
    X = [pd.get_dummies(dd[f], prefix=f, drop_first=True).astype(float) for f in fe]
    M = sm.add_constant(pd.concat([dd[list(terms)].reset_index(drop=True)] +
                                  [x.reset_index(drop=True) for x in X], axis=1))
    r = sm.OLS(dd[outcome].to_numpy(float), M.to_numpy(float)).fit(
        cov_type="cluster", cov_kwds={"groups": dd["sub_id"].to_numpy()})
    ci = np.asarray(r.conf_int())
    out = {"n": int(len(dd))}
    for i, t in enumerate(terms):
        out[t] = {"beta": float(r.params[i + 1]), "se": float(r.bse[i + 1]), "p": float(r.pvalues[i + 1]),
                  "ci": [float(ci[i + 1, 0]), float(ci[i + 1, 1])], "mde_80": mde(float(r.bse[i + 1]))}
    return out


# ---- R1 within trials that contain MW somewhere
sub = D[D["has_mw_page"] == 1].copy()
for c in ["mw_ev_z", "mw_else_z", "page_cov_z"]:
    sub[c] = z(sub[c])
rep["R1_within_mw_trials"] = {"n_trials": int(len(sub)),
                              "model": lpm(sub, "correct", ["mw_ev_z", "mw_else_z", "page_cov_z"])}
rep["R1_descriptive"] = {
    "acc_mw_page_no_mw_on_evidence": float(sub.loc[sub.mw_frac_evidence.fillna(0) == 0, "correct"].mean()),
    "n_no_mw_on_evidence": int((sub.mw_frac_evidence.fillna(0) == 0).sum()),
    "acc_mw_on_evidence": float(sub.loc[sub.mw_frac_evidence.fillna(0) > 0, "correct"].mean()),
    "n_mw_on_evidence": int((sub.mw_frac_evidence.fillna(0) > 0).sum()),
    "acc_no_mw_anywhere": float(D.loc[D.has_mw_page == 0, "correct"].mean()),
    "n_no_mw_anywhere": int((D.has_mw_page == 0).sum())}

# ---- R2 reading amount
rep["R2_with_reading_amount"] = lpm(D, "correct",
    ["mw_ev_z", "mw_else_z", "ev_cov_z", "ev_nfix_z", "ev_dwell_z", "page_cov_z"])

# ---- R3 overlap gradient across permutations
X0 = pd.concat([pd.get_dummies(D["sub_id"], prefix="s", drop_first=True).astype(float).reset_index(drop=True),
                pd.get_dummies(D["item"], prefix="i", drop_first=True).astype(float).reset_index(drop=True),
                D[["mw_else_z", "page_cov_z"]].reset_index(drop=True)], axis=1)
X0 = sm.add_constant(X0, has_constant="add").to_numpy(float)
Q, _ = np.linalg.qr(X0)
res = lambda v: v - Q @ (Q.T @ v)
yv = res(D["correct"].to_numpy(float))


def mw_on(region):
    out = np.full(len(D), np.nan)
    for it, m in meta.items():
        stem, p_ = m["story_phys"], int(m["page"])
        S = set(region[it].tolist())
        for r in item_rows[it]:
            s_ = SEQ.get((D["sub_id"].iat[r], stem, p_))
            if s_ is None:
                continue
            j, mw = s_
            hit = np.isin(j, list(S))
            if hit.sum() >= 1:
                out[r] = mw[hit].mean()
    return out


def stat(v):
    v = np.where(np.isfinite(v), v, np.nanmean(v))
    x = res((v - v.mean()) / (v.std() + 1e-12))
    return float((x * yv).sum() / max((x * x).sum(), 1e-12))


rows = []
for b in range(NPERM):
    pick, ov = {}, []
    for it, m in meta.items():
        sents = sent_map[(m["story_phys"], int(m["page"]))]
        order = RNG.permutation(len(sents))
        sel, n = [], 0
        for jj in order:
            if n >= m["n_evidence_words"]:
                break
            sel += sents[jj]; n += len(sents[jj])
        pick[it] = np.array(sel[:max(m["n_evidence_words"], 1)])
        t = set(ev_idx[it].tolist())
        ov.append(len(t & set(pick[it].tolist())) / max(len(t), 1))
    rows.append({"stat": stat(mw_on(pick)), "overlap": float(np.mean(ov))})
    if (b + 1) % 200 == 0:
        print(f"  gradient perm {b+1}/{NPERM}", flush=True)
G = pd.DataFrame(rows)
G.to_csv(RES / "mw_overlap_gradient.csv", index=False)
sl, ic, r_, p_, se_ = stats.linregress(G["overlap"], G["stat"])
rep["R3_overlap_gradient"] = {"n_perm": NPERM, "slope": float(sl), "intercept": float(ic),
                              "r": float(r_), "p": float(p_),
                              "stat_at_zero_overlap": float(ic),
                              "stat_at_full_overlap_extrapolated": float(ic + sl),
                              "observed_true_span": stat(mw_on(ev_idx)),
                              "mean_overlap": float(G["overlap"].mean())}

# ---- R4 outcome decomposition
rep["R4_outcome_decomposition"] = {
    "on_correct": lpm(D, "correct", ["mw_ev_z", "mw_else_z", "page_cov_z"]),
    "on_skipped_notsure": lpm(D, "skipped", ["mw_ev_z", "mw_else_z", "page_cov_z"]),
    "on_correct_given_answered": lpm(D[D["skipped"] == 0], "correct",
                                     ["mw_ev_z", "mw_else_z", "page_cov_z"])}

# ---- R6 item type
rep["R6_by_item_type"] = {}
for t in ["single_fact", "negated"]:
    s_ = D[D["item_type"] == t].copy()
    for c in ["mw_ev_z", "mw_else_z", "page_cov_z"]:
        s_[c] = z(s_[c])
    rep["R6_by_item_type"][t] = lpm(s_, "correct", ["mw_ev_z", "mw_else_z", "page_cov_z"])

(RES / "mw_localisation_deepen.json").write_text(json.dumps(rep, indent=2, default=str) + "\n")

print("\n=== R1 within trials that already contain MW somewhere on the page ===")
m = rep["R1_within_mw_trials"]["model"]
print(f"  n={rep['R1_within_mw_trials']['n_trials']}")
for k in ["mw_ev_z", "mw_else_z"]:
    v = m[k]
    print(f"  {k:11s} {v['beta']:+.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] p={v['p']:.3g}")
print(f"  descriptive: {rep['R1_descriptive']}")
print("\n=== R2 adding evidence coverage / fixations / dwell ===")
for k in ["mw_ev_z", "mw_else_z", "ev_cov_z", "ev_nfix_z", "ev_dwell_z"]:
    v = rep["R2_with_reading_amount"][k]
    print(f"  {k:11s} {v['beta']:+.4f} p={v['p']:.3g}")
g = rep["R3_overlap_gradient"]
print(f"\n=== R3 overlap gradient across {g['n_perm']} random regions ===")
print(f"  slope {g['slope']:+.4f} per unit overlap (r={g['r']:+.3f}, p={g['p']:.2g}); "
      f"stat at zero overlap {g['stat_at_zero_overlap']:+.4f}, extrapolated to full overlap "
      f"{g['stat_at_full_overlap_extrapolated']:+.4f}; true span {g['observed_true_span']:+.4f}")
print("\n=== R4 what does MW-on-answer do to the response? ===")
for k, m in rep["R4_outcome_decomposition"].items():
    v = m["mw_ev_z"]
    print(f"  {k:26s} mw_ev {v['beta']:+.4f} p={v['p']:.3g}   (n={m['n']})")
print("\n=== R6 by item type ===")
for t, m in rep["R6_by_item_type"].items():
    print(f"  {t:12s} mw_ev {m['mw_ev_z']['beta']:+.4f} (p={m['mw_ev_z']['p']:.3g})  "
          f"mw_else {m['mw_else_z']['beta']:+.4f} (p={m['mw_else_z']['p']:.3g})  n={m['n']}")
