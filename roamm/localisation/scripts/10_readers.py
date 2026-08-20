#!/usr/bin/env python3
"""H3 and the reader-level question: who tracks importance, who lapses on it, and does it matter?

Two parts, in the order the preregistration demands:

  A  RELIABILITY FIRST. Per-reader importance-tracking slopes are only worth interpreting if they
     are a stable property of the reader. Split-half reliability by odd/even stories with a
     Spearman-Brown correction; if r < 0.3 the typology is abandoned rather than clustered on
     noise. Same test for the lexical channel, which gives a reference point for how reliable an
     individual-difference measure CAN be in this dataset.

  B  WHO LOSES WHAT. For each reader, what fraction of their mind-wandering landed on the
     answer-bearing text, and what did it cost them? This is the reader-level face of the
     localised-MW result: two readers with identical MW rates can differ in whether their lapses
     happened to fall on the words that were later asked about.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy import stats
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common60 import ART, RES, boot_ci, fmt, absorb, z, load_word_measures

COV = ["zipf_z", "length_z", "surprisal_z", "n_words_sentence_z", "rel_pos_in_sentence_z",
       "sent_pos_on_page_z", "line_pos_z", "is_line_first", "is_line_last", "is_page_boundary_sentence"]
rep = {}

D = load_word_measures()
D["log_gaze"] = np.log(D["gaze_dur"])
D["refix"] = (D["n_refix"] > 0).astype(float)
D["page_id"] = D["story_phys"] + "_" + D["page"].astype(str)
for c in ["importance_llm", "zipf", "length", "surprisal", "n_words_sentence",
          "rel_pos_in_sentence", "sent_pos_on_page", "line_pos"]:
    D[c + "_z"] = z(D[c])
D["surprisal_z"] = D["surprisal_z"].fillna(0.0)
D = D.dropna(subset=["log_gaze", "refix", "importance_llm_z"] + COV)
stories = sorted(D["story_phys"].unique())
D["half"] = D["story_phys"].map({s: i % 2 for i, s in enumerate(stories)})


def slope(g, outcome, pred):
    # the predictor must not also appear as a covariate: a duplicated column makes the
    # normal equations singular and the split between the two copies numerically arbitrary
    cols = [pred] + [c for c in COV if c != pred]
    if len(g) < 200:
        return np.nan
    M = absorb(np.column_stack([g[outcome].to_numpy(float)] + [g[c].to_numpy(float) for c in cols]),
               [g["page_id"].to_numpy(), g["lemma"].to_numpy()])
    return float((np.linalg.pinv(M[:, 1:].T @ M[:, 1:]) @ (M[:, 1:].T @ M[:, 0]))[0])


# ---------------------------------------------------------------- A reliability
rel = {}
for outcome in ["refix", "log_gaze"]:
    for pred in ["importance_llm_z", "zipf_z"]:
        rows = []
        for s, g in D.groupby("subject"):
            rows.append({"subject": s, "full": slope(g, outcome, pred),
                         "h0": slope(g[g.half == 0], outcome, pred),
                         "h1": slope(g[g.half == 1], outcome, pred)})
        R = pd.DataFrame(rows).dropna()
        r_half = float(R["h0"].corr(R["h1"]))
        sb = 2 * r_half / (1 + r_half) if r_half > -1 else np.nan
        rel[f"{outcome}__{pred}"] = {
            "n_readers": int(len(R)), "split_half_r": r_half, "spearman_brown": float(sb),
            "between_reader_sd": float(R["full"].std()), "mean": float(R["full"].mean())}
        R.to_csv(RES / f"reader_slopes_{outcome}_{pred}.csv", index=False)
rep["A_reliability"] = rel
rep["A_VERDICT"] = ("typology abandoned: per-reader importance slopes are not reliable"
                    if rel["refix__importance_llm_z"]["spearman_brown"] < 0.3 else
                    "typology admissible")

# ---------------------------------------------------------------- B who loses what
T = pd.read_parquet(ART / "evidence_trials_llm.parquet")
T = T[(T["n_fix_evidence"] >= 3) & (T["n_fix_elsewhere"] >= 10)]
per = T.groupby("sub_id").apply(lambda g: pd.Series({
    "n_trials": len(g),
    "mw_rate_page": float((g["mw_frac_page"].fillna(0) > 0).mean()),
    "mw_frac_all": float(g["mw_frac_page"].fillna(0).mean()),
    "frac_trials_mw_on_answer": float((g["mw_frac_evidence"].fillna(0) > 0).mean()),
    "hit_rate_given_mw": (float((g["mw_frac_evidence"].fillna(0) > 0).sum() /
                                max((g["mw_frac_page"].fillna(0) > 0).sum(), 1))),
    "acc": float(g["correct"].mean()),
    "acc_when_mw_on_answer": (float(g.loc[g["mw_frac_evidence"].fillna(0) > 0, "correct"].mean())
                              if (g["mw_frac_evidence"].fillna(0) > 0).sum() >= 3 else np.nan),
    "acc_when_clean": (float(g.loc[g["mw_frac_evidence"].fillna(0) == 0, "correct"].mean())
                       if (g["mw_frac_evidence"].fillna(0) == 0).sum() >= 3 else np.nan),
}), include_groups=False).reset_index()
per["within_reader_cost"] = per["acc_when_clean"] - per["acc_when_mw_on_answer"]
per.to_csv(RES / "reader_level.csv", index=False)

ok = per.dropna(subset=["within_reader_cost"])
rep["B_reader_level"] = {
    "n_readers": int(len(per)),
    "n_readers_with_both_cells": int(len(ok)),
    "within_reader_cost": boot_ci(ok["within_reader_cost"].to_numpy()),
    "n_readers_positive_cost": int((ok["within_reader_cost"] > 0).sum()),
    "mw_rate_range": [float(per.mw_rate_page.min()), float(per.mw_rate_page.max())],
    "frac_mw_landing_on_answer_range": [float(per.hit_rate_given_mw.min()),
                                        float(per.hit_rate_given_mw.max())],
    "frac_mw_landing_on_answer_mean": float(per.hit_rate_given_mw.mean()),
    "corr_mwrate_vs_accuracy": {"r": float(per.mw_rate_page.corr(per.acc)),
                                "p": float(stats.pearsonr(per.mw_rate_page, per.acc)[1])},
    "corr_hitrate_vs_accuracy": {"r": float(per.hit_rate_given_mw.corr(per.acc)),
                                 "p": float(stats.pearsonr(per.hit_rate_given_mw, per.acc)[1])},
}
# does WHERE the lapses land add to HOW MUCH the reader lapses, across readers?
import statsmodels.api as sm
X = sm.add_constant(per[["mw_frac_all", "hit_rate_given_mw"]].to_numpy(float))
m = sm.OLS(per["acc"].to_numpy(float), X).fit()
rep["B_reader_regression"] = {"mw_frac_all": {"beta": float(m.params[1]), "p": float(m.pvalues[1])},
                              "hit_rate_given_mw": {"beta": float(m.params[2]), "p": float(m.pvalues[2])},
                              "r2": float(m.rsquared), "n": int(len(per))}

(RES / "readers.json").write_text(json.dumps(rep, indent=2, default=str) + "\n")

print("=== A per-reader slope reliability (split-half by story, Spearman-Brown) ===")
for k, v in rel.items():
    print(f"  {k:28s} n={v['n_readers']}  split-half r={v['split_half_r']:+.3f}  "
          f"SB={v['spearman_brown']:+.3f}  between-reader SD={v['between_reader_sd']:.4f}")
print(f"  VERDICT: {rep['A_VERDICT']}")
print("\n=== B reader level: where do the lapses land? ===")
b = rep["B_reader_level"]
print(f"  {b['n_readers']} readers; MW-page rate ranges {b['mw_rate_range'][0]:.2f}-{b['mw_rate_range'][1]:.2f}")
print(f"  of a reader's MW pages, fraction where the lapse touched the answer text: "
      f"mean {b['frac_mw_landing_on_answer_mean']:.2f}, range "
      f"{b['frac_mw_landing_on_answer_range'][0]:.2f}-{b['frac_mw_landing_on_answer_range'][1]:.2f}")
print(fmt("  within-reader accuracy cost", b["within_reader_cost"]))
print(f"  readers with a positive cost: {b['n_readers_positive_cost']}/{b['n_readers_with_both_cells']}")
print(f"  across readers: MW rate vs accuracy r={b['corr_mwrate_vs_accuracy']['r']:+.3f} "
      f"(p={b['corr_mwrate_vs_accuracy']['p']:.3g}); lapse-hits-answer rate vs accuracy "
      f"r={b['corr_hitrate_vs_accuracy']['r']:+.3f} (p={b['corr_hitrate_vs_accuracy']['p']:.3g})")
rr = rep["B_reader_regression"]
print(f"  joint reader model (R2={rr['r2']:.3f}): how much MW beta={rr['mw_frac_all']['beta']:+.3f} "
      f"(p={rr['mw_frac_all']['p']:.3g});  where it lands beta={rr['hit_rate_given_mw']['beta']:+.3f} "
      f"(p={rr['hit_rate_given_mw']['p']:.3g})")
