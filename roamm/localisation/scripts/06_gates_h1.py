#!/usr/bin/env python3
"""Gates G1-G3, then H1: does free reading allocate time by semantic importance?

The alternative explanations are handled in this order, deliberately:
  1  every model carries zipf/length/surprisal + word/sentence/line position + sentence length
  2  subject and page fixed effects, so nothing is a between-reader or between-page difference
  3  LEMMA fixed effects -- the identification. The same word type appears in high- and
     low-importance sentences (80.8% of tokens), so this contrast holds word identity fixed and
     asks only whether the sentence it sits in matters. If the effect dies here it was lexical.
  4  the sentence-mean-word-property control: important sentences might just be sentences whose
     words are collectively rarer; the sentence's mean zipf/length/surprisal enter as covariates.

Primary inference is the subject-level bootstrap over per-reader slopes (44 readers), matching
the selection and repair analysis; the pooled two-way FE fit with subject-clustered SE is reported alongside.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy import stats
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common60 import ART, RES, boot_ci, fmt, holm, absorb, ols_cluster, z, mde, load_word_measures

rep: dict = {}
W = pd.read_parquet(ART / "word_importance.parquet")

# ---------------------------------------------------------------- G1 convergent validity
g1 = {}
have_qwen = "importance_qwen" in W.columns and W["importance_qwen"].notna().any()
S = W.drop_duplicates(["story_phys", "page", "sent_idx"]).copy()
pairs = [("importance_llm", "importance_qwen_exp"), ("importance_llm", "centrality_lm"),
         ("importance_qwen_exp", "centrality_lm"), ("importance_llm", "in_summary_llm")]
for a, b in pairs:
    if a not in S.columns or b not in S.columns:
        continue
    m = S[[a, b]].dropna()
    rho, p = stats.spearmanr(m[a], m[b])
    # within-page (page-demeaned) version: the variance the analysis actually uses
    d = S[["story_phys", "page", a, b]].dropna()
    da = d[a] - d.groupby(["story_phys", "page"])[a].transform("mean")
    db = d[b] - d.groupby(["story_phys", "page"])[b].transform("mean")
    rho_w, p_w = stats.spearmanr(da, db)
    g1[f"{a}__{b}"] = dict(rho=float(rho), p=float(p), rho_within_page=float(rho_w),
                           p_within_page=float(p_w), n=int(len(m)))
# weighted kappa between the two LLM raters
if have_qwen:
    m = S[["importance_llm", "importance_qwen"]].dropna()
    from sklearn.metrics import cohen_kappa_score
    g1["kappa_quadratic_llm_vs_qwen"] = float(cohen_kappa_score(m.importance_llm, m.importance_qwen,
                                                                weights="quadratic"))
    g1["qwen_rating_distribution"] = m.importance_qwen.value_counts().sort_index().to_dict()
rep["G1_convergent_validity"] = g1
key = g1.get("importance_llm__importance_qwen_exp", {}).get("rho_within_page", np.nan)
rep["G1_PASS"] = bool(np.isfinite(key) and key >= 0.30)

# ---------------------------------------------------------------- G2 not a lexical confound
sm = W.groupby(["story_phys", "page", "sent_idx"]).agg(
    imp=("importance_llm", "first"), zipf=("zipf", "mean"), length=("length", "mean"),
    surprisal=("surprisal", "mean"), n_words=("n_words_sentence", "first")).reset_index()
rep["G2_sentence_level_confounds"] = {
    c: dict(r=float(sm["imp"].corr(sm[c])), rho=float(stats.spearmanr(sm["imp"], sm[c], nan_policy="omit")[0]))
    for c in ["zipf", "length", "surprisal", "n_words"]}
rep["G2_word_level_confounds"] = {
    c: float(stats.spearmanr(W["importance_llm"], W[c], nan_policy="omit")[0])
    for c in ["zipf", "length", "surprisal", "rel_pos_in_sentence", "sent_pos_on_page",
              "line_pos", "is_line_first", "n_words_sentence"]}
gl = W.groupby("lemma")["importance_llm"]
n_within = int(((gl.transform("size") >= 2) & (gl.transform("std").fillna(0) > 0)).sum())
rep["G2_within_lemma_tokens"] = n_within
rep["G2_PASS"] = bool(max(abs(v["r"]) for v in rep["G2_sentence_level_confounds"].values()) < 0.7
                      and n_within >= 20000)

# ---------------------------------------------------------------- data for H1
D = load_word_measures()
D["log_gaze"] = np.log(D["gaze_dur"])
D["log_ffd"] = np.log(D["ffd"])
D["refix"] = (D["n_refix"] > 0).astype(float)
D["regr_out"] = D["regression_out"].fillna(0).astype(float)
D["page_id"] = D["story_phys"] + "_" + D["page"].astype(str)
for c in ["importance_llm", "zipf", "length", "surprisal", "n_words_sentence",
          "rel_pos_in_sentence", "sent_pos_on_page", "line_pos", "importance_qwen_exp", "centrality_lm"]:
    if c in D.columns:
        D[c + "_z"] = z(D[c])
D["surprisal_z"] = D["surprisal_z"].fillna(0.0)
D["centrality_lm_z"] = D["centrality_lm_z"].fillna(0.0) if "centrality_lm_z" in D else 0.0

# MW overlap gate G3
tw = pd.read_parquet(ART.parents[1] / "selection_repair/artifacts/words_traversal.parquet")
both = D.groupby("word_key")["is_mw"].agg(["min", "max"])
rep["G3_mw_overlap"] = {
    "n_word_instances_read_in_both_states": int((both["min"] != both["max"]).sum()),
    "n_rows_on_those_instances": int(D["word_key"].isin(both[both["min"] != both["max"]].index).sum()),
    "mw_row_rate": float(D["is_mw"].mean())}
rep["G3_PASS"] = rep["G3_mw_overlap"]["n_word_instances_read_in_both_states"] >= 300

COV = ["zipf_z", "length_z", "surprisal_z", "n_words_sentence_z", "rel_pos_in_sentence_z",
       "sent_pos_on_page_z", "line_pos_z", "is_line_first", "is_line_last",
       "is_page_boundary_sentence"]


def per_subject_slope(dd, outcome, pred, extra=(), within_lemma=False):
    """Per-reader slope of `outcome` on `pred`, page-demeaned (and lemma-demeaned if asked)."""
    out = []
    cols = [pred] + list(extra) + COV
    for s, g in dd.groupby("subject"):
        g = g.dropna(subset=[outcome] + cols)
        if len(g) < 200:
            continue
        gg = [g["page_id"].to_numpy()] + ([g["lemma"].to_numpy()] if within_lemma else [])
        M = absorb(np.column_stack([g[outcome].to_numpy(float)] + [g[c].to_numpy(float) for c in cols]), gg)
        y, X = M[:, 0], M[:, 1:]
        b = np.linalg.pinv(X.T @ X) @ (X.T @ y)
        out.append((s, float(b[0])))
    return pd.DataFrame(out, columns=["subject", "slope"])


def pooled(dd, outcome, preds, within_lemma=False):
    cols = list(preds) + COV
    g = dd.dropna(subset=[outcome] + cols)
    groups = [g["subject"].to_numpy(), g["page_id"].to_numpy()] + ([g["lemma"].to_numpy()] if within_lemma else [])
    M = absorb(np.column_stack([g[outcome].to_numpy(float)] + [g[c].to_numpy(float) for c in cols]), groups)
    r = ols_cluster(M[:, 0], M[:, 1:], g["subject"].to_numpy(), names=cols)
    return {k: r[k] for k in list(preds) + ["_n", "_n_clusters"]}


# ---------------------------------------------------------------- H1
h1 = {}
for outcome, lab in [("log_gaze", "log first-pass gaze duration"), ("log_ffd", "log first fixation dur"),
                     ("refix", "P(refixation)"), ("regr_out", "P(regression out)")]:
    s_plain = per_subject_slope(D, outcome, "importance_llm_z")
    s_lemma = per_subject_slope(D, outcome, "importance_llm_z", within_lemma=True)
    h1[outcome] = {
        "label": lab,
        "subject_bootstrap": boot_ci(s_plain["slope"].to_numpy()),
        "subject_bootstrap_within_lemma": boot_ci(s_lemma["slope"].to_numpy()),
        "pooled_2wayFE": pooled(D, outcome, ["importance_llm_z"]),
        "pooled_3wayFE_within_lemma": pooled(D, outcome, ["importance_llm_z"], within_lemma=True),
    }
    s_plain.to_csv(RES / f"subject_slopes_{outcome}.csv", index=False)

# convergent measures on the primary outcome
h1["convergent_log_gaze"] = {}
for p in ["importance_qwen_exp_z", "centrality_lm_z", "in_summary_llm"]:
    if p.replace("_z", "") in D.columns or p in D.columns:
        if p not in D.columns:
            continue
        h1["convergent_log_gaze"][p] = {
            "subject_bootstrap": boot_ci(per_subject_slope(D, "log_gaze", p)["slope"].to_numpy()),
            "pooled_3wayFE_within_lemma": pooled(D, "log_gaze", [p], within_lemma=True)}

# both importance measures in the same model (do they carry separate information?)
if "importance_qwen_exp_z" in D.columns:
    h1["joint_llm_plus_qwen"] = pooled(D, "log_gaze", ["importance_llm_z", "importance_qwen_exp_z"],
                                      within_lemma=True)
    h1["joint_llm_plus_centrality"] = pooled(D, "log_gaze", ["importance_llm_z", "centrality_lm_z"],
                                            within_lemma=True)

# effect size in ms: importance 5 vs importance <=3, matched by the same models
lo, hi = D[D.importance_llm <= 3], D[D.importance_llm == 5]
h1["descriptive_ms"] = {
    "gaze_dur_importance_le3": float(lo.gaze_dur.mean()), "gaze_dur_importance_5": float(hi.gaze_dur.mean()),
    "n_le3": int(len(lo)), "n_5": int(len(hi)),
    "pct_diff": float(100 * (hi.gaze_dur.mean() / lo.gaze_dur.mean() - 1))}
rep["H1"] = h1

(RES / "gates_h1.json").write_text(json.dumps(rep, indent=2, default=str) + "\n")

print("=== G1 convergent validity (within-page Spearman) ===")
for k, v in g1.items():
    if isinstance(v, dict) and "rho_within_page" in v:
        print(f"  {k:44s} rho={v['rho']:+.3f}  within-page rho={v['rho_within_page']:+.3f} (p={v['p_within_page']:.2g}) n={v['n']}")
if "kappa_quadratic_llm_vs_qwen" in g1:
    print(f"  quadratic-weighted kappa (Opus5 vs Qwen2.5): {g1['kappa_quadratic_llm_vs_qwen']:.3f}")
print(f"  G1 PASS={rep['G1_PASS']}")
print("\n=== G2 confounds ===")
print("  sentence level:", {k: round(v["r"], 3) for k, v in rep["G2_sentence_level_confounds"].items()})
print("  word level    :", {k: round(v, 3) for k, v in rep["G2_word_level_confounds"].items()})
print(f"  within-lemma tokens with importance variation: {n_within}   G2 PASS={rep['G2_PASS']}")
print(f"\n=== G3 MW overlap === {rep['G3_mw_overlap']}  PASS={rep['G3_PASS']}")
print("\n=== H1 importance -> reading (per-reader slopes, 10k bootstrap) ===")
for k, v in h1.items():
    if not isinstance(v, dict) or "subject_bootstrap" not in v:
        continue
    print(f"\n {v['label']}:")
    print(fmt("importance (page FE)", v["subject_bootstrap"]))
    print(fmt("importance (page+LEMMA FE)", v["subject_bootstrap_within_lemma"]))
    p = v["pooled_3wayFE_within_lemma"]["importance_llm_z"]
    print(f"   pooled 3-way FE within-lemma: beta={p['beta']:+.4f} SE={p['se']:.4f} "
          f"p={p['p']:.3g}  n={v['pooled_3wayFE_within_lemma']['_n']}")
print("\n convergent measures on log gaze (within-lemma pooled):")
for k, v in h1.get("convergent_log_gaze", {}).items():
    b = v["pooled_3wayFE_within_lemma"][k]
    print(f"  {k:26s} beta={b['beta']:+.4f} p={b['p']:.3g} | subject boot p={v['subject_bootstrap']['p']:.3g}")
print(f"\n descriptive: gaze {h1['descriptive_ms']['gaze_dur_importance_le3']:.1f} ms (imp<=3) -> "
      f"{h1['descriptive_ms']['gaze_dur_importance_5']:.1f} ms (imp=5), {h1['descriptive_ms']['pct_diff']:+.1f}%")
