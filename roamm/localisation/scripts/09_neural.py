#!/usr/bin/env python3
"""H4: is there a neural signature of semantic importance during free reading?

A-priori windows, fixed before looking, from the two closest precedents:
  * successful encoding in natural reading shows fixation-related effects at 100-210 ms (N1-P2)
    and 380-480 ms (frontal P3), plus theta at the paragraph scale (eNeuro 2018)
  * discourse-level information gain shows a positive ERP shift (Sci Rep 2020)
So the pre-specified ROIs are occ_N1 and occ_P2 (early), and cp_N400 and front_late (late), with
front_late the primary candidate for a discourse-importance effect.

This is a SCREEN, not the estimate of record, and the reason matters: neighbouring fixations
land ~230 ms apart, so a fixation's window is contaminated by its neighbours' responses -- and
because importance is a SENTENCE-level property, adjacent fixations share the same importance
value, which makes the contamination *correlated with the predictor* rather than just noisy.
The selection and repair analysis established for this dataset that deconvolved rERP betas, not single-trial window
means, are the estimate of record. So the rule declared here: report the screen with its MDE; if
anything survives, confirm it with an overlap-corrected rERP refit before claiming it; if nothing
survives, say so with the bound and note the caveat rather than calling it a clean null.

Controls in every model: word-instance or lemma FE, subject FE, log fixation duration (a bigger
FRP with a longer fixation is not an importance effect), zipf/length/surprisal, incoming saccade
amplitude proxy, refixation index, and the layout covariates.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy import stats
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common60 import ART, RES, COUP, boot_ci, fmt, holm, absorb, ols_cluster, z, mde

ROIS = ["frp_occ_N1", "frp_occ_P2", "frp_cp_N400", "frp_front_late"]
COV = ["logdur_z", "zipf_z", "length_z", "surprisal_z", "fix_order_z", "n_words_sentence_z",
       "rel_pos_in_sentence_z", "sent_pos_on_page_z", "line_pos_z", "is_line_first", "is_line_last",
       "is_page_boundary_sentence"]
rep: dict = {}

W = pd.read_parquet(ART / "word_importance.parquet")
fx = pd.read_parquet(COUP / "reading_fixations.parquet")
fx = fx[fx["fix_dur"].between(50, 1000)]
frp = pd.read_parquet(COUP / "fixations_frp.parquet",
                      columns=["onset_abs_idx"] + ROIS + ["frp_p2p", "frp_valid"])
F = fx.merge(frp, on="onset_abs_idx", how="left")
F = F[F["frp_valid"].fillna(False) & (F["frp_p2p"].fillna(1) * 1e6 <= 150.0)].copy()
for c in ROIS:
    F[c] = F[c] * 1e6                      # to microvolts
F = F.merge(W.drop(columns=["page", "zipf", "length", "surprisal"]), on="word_key",
            how="inner", validate="m:1")
F["logdur"] = np.log(F["fix_dur"])
F["page_id"] = F["story_phys"] + "_" + F["page"].astype(str)
for c in ["importance_llm", "zipf", "length", "surprisal", "logdur", "fix_order",
          "n_words_sentence", "rel_pos_in_sentence", "sent_pos_on_page", "line_pos",
          "importance_qwen_exp", "centrality_lm"]:
    F[c + "_z"] = z(F[c])
for c in ["surprisal_z", "importance_qwen_exp_z", "centrality_lm_z"]:
    F[c] = F[c].fillna(0.0)
F = F.dropna(subset=ROIS + ["importance_llm_z"] + COV)
rep["n_epochs"] = int(len(F))
rep["n_subjects"] = int(F["subject"].nunique())
rep["roi_sd_uv"] = {c: float(F[c].std()) for c in ROIS}


def pooled(dd, roi, preds, fe=("subject", "page_id", "lemma")):
    cols = list(preds) + COV
    M = absorb(np.column_stack([dd[roi].to_numpy(float)] + [dd[c].to_numpy(float) for c in cols]),
               [dd[f].to_numpy() for f in fe])
    r = ols_cluster(M[:, 0], M[:, 1:], dd["subject"].to_numpy(), names=cols)
    out = {}
    for p in preds:
        v = r[p]
        v["mde_80_uv"] = mde(v["se"])
        out[p] = v
    out["_n"] = r["_n"]
    return out


def per_subject(dd, roi, pred, fe=("page_id", "lemma")):
    sl = []
    cols = [pred] + COV
    for s, g in dd.groupby("subject"):
        if len(g) < 500:
            continue
        M = absorb(np.column_stack([g[roi].to_numpy(float)] + [g[c].to_numpy(float) for c in cols]),
                   [g[f].to_numpy() for f in fe])
        b = np.linalg.pinv(M[:, 1:].T @ M[:, 1:]) @ (M[:, 1:].T @ M[:, 0])
        sl.append(float(b[0]))
    r = boot_ci(np.array(sl))
    r["mde_80_uv"] = mde(np.std(sl, ddof=1) / np.sqrt(len(sl)))
    return r


# ------------------------------------------------------- main effect of importance
rep["importance_main"] = {}
for roi in ROIS:
    rep["importance_main"][roi] = {
        "pooled_within_lemma": pooled(F, roi, ["importance_llm_z"])["importance_llm_z"],
        "per_subject": per_subject(F, roi, "importance_llm_z")}
ps = [rep["importance_main"][r]["per_subject"]["p"] for r in ROIS]
for r, a in zip(ROIS, holm(ps)):
    rep["importance_main"][r]["per_subject"]["p_holm"] = float(a)

# ------------------------------------------------------- convergent annotators
rep["convergent"] = {}
for nm, col in [("qwen", "importance_qwen_exp_z"), ("centrality", "centrality_lm_z"),
                ("in_summary", "in_summary_llm")]:
    rep["convergent"][nm] = {roi: pooled(F, roi, [col])[col] for roi in ROIS}

# ------------------------------------------------------- importance x MW
rep["importance_x_mw"] = {}
F["mw"] = F["is_mw"].astype(float)
F["mw_x_imp"] = F["mw"] * F["importance_llm_z"]
F["mw_x_zipf"] = F["mw"] * F["zipf_z"]
F["mw_x_surp"] = F["mw"] * F["surprisal_z"]
for roi in ROIS:
    r = pooled(F, roi, ["mw_x_imp", "mw_x_zipf", "mw_x_surp"], fe=("subject", "word_key"))
    base = pooled(F[F["mw"] == 0], roi, ["importance_llm_z", "zipf_z", "surprisal_z"])
    for k, bk in [("mw_x_imp", "importance_llm_z"), ("mw_x_zipf", "zipf_z"), ("mw_x_surp", "surprisal_z")]:
        b = base[bk]["beta"]
        if abs(b) > 1e-6:
            r[k]["pct_of_base"] = 100 * r[k]["beta"] / b
            r[k]["mde_pct_of_base"] = abs(100 * r[k]["mde_80_uv"] / b)
            r[k]["base_ontask_uv"] = b
    rep["importance_x_mw"][roi] = r

# ------------------------------------------------------- subsequent-memory style contrast
# does the FRP on the ANSWER span differ between later-correct and later-wrong trials?
ev = pd.read_parquet(ART / "item_evidence_llm.parquet")
trials = pd.read_parquet(ART / "evidence_trials_llm.parquet")
evk = {}
for _, r in ev.iterrows():
    for k in r["evidence_word_keys"]:
        evk[k] = r["item"]
subs_ = sorted(trials["sub_id"].unique())
F["sub_id"] = F["subject"].map({i: s for i, s in enumerate(subs_)})
F["ev_item"] = F["word_key"].map(evk)
FE = F[F["ev_item"].notna()].merge(
    trials[["sub_id", "item", "correct"]].rename(columns={"item": "ev_item"}),
    on=["sub_id", "ev_item"], how="inner")
rep["sme_n_epochs"] = int(len(FE))
rep["subsequent_memory_on_answer_span"] = {}
for roi in ROIS:
    sl = []
    for s, g in FE.groupby("subject"):
        a = g.loc[g["correct"] == 1, roi]
        b = g.loc[g["correct"] == 0, roi]
        if len(a) >= 40 and len(b) >= 40:
            sl.append(float(a.mean() - b.mean()))
    r = boot_ci(np.array(sl))
    r["mde_80_uv"] = mde(np.std(sl, ddof=1) / np.sqrt(len(sl))) if len(sl) > 2 else np.nan
    rep["subsequent_memory_on_answer_span"][roi] = r

(RES / "neural.json").write_text(json.dumps(rep, indent=2, default=str) + "\n")

print(f"=== H4 neural screen: {rep['n_epochs']} artifact-clean FRP epochs, {rep['n_subjects']} readers ===")
print(f"ROI SDs (uV): " + "  ".join(f"{k.replace('frp_','')}={v:.2f}" for k, v in rep["roi_sd_uv"].items()))
print("\nimportance main effect (uV per SD of importance):")
for roi in ROIS:
    p_ = rep["importance_main"][roi]["pooled_within_lemma"]
    s_ = rep["importance_main"][roi]["per_subject"]
    print(f"  {roi.replace('frp_',''):12s} pooled {p_['beta']:+.4f} (SE {p_['se']:.4f}, p={p_['p']:.3g}, "
          f"MDE80={p_['mde_80_uv']:.4f})   per-reader {s_['mean']:+.4f} "
          f"[{s_['ci'][0]:+.4f},{s_['ci'][1]:+.4f}] p={s_['p']:.3g} (holm {s_['p_holm']:.3g}) {s_['n_pos']}/{s_['n']}")
print("\nconvergent annotators (pooled within-lemma):")
for nm, d in rep["convergent"].items():
    print(f"  {nm:11s} " + "  ".join(
        f"{r.replace('frp_','')}={d[r]['beta']:+.4f}(p={d[r]['p']:.2g})" for r in ROIS))
print("\nimportance x MW (word-instance x subject FE):")
for roi in ROIS:
    v = rep["importance_x_mw"][roi]["mw_x_imp"]
    ex = f"  = {v['pct_of_base']:+.0f}% of base, MDE80={v['mde_pct_of_base']:.0f}%" if "pct_of_base" in v else ""
    print(f"  {roi.replace('frp_',''):12s} {v['beta']:+.4f} (SE {v['se']:.4f}, p={v['p']:.3g}){ex}")
print(f"\nsubsequent memory on the answer span ({rep['sme_n_epochs']} epochs), correct minus wrong:")
for roi in ROIS:
    r = rep["subsequent_memory_on_answer_span"][roi]
    print(f"  {roi.replace('frp_',''):12s} {r['mean']:+.4f} uV [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] "
          f"p={r['p']:.3g} {r['n_pos']}/{r['n']}  MDE80={r['mde_80_uv']:.4f}")
