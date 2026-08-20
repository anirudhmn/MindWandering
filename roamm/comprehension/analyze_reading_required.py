#!/usr/bin/env python3
"""Re-run the comprehension tests inside the reading-dependent headroom.

A language model with no passage answers these items at ~.52 while humans who read the page
reach .618. So most of the outcome variance is not about reading at all, and every earlier
null was evaluated against a dynamic range nobody had measured. Two consequences:

  A. Items differ enormously in how much reading they require. `reading_headroom` =
     1 - p(correct option | no passage) is a per-item measure of that. Every predictor should
     be tested for an INTERACTION with headroom: an effect of attention or physiology should
     concentrate in items that actually require having read the page.
  B. The subset of high-headroom items is where a physiological effect could live. The full
     ladder is re-run there.

This file also tests a page-level predictor the earlier analyses simply never included:
MEAN FIXATION-RELATED POTENTIAL AMPLITUDE. The earlier sweep tested alignment (ISC) and
coupling SLOPES, not the amplitude of the evoked response itself, which is the classic
subsequent-memory measure in the natural-reading literature.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

ROOT = Path(__file__).resolve().parents[1]
COUP = ROOT / "roamm" / "artifacts" / "coupling"
OUT = ROOT / "roamm" / "artifacts" / "comprehension"

FIX_RANGE = (50, 1000)
P2P_MAX_UV = 150.0
FRP_COLS = ["frp_occ_P1", "frp_occ_N1", "frp_occ_P2", "frp_cp_mid", "frp_cp_N400", "frp_front_late"]


def z(s):
    return (s - s.mean()) / s.std()


def glmm(dd, outcome, terms):
    dd = dd.dropna(subset=[outcome] + terms).copy()
    vcf = {"subject": "0 + C(sub_id)", "item": "0 + C(item)"}
    m = BinomialBayesMixedGLM.from_formula(f"{outcome} ~ " + " + ".join(terms), vcf, dd).fit_vb(verbose=False)
    names = list(m.model.exog_names)
    out = {"n": int(len(dd))}
    for k in terms:
        i = names.index(k)
        mu, sd = float(m.fe_mean[i]), float(m.fe_sd[i])
        out[k] = {"beta": mu, "sd": sd, "z": mu / sd, "p": float(2 * stats.norm.sf(abs(mu / sd))),
                  "OR": float(np.exp(mu)),
                  "p_tost_0p2": float(max(stats.norm.sf((mu + 0.2) / sd), stats.norm.cdf((mu - 0.2) / sd)))}
    return out


# ------------------------------------------------- page-level mean FRP (never tested before)
frp = pd.read_parquet(COUP / "fixations_frp.parquet")
frp = frp[frp["frp_valid"] & frp["fix_dur"].between(*FIX_RANGE) & (frp["frp_p2p"] * 1e6 <= P2P_MAX_UV)].copy()
for c in FRP_COLS:
    frp[c] = frp[c] * 1e6
P = frp.groupby(["subject", "story", "page"], observed=True).agg(
    **{f"page_{c}": (c, "mean") for c in FRP_COLS}, n_frp_page=("frp_p2p", "size")
).reset_index()

pages = pd.read_parquet(OUT / "pages_full.parquet")
subs = sorted(pages["sub_id"].unique())
P["sub_id"] = P["subject"].map({i: s for i, s in enumerate(subs)})
d = pages.merge(P.drop(columns=["subject"]), left_on=["sub_id", "story_phys", "page"],
                right_on=["sub_id", "story", "page"], how="left").drop(columns=["story"])

L = pd.read_parquet(OUT / "llm_answerability_v2.parquet")
d = d.merge(L[["item", "p_gold_nopassage", "p_gold_fullpage", "p_gold_evidence",
               "reading_headroom", "llm_reading_gain", "item_type"]], on="item", how="left")
ev = pd.read_parquet(OUT / "evidence_trials.parquet")
d = d.merge(ev[["sub_id", "item", "evidence_cov", "control_cov", "evidence_n_fix",
                "evidence_occ_n1", "control_occ_n1", "evidence_n400", "evidence_n_frp"]],
            on=["sub_id", "item"], how="left")

d["headroom_z"] = z(d["reading_headroom"])
d["ev_cov_z"] = z(d["evidence_cov"])
d["page_cov_z"] = z(d["coverage"])
d["n_words_z"] = z(d["n_gaze_raw"])
for c in FRP_COLS:
    d["z_" + c] = z(d["page_" + c])
FRPZ = ["z_" + c for c in FRP_COLS]

rep: dict = {"n": int(len(d))}

# ------------------------------------------------- item difficulty decomposition
I = d.groupby("item").agg(human=("correct", "mean")).join(
    L.set_index("item")[["p_gold_nopassage", "p_gold_fullpage", "reading_headroom"]])
X = np.column_stack([np.ones(len(I)), I["p_gold_nopassage"], I["p_gold_fullpage"]])
beta, *_ = np.linalg.lstsq(X, I["human"].to_numpy(), rcond=None)
pred = X @ beta
rep["item_difficulty_decomposition"] = {
    "n_items": int(len(I)),
    "r_human_vs_nopassage": float(np.corrcoef(I["p_gold_nopassage"], I["human"])[0, 1]),
    "r_human_vs_fullpage": float(np.corrcoef(I["p_gold_fullpage"], I["human"])[0, 1]),
    "R2_both": float(1 - ((I["human"] - pred) ** 2).sum() / ((I["human"] - I["human"].mean()) ** 2).sum()),
    "beta_nopassage": float(beta[1]), "beta_fullpage": float(beta[2]),
}

# ------------------------------------------------- headroom interactions
rep["headroom_interactions"] = {}
for name, col in [("mw", "mw"), ("page_coverage", "page_cov_z"), ("evidence_coverage", "ev_cov_z"),
                  ("words_read", "n_words_z"), ("page_occ_N1", "z_frp_occ_N1"),
                  ("page_N400", "z_frp_cp_N400"), ("gaze_alignment", "isc_gaze_resid_z_s")]:
    if col not in d:
        continue
    dd = d.dropna(subset=[col, "headroom_z"]).copy()
    dd["inter"] = dd[col] * dd["headroom_z"]
    r = glmm(dd, "correct", [col, "headroom_z", "inter"])
    rep["headroom_interactions"][name] = {"main": r[col], "interaction": r["inter"], "n": r["n"]}

# ------------------------------------------------- page-level FRP amplitude, full sample
rep["page_frp_amplitude_full"] = {}
for c in FRPZ:
    rep["page_frp_amplitude_full"][c] = glmm(d, "correct", [c, "n_words_z", "log_page_dur_z"])[c]

# ------------------------------------------------- the reading-required subset
med = L["reading_headroom"].median()
hi = d[d["reading_headroom"] >= med].copy()
lo = d[d["reading_headroom"] < med].copy()
rep["subset_sizes"] = {"reading_required_trials": int(len(hi)), "knowledge_answerable_trials": int(len(lo)),
                       "reading_required_items": int(hi["item"].nunique()),
                       "headroom_median": float(med),
                       "human_acc_reading_required": float(hi["correct"].mean()),
                       "human_acc_knowledge_answerable": float(lo["correct"].mean())}

rep["reading_required_ladder"] = {}
for lab, sub in [("reading_required", hi), ("knowledge_answerable", lo)]:
    s = sub.copy()
    for c in ["page_cov_z", "ev_cov_z", "n_words_z"] + FRPZ + ["isc_gaze_resid_z_s", "isc_n400_resid_z_s"]:
        if c in s:
            s[c] = z(s[c])
    block = {}
    block["mw"] = glmm(s, "correct", ["mw", "n_words_z", "log_page_dur_z"])["mw"]
    block["page_coverage"] = glmm(s, "correct", ["page_cov_z", "log_page_dur_z"])["page_cov_z"]
    block["evidence_coverage"] = glmm(s, "correct", ["ev_cov_z", "page_cov_z"])["ev_cov_z"]
    for c in FRPZ:
        block[c] = glmm(s, "correct", [c, "n_words_z", "log_page_dur_z"])[c]
    for c in ["isc_gaze_resid_z_s", "isc_n400_resid_z_s"]:
        if c in s:
            block[c] = glmm(s, "correct", [c, "n_words_z", "log_page_dur_z"])[c]
    rep["reading_required_ladder"][lab] = block

d.to_parquet(OUT / "pages_headroom.parquet", index=False)
(OUT / "reading_required_report.json").write_text(json.dumps(rep, indent=2, default=str) + "\n")
print(json.dumps(rep, indent=2, default=str))
