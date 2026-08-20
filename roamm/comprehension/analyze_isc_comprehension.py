#!/usr/bin/env python3
"""Race per-page inter-subject alignment against per-page lexical/semantic coupling
for the prediction of whether that page's comprehension question was answered correctly.

PAPER_PLAN open item: "test whether ISC change predicts comprehension or another
independent outcome; otherwise retain 'shared-response alignment' language."

Claims tested, per page (one 4-alternative question per page):
  Q1  does alignment (gaze / N400) predict comprehension?
  Q2  does it survive the self-reported MW flag, page duration and coverage?
  Q3  do the coupling channels reported as PRESERVED during MW (frequency -> fixation
      duration, surprisal -> N400) predict comprehension? A null there, paired with a
      positive for alignment, is a double dissociation rather than a power failure --
      so nulls get TOST equivalence tests on the same standardized scale.
  Q4  is alignment on the mediating path between MW and comprehension?

Every effect is reported twice: a crossed random-effects GLMM (subject + item), and a
within-cell linear probability model absorbing subject x story and item, so that only
within-reader, within-story variation is used.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "roamm" / "artifacts" / "comprehension"
RNG = np.random.default_rng(77)
NBOOT = 4000
EQ_BOUND = 0.2  # TOST bound: |log-odds per SD| < 0.2 counts as practically null

pages = pd.read_parquet(OUT / "pages_aug.parquet")
isc = pd.read_parquet(OUT / "page_isc.parquet")
d = pages.merge(isc.drop(columns=["subject"]), on=["sub_id", "story_phys", "page"], how="left", validate="one_to_one")

PRED = [
    "isc_gaze_raw_z", "isc_gaze_resid_z", "isc_n400_raw_z", "isc_n400_resid_z", "isc_occp2_resid_z",
    "b_zipf_gaze", "b_surp_gaze", "b_surp_n400",
]
CTRL = ["log_page_dur_z", "coverage", "n_fixations", "mean_logdur", "n_gaze_raw", "page_z", "run_z"]


def z(s):
    return (s - s.mean()) / s.std()


for c in PRED + CTRL:
    d[c + "_s"] = z(d[c])
# within-reader standardisation: strips any between-subject ability contribution
for c in PRED:
    d[c + "_w"] = d.groupby("sub_id")[c].transform(lambda x: (x - x.mean()) / x.std())

CTRLS = [c + "_s" for c in CTRL]
rep: dict = {"n_pages_total": int(len(d)), "n_pages_with_isc": int(d["isc_gaze_raw_z"].notna().sum())}


# ------------------------------------------------------------------ estimators
def glmm(dd, outcome, terms):
    cols = [outcome] + terms + ["sub_id", "item"]
    dd = dd.dropna(subset=[c for c in cols if c not in ("sub_id", "item")]).copy()
    vcf = {"subject": "0 + C(sub_id)", "item": "0 + C(item)"}
    m = BinomialBayesMixedGLM.from_formula(f"{outcome} ~ " + " + ".join(terms), vcf, dd).fit_vb(verbose=False)
    names = list(m.model.exog_names)
    out = {"n": int(len(dd))}
    for k in terms:
        i = names.index(k)
        mu, sd = float(m.fe_mean[i]), float(m.fe_sd[i])
        zz = mu / sd
        out[k] = {
            "beta": mu, "sd": sd, "z": zz, "p": float(2 * stats.norm.sf(abs(zz))),
            "OR": float(np.exp(mu)), "OR_ci": [float(np.exp(mu - 1.96 * sd)), float(np.exp(mu + 1.96 * sd))],
            # TOST against +/- EQ_BOUND log-odds per SD
            "p_tost": float(max(stats.norm.sf((mu + EQ_BOUND) / sd), stats.norm.cdf((mu - EQ_BOUND) / sd))),
        }
    return out


def lpm(dd, outcome, terms, absorb=("subject_story", "item")):
    cols = [outcome] + terms
    dd = dd.dropna(subset=cols).copy()
    X = [dd[terms].reset_index(drop=True)]
    for a in absorb:
        X.append(pd.get_dummies(dd[a], prefix=a, drop_first=True).astype(float).reset_index(drop=True))
    X = sm.add_constant(pd.concat(X, axis=1), has_constant="add")
    m = sm.OLS(dd[outcome].to_numpy(float), X.to_numpy(float)).fit(
        cov_type="cluster", cov_kwds={"groups": dd["sub_id"].astype("category").cat.codes}
    )
    out = {"n": int(len(dd))}
    for k in terms:
        i = list(X.columns).index(k)
        b, se = float(m.params[i]), float(m.bse[i])
        out[k] = {
            "beta_accuracy_points": b, "se": se, "t": b / se,
            "p": float(2 * stats.norm.sf(abs(b / se))),
            "ci95": [b - 1.96 * se, b + 1.96 * se],
            "p_tost_0p03": float(max(stats.norm.sf((b + 0.03) / se), stats.norm.cdf((b - 0.03) / se))),
        }
    return out


# ------------------------------------------------------------------ Q1: alignment alone
rep["Q1_alignment_alone"] = {}
for p in ["isc_gaze_raw_z_s", "isc_gaze_resid_z_s", "isc_n400_raw_z_s", "isc_n400_resid_z_s", "isc_occp2_resid_z_s"]:
    rep["Q1_alignment_alone"][p] = {
        "glmm": glmm(d, "correct", [p] + CTRLS)[p],
        "lpm_within": lpm(d, "correct", [p] + CTRLS)[p],
    }

# within-reader standardised version of the same predictors
rep["Q1_within_reader_standardised"] = {
    p: glmm(d, "correct", [p] + CTRLS)[p]
    for p in ["isc_gaze_resid_z_w", "isc_n400_resid_z_w"]
}

# ------------------------------------------------------------------ Q2: + MW flag
rep["Q2_alignment_plus_mw"] = {
    "glmm": glmm(d, "correct", ["mw", "isc_gaze_resid_z_s", "isc_n400_resid_z_s"] + CTRLS),
    "lpm_within": lpm(d, "correct", ["mw", "isc_gaze_resid_z_s", "isc_n400_resid_z_s"] + CTRLS),
}

# ------------------------------------------------------------------ Q3: coupling channels
rep["Q3_coupling_channels"] = {}
for p in ["b_zipf_gaze_s", "b_surp_gaze_s", "b_surp_n400_s"]:
    rep["Q3_coupling_channels"][p] = {
        "glmm": glmm(d, "correct", [p] + CTRLS)[p],
        "lpm_within": lpm(d, "correct", [p] + CTRLS)[p],
    }
rep["Q3_horse_race"] = glmm(
    d, "correct",
    ["mw", "isc_gaze_resid_z_s", "isc_n400_resid_z_s", "b_zipf_gaze_s", "b_surp_n400_s"] + CTRLS,
)

# --------------------------------------------------- does MW reduce page alignment?
rep["mw_effect_on_page_alignment"] = {}
for p in ["isc_gaze_raw_z", "isc_gaze_resid_z", "isc_n400_raw_z", "isc_n400_resid_z",
          "b_zipf_gaze", "b_surp_n400"]:
    dd = d.dropna(subset=[p]).copy()
    X = pd.concat(
        [dd[["mw"] + CTRLS].reset_index(drop=True),
         pd.get_dummies(dd["subject_story"], prefix="ss", drop_first=True).astype(float).reset_index(drop=True),
         pd.get_dummies(dd["item"], prefix="it", drop_first=True).astype(float).reset_index(drop=True)],
        axis=1,
    )
    X = sm.add_constant(X, has_constant="add")
    m = sm.OLS(z(dd[p]).to_numpy(float), X.to_numpy(float)).fit(
        cov_type="cluster", cov_kwds={"groups": dd["sub_id"].astype("category").cat.codes}
    )
    b, se = float(m.params[0]), float(m.bse[0])
    rep["mw_effect_on_page_alignment"][p] = {
        "beta_sd_units": b, "se": se, "t": b / se, "p": float(2 * stats.norm.sf(abs(b / se))),
        "p_tost_0p2sd": float(max(stats.norm.sf((b + 0.2) / se), stats.norm.cdf((b - 0.2) / se))),
        "n": int(len(dd)),
    }

# ------------------------------------------------------------------ Q4: mediation
def mediation(dd, med, boot=NBOOT):
    """MW -> mediator -> correct, linear paths, subject-cluster bootstrap on the indirect."""
    dd = dd.dropna(subset=[med, "correct", "mw"] + CTRLS).copy()
    M = z(dd[med]).to_numpy(float)
    y = dd["correct"].to_numpy(float)
    mwv = dd["mw"].to_numpy(float)
    C = dd[CTRLS].to_numpy(float)
    Xa = np.column_stack([np.ones(len(dd)), mwv, C])                 # mediator ~ mw + ctrl
    Xb = np.column_stack([np.ones(len(dd)), M, mwv, C])              # correct ~ med + mw + ctrl
    groups = [np.flatnonzero(dd["sub_id"].to_numpy() == u) for u in dd["sub_id"].unique()]

    def paths(idx):
        a = np.linalg.lstsq(Xa[idx], M[idx], rcond=None)[0][1]
        bb = np.linalg.lstsq(Xb[idx], y[idx], rcond=None)[0]
        c = np.linalg.lstsq(Xa[idx], y[idx], rcond=None)[0][1]
        return a, bb[1], bb[2], c

    all_idx = np.arange(len(dd))
    a, b, cp, c = paths(all_idx)
    ind = np.empty(boot)
    for i in range(boot):
        take = RNG.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[t] for t in take])
        try:
            aa, bbv, _, _ = paths(idx)
            ind[i] = aa * bbv
        except Exception:
            ind[i] = np.nan
    ind = ind[np.isfinite(ind)]
    return {
        "a_mw_to_mediator_sd": float(a), "b_mediator_to_correct": float(b),
        "c_total": float(c), "c_prime_direct": float(cp), "indirect_ab": float(a * b),
        "indirect_ci95": [float(np.nanpercentile(ind, 2.5)), float(np.nanpercentile(ind, 97.5))],
        "prop_mediated": float(a * b / c) if c != 0 else np.nan,
        "n": int(len(dd)),
    }


rep["Q4_mediation"] = {
    "isc_gaze_resid_z": mediation(d, "isc_gaze_resid_z"),
    "isc_n400_resid_z": mediation(d, "isc_n400_resid_z"),
    "coverage": mediation(d, "coverage"),
}

# ------------------------------------------------------------------ descriptives
q = pd.qcut(d["isc_gaze_resid_z"], 5, labels=False, duplicates="drop")
rep["accuracy_by_alignment_quintile"] = (
    d.assign(q=q).dropna(subset=["q"]).groupby("q")
    .agg(n=("correct", "size"), p_correct=("correct", "mean"), p_mw=("mw", "mean"),
         isc=("isc_gaze_resid", "mean"), page_dur=("page_dur", "mean"))
    .round(4).to_dict("index")
)

d.to_parquet(OUT / "pages_full.parquet", index=False)
(OUT / "isc_comprehension_report.json").write_text(json.dumps(rep, indent=2, default=str) + "\n")
print(json.dumps(rep, indent=2, default=str))
