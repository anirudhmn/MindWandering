#!/usr/bin/env python3
"""Does self-reported mind-wandering on a page predict failing that page's question?

This is the first outcome-based (non-physiological) test in the program: every prior
claim is physiology -> physiology. The ROAMM item bank gives one 4-alternative question
per page, answerable only from that page, so page-level MW report and page-level
comprehension can be crossed within subject and within story.

Estimators, weakest-assumption first:
  1. subject-level paired delta  p(correct|MW) - p(correct|on-task), bootstrap CI + Wilcoxon
  2. crossed random-effects Bayesian GLMM (subject + item)
  3. fixed-effects logit with subject and item dummies, subject-cluster-robust SE
  4. TIGHTEST: subject x story fixed effects (+ item) -- only within-story, within-reader
     variation in MW is used, so trait-level and story-level confounds are differenced out
  5. permutation null: shuffle the MW flag within subject x story

Outcomes: correct (chance .25), skipped ("I am not sure"), correct|answered.
Also: dose (fraction of the page spent MW) and MW timing within the page.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "roamm" / "artifacts" / "comprehension"
RNG = np.random.default_rng(2024)
NBOOT = 10000
NPERM = 10000

d = pd.read_parquet(OUT / "pages.parquet")
d["mw"] = d["is_MWreported"].astype(int)
d["subject_story"] = d["sub_id"] + "_" + d["reading"]
for c in ["understand", "prior_knowledge"]:
    d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std()
d["log_page_dur_z"] = (d["log_page_dur"] - d["log_page_dur"].mean()) / d["log_page_dur"].std()

rep: dict = {"n_pages": int(len(d)), "n_subjects": int(d.sub_id.nunique()), "n_items": int(d.item.nunique())}


# ------------------------------------------------------------------ 1. paired delta
def paired_delta(dd, outcome, group="sub_id", min_n=3):
    piv = dd.groupby([group, "mw"])[outcome].agg(["mean", "count"]).unstack("mw")
    ok = (piv[("count", 0)] >= min_n) & (piv[("count", 1)] >= min_n)
    piv = piv[ok]
    delta = (piv[("mean", 1)] - piv[("mean", 0)]).dropna().to_numpy()
    boot = np.array([RNG.choice(delta, delta.size).mean() for _ in range(NBOOT)])
    t, p = stats.ttest_1samp(delta, 0)
    try:
        w, pw = stats.wilcoxon(delta)
    except ValueError:
        w, pw = np.nan, np.nan
    return {
        "n_units": int(delta.size),
        "mean_ontask": float(piv[("mean", 0)].mean()),
        "mean_mw": float(piv[("mean", 1)].mean()),
        "delta": float(delta.mean()),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "t": float(t),
        "p_t": float(p),
        "p_wilcoxon": float(pw),
        "frac_units_negative": float((delta < 0).mean()),
        "cohens_dz": float(delta.mean() / delta.std(ddof=1)),
    }


rep["paired_subject"] = {o: paired_delta(d, o) for o in ["correct", "skipped"]}
rep["paired_subject"]["correct_answered"] = paired_delta(d.dropna(subset=["correct_answered"]), "correct_answered")
# within subject x story (removes any story-level or trait-level confound)
rep["paired_subject_story"] = {
    o: paired_delta(d, o, group="subject_story", min_n=2) for o in ["correct", "skipped"]
}

# ------------------------------------------------------------------ 2. Bayesian GLMM
FIX = "mw + log_page_dur_z + page_z + run_z + understand_z + prior_knowledge_z"


def glmm(dd, outcome, fixed=FIX):
    vcf = {"subject": "0 + C(sub_id)", "item": "0 + C(item)"}
    m = BinomialBayesMixedGLM.from_formula(f"{outcome} ~ {fixed}", vcf, dd)
    r = m.fit_vb(verbose=False)
    names = list(r.model.exog_names)
    out = {}
    for k in names:
        i = names.index(k)
        mu, sd = float(r.fe_mean[i]), float(r.fe_sd[i])
        z = mu / sd
        out[k] = {
            "beta": mu,
            "sd": sd,
            "z": z,
            "p_approx": float(2 * stats.norm.sf(abs(z))),
            "OR": float(np.exp(mu)),
            "OR_ci": [float(np.exp(mu - 1.96 * sd)), float(np.exp(mu + 1.96 * sd))],
        }
    return out


rep["glmm_correct"] = glmm(d, "correct")
rep["glmm_skipped"] = glmm(d, "skipped")
rep["glmm_correct_unadjusted"] = glmm(d, "correct", fixed="mw")
rep["glmm_correct_answered"] = glmm(d.dropna(subset=["correct_answered"]).assign(ca=lambda x: x.correct_answered.astype(int)), "ca")


# ---------------------------------------------- 3/4. fixed-effects logit, cluster-robust
def felogit(dd, outcome, absorb, extra=""):
    f = f"{outcome} ~ mw + log_page_dur_z + page_z + run_z{extra} + C({absorb})"
    if absorb != "item":
        f += " + C(item)"
    m = smf.logit(f, data=dd).fit(
        disp=0, maxiter=200, cov_type="cluster",
        cov_kwds={"groups": dd["sub_id"].astype("category").cat.codes.to_numpy()},
    )
    b, se = float(m.params["mw"]), float(m.bse["mw"])
    return {
        "beta": b,
        "se_cluster": se,
        "z": b / se,
        "p": float(2 * stats.norm.sf(abs(b / se))),
        "OR": float(np.exp(b)),
        "OR_ci": [float(np.exp(b - 1.96 * se)), float(np.exp(b + 1.96 * se))],
        "n": int(len(dd)),
    }


rep["felogit_subject_item"] = felogit(d, "correct", "sub_id", extra=" + understand_z + prior_knowledge_z")

# 4. tightest specification: linear probability model absorbing subject x story AND item.
# LPM rather than logit here because ~220 cell dummies invite separation; the coefficient
# reads directly in accuracy points, and SEs are clustered on subject.
def lpm(dd, outcome, absorb=("subject_story", "item"), covars=("log_page_dur_z", "page_z")):
    X = [dd[list(covars)].reset_index(drop=True)]
    for a in absorb:
        X.append(pd.get_dummies(dd[a], prefix=a, drop_first=True).astype(float).reset_index(drop=True))
    X = sm.add_constant(pd.concat(X, axis=1), has_constant="add")
    X.insert(0, "mw", dd["mw"].to_numpy(float))
    m = sm.OLS(dd[outcome].to_numpy(float), X.to_numpy(float)).fit(
        cov_type="cluster", cov_kwds={"groups": dd["sub_id"].astype("category").cat.codes}
    )
    return {
        "beta_accuracy_points": float(m.params[0]),
        "se_cluster": float(m.bse[0]),
        "t": float(m.tvalues[0]),
        "p": float(m.pvalues[0]),
        "ci95": [float(m.params[0] - 1.96 * m.bse[0]), float(m.params[0] + 1.96 * m.bse[0])],
        "n": int(len(dd)),
    }


rep["lpm_subject_x_story_plus_item"] = lpm(d, "correct")
rep["lpm_skipped_subject_x_story_plus_item"] = lpm(d, "skipped")
rep["lpm_correct_no_duration_control"] = lpm(d, "correct", covars=("page_z",))

# conditional (within-cell) logit as the odds-scale version of the same contrast
try:
    from statsmodels.discrete.conditional_models import ConditionalLogit

    # item effects need no dummies here: every subject x story cell contains each of that
    # story's 10 items exactly once, so item is orthogonal to the conditioning strata
    Xc_ = d[["mw", "log_page_dur_z", "page_z"]].reset_index(drop=True)
    cl = ConditionalLogit(d["correct"].to_numpy(float), Xc_.to_numpy(float),
                          groups=d.groupby("subject_story").ngroup().to_numpy()).fit(disp=0)
    rep["conditional_logit_subject_x_story"] = {
        "beta": float(cl.params[0]), "se": float(cl.bse[0]),
        "OR": float(np.exp(cl.params[0])), "p": float(cl.pvalues[0]),
    }
except Exception as e:  # pragma: no cover
    rep["conditional_logit_subject_x_story"] = {"error": str(e)}

# ------------------------------------------------------------------ 5. permutation null
# Shuffle the MW flag within subject x story: preserves each reader's per-story MW count
# and every page/item/subject marginal. Statistic = the LPM coefficient above, computed by
# Frisch-Waugh residualisation so 10k permutations are one matmul.
codes = d.groupby("subject_story").ngroup().to_numpy()
mwv = d["mw"].to_numpy(float)
order = np.argsort(codes, kind="stable")
blocks = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)

Xc = pd.concat(
    [d[["log_page_dur_z", "page_z"]].reset_index(drop=True),
     pd.get_dummies(d["subject_story"], prefix="ss", drop_first=True).astype(float).reset_index(drop=True),
     pd.get_dummies(d["item"], prefix="it", drop_first=True).astype(float).reset_index(drop=True)],
    axis=1,
)
Xc = sm.add_constant(Xc, has_constant="add").to_numpy(float)
Q, _ = np.linalg.qr(Xc)
resid = lambda V: V - Q @ (Q.T @ V)
y = d["correct"].to_numpy(float)
yr = resid(y)

P = np.empty((len(d), NPERM))
for b in range(NPERM):
    col = mwv.copy()
    for blk in blocks:
        col[blk] = RNG.permutation(mwv[blk])
    P[:, b] = col
Pr = resid(P)
null = (Pr * yr[:, None]).sum(0) / np.maximum((Pr * Pr).sum(0), 1e-12)
mr = resid(mwv)
obs = float((mr * yr).sum() / (mr * mr).sum())
rep["permutation"] = {
    "n_perm": int(NPERM),
    "scheme": "shuffle MW within subject x story; statistic = FWL/LPM coefficient",
    "obs_beta": obs,
    "null_mean": float(null.mean()),
    "null_sd": float(null.std()),
    "p_two_sided": float(((np.abs(null) >= abs(obs)).sum() + 1) / (NPERM + 1)),
    "p_one_sided_negative": float(((null <= obs).sum() + 1) / (NPERM + 1)),
}

# ------------------------------------------------------------------ dose & timing
mwp = d[d["mw"] == 1].copy()
mwp["mw_frac_z"] = (mwp["mw_frac_page"] - mwp["mw_frac_page"].mean()) / mwp["mw_frac_page"].std()
mwp["mw_dur_z"] = (np.log(mwp["mw_dur"].clip(lower=0.2)) - np.log(mwp["mw_dur"].clip(lower=0.2)).mean()) / np.log(mwp["mw_dur"].clip(lower=0.2)).std()
mwp["recency_z"] = (np.log(mwp["offset2page_end"].clip(lower=0.05)) - np.log(mwp["offset2page_end"].clip(lower=0.05)).mean()) / np.log(mwp["offset2page_end"].clip(lower=0.05)).std()
rep["dose_within_mw_pages"] = {}
for v in ["mw_frac_z", "mw_dur_z", "recency_z"]:
    sub = mwp.dropna(subset=[v, "correct", "log_page_dur_z", "page_z", "run_z"]).copy()
    vcf = {"subject": "0 + C(sub_id)", "item": "0 + C(item)"}
    m = BinomialBayesMixedGLM.from_formula(f"correct ~ {v} + log_page_dur_z + page_z + run_z", vcf, sub).fit_vb(verbose=False)
    names = list(m.model.exog_names)
    i = names.index(v)
    mu, sd = float(m.fe_mean[i]), float(m.fe_sd[i])
    rep["dose_within_mw_pages"][v] = {
        "beta": mu, "sd": sd, "z": mu / sd, "p_approx": float(2 * stats.norm.sf(abs(mu / sd))),
        "n": int(len(sub)),
    }

# ---------------------------------------------- MW elsewhere in the story (specificity)
other = d.groupby(["sub_id", "reading"])["mw"].transform("sum")
d["mw_other_pages"] = other - d["mw"]
d["mw_other_z"] = (d["mw_other_pages"] - d["mw_other_pages"].mean()) / d["mw_other_pages"].std()
vcf = {"subject": "0 + C(sub_id)", "item": "0 + C(item)"}
m = BinomialBayesMixedGLM.from_formula(
    "correct ~ mw + mw_other_z + log_page_dur_z + page_z + run_z", vcf, d
).fit_vb(verbose=False)
names = list(m.model.exog_names)
rep["specificity_this_page_vs_rest_of_story"] = {
    k: {"beta": float(m.fe_mean[names.index(k)]), "sd": float(m.fe_sd[names.index(k)]),
        "z": float(m.fe_mean[names.index(k)] / m.fe_sd[names.index(k)]),
        "p_approx": float(2 * stats.norm.sf(abs(m.fe_mean[names.index(k)] / m.fe_sd[names.index(k)])))}
    for k in ["mw", "mw_other_z"]
}

# ---------------------------------------------- descriptive cross-tab
ct = d.groupby("mw").agg(n=("correct", "size"), p_correct=("correct", "mean"),
                         p_skip=("skipped", "mean"),
                         p_correct_answered=("correct_answered", "mean"),
                         mean_page_dur=("page_dur", "mean"))
rep["crosstab"] = ct.round(4).to_dict("index")

(OUT / "mw_comprehension_report.json").write_text(json.dumps(rep, indent=2, default=str) + "\n")
d.to_parquet(OUT / "pages_aug.parquet", index=False)
print(json.dumps(rep, indent=2, default=str))
