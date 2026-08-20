#!/usr/bin/env python3
"""Did the reader encode THIS fact? -- item-anchored analysis of comprehension.

Page-level averaging found nothing physiological. This asks the sharper question: for each
question, did the reader's eyes and brain engage the ~26 words that actually answer it, and
does that predict answering it correctly several minutes later?

Five tests, each with its own control:

  T1  evidence-region coverage vs a MATCHED control region on the same page. Both regions
      are equal-size and equally covered on average, so a difference cannot be "read more".
  T2  RANDOM-REGION PERMUTATION. The evidence region is replaced by a random equal-size set
      of sentences from the same page, 1000 times, refitting the same estimator. This is the
      decisive control on the localisation: if my sentence-picking were noise, the real
      evidence region would sit in the middle of that null.
  T3  the encoding lesion -- accuracy when the answer text was never fixated at all, within
      reader and within item.
  T4  item-type dissociation. single_fact items should depend on evidence dwell; NEGATED
      items ("which is NOT true") should depend on broad page coverage instead, because the
      correct response is the option the page never stated. A dissociation here is internal
      validation that the spans mean something.
  T5  gaze-matched neural subsequent-memory effect: among trials where the evidence region
      WAS read, does the fixation-related potential there separate later-correct from
      later-wrong -- and does it do so more than the same contrast on the control region?
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

ROOT = Path(__file__).resolve().parents[2]
STIM = ROOT / "data" / "derivatives" / "stimuli" / "wiki_stories"
COUP = ROOT / "roamm" / "artifacts" / "coupling"
OUT = ROOT / "roamm" / "artifacts" / "comprehension"
RNG = np.random.default_rng(31337)
NPERM = 1000

D = pd.read_parquet(OUT / "evidence_trials.parquet")
E = pd.read_parquet(OUT / "item_evidence.parquet")
rep: dict = {}


def z(s):
    return (s - s.mean()) / s.std()


D["ev_cov_z"] = z(D["evidence_cov"])
D["ctrl_cov_z"] = z(D["control_cov"].fillna(D["control_cov"].mean()))
D["page_cov_z"] = z(D["coverage"])
D["ev_dwell_z"] = z(np.log1p(D["evidence_dwell_per_word"]))
D["ctrl_dwell_z"] = z(np.log1p(D["control_dwell_per_word"].fillna(0)))
D["subject_story"] = D["sub_id"] + "_" + D["reading"]


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
                  "OR": float(np.exp(mu))}
    return out


# ------------------------------------------------------- T1 evidence vs matched control
rep["T1_evidence_vs_matched_control"] = glmm(D, "correct", ["ev_cov_z", "ctrl_cov_z"])
rep["T1_with_page_coverage"] = glmm(D, "correct", ["ev_cov_z", "ctrl_cov_z", "page_cov_z"])
rep["T1_dwell_version"] = glmm(D, "correct", ["ev_dwell_z", "ctrl_dwell_z"])

d1 = rep["T1_evidence_vs_matched_control"]
diff = d1["ev_cov_z"]["beta"] - d1["ctrl_cov_z"]["beta"]
sed = np.hypot(d1["ev_cov_z"]["sd"], d1["ctrl_cov_z"]["sd"])  # conservative, ignores covariance
rep["T1_difference"] = {"beta_diff": float(diff), "se_upper_bound": float(sed),
                        "z_conservative": float(diff / sed),
                        "p_conservative": float(2 * stats.norm.sf(abs(diff / sed)))}

# descriptive curve
for col, lab in [("evidence_cov", "evidence"), ("control_cov", "control")]:
    b = pd.cut(D[col], [-0.01, 0.001, 0.25, 0.5, 0.75, 1.01],
               labels=["never read", "<25%", "25-50%", "50-75%", ">75%"])
    rep[f"accuracy_by_{lab}_coverage"] = (
        D.assign(b=b).groupby("b", observed=True)
        .agg(n=("correct", "size"), acc=("correct", "mean"), skip=("skipped", "mean")).round(4).to_dict("index")
    )

# ------------------------------------------------------- T2 random-region permutation
# per (reader, page): boolean vector over the page's words, in page order
sent_map, page_words = {}, {}
for stem in E["story_phys"].unique():
    c = pd.read_csv(STIM / f"{stem}_coordinates.csv")
    for pg, g in c.groupby("page"):
        keys = g["word_key"].tolist()
        page_words[(stem, int(pg))] = keys
        sids = list(dict.fromkeys(g["sentence_id"]))
        sent_map[(stem, int(pg))] = [
            [keys.index(k) for k in g.loc[g["sentence_id"] == s, "word_key"]] for s in sids
        ]

fx = pd.read_parquet(COUP / "reading_fixations.parquet", columns=["subject", "story", "page", "word_key", "fix_dur"])
fx = fx[fx["fix_dur"].between(50, 1000)]
subs = sorted(D["sub_id"].unique())
fx["sub_id"] = fx["subject"].map({i: s for i, s in enumerate(subs)})
seen = fx.groupby(["sub_id", "story", "page"])["word_key"].apply(set).to_dict()

FIX = {}
for (stem, pg), keys in page_words.items():
    idx = {k: i for i, k in enumerate(keys)}
    for s in subs:
        v = np.zeros(len(keys), bool)
        for k in seen.get((s, stem, pg), ()):
            j = idx.get(k)
            if j is not None:
                v[j] = True
        FIX[(s, stem, pg)] = v

item_meta = E.set_index("item")[["story_phys", "page", "n_evidence_words"]].to_dict("index")
key_order = list(zip(D["sub_id"], D["item"]))

# fixed-effects LPM statistic via Frisch-Waugh, matching the T1 contrast in spirit
X0 = pd.concat([
    pd.get_dummies(D["sub_id"], prefix="s", drop_first=True).astype(float).reset_index(drop=True),
    pd.get_dummies(D["item"], prefix="i", drop_first=True).astype(float).reset_index(drop=True),
    D[["ctrl_cov_z"]].reset_index(drop=True),
], axis=1)
X0 = sm.add_constant(X0, has_constant="add").to_numpy(float)
Q, _ = np.linalg.qr(X0)
res = lambda v: v - Q @ (Q.T @ v)
yv = res(D["correct"].to_numpy(float))
obs_stat = float((res(D["ev_cov_z"].to_numpy(float)) * yv).sum() / (res(D["ev_cov_z"].to_numpy(float)) ** 2).sum())

null = np.empty(NPERM)
for b in range(NPERM):
    cov = np.empty(len(D))
    for it, meta in item_meta.items():
        stem, pg, nw = meta["story_phys"], meta["page"], meta["n_evidence_words"]
        sents = sent_map[(stem, pg)]
        order = RNG.permutation(len(sents))
        pick, n = [], 0
        for j in order:
            if n >= nw:
                break
            pick += sents[j]
            n += len(sents[j])
        pick = np.array(pick[: max(nw, 1)])
        rows = np.flatnonzero((D["item"] == it).to_numpy())
        for r in rows:
            v = FIX[(D["sub_id"].iat[r], stem, pg)]
            cov[r] = v[pick].mean() if pick.size else 0.0
    cz = (cov - cov.mean()) / (cov.std() + 1e-12)
    rz = res(cz)
    null[b] = (rz * yv).sum() / max((rz * rz).sum(), 1e-12)

rep["T2_random_region_permutation"] = {
    "n_perm": NPERM,
    "obs_evidence_stat": obs_stat,
    "null_mean": float(null.mean()), "null_sd": float(null.std()),
    "p_one_sided": float(((null >= obs_stat).sum() + 1) / (NPERM + 1)),
    "percentile_of_observed": float((null < obs_stat).mean() * 100),
}

# ------------------------------------------------------- T3 the encoding lesion
D["ev_never"] = (D["evidence_n_fix"] == 0).astype(int)
lesion = D.groupby("ev_never").agg(n=("correct", "size"), acc=("correct", "mean"),
                                   skip=("skipped", "mean"), page_cov=("coverage", "mean")).round(4)
rep["T3_lesion_descriptive"] = lesion.to_dict("index")
rep["T3_lesion_model"] = glmm(D, "correct", ["ev_never", "page_cov_z", "ctrl_cov_z"])

# ------------------------------------------------------- T4 item-type dissociation
rep["T4_by_item_type"] = {}
for t in ["single_fact", "negated"]:
    sub = D[D["item_type"] == t].copy()
    for c in ["ev_cov_z", "ctrl_cov_z", "page_cov_z"]:
        sub[c] = z(sub[c])
    rep["T4_by_item_type"][t] = glmm(sub, "correct", ["ev_cov_z", "page_cov_z"])
# formal interaction on the pooled set
D["is_single"] = (D["item_type"] == "single_fact").astype(float)
D["ev_x_single"] = D["ev_cov_z"] * D["is_single"]
rep["T4_interaction"] = glmm(D[D["item_type"].isin(["single_fact", "negated"])],
                             "correct", ["ev_cov_z", "ev_x_single", "is_single", "page_cov_z"])

# ------------------------------------------------------- T5 neural subsequent memory
N = D[(D["evidence_n_frp"] >= 5) & (D["control_n_frp"] >= 5)].copy()
rep["T5_n_trials"] = int(len(N))
rep["T5_frp_dm"] = {}
for band in ["n400", "occ_n1", "occ_p2", "front_late"]:
    ev, ct = f"evidence_{band}", f"control_{band}"
    sub = N.dropna(subset=[ev, ct]).copy()
    sub["ev_z"] = z(sub[ev])
    sub["ct_z"] = z(sub[ct])
    sub["diff_z"] = z(sub[ev] - sub[ct])
    rep["T5_frp_dm"][band] = {
        "evidence_region": glmm(sub, "correct", ["ev_z", "ct_z", "ev_dwell_z", "ev_cov_z"])["ev_z"],
        "evidence_minus_control": glmm(sub, "correct", ["diff_z", "ev_dwell_z", "ev_cov_z"])["diff_z"],
        "n": int(len(sub)),
    }

(OUT / "evidence_region_report.json").write_text(json.dumps(rep, indent=2, default=str) + "\n")
print(json.dumps(rep, indent=2, default=str))
