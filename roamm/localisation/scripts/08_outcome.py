#!/usr/bin/env python3
"""H5: does encoding the answer predict answering the question -- and is mind-wandering's cost
INFORMATIONALLY SPECIFIC?

Page-level physiology already failed to predict comprehension (the selection and repair analysis comprehension
arbiter: alignment, coupling, ISC all equivalent-to-null; the only page-level predictor of
understanding was how many words you fixated). That analysis averaged over ~217 words when the
question is answered by a median 31-word stretch. This asks the localised question instead.

Tests, each with the control that makes it a test rather than a restatement:

  T1  evidence-region coverage/dwell vs a MATCHED equal-size control region on the SAME page,
      drawn from sentences no option maps to. "Did you read the answer" cannot then reduce to
      "did you read the page".
  T2  RANDOM-REGION PERMUTATION, 1000 refits with the evidence span replaced by a random
      equal-size set of sentences from the same page. This is the decisive control on the
      localisation itself: if my span annotation were noise the observed statistic would sit in
      the middle of this null.
  T3  the encoding lesion: accuracy when the answer text was never fixated at all.
  T4  ITEM-TYPE DISSOCIATION. single_fact items should depend on evidence dwell; negated
      ("which is NOT true") items should depend on broad page coverage instead, because the
      keyed option is the one the page never states and the reader must verify three others.
      Built-in validation: noise cannot produce a dissociation.
  T5  ** MW LOCALISATION -- the new claim. ** MW while the eyes were ON the evidence span, vs MW
      elsewhere on the same page, both in the same model. Claim C0 (MW on a page predicts failing
      that page's item) is agnostic about mechanism; if the cost is informational rather than
      global, only the first term should carry it.
  T6  importance-weighted reading -> accuracy, since importance is available to the reader online
      and question relevance is not.

Sensitivity, predeclared: drop the 11 annotator-exposed items and the 1 mis-keyed item.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common60 import ART, RES, IT, COUP, boot_ci, fmt, holm, z, mde

RNG = np.random.default_rng(608)
NPERM = 1000
FIX_RANGE = (50, 1000)
P2P_MAX_UV = 150.0
COMP = IT.parent / "artifacts" / "comprehension"

E = pd.read_parquet(ART / "item_evidence_llm.parquet")
pages = pd.read_parquet(COMP / "pages_full.parquet")
Wimp = pd.read_parquet(ART / "word_importance.parquet")
subs = sorted(pages["sub_id"].unique())

fx = pd.read_parquet(COUP / "reading_fixations.parquet")
fx = fx[fx["fix_dur"].between(*FIX_RANGE)].copy()
frp = pd.read_parquet(COUP / "fixations_frp.parquet",
                      columns=["onset_abs_idx", "frp_cp_N400", "frp_occ_P2", "frp_occ_N1",
                               "frp_front_late", "frp_p2p", "frp_valid"])
fx = fx.merge(frp, on="onset_abs_idx", how="left")
fx["frp_ok"] = fx["frp_valid"].fillna(False) & (fx["frp_p2p"].fillna(1) * 1e6 <= P2P_MAX_UV)
for c in ["frp_cp_N400", "frp_occ_P2", "frp_occ_N1", "frp_front_late"]:
    fx[c] = np.where(fx["frp_ok"], fx[c] * 1e6, np.nan)
fx["sub_id"] = fx["subject"].map({i: s for i, s in enumerate(subs)})

memb = []
for _, r in E.iterrows():
    memb += [{"word_key": k, "item": r["item"], "region": "evidence"} for k in r["evidence_word_keys"]]
    memb += [{"word_key": k, "item": r["item"], "region": "control"} for k in r["control_word_keys"]]
memb = pd.DataFrame(memb).drop_duplicates(["word_key", "item", "region"])
F = fx.merge(memb, on="word_key", how="inner")


def agg(g):
    return pd.Series({"n_words": g["word_key"].nunique(), "n_fix": len(g),
                      "dwell_ms": g["fix_dur"].sum(), "mean_fix_ms": g["fix_dur"].mean(),
                      "n_refix": int((g["fix_order"] > 0).sum()),
                      "mw_frac": float((g["is_mw"] == 1).mean()),
                      "n_frp": int(g["frp_ok"].sum()),
                      "n400": g.loc[g["frp_ok"], "frp_cp_N400"].mean(),
                      "occ_n1": g.loc[g["frp_ok"], "frp_occ_N1"].mean(),
                      "occ_p2": g.loc[g["frp_ok"], "frp_occ_P2"].mean(),
                      "front_late": g.loc[g["frp_ok"], "frp_front_late"].mean()})


A = F.groupby(["sub_id", "item", "region"], observed=True).apply(agg, include_groups=False).reset_index()
W = A.pivot(index=["sub_id", "item"], columns="region")
W.columns = [f"{b}_{a}" for a, b in W.columns]
W = W.reset_index()
grid = pd.MultiIndex.from_product([subs, E["item"].tolist()], names=["sub_id", "item"]).to_frame(index=False)
W = grid.merge(W, on=["sub_id", "item"], how="left")
for c in W.columns:
    if c.startswith(("evidence_", "control_")) and c.split("_", 1)[1] in (
            "n_words", "n_fix", "dwell_ms", "n_refix", "n_frp"):
        W[c] = W[c].fillna(0.0)

# MW elsewhere on the page (everything outside the evidence span)
ev_keys = set(k for ks in E["evidence_word_keys"] for k in ks)
item_of_page = E.set_index(["story_phys", "page"])["item"].to_dict()
fx2 = fx.merge(Wimp[["word_key", "story_phys", "page", "importance_llm"]].rename(
    columns={"page": "page_stim"}), on="word_key", how="inner")
fx2["item"] = [item_of_page.get((st, int(pp))) for st, pp in zip(fx2["story_phys"], fx2["page_stim"])]
fx2["on_evidence"] = fx2["word_key"].isin(ev_keys)
pg = fx2.groupby(["sub_id", "item"], observed=True).apply(lambda g: pd.Series({
    "page_n_fix": len(g),
    "mw_frac_evidence": float(g.loc[g.on_evidence, "is_mw"].mean()) if g.on_evidence.any() else np.nan,
    "mw_frac_elsewhere": float(g.loc[~g.on_evidence, "is_mw"].mean()) if (~g.on_evidence).any() else np.nan,
    "n_fix_evidence": int(g.on_evidence.sum()), "n_fix_elsewhere": int((~g.on_evidence).sum()),
    "imp_weighted_dwell": float((g["fix_dur"] * g["importance_llm"]).sum() / max(g["fix_dur"].sum(), 1)),
    "dwell_top_imp": float(g.loc[g.importance_llm >= 4, "fix_dur"].sum()),
    "dwell_low_imp": float(g.loc[g.importance_llm <= 3, "fix_dur"].sum()),
}), include_groups=False).reset_index()

D = (W.merge(E.drop(columns=["evidence_word_keys", "control_word_keys"]), on="item", how="left")
       .merge(pg, on=["sub_id", "item"], how="left")
       .merge(pages[["sub_id", "item", "correct", "skipped", "mw", "page_dur", "coverage",
                     "n_fixations", "n_words_fixated", "understand", "prior_knowledge",
                     "mw_frac_page", "run", "page"]], on=["sub_id", "item"], how="inner"))
D["evidence_cov"] = D["evidence_n_words"] / D["n_evidence_words"]
D["control_cov"] = D["control_n_words"] / D["n_control_words"].replace(0, np.nan)
D["evidence_dwell_per_word"] = D["evidence_dwell_ms"] / D["n_evidence_words"]
D["control_dwell_per_word"] = D["control_dwell_ms"] / D["n_control_words"].replace(0, np.nan)
D["ev_never"] = (D["evidence_n_fix"] == 0).astype(int)
D.to_parquet(ART / "evidence_trials_llm.parquet", index=False)

for c, nm in [("evidence_cov", "ev_cov_z"), ("control_cov", "ctrl_cov_z"), ("coverage", "page_cov_z"),
              ("imp_weighted_dwell", "imp_dwell_z"), ("mw_frac_page", "mw_page_z")]:
    D[nm] = z(D[c].fillna(D[c].mean()))
D["ev_dwell_z"] = z(np.log1p(D["evidence_dwell_per_word"]))
D["ctrl_dwell_z"] = z(np.log1p(D["control_dwell_per_word"].fillna(0)))
D["mw_ev_z"] = z(D["mw_frac_evidence"].fillna(0))
D["mw_else_z"] = z(D["mw_frac_elsewhere"].fillna(0))

rep = {"n_trials": int(len(D)), "n_readers": int(D.sub_id.nunique()), "n_items": int(D.item.nunique()),
       "evidence_coverage_describe": {k: float(v) for k, v in D.evidence_cov.describe().items()},
       "frac_evidence_never_fixated": float(D.ev_never.mean())}


def lpm(dd, outcome, terms, fe=("sub_id", "item")):
    """Linear probability model absorbing the FE, HC1 SE, clustered by reader."""
    dd = dd.dropna(subset=[outcome] + list(terms)).copy()
    X = [pd.get_dummies(dd[f], prefix=f, drop_first=True).astype(float) for f in fe]
    M = pd.concat([dd[list(terms)].reset_index(drop=True)] + [x.reset_index(drop=True) for x in X], axis=1)
    M = sm.add_constant(M)
    r = sm.OLS(dd[outcome].to_numpy(float), M.to_numpy(float)).fit(
        cov_type="cluster", cov_kwds={"groups": dd["sub_id"].to_numpy()})
    out = {"n": int(len(dd))}
    ci = np.asarray(r.conf_int())
    for i, t in enumerate(terms):
        out[t] = {"beta": float(r.params[i + 1]), "se": float(r.bse[i + 1]),
                  "p": float(r.pvalues[i + 1]),
                  "ci": [float(ci[i + 1, 0]), float(ci[i + 1, 1])],
                  "mde_80": mde(float(r.bse[i + 1]))}
    return out


# ---------------------------------------------------------------- T1
rep["T1_evidence_vs_matched_control"] = lpm(D, "correct", ["ev_cov_z", "ctrl_cov_z"])
rep["T1_with_page_coverage"] = lpm(D, "correct", ["ev_cov_z", "ctrl_cov_z", "page_cov_z"])
rep["T1_dwell"] = lpm(D, "correct", ["ev_dwell_z", "ctrl_dwell_z"])
d1 = rep["T1_evidence_vs_matched_control"]
diff = d1["ev_cov_z"]["beta"] - d1["ctrl_cov_z"]["beta"]
sed = np.hypot(d1["ev_cov_z"]["se"], d1["ctrl_cov_z"]["se"])
rep["T1_difference_conservative"] = {"beta_diff": float(diff), "se_upper": float(sed),
                                     "p": float(2 * stats.norm.sf(abs(diff / sed)))}
b = pd.cut(D["evidence_cov"], [-.01, .001, .25, .5, .75, 1.01],
           labels=["never read", "<25%", "25-50%", "50-75%", ">75%"])
rep["accuracy_by_evidence_coverage"] = (D.assign(b=b).groupby("b", observed=True)
    .agg(n=("correct", "size"), acc=("correct", "mean"), skip=("skipped", "mean")).round(4).to_dict("index"))
b2 = pd.cut(D["control_cov"], [-.01, .001, .25, .5, .75, 1.01],
            labels=["never read", "<25%", "25-50%", "50-75%", ">75%"])
rep["accuracy_by_control_coverage"] = (D.assign(b=b2).groupby("b", observed=True)
    .agg(n=("correct", "size"), acc=("correct", "mean")).round(4).to_dict("index"))

# ---------------------------------------------------------------- T2 random-region permutation
STIM = IT.parents[1] / "data" / "derivatives" / "stimuli" / "wiki_stories"
sent_map, page_words = {}, {}
for stem in E["story_phys"].unique():
    c = pd.read_csv(STIM / f"{stem}_coordinates.csv")
    for p_, g in c.groupby("page"):
        keys = g["word_key"].tolist()
        page_words[(stem, int(p_))] = keys
        sids = list(dict.fromkeys(g["sentence_id"]))
        sent_map[(stem, int(p_))] = [[keys.index(k) for k in g.loc[g["sentence_id"] == s, "word_key"]]
                                     for s in sids]
seen = fx.groupby(["sub_id", "story", "page"])["word_key"].apply(set).to_dict()
FIXV = {}
for (stem, p_), keys in page_words.items():
    idx = {k: i for i, k in enumerate(keys)}
    for s in subs:
        v = np.zeros(len(keys), bool)
        for k in set(seen.get((s, stem, float(p_)), set())) | set(seen.get((s, stem, p_), set())):
            j = idx.get(k)
            if j is not None:
                v[j] = True
        FIXV[(s, stem, p_)] = v

meta = E.set_index("item")[["story_phys", "page", "n_evidence_words"]].to_dict("index")
X0 = pd.concat([pd.get_dummies(D["sub_id"], prefix="s", drop_first=True).astype(float).reset_index(drop=True),
                pd.get_dummies(D["item"], prefix="i", drop_first=True).astype(float).reset_index(drop=True),
                D[["ctrl_cov_z"]].reset_index(drop=True)], axis=1)
X0 = sm.add_constant(X0, has_constant="add").to_numpy(float)
Q, _ = np.linalg.qr(X0)
res = lambda v: v - Q @ (Q.T @ v)
yv = res(D["correct"].to_numpy(float))
xe = res(D["ev_cov_z"].to_numpy(float))
obs_stat = float((xe * yv).sum() / (xe ** 2).sum())
null = np.empty(NPERM)
item_rows = {it: np.flatnonzero((D["item"] == it).to_numpy()) for it in meta}
for bi in range(NPERM):
    cov = np.zeros(len(D))
    for it, m in meta.items():
        stem, p_, nw = m["story_phys"], m["page"], m["n_evidence_words"]
        sents = sent_map[(stem, p_)]
        order = RNG.permutation(len(sents))
        pick, n = [], 0
        for j in order:
            if n >= nw:
                break
            pick += sents[j]; n += len(sents[j])
        pick = np.array(pick[:max(nw, 1)])
        for r_ in item_rows[it]:
            v = FIXV[(D["sub_id"].iat[r_], stem, p_)]
            cov[r_] = v[pick].mean() if pick.size else 0.0
    cz = (cov - cov.mean()) / (cov.std() + 1e-12)
    rz = res(cz)
    null[bi] = (rz * yv).sum() / max((rz * rz).sum(), 1e-12)
    if (bi + 1) % 250 == 0:
        print(f"  random-region perm {bi+1}/{NPERM}", flush=True)
rep["T2_random_region_permutation"] = {
    "n_perm": NPERM, "observed": obs_stat, "null_mean": float(null.mean()), "null_sd": float(null.std()),
    "p_one_sided": float(((null >= obs_stat).sum() + 1) / (NPERM + 1)),
    "percentile": float((null < obs_stat).mean() * 100)}

# ---------------------------------------------------------------- T3 lesion
rep["T3_lesion_descriptive"] = D.groupby("ev_never").agg(
    n=("correct", "size"), acc=("correct", "mean"), skip=("skipped", "mean"),
    page_cov=("coverage", "mean")).round(4).to_dict("index")
rep["T3_lesion_model"] = lpm(D, "correct", ["ev_never", "page_cov_z", "ctrl_cov_z"])

# ---------------------------------------------------------------- T4 item type
rep["T4_by_item_type"] = {}
for t in ["single_fact", "negated"]:
    sub = D[D["item_type"] == t].copy()
    for c in ["ev_cov_z", "ctrl_cov_z", "page_cov_z"]:
        sub[c] = z(sub[c])
    rep["T4_by_item_type"][t] = lpm(sub, "correct", ["ev_cov_z", "page_cov_z"])
D["is_single"] = (D["item_type"] == "single_fact").astype(float)
D["ev_x_single"] = D["ev_cov_z"] * D["is_single"]
rep["T4_interaction"] = lpm(D[D["item_type"].isin(["single_fact", "negated"])], "correct",
                            ["ev_cov_z", "ev_x_single", "page_cov_z"])

# ---------------------------------------------------------------- T5 MW localisation (new)
sub = D[(D["n_fix_evidence"] >= 3) & (D["n_fix_elsewhere"] >= 10)].copy()
rep["T5_n_trials"] = int(len(sub))
rep["T5_mw_localisation"] = lpm(sub, "correct", ["mw_ev_z", "mw_else_z"])
rep["T5_mw_localisation_with_reading"] = lpm(sub, "correct", ["mw_ev_z", "mw_else_z", "ev_cov_z", "page_cov_z"])
rep["T5_mw_page_flag_only"] = lpm(sub, "correct", ["mw_page_z"])
m5 = rep["T5_mw_localisation"]
dd5 = m5["mw_ev_z"]["beta"] - m5["mw_else_z"]["beta"]
sd5 = np.hypot(m5["mw_ev_z"]["se"], m5["mw_else_z"]["se"])
rep["T5_difference_conservative"] = {"beta_diff": float(dd5), "se_upper": float(sd5),
                                     "p": float(2 * stats.norm.sf(abs(dd5 / sd5))),
                                     "mde_80": mde(float(sd5))}
rep["T5_descriptive"] = {
    "mw_on_evidence_mean": float(sub.mw_frac_evidence.mean()),
    "mw_elsewhere_mean": float(sub.mw_frac_elsewhere.mean()),
    "acc_mw_ev_zero": float(sub.loc[sub.mw_frac_evidence == 0, "correct"].mean()),
    "acc_mw_ev_pos": float(sub.loc[sub.mw_frac_evidence > 0, "correct"].mean()),
    "n_mw_ev_pos": int((sub.mw_frac_evidence > 0).sum())}

# ---------------------------------------------------------------- T6 importance-weighted reading
rep["T6_importance_weighted"] = lpm(D, "correct", ["imp_dwell_z", "page_cov_z"])
D["dwell_ratio_imp"] = np.log1p(D["dwell_top_imp"]) - np.log1p(D["dwell_low_imp"])
D["dwell_ratio_z"] = z(D["dwell_ratio_imp"])
rep["T6_high_vs_low_importance_dwell"] = lpm(D, "correct", ["dwell_ratio_z", "page_cov_z"])

# ---------------------------------------------------------------- sensitivity
keep = D[(~D["exposed_to_annotator"]) & (~D["mis_keyed"])]
rep["S_unexposed_items"] = {"n_items": int(keep.item.nunique()),
                            "T1": lpm(keep, "correct", ["ev_cov_z", "ctrl_cov_z"]),
                            "T5": lpm(keep[(keep.n_fix_evidence >= 3) & (keep.n_fix_elsewhere >= 10)],
                                      "correct", ["mw_ev_z", "mw_else_z"])}

(RES / "outcome.json").write_text(json.dumps(rep, indent=2, default=str) + "\n")

print(f"\n=== H5 outcome, {rep['n_trials']} trials x {rep['n_items']} items x {rep['n_readers']} readers ===")
print(f"evidence never fixated on {rep['frac_evidence_never_fixated']:.1%} of trials")
print("\nT1 evidence vs matched control (LPM, subject+item FE, reader-clustered SE):")
for k in ["ev_cov_z", "ctrl_cov_z"]:
    v = d1[k]
    print(f"  {k:12s} {v['beta']:+.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] p={v['p']:.3g}  MDE80={v['mde_80']:.4f}")
print(f"  difference (conservative SE): {rep['T1_difference_conservative']['beta_diff']:+.4f} "
      f"p={rep['T1_difference_conservative']['p']:.3g}")
print("\n  accuracy by evidence coverage:", {k: v["acc"] for k, v in rep["accuracy_by_evidence_coverage"].items()})
print("  accuracy by control  coverage:", {k: v["acc"] for k, v in rep["accuracy_by_control_coverage"].items()})
t2 = rep["T2_random_region_permutation"]
print(f"\nT2 random-region permutation: observed {t2['observed']:+.4f} vs null {t2['null_mean']:+.4f} "
      f"(SD {t2['null_sd']:.4f}) -> p={t2['p_one_sided']:.4f} ({t2['percentile']:.1f}th pct)")
print(f"\nT3 lesion: {rep['T3_lesion_descriptive']}")
v = rep["T3_lesion_model"]["ev_never"]
print(f"  model beta={v['beta']:+.4f} p={v['p']:.3g}")
print("\nT4 item-type dissociation:")
for t, m in rep["T4_by_item_type"].items():
    print(f"  {t:12s} ev_cov {m['ev_cov_z']['beta']:+.4f} (p={m['ev_cov_z']['p']:.3g})   "
          f"page_cov {m['page_cov_z']['beta']:+.4f} (p={m['page_cov_z']['p']:.3g})  n={m['n']}")
print(f"  interaction ev x single_fact: {rep['T4_interaction']['ev_x_single']['beta']:+.4f} "
      f"p={rep['T4_interaction']['ev_x_single']['p']:.3g}")
print(f"\nT5 MW localisation (n={rep['T5_n_trials']}):")
for k in ["mw_ev_z", "mw_else_z"]:
    v = rep["T5_mw_localisation"][k]
    print(f"  {k:11s} {v['beta']:+.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] p={v['p']:.3g}  MDE80={v['mde_80']:.4f}")
print(f"  difference: {rep['T5_difference_conservative']['beta_diff']:+.4f} "
      f"p={rep['T5_difference_conservative']['p']:.3g} MDE80={rep['T5_difference_conservative']['mde_80']:.4f}")
print(f"  descriptive: {rep['T5_descriptive']}")
print(f"\nT6 importance-weighted dwell: {rep['T6_importance_weighted']['imp_dwell_z']['beta']:+.4f} "
      f"p={rep['T6_importance_weighted']['imp_dwell_z']['p']:.3g}; high-vs-low dwell ratio "
      f"{rep['T6_high_vs_low_importance_dwell']['dwell_ratio_z']['beta']:+.4f} "
      f"p={rep['T6_high_vs_low_importance_dwell']['dwell_ratio_z']['p']:.3g}")
s_ = rep["S_unexposed_items"]
print(f"\nsensitivity, {s_['n_items']} unexposed items: T1 ev_cov {s_['T1']['ev_cov_z']['beta']:+.4f} "
      f"(p={s_['T1']['ev_cov_z']['p']:.3g}); T5 mw_ev {s_['T5']['mw_ev_z']['beta']:+.4f} "
      f"(p={s_['T5']['mw_ev_z']['p']:.3g})")
