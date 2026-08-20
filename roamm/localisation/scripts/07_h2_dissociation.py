#!/usr/bin/env python3
"""H2: does mind-wandering decouple the eyes from IMPORTANCE while leaving the words coupled?

The selection and repair analysis established, on these rows, that MW leaves LEXICAL selectivity intact (frequency /
length / surprisal -> gaze and -> skipping, equivalent within +-10%) while reading gets longer,
more regressive and less skipping, and that the drop in inter-reader alignment lives entirely in
the NON-lexical residual. Its reading -- "the reader loses the thread, not the words" -- makes
one strong prediction it had no way to test: a DISCOURSE-level property should decouple where
word-level properties do not. Importance is that property.

The lexical channels are the internal control, measured on the same rows in the same model, so
the contrast is immune to anything that rescales all coupling (arousal, fatigue, general slowing).

Two outcome channels, because the selection and repair analysis showed they behave differently:
  DURATION   log first-pass gaze duration
  SELECTION  whether the word is skipped (scan-path definition: stepped over in one forward
             saccade, the selection and repair analysis corrected variable -- the legacy "any unfixated word"
             version manufactured a fake effect)

Estimators:
  A  per reader, per state, slopes with PAGE fixed effects. Lemma FE are deliberately NOT used
     here: zipf and length are properties of the lemma, so lemma FE would absorb the control
     channels completely and make the cross-channel comparison meaningless (a first pass at this
     did exactly that and produced base slopes of 0.00000 with exploding ratios).
  B  pooled two-way (word instance x subject) FE with subject-clustered SE -- the primary.
     Word-instance FE absorbs every time-invariant word property, including importance itself,
     so only interactions with MW are identified, which is exactly the hypothesis.
  C  Somers' D selectivity per reader per state (rank-based, base-rate free), matching
     the primary selection measure of the selection analysis, so the numbers are comparable.

Every estimate is reported with its MDE as a percentage of the on-task base effect. The hard
lesson from the selection analysis: a contrast can look supportive while being unable to detect
total abolition.
"""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
from scipy import stats
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common60 import (ART, RES, IT, boot_ci, fmt, holm, absorb, ols_cluster, z, mde,
                      load_word_measures)

CH = {"importance": "importance_llm_z", "zipf": "zipf_z", "length": "length_z", "surprisal": "surprisal_z"}
COV = ["n_words_sentence_z", "rel_pos_in_sentence_z", "sent_pos_on_page_z", "line_pos_z",
       "is_line_first", "is_line_last", "is_page_boundary_sentence"]
ZCOLS = ["importance_llm", "zipf", "length", "surprisal", "n_words_sentence",
         "rel_pos_in_sentence", "sent_pos_on_page", "line_pos", "importance_qwen_exp", "centrality_lm"]
rep: dict = {}


def prep(D):
    D = D.copy()
    D["page_id"] = D["story_phys"] + "_" + D["page"].astype(str)
    for c in ZCOLS:
        if c in D.columns:
            D[c + "_z"] = z(D[c])
    for c in ["surprisal_z", "importance_qwen_exp_z", "centrality_lm_z"]:
        if c in D.columns:
            D[c] = D[c].fillna(0.0)
    return D


def somers_d(x, y):
    x = np.asarray(x, float); y = np.asarray(y).astype(int)
    m = np.isfinite(x); x, y = x[m], y[m]
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 < 20 or n0 < 20:
        return np.nan
    r = stats.rankdata(x)
    return 2 * ((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)) - 1


# ============================================================ DURATION channel
D = prep(load_word_measures())
D["log_gaze"] = np.log(D["gaze_dur"])
D = D.dropna(subset=["log_gaze"] + list(CH.values()) + COV)

# ---- A per-reader state slopes, PAGE FE only
def slopes_by_state(dd, outcome, chans, min_n=300):
    rows = []
    allc = list(chans.values())
    for (s, m), g in dd.groupby(["subject", "is_mw"]):
        g = g.dropna(subset=[outcome] + allc + COV)
        if len(g) < min_n:
            continue
        M = absorb(np.column_stack([g[outcome].to_numpy(float)] +
                                   [g[c].to_numpy(float) for c in allc + COV]),
                   [g["page_id"].to_numpy()])
        b = np.linalg.pinv(M[:, 1:].T @ M[:, 1:]) @ (M[:, 1:].T @ M[:, 0])
        rows.append(dict(subject=s, is_mw=int(m), n=len(g), **{k: float(b[i]) for i, k in enumerate(chans)}))
    return pd.DataFrame(rows)


def attenuation(S, chans):
    w = S.pivot(index="subject", columns="is_mw", values=list(chans)).dropna()
    out = {"_n_readers": int(len(w))}
    rel = {}
    for k in chans:
        base = float(w[(k, 0)].mean())
        d = (w[(k, 1)] - w[(k, 0)]).to_numpy()
        rel[k] = d / base if abs(base) > 1e-8 else np.full(len(d), np.nan)
        out[k] = dict(base_ontask=base, mw_slope=float(w[(k, 1)].mean()), delta=boot_ci(d),
                      retention_pct=boot_ci(100 * (1 + rel[k])))
    for lex in [c for c in chans if c != "importance"]:
        c = rel["importance"] - rel[lex]
        if not np.all(np.isfinite(c)):
            continue
        r = boot_ci(100 * c)
        r["mde_80pct_points"] = mde(np.std(100 * c, ddof=1) / np.sqrt(len(c)))
        out[f"contrast_importance_minus_{lex}"] = r
    return out


S = slopes_by_state(D, "log_gaze", CH)
S.to_csv(RES / "slopes_by_state_duration.csv", index=False)
rep["duration_A_per_reader"] = attenuation(S, CH)

# ---- B pooled two-way FE with interactions
def pooled_interactions(dd, outcome, chans, base_slopes=None, extra_cov=()):
    g = dd.copy()
    inter = []
    for k, c in chans.items():
        g[f"mw_x_{k}"] = g["is_mw"] * g[c]
        inter.append(f"mw_x_{k}")
    cols = inter + list(COV) + list(extra_cov)
    M = absorb(np.column_stack([g[outcome].to_numpy(float)] + [g[c].to_numpy(float) for c in cols]),
               [g["word_key"].to_numpy(), g["subject"].to_numpy()])
    r = ols_cluster(M[:, 0], M[:, 1:], g["subject"].to_numpy(), names=cols)
    out = {k: r[k] for k in inter}
    out["_n"], out["_n_clusters"] = r["_n"], r["_n_clusters"]
    if base_slopes:
        for k in chans:
            b = base_slopes.get(k)
            v = out[f"mw_x_{k}"]
            if b and abs(b) > 1e-8:
                v["pct_of_base"] = 100 * v["beta"] / b
                v["pct_of_base_ci"] = [100 * v["ci"][0] / b, 100 * v["ci"][1] / b]
                v["mde_pct_of_base"] = abs(100 * mde(v["se"]) / b)
                v["base_ontask"] = b
    return out


# on-task base slopes for scaling (page FE, on-task rows only, pooled)
def base_pooled(dd, outcome, chans):
    g = dd[dd["is_mw"] == 0]
    cols = list(chans.values()) + COV
    M = absorb(np.column_stack([g[outcome].to_numpy(float)] + [g[c].to_numpy(float) for c in cols]),
               [g["subject"].to_numpy(), g["page_id"].to_numpy()])
    r = ols_cluster(M[:, 0], M[:, 1:], g["subject"].to_numpy(), names=cols)
    return {k: r[v]["beta"] for k, v in chans.items()}, {k: r[v] for k, v in chans.items()}


base_dur, base_dur_full = base_pooled(D, "log_gaze", CH)
rep["duration_base_ontask"] = base_dur_full
rep["duration_B_pooled"] = pooled_interactions(D, "log_gaze", CH, base_dur)
g = D.copy()
M = absorb(np.column_stack([g["log_gaze"].to_numpy(float), g["is_mw"].to_numpy(float)] +
                           [g[c].to_numpy(float) for c in COV]),
           [g["word_key"].to_numpy(), g["subject"].to_numpy()])
rep["duration_additive_mw_shift"] = ols_cluster(M[:, 0], M[:, 1:], g["subject"].to_numpy(),
                                               names=["mw"] + COV)["mw"]

# ============================================================ SELECTION channel
tw = pd.read_parquet(IT.parent / "selection_repair/artifacts/words_traversal.parquet")
Wimp = pd.read_parquet(ART / "word_importance.parquet")
T = tw.drop(columns=["zipf", "surprisal", "length", "page", "story"]).merge(
    Wimp, on="word_key", how="inner", validate="m:1")
T = T[T["state_agree"]].copy()
T = prep(T)
T = T.dropna(subset=["skipped"] + list(CH.values()) + COV)
rep["selection_n_rows"] = int(len(T))
rep["selection_skip_rate"] = {"on_task": float(T.loc[T.is_mw == 0, "skipped"].mean()),
                              "mw": float(T.loc[T.is_mw == 1, "skipped"].mean())}

base_sel, base_sel_full = base_pooled(T, "skipped", CH)
rep["selection_base_ontask"] = base_sel_full
rep["selection_B_pooled"] = pooled_interactions(T, "skipped", CH, base_sel)
S2 = slopes_by_state(T, "skipped", CH, min_n=300)
S2.to_csv(RES / "slopes_by_state_selection.csv", index=False)
rep["selection_A_per_reader"] = attenuation(S2, CH)

# ---- C Somers' D selectivity (comparable with the selection analysis)
sk = []
for (s, m), g in T.groupby(["subject", "is_mw"]):
    if len(g) < 300:
        continue
    sk.append(dict(subject=s, is_mw=int(m), n=len(g), skip_rate=float(g["skipped"].mean()),
                   **{k: somers_d(g[v.replace("_z", "")], g["skipped"]) for k, v in CH.items()}))
SK = pd.DataFrame(sk)
SK.to_csv(RES / "somersD_by_state.csv", index=False)
pv = SK.pivot(index="subject", columns="is_mw", values=list(CH)).dropna()
selc = {"_n_readers": int(len(pv))}
rel = {}
for k in CH:
    base = float(pv[(k, 0)].mean())
    d = (pv[(k, 1)] - pv[(k, 0)]).to_numpy()
    rel[k] = d / base
    r = boot_ci(d)
    r["mde_80pct_D"] = mde(np.std(d, ddof=1) / np.sqrt(len(d)))
    r["mde_pct_of_base"] = abs(100 * r["mde_80pct_D"] / base)
    selc[k] = dict(ontask_D=base, mw_D=float(pv[(k, 1)].mean()), delta=r,
                   retention_pct=boot_ci(100 * (1 + rel[k])),
                   n_readers_sign_flip=int(((pv[(k, 0)] < 0) & (pv[(k, 1)] > 0)).sum()) if base < 0
                   else int(((pv[(k, 0)] > 0) & (pv[(k, 1)] < 0)).sum()))
for lex in ["zipf", "length", "surprisal"]:
    c = rel["importance"] - rel[lex]
    r = boot_ci(100 * c)
    r["mde_80pct_points"] = mde(np.std(100 * c, ddof=1) / np.sqrt(len(c)))
    selc[f"contrast_importance_minus_{lex}"] = r
rep["selection_C_somersD"] = selc

# ---- D is the raw Somers' D shift semantic, or structural?
# Two ways it could be an artefact rather than a loss of importance tracking:
#  (i) importance correlates with sentence length and line position, and MW skipping is
#      structurally different (the selection and repair analysis: 74% of MW stepped-over words sit in large
#      same-page jumps = return sweeps, vs 48% on-task). Residualise importance on every
#      covariate first, then recompute D on the residual.
#  (ii) restrict to LINE-INTERIOR steps (gap <= 4 words; the selection and repair analysis showed steps > 4 cross a
#      text line in 96.6% of cases), which removes return sweeps entirely.
Tr = T.copy()
rescols = ["zipf_z", "length_z", "surprisal_z"] + COV
Mr = np.column_stack([Tr[c].to_numpy(float) for c in rescols] + [np.ones(len(Tr))])
imp_res = Tr["importance_llm_z"].to_numpy(float)
imp_res = imp_res - Mr @ (np.linalg.pinv(Mr.T @ Mr) @ (Mr.T @ imp_res))
Tr["imp_resid"] = imp_res

def somers_panel(dd, col, min_n=300):
    rows = []
    for (s, m), g in dd.groupby(["subject", "is_mw"]):
        if len(g) < min_n:
            continue
        rows.append(dict(subject=s, is_mw=int(m), n=len(g), D=somers_d(g[col], g["skipped"])))
    R = pd.DataFrame(rows)
    w = R.pivot(index="subject", columns="is_mw", values="D").dropna()
    d = (w[1] - w[0]).to_numpy()
    r = boot_ci(d)
    r["mde_80pct_D"] = mde(np.std(d, ddof=1) / np.sqrt(len(d)))
    return dict(ontask_D=float(w[0].mean()), mw_D=float(w[1].mean()), delta=r,
                n_readers=int(len(w)))

rep["selection_D_robustness"] = {
    "raw_importance": somers_panel(Tr, "importance_llm"),
    "residualised_importance": somers_panel(Tr, "imp_resid"),
    "line_interior_raw": somers_panel(Tr[Tr["gap"] <= 4], "importance_llm"),
    "line_interior_residualised": somers_panel(Tr[Tr["gap"] <= 4], "imp_resid"),
    "zipf_line_interior": somers_panel(Tr[Tr["gap"] <= 4], "zipf"),
}
rep["selection_skip_structure"] = {
    "frac_gap_gt4_ontask": float((Tr.loc[Tr.is_mw == 0, "gap"] > 4).mean()),
    "frac_gap_gt4_mw": float((Tr.loc[Tr.is_mw == 1, "gap"] > 4).mean()),
    "skip_rate_line_interior_ontask": float(Tr[(Tr.gap <= 4) & (Tr.is_mw == 0)]["skipped"].mean()),
    "skip_rate_line_interior_mw": float(Tr[(Tr.gap <= 4) & (Tr.is_mw == 1)]["skipped"].mean())}

# ---- convergent annotators on the selection channel
rep["selection_convergent"] = {}
for nm, col in [("qwen", "importance_qwen_exp_z"), ("centrality", "centrality_lm_z"),
                ("in_summary", "in_summary_llm")]:
    if col not in T.columns:
        continue
    ch2 = {"importance": col, "zipf": "zipf_z", "length": "length_z", "surprisal": "surprisal_z"}
    b2, _ = base_pooled(T, "skipped", ch2)
    rep["selection_convergent"][nm] = pooled_interactions(T, "skipped", ch2, b2)["mw_x_importance"]

# ---- deep MW
deep = T[(T["is_mw"] == 0) | (T["mw_frac"] >= 0.99)]
bd, _ = base_pooled(deep, "skipped", CH)
rep["selection_deep_mw"] = pooled_interactions(deep, "skipped", CH, bd)

(RES / "h2_dissociation.json").write_text(json.dumps(rep, indent=2, default=str) + "\n")

print("=" * 78)
print("DURATION CHANNEL  (log first-pass gaze duration)")
print(f"  on-task base slopes: " + "  ".join(f"{k}={v:+.4f}" for k, v in base_dur.items()))
print(f"  additive MW shift {rep['duration_additive_mw_shift']['beta']:+.4f} "
      f"(p={rep['duration_additive_mw_shift']['p']:.2g})")
b = rep["duration_B_pooled"]
print(f"  pooled word-instance x subject FE, n={b['_n']}:")
for k in CH:
    v = b[f"mw_x_{k}"]
    ex = (f"  = {v['pct_of_base']:+.0f}% of base [{v['pct_of_base_ci'][0]:+.0f},{v['pct_of_base_ci'][1]:+.0f}]"
          f"  MDE80={v['mde_pct_of_base']:.0f}%") if "pct_of_base" in v else ""
    print(f"    mw x {k:11s} {v['beta']:+.5f} (SE {v['se']:.5f}, p={v['p']:.3g}){ex}")

print("\n" + "=" * 78)
print(f"SELECTION CHANNEL  (skipping; n={rep['selection_n_rows']} traversals, "
      f"skip rate {rep['selection_skip_rate']['on_task']:.3f} on-task -> {rep['selection_skip_rate']['mw']:.3f} MW)")
print(f"  on-task base slopes: " + "  ".join(f"{k}={v:+.4f}" for k, v in base_sel.items()))
b = rep["selection_B_pooled"]
print(f"  pooled word-instance x subject FE, n={b['_n']}:")
for k in CH:
    v = b[f"mw_x_{k}"]
    ex = (f"  = {v['pct_of_base']:+.0f}% of base [{v['pct_of_base_ci'][0]:+.0f},{v['pct_of_base_ci'][1]:+.0f}]"
          f"  MDE80={v['mde_pct_of_base']:.0f}%") if "pct_of_base" in v else ""
    print(f"    mw x {k:11s} {v['beta']:+.5f} (SE {v['se']:.5f}, p={v['p']:.3g}){ex}")
print(f"\n  Somers' D selectivity per reader (n={selc['_n_readers']}):")
for k in CH:
    f_ = selc[k]
    print(f"    {k:11s} on-task D={f_['ontask_D']:+.4f} -> MW D={f_['mw_D']:+.4f}   "
          f"delta {f_['delta']['mean']:+.4f} [{f_['delta']['ci'][0]:+.4f},{f_['delta']['ci'][1]:+.4f}] "
          f"p={f_['delta']['p']:.3g} {f_['delta']['n_pos']}/{f_['delta']['n']}  "
          f"retention {f_['retention_pct']['mean']:.0f}%  MDE80={f_['delta']['mde_pct_of_base']:.0f}% of base")
print("\n  CONTRAST (importance vs lexical relative attenuation, % points):")
for lex in ["zipf", "length", "surprisal"]:
    c = selc[f"contrast_importance_minus_{lex}"]
    print(fmt(f"   importance - {lex}", c) + f"  MDE80={c['mde_80pct_points']:.0f}pts")
print("\n  IS THE RAW D SHIFT SEMANTIC OR STRUCTURAL?")
for k, v in rep["selection_D_robustness"].items():
    d = v["delta"]
    print(f"    {k:28s} D {v['ontask_D']:+.4f} -> {v['mw_D']:+.4f}  delta {d['mean']:+.4f} "
          f"[{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}] p={d['p']:.3g} {d['n_pos']}/{d['n']}")
ss = rep["selection_skip_structure"]
print(f"    step gap>4 words (return sweeps): on-task {ss['frac_gap_gt4_ontask']:.3f} -> MW {ss['frac_gap_gt4_mw']:.3f}")
print("\n  convergent annotators, mw x importance on skipping:")
for nm, v in rep["selection_convergent"].items():
    print(f"    {nm:12s} {v['beta']:+.5f} (SE {v['se']:.5f}, p={v['p']:.3g})"
          + (f"  = {v['pct_of_base']:+.0f}% of base" if "pct_of_base" in v else ""))
dv = rep["selection_deep_mw"]["mw_x_importance"]
print(f"\n  deep MW only: mw x importance {dv['beta']:+.5f} (p={dv['p']:.3g})"
      + (f" = {dv['pct_of_base']:+.0f}% of base" if "pct_of_base" in dv else ""))
