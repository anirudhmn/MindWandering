#!/usr/bin/env python3
"""Tests that discriminate between accounts of what mind-wandering does.

The candidate accounts:
  (i)   sensory gating: input attenuated              [already rejected]
  (ii)  downstream integration failure + compensation: input intact, integration fails,
        reader re-reads to recover
  (iii) uniform state change: everything slows and repeats more, with no difficulty-specific
        or integration-specific signature

A. Spillover. Difficulty at word N-1 carries into the fixation on word N. This is a
   behavioural index of integration that does not require the N400. Integration failure
   predicts a change; a uniform state change predicts none.
B. Targeting. Is the extra effort concentrated on difficult words, as compensation would
   imply, or spread evenly?
C. Time course. Does effort build across an episode, as a reader falling behind would
   predict, or is it a step change?
D. Shared-response decomposition. Does the inter-subject alignment drop survive removing
   the additive shift and the lexical component, or is it a byproduct of them?
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy import stats
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ART, RES, COUP, boot_ci, fmt, holm

RNG = np.random.default_rng(59)
rep = {}

f = pd.read_parquet(COUP / "reading_fixations.parquet").sort_values(["subject", "run", "tStart"])
g = f.groupby(["subject", "run"], sort=False)
f["refix"] = (g["pos"].shift(-1) == f["pos"]).astype(float)
f.loc[g["pos"].shift(-1).isna(), "refix"] = np.nan
f["logdur"] = np.log(f.fix_dur.clip(lower=1))

# =============================================================== A. spillover
print("=== A. Spillover: does difficulty at word N-1 carry into word N differently? ===")
MIN_SPILL = 120
cols = ["zipf", "surprisal", "length", "zipf_prev", "surprisal_prev", "length_prev"]
d = f.dropna(subset=cols + ["logdur"]).copy()
# only consecutive forward steps, so 'prev' is genuinely the previously read word
d = d[d.pos - d.prev_pos == 1]
rows = []
for s, gg in d.groupby("subject"):
    rec = {"subject": s}
    ok = True
    for st, tag in [(0, "on"), (1, "mw")]:
        gs = gg[gg.is_mw == st]
        if len(gs) < MIN_SPILL:
            ok = False
            break
        X = np.column_stack([(gs[c] - gs[c].mean()) / (gs[c].std() + 1e-9) for c in cols])
        X = np.column_stack([np.ones(len(X)), X])
        b = np.linalg.lstsq(X, gs.logdur.to_numpy(), rcond=None)[0]
        for c, v in zip(cols, b[1:]):
            rec[f"{c}_{tag}"] = float(v)
        rec[f"n_{tag}"] = len(gs)
    if ok:
        rows.append(rec)
A = pd.DataFrame(rows)
print(f"  {len(A)} readers, median MW fixations {int(A.n_mw.median())}")
rep["A_spillover"] = {}
ps = []
for c in cols:
    diff = (A[f"{c}_mw"] - A[f"{c}_on"]).to_numpy()
    base = A[f"{c}_on"].mean()
    r = boot_ci(diff)
    r["on"] = float(base)
    r["mw"] = float(A[f"{c}_mw"].mean())
    r["pct_change"] = float(diff.mean() / abs(base) * 100 * np.sign(base)) if base else np.nan
    rep["A_spillover"][c] = r
    ps.append(r["p"])
    kind = "lag-1" if c.endswith("_prev") else "current"
    print(f"  {c:16s} ({kind:7s}) on={base:+.5f} mw={A[f'{c}_mw'].mean():+.5f} "
          f"change={r['pct_change']:+6.1f}%  p={r['p']:.3f}")
for c, a in zip(cols, holm(ps)):
    rep["A_spillover"][c]["p_holm"] = float(a)
lagp = [rep["A_spillover"][c]["p_holm"] for c in cols if c.endswith("_prev")]
print(f"  smallest Holm-corrected p among the three lag-1 terms: {min(lagp):.3f}")

# =============================================================== B. targeting
print("\n=== B. Is the extra effort concentrated on difficult words? ===")
d2 = f.dropna(subset=["surprisal", "zipf", "logdur"]).copy()
d2["diff_q"] = d2.groupby("subject").surprisal.transform(
    lambda x: pd.qcut(x, 4, labels=[0, 1, 2, 3], duplicates="drop"))
rep["B_targeting"] = {}
for meas, lab in [("logdur", "fixation duration (log)"), ("regression_out", "regression rate"),
                  ("refix", "refixation rate")]:
    sub = d2.dropna(subset=[meas])
    piv = sub.groupby(["subject", "diff_q", "is_mw"], observed=True)[meas].mean().unstack()
    piv = piv.dropna()
    piv["delta"] = piv[1] - piv[0]
    wide = piv.reset_index().pivot(index="subject", columns="diff_q", values="delta").dropna()
    slope = []
    for _, row in wide.iterrows():
        q = np.arange(len(row))
        slope.append(np.polyfit(q, row.to_numpy(), 1)[0])
    r = boot_ci(np.array(slope))
    rep["B_targeting"][meas] = {"per_quartile_delta": [float(wide[c].mean()) for c in wide.columns],
                                "slope_across_quartiles": r}
    print(f"  {lab:26s} MW-minus-on-task by difficulty quartile: " +
          "  ".join(f"{wide[c].mean():+.4f}" for c in wide.columns))
    print(fmt("     trend across quartiles", r))

# =============================================================== C. time course
print("\n=== C. Does effort build across an episode? ===")
ep = []
for (s, r_), gg in f.groupby(["subject", "run"], sort=False):
    mw = gg.is_mw.to_numpy().astype(int)
    t = gg.tStart.to_numpy()
    i = 0
    eid = 0
    lab = np.full(len(mw), np.nan)
    tsince = np.full(len(mw), np.nan)
    while i < len(mw):
        if mw[i] == 1:
            j = i
            while j + 1 < len(mw) and mw[j + 1] == 1:
                j += 1
            lab[i:j + 1] = eid
            tsince[i:j + 1] = t[i:j + 1] - t[i]
            eid += 1
            i = j + 1
        else:
            i += 1
    h = gg.copy()
    h["episode"] = lab
    h["t_since"] = tsince
    ep.append(h)
E = pd.concat(ep, ignore_index=True)
mwf = E[E.is_mw == 1].dropna(subset=["t_since"])
bins = [0, 2, 5, 10, 1e9]
labs = ["0-2 s", "2-5 s", "5-10 s", ">10 s"]
mwf["tb"] = pd.cut(mwf.t_since, bins, labels=labs, right=False)
base = E[E.is_mw == 0].groupby("subject").agg(
    logdur=("logdur", "mean"), reg=("regression_out", "mean"), refix=("refix", "mean"))
rep["C_timecourse"] = {}
print(f"  {int(mwf.episode.notna().sum())} MW fixations in "
      f"{int(E.groupby('subject').episode.nunique().sum())} episodes")
for meas, key, lab in [("logdur", "logdur", "fixation duration (log, vs own baseline)"),
                       ("regression_out", "reg", "regression rate (vs own baseline)"),
                       ("refix", "refix", "refixation rate (vs own baseline)")]:
    out = {}
    for b in labs:
        sub = mwf[mwf.tb == b].groupby("subject")[meas].mean()
        common = sub.index.intersection(base.index)
        v = (sub.loc[common] - base.loc[common, key]).to_numpy()
        v = v[np.isfinite(v)]
        out[b] = boot_ci(v)
    rep["C_timecourse"][meas] = out
    print(f"  {lab:42s} " + "  ".join(f"{b}: {out[b]['mean']:+.4f}" for b in labs))
    # linear trend across bins, per subject
    piv = mwf.groupby(["subject", "tb"], observed=True)[meas].mean().unstack().dropna()
    tr = np.array([np.polyfit(np.arange(piv.shape[1]), row.to_numpy(), 1)[0] for _, row in piv.iterrows()])
    rep["C_timecourse"][meas]["trend"] = boot_ci(tr)
    print(fmt("     linear trend across bins", rep["C_timecourse"][meas]["trend"]))

# =============================================================== D. shared response
print("\n=== D. Does the alignment drop survive removing the effort shift and lexical coupling? ===")
W = (f[f.is_firstpass == 1]
     .groupby(["subject", "word_key"])
     .agg(GD=("fix_dur", "sum"), is_mw=("is_mw", "first")).reset_index())
wf = pd.read_parquet(COUP / "word_features.parquet")[["word_key", "zipf", "surprisal", "length"]]
W = W.merge(wf, on="word_key", how="left").dropna(subset=["zipf", "surprisal", "length"])
W["y_raw"] = np.log(W.GD.clip(lower=1))
# within state and subject: z-score (removes the additive and scale shift)
W["y_z"] = W.groupby(["subject", "is_mw"]).y_raw.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
# residual after removing word properties, fitted within subject and state
res = []
for (s, st), gg in W.groupby(["subject", "is_mw"]):
    X = np.column_stack([np.ones(len(gg))] + [gg[c].to_numpy() for c in ["zipf", "surprisal", "length"]])
    b = np.linalg.lstsq(X, gg.y_z.to_numpy(), rcond=None)[0]
    r = gg.y_z.to_numpy() - X @ b
    res.append(pd.Series(r, index=gg.index))
W["y_resid"] = pd.concat(res).sort_index()

rep["D_isc"] = {}
for ycol, lab in [("y_raw", "raw log gaze duration"),
                  ("y_z", "within-state z-scored (effort shift removed)"),
                  ("y_resid", "residual after word properties (lexical component removed)")]:
    # template from OTHER readers' on-task readings only
    on = W[W.is_mw == 0]
    tot = on.groupby("word_key")[ycol].agg(["sum", "count"])
    vals = []
    for s, gg in W.groupby("subject"):
        own = gg[gg.is_mw == 0].groupby("word_key")[ycol].agg(["sum", "count"])
        t = tot.join(own, rsuffix="_own", how="left").fillna(0)
        loo = (t["sum"] - t["sum_own"]) / (t["count"] - t["count_own"]).replace(0, np.nan)
        n_other = (t["count"] - t["count_own"])
        loo = loo[n_other >= 5]
        gg = gg[gg.word_key.isin(loo.index)]
        a = gg[gg.is_mw == 1]
        b = gg[gg.is_mw == 0]
        if len(a) < 60 or len(b) < 60:
            continue
        r_mw = np.corrcoef(a[ycol], loo.loc[a.word_key])[0, 1]
        # match N by subsampling the on-task words
        rs = []
        for _ in range(50):
            idx = RNG.choice(len(b), size=len(a), replace=False)
            bb = b.iloc[idx]
            rs.append(np.corrcoef(bb[ycol], loo.loc[bb.word_key])[0, 1])
        vals.append((s, np.mean(rs), r_mw, len(a)))
    V = np.array([(v[1], v[2]) for v in vals])
    r = boot_ci(V[:, 1] - V[:, 0])
    rep["D_isc"][ycol] = {"label": lab, "n_readers": len(V), "isc_on": float(V[:, 0].mean()),
                          "isc_mw": float(V[:, 1].mean()), "diff": r,
                          "retention_pct": float(V[:, 1].mean() / V[:, 0].mean() * 100)}
    print(f"  {lab:52s} n={len(V)}")
    print(f"     on-task {V[:,0].mean():.4f}  MW {V[:,1].mean():.4f}  "
          f"retention {V[:,1].mean()/V[:,0].mean()*100:.1f}%")
    print(fmt("     difference", r))

json.dump(rep, open(RES / "mechanism_tests.json", "w"), indent=2, default=float)
print(f"\nwrote {RES/'mechanism_tests.json'}")
