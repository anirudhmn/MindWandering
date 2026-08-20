#!/usr/bin/env python3
"""G5 — within-token-instance identification of the MW duration-coupling result.
G6 — measurement-scale audit (raw ms vs log) for ROAMM and ZuCo.

word_key is a unique corpus token instance read by ~44 readers, so MW status varies WITHIN
token instance. A two-way (word_key x subject) fixed-effects model therefore identifies the
property x MW interaction while holding the exact word, its context, and its page position
fixed.
"""
from __future__ import annotations
import json
import glob
import os
import numpy as np
import pandas as pd
from scipy import stats
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ART, RES, COUP, boot_ci, fmt

PROPS = ["zipf", "surprisal", "length"]
rep = {}

# ---------------------------------------------------------------- data
fx = pd.read_parquet(COUP / "fixations_frp.parquet")
wf = pd.read_parquet(COUP / "word_features.parquet")[["word_key", "zipf", "surprisal", "length"]]
fp = fx[fx.is_firstpass == 1]
W = (fp.sort_values(["subject", "word_key", "fix_order"])
       .groupby(["subject", "word_key"])
       .agg(GD=("fix_dur", "sum"), FFD=("fix_dur", "first"), is_mw=("is_mw", "first"),
            mw_frac=("mw_frac", "first"), occ_N1=("frp_occ_N1", "first"),
            cp_N400=("frp_cp_N400", "first"))
       .reset_index()
       .merge(wf, on="word_key", how="left")
       .dropna(subset=PROPS))
W["logGD"] = np.log(W.GD.clip(lower=1))
W["logFFD"] = np.log(W.FFD.clip(lower=1))
for p in PROPS:
    W[f"z{p}"] = (W[p] - W[p].mean()) / W[p].std()
print(f"ROAMM word-level rows {len(W)}, tokens {W.word_key.nunique()}, subjects {W.subject.nunique()}, "
      f"MW rate {W.is_mw.mean():.4f}")
mwpk = W.groupby("word_key").is_mw.agg(["mean", "size"])
both = ((mwpk["mean"] > 0) & (mwpk["mean"] < 1)).sum()
rep["tokens_with_within_token_MW_variation"] = int(both)
print(f"token instances read both on-task and during MW by different readers: {both} "
      f"({both/len(mwpk)*100:.1f}%)")


def twoway_fe(df, y, xcols, fe1="word_key", fe2="subject", tol=1e-9, maxit=200):
    """Absorb two fixed effects by alternating within-transformation."""
    cols = [y] + xcols
    M = df[cols].to_numpy(float).copy()
    g1 = df[fe1].astype("category").cat.codes.to_numpy()
    g2 = df[fe2].astype("category").cat.codes.to_numpy()
    n1, n2 = g1.max() + 1, g2.max() + 1
    c1 = np.bincount(g1, minlength=n1).astype(float)
    c2 = np.bincount(g2, minlength=n2).astype(float)
    for _ in range(maxit):
        prev = M[:, 0].copy()
        for gi, ci, ni in ((g1, c1, n1), (g2, c2, n2)):
            for j in range(M.shape[1]):
                s = np.bincount(gi, weights=M[:, j], minlength=ni)
                M[:, j] -= (s / ci)[gi]
        if np.max(np.abs(M[:, 0] - prev)) < tol:
            break
    Y, X = M[:, 0], M[:, 1:]
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ Y)
    resid = Y - X @ beta
    # subject-clustered covariance
    XtXi = np.linalg.inv(XtX)
    meat = np.zeros_like(XtX)
    for s in np.unique(g2):
        m = g2 == s
        u = (X[m] * resid[m][:, None]).sum(axis=0)
        meat += np.outer(u, u)
    V = XtXi @ meat @ XtXi
    se = np.sqrt(np.diag(V))
    nclust = len(np.unique(g2))
    V *= nclust / (nclust - 1)
    se = np.sqrt(np.diag(V))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), nclust - 1)
    return dict(zip(xcols, [dict(beta=float(b), se=float(s), t=float(tt), p=float(pp))
                            for b, s, tt, pp in zip(beta, se, t, p)]))


# ---------------------------------------------------------------- G5
print("\n=== G5: two-way (token x subject) fixed-effects interaction model ===")
rep["G5"] = {}
for ycol in ["logGD", "logFFD"]:
    d = W.dropna(subset=[ycol]).copy()
    d["mw"] = d.is_mw.astype(float)
    xcols = ["mw"] + [f"mw_x_{p}" for p in PROPS]
    for p in PROPS:
        d[f"mw_x_{p}"] = d["mw"] * d[f"z{p}"]
    r = twoway_fe(d, ycol, xcols)
    rep["G5"][ycol] = r
    print(f"\n  {ycol}  (n={len(d)}, token FE + subject FE, subject-clustered SE)")
    for k, v in r.items():
        print(f"    {k:14s} beta={v['beta']:+.5f} se={v['se']:.5f} t={v['t']:+.2f} p={v['p']:.3g}")

# across-word (no token FE) comparison, same rows: per-subject slope retention, SIGNED
print("\n  Comparison — across-word per-subject SIGNED slope retention (no token FE):")
rep["G5_across_word_signed_retention"] = {}
for p in PROPS:
    rr = []
    for s, g in W.groupby("subject"):
        ge, gd = g[g.is_mw == 0], g[g.is_mw == 1]
        if len(ge) < 200 or len(gd) < 80:
            continue
        be = np.polyfit(ge[f"z{p}"], ge.logGD, 1)[0]
        bd = np.polyfit(gd[f"z{p}"], gd.logGD, 1)[0]
        rr.append((be, bd))
    rr = np.array(rr)
    ret = rr[:, 1].mean() / rr[:, 0].mean() * 100
    t, pv = stats.ttest_rel(rr[:, 1], rr[:, 0])
    rep["G5_across_word_signed_retention"][p] = {"beta_on": float(rr[:, 0].mean()),
                                                 "beta_mw": float(rr[:, 1].mean()),
                                                 "retention_pct": float(ret),
                                                 "t": float(t), "p": float(pv), "n": int(len(rr))}
    print(f"    {p:10s} on={rr[:,0].mean():+.5f} mw={rr[:,1].mean():+.5f} "
          f"retention={ret:6.1f}%  paired t={t:+.2f} p={pv:.3g} n={len(rr)}")

# ---------------------------------------------------------------- G6
print("\n=== G6: measurement-scale audit ===")


def scale_pair(df, state_col, ycol_raw, xcol, minn=80):
    out = []
    for s, g in df.groupby("subject"):
        ge = g[g[state_col] == 0].dropna(subset=[ycol_raw, xcol])
        gd = g[g[state_col] == 1].dropna(subset=[ycol_raw, xcol])
        if len(ge) < minn or len(gd) < minn:
            continue
        ze = (ge[xcol] - ge[xcol].mean()) / ge[xcol].std()
        zd = (gd[xcol] - gd[xcol].mean()) / gd[xcol].std()
        out.append((np.polyfit(ze, ge[ycol_raw], 1)[0], np.polyfit(zd, gd[ycol_raw], 1)[0],
                    np.polyfit(ze, np.log(ge[ycol_raw].clip(lower=1)), 1)[0],
                    np.polyfit(zd, np.log(gd[ycol_raw].clip(lower=1)), 1)[0],
                    ge[ycol_raw].mean(), gd[ycol_raw].mean()))
    return np.array(out)


rep["G6"] = {}
Wm = W.copy()
Wm["state"] = Wm.is_mw.astype(int)
for p in ["zipf", "surprisal"]:
    a = scale_pair(Wm, "state", "GD", p)
    raw_ret = a[:, 1].mean() / a[:, 0].mean() * 100
    log_ret = a[:, 3].mean() / a[:, 2].mean() * 100
    mean_shift = a[:, 5].mean() / a[:, 4].mean()
    rep["G6"][f"ROAMM_MW_{p}"] = {"raw_retention_pct": float(raw_ret), "log_retention_pct": float(log_ret),
                                  "mean_duration_ratio": float(mean_shift),
                                  "log_retention_predicted_by_additive_shift_pct": float(100 / mean_shift),
                                  "n": int(len(a))}
    print(f"  ROAMM MW  {p:10s} raw-scale retention {raw_ret:6.1f}%  log-scale {log_ret:6.1f}%  "
          f"(mean GD ratio MW/on = {mean_shift:.3f}; a purely additive shift predicts "
          f"log retention {100/mean_shift:.1f}%)")

# ZuCo NR -> TSR
ZA = str(ROOT / "zuco" / "artifacts")
recs = []
for task in ["NR", "TSR"]:
    ling = pd.read_parquet(f"{ZA}/linguistic_{task}.parquet")
    for mp in sorted(glob.glob(f"{ZA}/frp/meta_*_{task}.parquet")):
        subj = os.path.basename(mp).split("_")[1]
        m = pd.read_parquet(mp)
        m = m.copy()
        m["subject"] = subj
        m["state"] = 0 if task == "NR" else 1
        recs.append(m.merge(ling, on=["task", "sent_idx", "word_idx"], how="left"))
Z = pd.concat(recs, ignore_index=True)
Z["GDms"] = Z["GD"] / 500 * 1000
for p in ["zipf", "surprisal"]:
    a = scale_pair(Z, "state", "GDms", p)
    raw_ret = a[:, 1].mean() / a[:, 0].mean() * 100
    log_ret = a[:, 3].mean() / a[:, 2].mean() * 100
    mean_shift = a[:, 5].mean() / a[:, 4].mean()
    rep["G6"][f"ZuCo_TSR_{p}"] = {"raw_retention_pct": float(raw_ret), "log_retention_pct": float(log_ret),
                                  "mean_duration_ratio": float(mean_shift),
                                  "log_retention_predicted_by_additive_shift_pct": float(100 / mean_shift),
                                  "n": int(len(a))}
    print(f"  ZuCo TSR  {p:10s} raw-scale retention {raw_ret:6.1f}%  log-scale {log_ret:6.1f}%  "
          f"(mean GD ratio TSR/NR = {mean_shift:.3f}; additive shift predicts "
          f"log retention {100/mean_shift:.1f}%)")

json.dump(rep, open(RES / "g5_g6_duration.json", "w"), indent=2, default=float)
print(f"\nwrote {RES/'g5_g6_duration.json'}")
