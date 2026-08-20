#!/usr/bin/env python3
"""G7 — ZuCo session control.

ZuCo 1.0 task order is fixed and identical for every subject (Hollenstein et al. 2018):
  session 1 = Task 2 (NR) then the FIRST half of Task 1 (SR)
  session 2 = Task 3 (TSR) then the SECOND half of Task 1 (SR)
So the headline NR-vs-TSR goal contrast is also a session contrast. Splitting SR at its
sentence-index midpoint yields SR-half-1 (session 1) and SR-half-2 (session 2), both deep
tasks, which isolates session from task.
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
from common import RES, boot_ci, fmt

ZA = str(ROOT / "zuco" / "artifacts")
PROPS = ["zipf", "surprisal"]

recs = []
for task in ["NR", "SR", "TSR"]:
    ling = pd.read_parquet(f"{ZA}/linguistic_{task}.parquet")
    for mp in sorted(glob.glob(f"{ZA}/frp/meta_*_{task}.parquet")):
        subj = os.path.basename(mp).split("_")[1]
        m = pd.read_parquet(mp).copy()
        m["subject"] = subj
        recs.append(m.merge(ling, on=["task", "sent_idx", "word_idx"], how="left"))
Z = pd.concat(recs, ignore_index=True).dropna(subset=PROPS)
Z["GDms"] = Z["GD"] / 500 * 1000
Z["logGD"] = np.log(Z.GDms.clip(lower=1))

sr_max = Z.loc[Z.task == "SR", "sent_idx"].max()
mid = sr_max / 2.0
Z["cond"] = Z["task"]
Z.loc[(Z.task == "SR") & (Z.sent_idx <= mid), "cond"] = "SR1"
Z.loc[(Z.task == "SR") & (Z.sent_idx > mid), "cond"] = "SR2"
Z["session"] = Z["cond"].map({"NR": 1, "SR1": 1, "TSR": 2, "SR2": 2})
Z["depth"] = Z["cond"].map({"NR": "deep", "SR1": "deep", "SR2": "deep", "TSR": "shallow"})

print("ZuCo condition table (SR split at sent_idx midpoint = %.0f):" % mid)
print(Z.groupby(["cond", "session", "depth"]).agg(words=("logGD", "size"),
                                                  subjects=("subject", "nunique"),
                                                  meanGD=("GDms", "mean")).round(1).to_string())

rep = {"sr_midpoint_sent_idx": float(mid)}


def slopes(df, cond, prop, minn=80):
    out = {}
    for s, g in df[df.cond == cond].groupby("subject"):
        g = g.dropna(subset=["logGD", prop])
        if len(g) < minn:
            continue
        z = (g[prop] - g[prop].mean()) / g[prop].std()
        out[s] = float(np.polyfit(z, g.logGD, 1)[0])
    return out


print("\nPer-subject SIGNED zipf/surprisal -> logGD slopes by condition:")
rep["slopes"] = {}
S = {}
for prop in PROPS:
    S[prop] = {c: slopes(Z, c, prop) for c in ["NR", "SR1", "SR2", "TSR"]}
    rep["slopes"][prop] = {c: {"mean": float(np.mean(list(v.values()))), "n": len(v)}
                           for c, v in S[prop].items()}
    print(f"  {prop:10s} " + "  ".join(
        f"{c}={np.mean(list(S[prop][c].values())):+.4f}(n={len(S[prop][c])})" for c in ["NR", "SR1", "SR2", "TSR"]))


def contrast(prop, a, b, label):
    sa, sb = S[prop][a], S[prop][b]
    common = sorted(set(sa) & set(sb))
    va = np.array([sa[k] for k in common])
    vb = np.array([sb[k] for k in common])
    ret = vb.mean() / va.mean() * 100
    t, p = stats.ttest_rel(vb, va)
    r = boot_ci(vb - va)
    r.update({"retention_pct": float(ret), "t_paired": float(t), "p_paired": float(p),
              "n_common": len(common), "ref": a, "test": b})
    print(f"  {label:38s} {a}->{b}: retention {ret:6.1f}%  Δ={r['mean']:+.5f} "
          f"[{r['ci'][0]:+.5f},{r['ci'][1]:+.5f}] paired t={t:+.2f} p={p:.3g} n={len(common)}")
    return r


print("\n=== G7 CONTRASTS (retention = |slope_test| / |slope_ref| on signed slopes) ===")
rep["contrasts"] = {}
for prop in PROPS:
    print(f"\n {prop}:")
    rep["contrasts"][prop] = {
        "session_effect_SR1_to_SR2": contrast(prop, "SR1", "SR2", "SESSION effect (deep, both halves)"),
        "task_within_session2_SR2_to_TSR": contrast(prop, "SR2", "TSR", "TASK within SESSION 2 (deep->shallow)"),
        "headline_NR_to_TSR": contrast(prop, "NR", "TSR", "headline NR->TSR (task AND session)"),
        "materials_NR_to_SR1": contrast(prop, "NR", "SR1", "MATERIALS within session 1 (wiki->movie)"),
    }

print("\n=== GATE G7 ===")
ok = []
for prop in PROPS:
    c = rep["contrasts"][prop]
    sess_ok = c["session_effect_SR1_to_SR2"]["p_paired"] > 0.05
    task_ok = (c["task_within_session2_SR2_to_TSR"]["retention_pct"] < 80 and
               c["task_within_session2_SR2_to_TSR"]["p_paired"] < 0.05)
    ok.append(sess_ok and task_ok)
    print(f"  {prop:10s} session effect n.s.: {sess_ok}   "
          f"within-session-2 task decoupling: {task_ok} "
          f"({c['task_within_session2_SR2_to_TSR']['retention_pct']:.1f}%, "
          f"p={c['task_within_session2_SR2_to_TSR']['p_paired']:.3g})")
rep["GATE_G7_PASS"] = bool(all(ok))
print(f"  GATE G7: {'PASS' if all(ok) else 'FAIL'}")

json.dump(rep, open(RES / "g7_zuco_session.json", "w"), indent=2, default=float)
print(f"\nwrote {RES/'g7_zuco_session.json'}")
