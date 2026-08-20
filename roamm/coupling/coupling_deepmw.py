#!/usr/bin/env python3
"""Power/robustness check: is the decoupling null caused by fuzzy MW-span boundaries?

Define, per subject-run (fixations ordered by time):
  deep MW    = fixation inside a run of >=4 consecutive MW fixations, not at the edge
  clean task = fixation with no MW fixation within +/-3 neighbours
Then re-run the behavioral frequency-coupling interaction and the neural occ_N1/occ_P2
frequency-coupling interaction using only these sharpened conditions. If decoupling is
real but blurred by label timing, the sharpened contrast should reveal it.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path("roamm/artifacts/coupling")
RNG = np.random.default_rng(11)

fix = pd.read_parquet(OUT / "fixations_frp.parquet")
wf = pd.read_parquet(OUT / "word_features.parquet")[["word_key","length","zipf","clean"]]
df = fix.merge(wf, on="word_key", how="left")
for r in ["occ_N1","occ_P2"]:
    df["frp_"+r] = df["frp_"+r]*1e6
df["p2p_uV"] = df["frp_p2p"]*1e6

# order within subject-run and compute MW-run structure
df = df.sort_values(["subject","run","tStart"]).reset_index(drop=True)
deep = np.zeros(len(df), bool); clean = np.zeros(len(df), bool)
for _, g in df.groupby(["subject","run"], sort=False):
    idx = g.index.to_numpy()
    mw = g["is_mw"].to_numpy().astype(int)
    # consecutive-MW run id
    brk = np.r_[0, np.cumsum(mw[1:] != mw[:-1])]
    for rid in np.unique(brk[mw == 1]):
        sel = np.flatnonzero((brk == rid) & (mw == 1))
        if len(sel) >= 4:
            deep[idx[sel[1:-1]]] = True          # drop edges
    # clean on-task: no MW within +/-3
    pad = np.r_[np.zeros(3,int), mw, np.zeros(3,int)]
    near = np.array([pad[i:i+7].sum() for i in range(len(mw))])
    clean[idx[(mw == 0) & (near == 0)]] = True

df["cond"] = np.where(deep, 1, np.where(clean, 0, -1))
sub = df[df["cond"] >= 0].copy()
sub = sub[(sub["clean"].str.len() >= 1) & (sub["zipf"] > 0) & sub["fix_dur"].between(50,1000)]
sub["log_dur"] = np.log(sub["fix_dur"].to_numpy())
print(f"deep-MW fixations: {int(deep.sum())}  clean-task: {int(clean.sum())}  "
      f"used: {len(sub)}")

def interaction(resp, artifact=False):
    """per-subject slope-change (cond=1 deep-MW minus cond=0) for zipf; group test."""
    recs = []
    for s, gg in sub.groupby("subject"):
        g = gg
        if artifact:
            g = g[g["p2p_uV"] <= 150]
        c = g["cond"].to_numpy().astype(float)
        if (c == 1).sum() < 30 or (c == 0).sum() < 100:
            continue
        zc = g["zipf"].to_numpy()-g["zipf"].mean()
        lc = g["length"].to_numpy()-g["length"].mean()
        dc = np.log(g["fix_dur"].to_numpy())-np.log(g["fix_dur"].to_numpy()).mean()
        y = g[resp].to_numpy().astype(float); ok = np.isfinite(y)
        if resp == "log_dur":
            X = np.column_stack([np.ones(len(g)), zc, lc, c, zc*c, lc*c])
        else:
            X = np.column_stack([np.ones(len(g)), zc, lc, dc, c, zc*c, lc*c, dc*c])
        b,*_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        recs.append({"subject": s, "zipf": b[1], "zipf_x_mw": b[4] if resp=="log_dur" else b[5]})
    R = pd.DataFrame(recs)
    v = R["zipf_x_mw"].to_numpy(); t,p = stats.ttest_1samp(v,0)
    boot = np.array([RNG.choice(v,len(v)).mean() for _ in range(10000)])
    ci = np.percentile(boot,[2.5,97.5])
    ont = R["zipf"].to_numpy()
    return {"resp": resp, "n": len(R), "on_task_zipf": float(ont.mean()),
            "interaction": float(v.mean()), "ci": [float(ci[0]),float(ci[1])],
            "t": float(t), "p": float(p), "frac_pos": float((v>0).mean())}

res = {"behavioral_logdur": interaction("log_dur"),
       "neural_occ_N1": interaction("frp_occ_N1", artifact=True),
       "neural_occ_P2": interaction("frp_occ_P2", artifact=True)}
(OUT / "deepmw_report.json").write_text(json.dumps(res, indent=2)+"\n")
for k,v in res.items():
    print(f"\n{k}: n={v['n']} on-task zipf slope={v['on_task_zipf']:+.4f}")
    print(f"    deep-MW interaction={v['interaction']:+.4f} CI[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] "
          f"t={v['t']:+.2f} p={v['p']:.3f} frac_pos={v['frac_pos']:.2f}")
