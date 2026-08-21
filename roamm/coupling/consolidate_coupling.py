#!/usr/bin/env python3
"""Consolidate the coupling result: additive slowing vs multiplicative decoupling.

(1) Robustness of the MW fixation-lengthening main effect: per-subject, controlling
    for within-run time-on-task (fix_order) AND page position, with subject bootstrap
    and a within-subject label-shuffle control.
(2) Assemble the final numbers table across behavioral + neural + LMM analyses.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path("roamm/artifacts/coupling")
RNG = np.random.default_rng(41)
fix = pd.read_parquet(OUT/"fixations_frp.parquet")
wf = pd.read_parquet(OUT/"word_features.parquet")[["word_key","length","zipf","surprisal","clean"]]
df = fix.merge(wf, on="word_key", how="left")
df = df[(df["clean"].str.len()>=1)&(df["zipf"]>0)&df["surprisal"].notna()&df["fix_dur"].between(50,1000)].copy()
df["log_dur"] = np.log(df["fix_dur"].to_numpy())

def mw_mean_effect(shuffle=False):
    """per-subject MW effect on log_dur controlling zipf,length,surprisal,order,page."""
    rows=[]
    for s,g in df.groupby("subject"):
        mw=g["is_mw"].to_numpy().astype(float)
        if shuffle: mw=RNG.permutation(mw)
        if mw.sum()<40: continue
        def c(x): x=x.to_numpy().astype(float); return x-x.mean()
        o=g["fix_order"].to_numpy().astype(float); oz=(o-o.mean())/(o.std()+1e-9)
        pg=g["page"].to_numpy().astype(float); pz=(pg-np.nanmean(pg))/(np.nanstd(pg)+1e-9)
        y=g["log_dur"].to_numpy()
        X=np.column_stack([np.ones(len(g)),c(g["zipf"]),c(g["length"]),c(g["surprisal"]),oz,np.nan_to_num(pz),mw])
        b,*_=np.linalg.lstsq(X,y,rcond=None)
        rows.append(b[-1])
    v=np.array(rows); t,p=stats.ttest_1samp(v,0)
    boot=np.array([RNG.choice(v,len(v)).mean() for _ in range(10000)]); ci=np.percentile(boot,[2.5,97.5])
    return {"mean_logdur":float(v.mean()),"pct":float((np.exp(v.mean())-1)*100),
            "ci_pct":[float((np.exp(ci[0])-1)*100),float((np.exp(ci[1])-1)*100)],
            "t":float(t),"p":float(p),"frac_pos":float((v>0).mean()),"n":int(len(v))}

real=mw_mean_effect(False); shuf=mw_mean_effect(True)
print("MW fixation-lengthening (time-on-task + page controlled):")
print(f"  real:    +{real['pct']:.1f}% CI[{real['ci_pct'][0]:+.1f}%,{real['ci_pct'][1]:+.1f}%] "
      f"t={real['t']:+.2f} p={real['p']:.2e} frac_pos={real['frac_pos']:.2f} n={real['n']}")
print(f"  shuffle: {shuf['pct']:+.2f}% p={shuf['p']:.2f} (control, should be ~0)")

# neural MW main effect on N1/N400 (uniform amplitude shift?), controlling word props + logdur
for r in ["occ_N1","cp_N400"]:
    df[r]=df["frp_"+r]*1e6
df["p2p_uV"]=df["frp_p2p"]*1e6
def neural_mw(roi):
    rows=[]
    for s,g in df[df["p2p_uV"]<=150].groupby("subject"):
        mw=g["is_mw"].to_numpy().astype(float)
        if mw.sum()<40: continue
        def c(x): x=x.to_numpy().astype(float); return x-x.mean()
        o=g["fix_order"].to_numpy().astype(float); oz=(o-o.mean())/(o.std()+1e-9)
        y=g[roi].to_numpy(); ok=np.isfinite(y)
        X=np.column_stack([np.ones(len(g)),c(g["zipf"]),c(g["length"]),c(g["surprisal"]),
                           np.log(g["fix_dur"].to_numpy()),oz,mw])
        b,*_=np.linalg.lstsq(X[ok],y[ok],rcond=None); rows.append(b[-1])
    v=np.array(rows); t,p=stats.ttest_1samp(v,0)
    return {"mean_uV":float(v.mean()),"t":float(t),"p":float(p),"n":int(len(v))}
neural={r:neural_mw(r) for r in ["occ_N1","cp_N400"]}
print("\nNeural MW main effect (uniform FRP shift):")
for r,d in neural.items():
    print(f"  {r}: {d['mean_uV']:+.4f} uV t={d['t']:+.2f} p={d['p']:.3f}")

summary={"mw_lengthening":real,"mw_lengthening_shuffle":shuf,"neural_mw_main":neural}
(OUT/"coupling_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print("\nwrote coupling_summary.json")
