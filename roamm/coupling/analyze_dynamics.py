#!/usr/bin/env python3
"""Analyze peri-MW dynamics: is there a consistent neural/ocular ramp into and out of
mind-wandering? Per-subject average trajectories -> group mean +/- SEM; test the
pre-onset vs post-onset change (and the peri-onset slope) against zero with subject-level
stats and a cluster-based permutation across time."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats

OUT = Path("roamm/artifacts/coupling")
d = np.load(OUT/"dynamics.npz")
t = d["t"]; RNG = np.random.default_rng(3)
SIGS = ["pupil","alpha","beta","blink","sacc","fixdur"]

def subj_avg(prefix, meta):
    """return [n_subj, NBIN] per-subject mean trajectory for a signal."""
    out = {}
    for sg in SIGS:
        arr = d[f"{prefix}_{sg}"]; subs = meta
        rows=[]; ids=[]
        for s in np.unique(subs):
            m = subs==s
            if m.sum()>=3:
                rows.append(np.nanmean(arr[m],axis=0)); ids.append(s)
        out[sg] = (np.array(rows), np.array(ids))
    return out

def cluster_perm(X, tmask, n_perm=5000):
    """one-sample cluster test that trajectory deviates from its pre-onset baseline."""
    K = X[:, tmask]
    def clust(x):
        tv = x.mean(0)/(x.std(0,ddof=1)/np.sqrt(x.shape[0])+1e-12)
        thr = stats.t.ppf(0.975, x.shape[0]-1); sig=np.abs(tv)>thr; out=[]; i=0
        while i<len(sig):
            if sig[i]:
                j=i; ss=0.0
                while j<len(sig) and sig[j] and np.sign(tv[j])==np.sign(tv[i]): ss+=tv[j]; j+=1
                out.append((i,j,ss)); i=j
            else: i+=1
        return out
    obs=clust(K)
    if not obs: return {"sig":[], "min_p":1.0}
    null=np.empty(n_perm)
    for b in range(n_perm):
        null[b]=max((abs(c[2]) for c in clust(K*RNG.choice([-1,1],K.shape[0])[:,None])),default=0)
    tt=t[tmask]
    res=[{"t0":float(tt[i]),"t1":float(tt[j-1]),"p":float((null>=abs(m)).mean())} for i,j,m in obs]
    return {"sig":[r for r in res if r["p"]<0.05], "min_p":min(r["p"] for r in res)}

on = subj_avg("on", d["meta_on"]); off = subj_avg("off", d["meta_off"])
pseudo = subj_avg("pseudo", d["meta_pseudo"]) if "meta_pseudo" in d.files else None
base = (t>=-8)&(t<=-4)   # pre-onset baseline
during = (t>=1)&(t<=5)   # into MW

report={"n_onset_spans":int(len(d["meta_on"])),"n_offset_spans":int(len(d["meta_off"]))}
print(f"onset spans: {len(d['meta_on'])}, offset spans: {len(d['meta_off'])}\n")
print("=== PERI-ONSET: pre-onset(-8..-4s) -> during(1..5s) change, per subject (+pseudo control) ===")
for sg in SIGS:
    X, ids = on[sg]
    Xb = X - X[:, base].mean(1, keepdims=True)     # baseline-correct per subject
    dur = Xb[:, during].mean(1)
    tval,p = stats.ttest_1samp(dur,0)
    cl = cluster_perm(Xb, (t>=-8)&(t<=8))
    ps_str = ""
    if pseudo is not None:
        Xp = pseudo[sg][0]; Xpb = Xp - Xp[:, base].mean(1, keepdims=True)
        dp = Xpb[:, during].mean(1); tp,pp = stats.ttest_1samp(dp,0)
        # real-vs-pseudo paired (by subject intersection)
        report[f"pseudo_{sg}"] = {"change":float(dp.mean()),"p":float(pp)}
        ps_str = f" | PSEUDO Δ={dp.mean():+.3f} p={pp:.3f}"
    report[f"onset_{sg}"] = {"during_change":float(dur.mean()),"t":float(tval),"p":float(p),
                             "frac_pos":float((dur>0).mean()),"cluster_min_p":cl["min_p"],
                             "sig_windows":cl["sig"]}
    star="***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
    print(f"  {sg:7} Δ={dur.mean():+.3f} t={tval:+.2f} p={p:.4f} clust_p={cl['min_p']:.3f} {star}{ps_str}")

np.savez(OUT/"dynamics_subjavg.npz", t=t,
         **{f"on_{sg}":on[sg][0] for sg in SIGS},
         **{f"off_{sg}":off[sg][0] for sg in SIGS},
         **({f"pseudo_{sg}":pseudo[sg][0] for sg in SIGS} if pseudo is not None else {}))
(OUT/"dynamics_report.json").write_text(json.dumps(report,indent=2)+"\n")
print("\nwrote dynamics_report.json + dynamics_subjavg.npz")
