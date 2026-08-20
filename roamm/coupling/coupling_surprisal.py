#!/usr/bin/env python3
"""Deep-vs-shallow decoupling test. Frequency = shallow lexical access (preserved,
shown earlier). Surprisal = deep predictive/semantic processing (the N400 variable).
Hypothesis: surprisal coupling decouples during MW even though frequency does not.

Per subject, unique-variance model (all lexical predictors centered & entered together):
  behavioral:  log_dur ~ zipf + length + surprisal + order
                         + is_mw*(zipf, length, surprisal, order)
  neural:      frp_roi ~ zipf + length + surprisal + logdur + order + is_mw*(...)
Key term: surprisal:is_mw. Group one-sample test across subjects; also deep-MW subset.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path("roamm/artifacts/coupling")
RNG = np.random.default_rng(23)

fix = pd.read_parquet(OUT / "fixations_frp.parquet")
wf = pd.read_parquet(OUT / "word_features.parquet")[["word_key","length","zipf","surprisal","clean"]]
df = fix.merge(wf, on="word_key", how="left")
for r in ["occ_N1","occ_P2","cp_mid","cp_N400"]:
    df["frp_"+r] = df["frp_"+r]*1e6
df["p2p_uV"] = df["frp_p2p"]*1e6
df = df[(df["clean"].str.len()>=1) & (df["zipf"]>0) & df["surprisal"].notna()
        & df["fix_dur"].between(50,1000)]
df["log_dur"] = np.log(df["fix_dur"].to_numpy())

# deep-MW / clean-task labels (reuse logic)
df = df.sort_values(["subject","run","tStart"]).reset_index(drop=True)
deep = np.zeros(len(df), bool); clean = np.zeros(len(df), bool)
for _, g in df.groupby(["subject","run"], sort=False):
    idx = g.index.to_numpy(); mw = g["is_mw"].to_numpy().astype(int)
    brk = np.r_[0, np.cumsum(mw[1:] != mw[:-1])]
    for rid in np.unique(brk[mw==1]):
        sel = np.flatnonzero((brk==rid)&(mw==1))
        if len(sel) >= 4: deep[idx[sel[1:-1]]] = True
    pad = np.r_[np.zeros(3,int), mw, np.zeros(3,int)]
    near = np.array([pad[i:i+7].sum() for i in range(len(mw))])
    clean[idx[(mw==0)&(near==0)]] = True
df["deepcond"] = np.where(deep,1,np.where(clean,0,-1))

def center(x): x=x.to_numpy().astype(float); return x-x.mean()

def run_model(resp, mwcol, artifact, deeponly):
    recs=[]
    for s, g in df.groupby("subject"):
        if deeponly: g = g[g["deepcond"]>=0]
        if artifact: g = g[g["p2p_uV"]<=150]
        if len(g) < 300: continue
        mw = (g["deepcond"].to_numpy()==1).astype(float) if deeponly else g[mwcol].to_numpy().astype(float)
        if mw.sum()<40 or (1-mw).sum()<150: continue
        zc,lc,sc = center(g["zipf"]),center(g["length"]),center(g["surprisal"])
        o=g["fix_order"].to_numpy().astype(float); oz=(o-o.mean())/(o.std()+1e-9)
        y=g[resp].to_numpy().astype(float); ok=np.isfinite(y)
        cols=[np.ones(len(g)),zc,lc,sc,oz,mw,zc*mw,lc*mw,sc*mw,oz*mw]
        names=["int","zipf","length","surp","order","is_mw","zipf_x","len_x","surp_x","ord_x"]
        if resp!="log_dur":
            dc=np.log(g["fix_dur"].to_numpy()); dc=dc-dc.mean()
            cols=cols[:5]+[dc]+cols[5:]+[dc*mw]
            names=names[:5]+["logdur"]+names[5:]+["logdur_x"]
        X=np.column_stack(cols)
        if ok.sum()<300: continue
        b,*_=np.linalg.lstsq(X[ok],y[ok],rcond=None); d=dict(zip(names,b)); d["subject"]=s
        recs.append(d)
    R=pd.DataFrame(recs)
    def grp(col):
        v=R[col].to_numpy(); v=v[np.isfinite(v)]; t,p=stats.ttest_1samp(v,0)
        boot=np.array([RNG.choice(v,len(v)).mean() for _ in range(10000)]); ci=np.percentile(boot,[2.5,97.5])
        return {"mean":float(v.mean()),"ci":[float(ci[0]),float(ci[1])],"t":float(t),"p":float(p),"frac_pos":float((v>0).mean())}
    return {"n":len(R),"on_task_surp":grp("surp"),"surp_x_mw":grp("surp_x"),
            "on_task_zipf":grp("zipf"),"zipf_x_mw":grp("zipf_x")}

report={}
for tag,resp,deeponly in [("behavioral","log_dur",False),("behavioral_deep","log_dur",True),
                          ("neural_cp_mid","frp_cp_mid",False),("neural_cp_N400","frp_cp_N400",False),
                          ("neural_cp_N400_deep","frp_cp_N400",True)]:
    art = resp!="log_dur"
    report[tag]=run_model(resp,"is_mw",art,deeponly)
    r=report[tag]
    print(f"\n[{tag}] n={r['n']}")
    print(f"  SURPRISAL on-task={r['on_task_surp']['mean']:+.4f} p={r['on_task_surp']['p']:.1e} "
          f"| x_MW={r['surp_x_mw']['mean']:+.4f} CI[{r['surp_x_mw']['ci'][0]:+.4f},{r['surp_x_mw']['ci'][1]:+.4f}] "
          f"p={r['surp_x_mw']['p']:.3f} frac+={r['surp_x_mw']['frac_pos']:.2f}")
    print(f"  FREQ      on-task={r['on_task_zipf']['mean']:+.4f} p={r['on_task_zipf']['p']:.1e} "
          f"| x_MW={r['zipf_x_mw']['mean']:+.4f} p={r['zipf_x_mw']['p']:.3f}")
(OUT/"surprisal_coupling_report.json").write_text(json.dumps(report,indent=2)+"\n")
print("\nwrote surprisal_coupling_report.json")
