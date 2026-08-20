#!/usr/bin/env python3
"""(1) Equivalence/power analysis for the preserved-coupling claim, and
(2) grand-average FRP waveforms by word-property x MW condition for the figure.

Equivalence: express the MW x property interaction as a fraction of the on-task slope
(per subject), with subject-bootstrap CI, and a TOST-style check of whether we can
exclude a >=33% attenuation of the coupling during MW.

Waveforms: from frp_epochs.npy, grand-average occipitotemporal FRP by frequency
tercile and centroparietal FRP by surprisal tercile, each split MW vs on-task.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path("roamm/artifacts/coupling")
RNG = np.random.default_rng(31)
EEG = ['Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1','C1','C3','C5','T7','TP7','CP5','CP3','CP1','P1','P3','P5','P7','P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz','Fpz','Fp2','AF8','AF4','Afz','Fz','F2','F4','F6','F8','FT8','FC6','FC4','FC2','FCz','Cz','C2','C4','C6','T8','TP8','CP6','CP4','CP2','P2','P4','P6','P8','P10','PO8','PO4','O2']
CHI = {c:i for i,c in enumerate(EEG)}
OCC = ['PO7','PO8','PO3','PO4','O1','O2','Oz','POz','P7','P8','P9','P10','Iz']
CP  = ['Cz','CPz','Pz','CP1','CP2','C1','C2','P1','P2']

fix = pd.read_parquet(OUT/"fixations.parquet").reset_index(drop=True)
wf = pd.read_parquet(OUT/"word_features.parquet")[["word_key","length","zipf","surprisal","clean"]]
df = fix.merge(wf, on="word_key", how="left")
good = (df["clean"].str.len()>=1)&(df["zipf"]>0)&df["surprisal"].notna()&df["fix_dur"].between(50,1000)
df = df[good].reset_index(drop=True)
df["log_dur"] = np.log(df["fix_dur"].to_numpy())

# ---------- (1) equivalence on behavioral coupling ----------
def per_subject_slopes(prop):
    rows=[]
    for s,g in df.groupby("subject"):
        mw=g["is_mw"].to_numpy().astype(float)
        if mw.sum()<40 or (1-mw).sum()<150: continue
        def c(x): x=x.to_numpy().astype(float); return x-x.mean()
        zc,lc,sc=c(g["zipf"]),c(g["length"]),c(g["surprisal"])
        pc={"zipf":zc,"length":lc,"surprisal":sc}[prop]
        o=g["fix_order"].to_numpy().astype(float); oz=(o-o.mean())/(o.std()+1e-9)
        y=g["log_dur"].to_numpy()
        # full covariate model, isolate property slope + its MW interaction
        X=np.column_stack([np.ones(len(g)),zc,lc,sc,oz,mw,zc*mw,lc*mw,sc*mw,oz*mw])
        b,*_=np.linalg.lstsq(X,y,rcond=None)
        base={"zipf":b[1],"length":b[2],"surprisal":b[3]}[prop]
        inter={"zipf":b[6],"length":b[7],"surprisal":b[8]}[prop]
        rows.append((s,base,inter))
    return pd.DataFrame(rows,columns=["subject","base","inter"])

equiv={}
for prop in ["zipf","surprisal"]:
    R=per_subject_slopes(prop)
    base=R["base"].to_numpy(); inter=R["inter"].to_numpy()
    # MW slope = base+inter ; fractional retention = (base+inter)/base, but sign-aware:
    # attenuation fraction = -inter/base (positive = coupling weakened toward 0)
    boot_att=[]; boot_mwslope=[]
    for _ in range(10000):
        idx=RNG.integers(0,len(R),len(R))
        bb=base[idx].mean(); ii=inter[idx].mean()
        boot_att.append(-ii/bb); boot_mwslope.append(bb+ii)
    att=np.array(boot_att)
    equiv[prop]={
        "on_task_slope":float(base.mean()),
        "mw_slope":float((base+inter).mean()),
        "attenuation_frac_mean":float(np.median(att)),
        "attenuation_frac_ci":[float(np.percentile(att,2.5)),float(np.percentile(att,97.5))],
        "excludes_33pct_attenuation":bool(np.percentile(att,97.5)<0.33),
        "n_subjects":int(len(R)),
    }
    print(f"[{prop}] on-task slope={base.mean():+.4f} MW slope={(base+inter).mean():+.4f} "
          f"| attenuation={np.median(att)*100:+.0f}% CI[{np.percentile(att,2.5)*100:+.0f}%,{np.percentile(att,97.5)*100:+.0f}%] "
          f"exclude>=33%? {equiv[prop]['excludes_33pct_attenuation']}")

# ---------- (2) grand-average waveforms ----------
ep = np.load(OUT/"frp_epochs.npy")           # [n,64,T] uV, aligned to fixations.parquet
t = np.load(OUT/"frp_epochs_time.npy")
valid = np.load(OUT/"frp_epochs_valid.npy")
gi = df.index.to_numpy()                      # rows kept (into original fixations order)
sub_ok = valid[gi]
gi = gi[sub_ok]
sdf = df.loc[df.index.isin(gi)].copy()

def grand_avg(prop, chans, terciles=(0.33,0.66)):
    ch=[CHI[c] for c in chans]
    out={}
    for cond,name in ((0,"ontask"),(1,"mw")):
        sel=sdf[sdf["is_mw"]==cond]
        lo=sel[prop].quantile(terciles[0]); hi=sel[prop].quantile(terciles[1])
        for band,mask in (("low",sel[prop]<=lo),("high",sel[prop]>=hi)):
            rows=sel.index[mask.to_numpy()].to_numpy()
            w=ep[rows][:,ch,:].mean(axis=(0,1)).astype(float)   # avg ch & trials
            out[f"{name}_{band}"]=w
            out[f"{name}_{band}_n"]=int(len(rows))
    return out

waves={"time":t.tolist(),
       "occ_by_freq":{k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in grand_avg("zipf",OCC).items()},
       "cp_by_surprisal":{k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in grand_avg("surprisal",CP).items()}}
(OUT/"waveforms.json").write_text(json.dumps(waves))
(OUT/"equivalence_report.json").write_text(json.dumps(equiv,indent=2)+"\n")
print("\nwrote waveforms.json + equivalence_report.json")
