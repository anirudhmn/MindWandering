#!/usr/bin/env python3
"""Bulletproof the gaze-ISC-drop: pseudo-condition control (are the SPECIFIC MW words
lower-ISC than random same-count word sets?), template-variance check (range restriction),
and a combined neural-ISC test."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path("roamm/artifacts/coupling")
RNG = np.random.default_rng(404)

def loo_template(df, val):
    ot=df[df["is_mw"]==0]
    grp=ot.groupby(["story","pos"])[val].agg(["sum","count"])
    m=df.merge(grp,left_on=["story","pos"],right_index=True,how="left")
    own=((df["is_mw"].to_numpy()==0)).astype(float)*df[val].to_numpy()
    return (m["sum"].to_numpy()-own)/np.clip(m["count"].to_numpy()-(df["is_mw"].to_numpy()==0),1,None)

fx=pd.read_parquet(OUT/"reading_fixations.parquet")
fx=fx[fx["fix_dur"].between(50,1000)].copy(); fx["logdur"]=np.log(fx["fix_dur"])
gaze=fx.groupby(["subject","story","pos"]).agg(logdur=("logdur","mean"),is_mw=("is_mw","max"),
                                               tstart=("tStart","mean")).reset_index()
gaze["template"]=loo_template(gaze,"logdur")
gaze=gaze.dropna(subset=["template","logdur"])
# within-subject time-on-task decile (to position-match the pseudo control)
gaze["tdec"]=gaze.groupby("subject")["tstart"].transform(lambda s: pd.qcut(s.rank(method="first"),10,labels=False))

def isc(vals,temp):
    if len(vals)<80 or np.std(vals)<1e-9 or np.std(temp)<1e-9: return np.nan
    return np.corrcoef(vals,temp)[0,1]

# real per-subject diff + template SD by condition + pseudo-control
real=[]; tvar_mw=[]; tvar_ot=[]; pseudo_diffs=[]
K=1000
per_subj_pseudo={}
for s,g in gaze.groupby("subject"):
    mw=g["is_mw"].to_numpy(); v=g["logdur"].to_numpy(); tp=g["template"].to_numpy()
    if mw.sum()<80 or (mw==0).sum()<80: continue
    r_mw=isc(v[mw==1],tp[mw==1]); r_ot=isc(v[mw==0],tp[mw==0])
    if np.isnan(r_mw) or np.isnan(r_ot): continue
    real.append(r_mw-r_ot); tvar_mw.append(np.std(tp[mw==1])); tvar_ot.append(np.std(tp[mw==0]))
    # pseudo: subsets same size as MW, POSITION-MATCHED to MW's time-on-task deciles
    idx=np.arange(len(g)); tdec=g["tdec"].to_numpy(); mwdec=tdec[mw==1]
    dec_need=pd.Series(mwdec).value_counts().to_dict()
    ps=[]
    for _ in range(K):
        pick=[]
        for dd,need in dec_need.items():
            pool=idx[tdec==dd]
            pick.append(RNG.choice(pool,need,replace=len(pool)<need))
        sub=np.concatenate(pick); rest=np.setdiff1d(idx,sub)
        ps.append(isc(v[sub],tp[sub]) - isc(v[rest],tp[rest]))
    per_subj_pseudo[s]=np.nanmean(ps)
real=np.array(real)
pseudo=np.array(list(per_subj_pseudo.values()))
t,p=stats.ttest_1samp(real,0)
# real mean diff vs pseudo mean diff (paired-ish across subjects)
tp2,pp2=stats.ttest_rel(real, pseudo[:len(real)]) if len(pseudo)==len(real) else (np.nan,np.nan)
print("=== GAZE ISC robustness ===")
print(f"real Δ(MW-ontask) = {real.mean():+.4f}  p={p:.4f}  (n={len(real)}, frac_neg={np.mean(real<0):.2f})")
print(f"pseudo Δ (random same-count subsets) = {pseudo.mean():+.4f}")
print(f"real vs pseudo (paired) p={pp2:.4f}  -> MW-specific if real < pseudo")
print(f"template SD: MW={np.mean(tvar_mw):.3f}  on-task={np.mean(tvar_ot):.3f} "
      f"(similar => no range-restriction artifact)")

# combined neural ISC (meta): average the two neural diffs per subject
frp=pd.read_parquet(OUT/"fixations_frp.parquet")[["onset_abs_idx","subject","is_mw","frp_cp_N400","frp_occ_P2","frp_p2p","fix_dur"]]
posmap=fx[["onset_abs_idx","story","pos"]].drop_duplicates("onset_abs_idx")
frp=frp.merge(posmap,on="onset_abs_idx",how="inner")
frp=frp[(frp["frp_p2p"]*1e6<=150)&frp["fix_dur"].between(50,1000)]
neur_diffs={}
for col in ["frp_cp_N400","frp_occ_P2"]:
    frp[col]=frp[col]*1e6
    w=frp.groupby(["subject","story","pos"]).agg(**{col:(col,"mean"),"is_mw":("is_mw","max")}).reset_index()
    w["template"]=loo_template(w,col); w=w.dropna(subset=["template",col])
    dd={}
    for s,g in w.groupby("subject"):
        mw=g["is_mw"].to_numpy()
        if mw.sum()<80 or (mw==0).sum()<80: continue
        r_mw=isc(g[col].to_numpy()[mw==1],g["template"].to_numpy()[mw==1])
        r_ot=isc(g[col].to_numpy()[mw==0],g["template"].to_numpy()[mw==0])
        if not (np.isnan(r_mw) or np.isnan(r_ot)): dd[s]=r_mw-r_ot
    neur_diffs[col]=dd
common=set(neur_diffs["frp_cp_N400"])&set(neur_diffs["frp_occ_P2"])
comb=np.array([(neur_diffs["frp_cp_N400"][s]+neur_diffs["frp_occ_P2"][s])/2 for s in common])
tc,pc=stats.ttest_1samp(comb,0)
print(f"\ncombined NEURAL ISC Δ = {comb.mean():+.4f}  p={pc:.4f} (n={len(comb)}, frac_neg={np.mean(comb<0):.2f})")

json.dump({"gaze_real_diff":float(real.mean()),"gaze_p":float(p),"gaze_pseudo_diff":float(pseudo.mean()),
           "gaze_real_vs_pseudo_p":float(pp2),"template_sd_mw":float(np.mean(tvar_mw)),
           "template_sd_ontask":float(np.mean(tvar_ot)),"neural_combined_diff":float(comb.mean()),
           "neural_combined_p":float(pc)}, open(OUT/"isc_verify_report.json","w"),indent=2)
print("\nwrote isc_verify_report.json")
