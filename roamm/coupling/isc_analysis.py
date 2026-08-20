#!/usr/bin/env python3
"""Inter-subject correlation (ISC): does a mind-wandering reader's response decorrelate
from the group's shared, stimulus-driven response to the same text?

The shared timeline in reading is the TEXT (self-paced), so ISC is computed per WORD.
For each (story, word position) we build a leave-one-subject-out GROUP TEMPLATE — the
mean response of OTHER subjects to that word (on-task only). Then, per subject, we
correlate the subject's per-word response with that template SEPARATELY for their MW and
on-task words. Lower ISC during MW = the reader decouples from the shared reading response
(a form of decoupling not captured by word-property coupling).

Gaze ISC: response = log first-pass fixation duration (high SNR).
Neural ISC: response = centroparietal N400 amplitude and occipitotemporal amplitude.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path("roamm/artifacts/coupling")
RNG = np.random.default_rng(303)

def loo_template(df, val, key=("story","pos")):
    """leave-one-subject-out group mean of `val` over ON-TASK rows, per word."""
    ot = df[df["is_mw"]==0]
    grp = ot.groupby(list(key))[val].agg(["sum","count"])
    # map each row to (sum,count) then remove own contribution if that row is on-task
    m = df.merge(grp, left_on=list(key), right_index=True, how="left")
    loo = m["sum"].to_numpy().astype(float); cnt = m["count"].to_numpy().astype(float)
    own = ((df["is_mw"].to_numpy()==0)).astype(float)*df[val].to_numpy()
    loo_mean = (loo - own) / np.clip(cnt - (df["is_mw"].to_numpy()==0), 1, None)
    return loo_mean

def isc_by_condition(df, val):
    df = df.dropna(subset=[val]).copy()
    df["template"] = loo_template(df, val)
    df = df.dropna(subset=["template"])
    recs=[]
    for s,g in df.groupby("subject"):
        for cond,name in ((0,"ontask"),(1,"mw")):
            gc=g[g["is_mw"]==cond]
            if len(gc)<80 or gc["template"].std()<1e-9 or gc[val].std()<1e-9: continue
            r=np.corrcoef(gc[val],gc["template"])[0,1]
            recs.append({"subject":s,"cond":name,"isc":r,"n":len(gc)})
    R=pd.DataFrame(recs)
    piv=R.pivot_table(index="subject",columns="cond",values="isc")
    piv=piv.dropna()
    d=(piv["mw"]-piv["ontask"]).to_numpy()
    t,p=stats.ttest_1samp(d,0)
    boot=np.array([RNG.choice(d,len(d)).mean() for _ in range(10000)])
    return {"isc_ontask":float(piv["ontask"].mean()),"isc_mw":float(piv["mw"].mean()),
            "diff":float(d.mean()),"ci":[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))],
            "t":float(t),"p":float(p),"n":int(len(d)),"frac_neg":float((d<0).mean())}

# ---- gaze ISC ----
fx=pd.read_parquet(OUT/"reading_fixations.parquet")
fx=fx[fx["fix_dur"].between(50,1000)].copy()
fx["logdur"]=np.log(fx["fix_dur"].to_numpy())
# one row per (subject, word) — average refixations
gaze=fx.groupby(["subject","story","pos"]).agg(logdur=("logdur","mean"),is_mw=("is_mw","max")).reset_index()
res={}
res["gaze_fixdur"]=isc_by_condition(gaze,"logdur")
print("=== GAZE ISC (log fixation duration vs leave-one-out group template) ===")
r=res["gaze_fixdur"]
print(f"  ISC on-task={r['isc_ontask']:.3f}  MW={r['isc_mw']:.3f}  Δ={r['diff']:+.4f} "
      f"CI[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] t={r['t']:+.2f} p={r['p']:.4f} frac_neg={r['frac_neg']:.2f} (n={r['n']})")

# ---- neural ISC ----
frp=pd.read_parquet(OUT/"fixations_frp.parquet")[["onset_abs_idx","subject","is_mw","frp_cp_N400","frp_occ_P2","frp_p2p","fix_dur"]]
pos=fx[["onset_abs_idx","story","pos"]].drop_duplicates("onset_abs_idx")
frp=frp.merge(pos,on="onset_abs_idx",how="inner")
frp=frp[(frp["frp_p2p"]*1e6<=150)&frp["fix_dur"].between(50,1000)]
for col,lab in [("frp_cp_N400","neural_N400"),("frp_occ_P2","neural_occP2")]:
    frp[col]=frp[col]*1e6
    w=frp.groupby(["subject","story","pos"]).agg(**{col:(col,"mean"),"is_mw":("is_mw","max")}).reset_index()
    res[lab]=isc_by_condition(w,col)
    r=res[lab]
    print(f"=== NEURAL ISC ({lab}) ===")
    print(f"  ISC on-task={r['isc_ontask']:.3f}  MW={r['isc_mw']:.3f}  Δ={r['diff']:+.4f} "
          f"CI[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] t={r['t']:+.2f} p={r['p']:.4f} (n={r['n']})")

(OUT/"isc_report.json").write_text(json.dumps(res,indent=2)+"\n")
print("\nwrote isc_report.json")
