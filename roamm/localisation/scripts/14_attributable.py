"""How much comprehension failure is ATTRIBUTABLE to mind-wandering on the answer span?
AUC is a rank metric and is insensitive to a large effect on a minority of trials.
The right statistic is an adjusted counterfactual / attributable fraction."""
import numpy as np, pandas as pd
import pathlib
ROOT=pathlib.Path(__file__).resolve().parents[3]
SP=str(ROOT/'roamm/localisation/results')+'/'
rng=np.random.default_rng(20260819)
m=pd.read_parquet(SP+'merged.parquet').reset_index(drop=True)
CTRL=['mw_frac_elsewhere','coverage','evidence_cov','evidence_dwell_per_word']

def cf(df):
    """adjusted counterfactual: set MW-on-span to zero, holding everything else fixed"""
    D=pd.concat([df[['mw_frac_evidence']+CTRL].astype(float),
                 pd.get_dummies(df.sub_id,drop_first=True).astype(float),
                 pd.get_dummies(df['item'],drop_first=True).astype(float)],axis=1)
    D.insert(0,'const',1.0); A=D.values; y=df.correct.values.astype(float)
    b=np.linalg.lstsq(A,y,rcond=None)[0]
    obs=y.mean()
    A0=A.copy(); A0[:,1]=0.0                     # zero out MW on the answer span
    return obs, float((A0@b).mean()), float(b[1])

obs,cf_acc,beta=cf(m)
subs=m.sub_id.unique(); idx={s:np.where(m.sub_id.values==s)[0] for s in subs}
B=[]
for _ in range(3000):
    pick=rng.choice(subs,len(subs),replace=True)
    rows=np.concatenate([idx[s] for s in pick]); b_=m.iloc[rows].copy()
    b_['sub_id']=np.concatenate([[f"{s}_{k}"]*len(idx[s]) for k,s in enumerate(pick)])
    try:
        o,c,_=cf(b_); B.append((c-o, (c-o)/(1-o)))
    except Exception: pass
B=np.array(B); q=lambda v,a: float(np.quantile(v,a))
gain=cf_acc-obs; err=1-obs
print("=== attributable to mind-wandering on the answer span ===")
print(f"  observed accuracy                         {obs:.4f}")
print(f"  counterfactual (no MW on the answer span) {cf_acc:.4f}")
print(f"  accuracy points recovered                 {gain*100:+.2f} pts "
      f"[{q(B[:,0],.025)*100:+.2f},{q(B[:,0],.975)*100:+.2f}]")
print(f"  share of ALL errors eliminated            {gain/err*100:.1f}% "
      f"[{q(B[:,1],.025)*100:.1f}%,{q(B[:,1],.975)*100:.1f}%]")
print(f"  (per-SD beta = {beta*m.mw_frac_evidence.std():+.4f}; raw beta {beta:+.4f})")

# exposure and conditional cost
exp=(m.mw_frac_evidence>0).mean()
print(f"\n  exposure: {exp*100:.1f}% of trials have a lapse on the answer span")
print(f"  raw accuracy: no lapse {m[m.mw_frac_evidence==0].correct.mean():.3f} | "
      f"lapse {m[m.mw_frac_evidence>0].correct.mean():.3f}")

# reading amount comparison, same machinery
def cf_read(df):
    D=pd.concat([df[['evidence_cov','mw_frac_evidence','mw_frac_elsewhere','coverage',
                     'evidence_dwell_per_word']].astype(float),
                 pd.get_dummies(df.sub_id,drop_first=True).astype(float),
                 pd.get_dummies(df['item'],drop_first=True).astype(float)],axis=1)
    D.insert(0,'const',1.0); A=D.values; y=df.correct.values.astype(float)
    b=np.linalg.lstsq(A,y,rcond=None)[0]
    A1=A.copy(); A1[:,1]=1.0                      # full coverage of the answer span
    return y.mean(), float((A1@b).mean())
o2,c2=cf_read(m)
print(f"\n  for comparison — counterfactual 'read the whole answer span': "
      f"{(c2-o2)*100:+.2f} pts ({(c2-o2)/(1-o2)*100:.1f}% of errors)")
