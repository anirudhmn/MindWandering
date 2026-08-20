"""A NON-CIRCULAR attention index: how strongly is the reader's behaviour being driven by the
text right now? Never trained on MW labels. Then ask how it relates to mind-wandering."""
import numpy as np, pandas as pd
import pathlib
ROOT=pathlib.Path(__file__).resolve().parents[3]
SP=str(ROOT/'roamm/attention_index/results')+'/'  # the paper figure reads this
rng=np.random.default_rng(5)
fx=pd.read_parquet(str(ROOT)+'/roamm/artifacts/coupling/all_fixations.parquet')
wf=pd.read_parquet(str(ROOT)+'/roamm/artifacts/coupling/word_features.parquet')
d=fx.dropna(subset=['word_key']).merge(wf[['word_key','length','zipf','surprisal']],on='word_key',how='inner')
d=d[(d.fix_dur>=50)&(d.fix_dur<=1000)].dropna(subset=['length','zipf','surprisal'])
d=d.sort_values(['subject','run','fix_order_all']).reset_index(drop=True)
d['logdur']=np.log(d.fix_dur)
print(f"fixations with word features: {len(d)} | MW {d.is_mw.mean():.4f}")

# 1) population text->behaviour model, fit on ON-TASK fixations only, reader FE
tr=d[~d.is_mw]
X=np.column_stack([np.ones(len(tr)),tr.zipf,tr.length,tr.surprisal])
S=pd.get_dummies(tr.subject,drop_first=True).astype(float).values
b=np.linalg.lstsq(np.column_stack([X,S]),tr.logdur.values,rcond=None)[0]
coef=b[:4]
print(f"text model (on-task): zipf {coef[1]:+.4f}  length {coef[2]:+.4f}  surprisal {coef[3]:+.4f}")
d['pred']=coef[0]+coef[1]*d.zipf+coef[2]*d.length+coef[3]*d.surprisal

# 2) rolling LOCAL COUPLING: slope of observed on predicted within a rolling window of
#    W fixations, both centred inside the window -> removes any additive level shift
W=25
def local_slope(g):
    p=g.pred; o=g.logdur
    mp=p.rolling(W,center=True,min_periods=15).mean(); mo=o.rolling(W,center=True,min_periods=15).mean()
    cov=((p-mp)*(o-mo)).rolling(W,center=True,min_periods=15).mean()
    var=((p-mp)**2).rolling(W,center=True,min_periods=15).mean()
    return (cov/var.where(var>1e-4)).rename('coupling')
d['coupling']=d.groupby(['subject','run'],group_keys=False).apply(local_slope,include_groups=False)
c=d.dropna(subset=['coupling']).copy()
c['coupling']=c.coupling.clip(*np.percentile(c.coupling,[1,99]))
# 0-100 attention score = within-reader percentile of local text-coupling
c['attention']=c.groupby('subject').coupling.rank(pct=True)*100
print(f"usable fixations: {len(c)}")

print("\n=== does the non-circular attention index track mind-wandering? ===")
g=c.groupby(['subject','is_mw']).attention.mean().unstack()
g=g.dropna(); dd=(g[True]-g[False]).values
bs=np.array([np.mean(rng.choice(dd,len(dd),replace=True)) for _ in range(5000)])
q=lambda v,a: float(np.quantile(v,a))
print(f"  attention score, on-task {g[False].mean():.2f} vs MW {g[True].mean():.2f}")
print(f"  paired diff = {dd.mean():+.3f} points [{q(bs,.025):+.3f},{q(bs,.975):+.3f}] "
      f"p={2*min((bs>0).mean(),(bs<0).mean()):.4f}   (n={len(dd)} readers)")
print(f"  readers with LOWER attention during MW: {(dd<0).sum()}/{len(dd)}")

from sklearn.metrics import roc_auc_score
print(f"  AUC of attention index for MW = {roc_auc_score(c.is_mw, -c.attention):.4f}")

# contrast: a level-based index (raw fixation duration) which MW *does* shift
c['slow']=c.groupby('subject').logdur.rank(pct=True)*100
g2=c.groupby(['subject','is_mw']).slow.mean().unstack().dropna()
d2=(g2[True]-g2[False]).values
b2=np.array([np.mean(rng.choice(d2,len(d2),replace=True)) for _ in range(5000)])
print(f"\n  CONTRAST — level-based index (fixation duration percentile):")
print(f"  on-task {g2[False].mean():.2f} vs MW {g2[True].mean():.2f}  diff={d2.mean():+.3f} "
      f"[{q(b2,.025):+.3f},{q(b2,.975):+.3f}] p={2*min((b2>0).mean(),(b2<0).mean()):.4f}")
print(f"  AUC of duration index for MW = {roc_auc_score(c.is_mw, c.slow):.4f}")
c[['subject','run','story','fix_order_all','tStart','is_mw','attention','slow','coupling']].to_parquet(SP+'attention_index.parquet')
