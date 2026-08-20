#!/usr/bin/env python
"""GATE A1 — is the depth manipulation real? TSR (shallow relation-search) should be eye-shallower
than NR (deep comprehension), within subject. Compare per-subject mean FFD/GD/TRT/nFix (paired, 12
subjects present in both). Durations in samples->ms (/500*1000)."""
import glob, os, numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
A=str(Path(__file__).resolve().parents[1]/'artifacts')

def per_subject(task):
    rows={}
    for mp in sorted(glob.glob(f'{A}/frp/meta_*_{task}.parquet')):
        subj=os.path.basename(mp).split('_')[1]; m=pd.read_parquet(mp)
        rows[subj]=dict(FFD=m.FFD.mean()/500*1000, GD=m.GD.mean()/500*1000,
                        TRT=m.TRT.mean()/500*1000, nFix=m.nFix.mean(),
                        nwords_fixated=len(m)/m.sent_idx.nunique())
    return pd.DataFrame(rows).T

nr=per_subject('NR'); tsr=per_subject('TSR')
subs=sorted(set(nr.index)&set(tsr.index))
print(f'paired subjects: {len(subs)}\n')
print(f"{'metric':16s} {'NR':>8s} {'TSR':>8s} {'delta':>8s} {'t':>6s} {'p':>9s}  direction")
for c in ['FFD','GD','TRT','nFix','nwords_fixated']:
    a=nr.loc[subs,c].astype(float).values; b=tsr.loc[subs,c].astype(float).values
    t,p=stats.ttest_rel(b,a); d=(b-a).mean()
    print(f"{c:16s} {a.mean():8.2f} {b.mean():8.2f} {d:+8.2f} {t:6.2f} {p:9.2e}  {'TSR<NR' if d<0 else 'TSR>NR'}")
print('\nPASS if >=2 of {TRT,nFix,reading} lower in TSR (shallower).')
