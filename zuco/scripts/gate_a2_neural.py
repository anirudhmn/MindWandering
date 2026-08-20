#!/usr/bin/env python
"""GATE A2 (neural) — does the robust FREQUENCY FRP (lexical coupling) attenuate under shallow reading?
Compare per-subject deconvolved zipf-kernel amplitude in OCC 150-290 ms, NR vs TSR (12 paired subjects).
Also surprisal->CP (SNR-floored, reported for completeness). Corroborates the behavioral lexical-shedding."""
import glob, os, json, numpy as np
from scipy import stats
from pathlib import Path
A=str(Path(__file__).resolve().parents[1]/'artifacts')
import gate_a0 as G; occ,cp=G.rois()
PREDS=['intercept','zipf','surprisal','wlen']; pi={p:i for i,p in enumerate(PREDS)}

def load(task):
    K={}
    for f in sorted(glob.glob(f'{A}/rerp/rerp_*_{task}.npz')):
        d=np.load(f,allow_pickle=True); b=d['beta']; lags=d['lags']
        b=b-b[:,lags<0,:].mean(1,keepdims=True); K[os.path.basename(f).split('_')[1]]=(b,lags)
    return K

def winamp(b,lags,pred,roi,a,bb):
    t=lags/500*1000; w=(t>=a)&(t<=bb); return b[pi[pred]][w][:,roi].mean()

NR=load('NR'); TSR=load('TSR'); subs=sorted(set(NR)&set(TSR))
print(f'{len(subs)} paired subjects\n')
for pred,roi,a,b,name in [('zipf',occ,150,290,'Frequency FRP -> OCC (lexical)'),
                          ('surprisal',cp,300,450,'Surprisal -> CP (semantic, SNR-floored)')]:
    n=np.array([winamp(*NR[s],pred,roi,a,b) for s in subs])
    t_=np.array([winamp(*TSR[s],pred,roi,a,b) for s in subs])
    tt,p=stats.ttest_rel(t_,n)
    # attenuation toward zero
    att=np.abs(n)-np.abs(t_); ta,pa=stats.ttest_1samp(att,0)
    print(f'{name}:')
    print(f'  NR={n.mean():+.4f}  TSR={t_.mean():+.4f}  Δ(TSR-NR)={ (t_-n).mean():+.4f} t={tt:.2f} p={p:.3f}')
    print(f'  |attenuation| (NR-TSR toward 0)={att.mean():+.4f} t={ta:.2f} p={pa:.3f} ({np.mean(att>0)*100:.0f}% subj attenuated)\n')
