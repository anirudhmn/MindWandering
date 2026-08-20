#!/usr/bin/env python
"""Mass-univariate group test of rERP kernels: for each (channel,lag) a one-sample t across subjects,
with a sign-flip cluster-permutation test (channel-adjacency x time). Finds WHERE/WHEN frequency and
surprisal reliably modulate the deconvolved FRP, without assuming ROAMM's exact ROI transfers."""
import glob, os, json, numpy as np
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
A=str(Path(__file__).resolve().parents[1]/'artifacts')
PREDS=['intercept','zipf','surprisal','wlen']; pi={p:i for i,p in enumerate(PREDS)}
CH=json.load(open(f'{A}/chanlocs_105.json'))
POS=np.array([[c['X'],c['Y'],c['Z']] for c in CH]); LAB=[c['label'] for c in CH]

def load(task):
    ks=[]
    for f in sorted(glob.glob(f'{A}/rerp/rerp_*_{task}.npz')):
        d=np.load(f,allow_pickle=True); b=d['beta']; lags=d['lags']
        b=b-b[:,lags<0,:].mean(1,keepdims=True); ks.append(b)
    return np.array(ks), lags

def adjacency(thresh=3.5):
    D=np.sqrt(((POS[:,None,:]-POS[None,:,:])**2).sum(-1))
    return D<thresh
ADJ=adjacency()

def cluster_stat(tmap, tcrit):
    """cluster mass on (ch,lag) grid using channel adjacency + lag contiguity, signed."""
    nch,nl=tmap.shape; best=0.0
    for sgn in (1,-1):
        sig=(sgn*tmap>tcrit)
        seen=np.zeros_like(sig,bool)
        for c in range(nch):
            for l in range(nl):
                if sig[c,l] and not seen[c,l]:
                    stack=[(c,l)]; seen[c,l]=True; mass=0.0
                    while stack:
                        cc,ll=stack.pop(); mass+=abs(tmap[cc,ll])
                        for c2 in np.where(ADJ[cc])[0]:
                            for l2 in (ll-1,ll+1):
                                if 0<=l2<nl and sig[c2,l2] and not seen[c2,l2]:
                                    seen[c2,l2]=True; stack.append((c2,l2))
                    best=max(best,mass)
    return best

def test(K, pred, nperm=500, seed=0):
    X=K[:,pi[pred]]                       # [S, L, ch]
    S=len(X); t=X.mean(0)/(X.std(0,ddof=1)/np.sqrt(S)+1e-12)   # [L,ch]
    tmap=t.T                              # [ch,L]
    tcrit=stats.t.ppf(0.975,S-1)
    obs=cluster_stat(tmap,tcrit)
    rng=np.random.default_rng(seed); null=[]
    for _ in range(nperm):
        fl=rng.choice([-1,1],S)[:,None,None]
        Xp=X*fl; tp=(Xp.mean(0)/(Xp.std(0,ddof=1)/np.sqrt(S)+1e-12)).T
        null.append(cluster_stat(tp,tcrit))
    null=np.array(null); p=(np.sum(null>=obs)+1)/(nperm+1)
    return tmap, obs, p, null

if __name__=='__main__':
    K,lags=load('NR'); t_ms=lags/500*1000
    print(f'NR {len(K)} subjects')
    for pred in ['zipf','surprisal']:
        tmap,obs,p,null=test(K,pred)
        # peak
        ci,li=np.unravel_index(np.argmax(np.abs(tmap)),tmap.shape)
        print(f'{pred}: max|t|={np.abs(tmap).max():.2f} at {LAB[ci]} {t_ms[li]:.0f}ms | cluster mass={obs:.0f} p_perm={p:.4f} (null95={np.percentile(null,95):.0f})')
        # report strongest early(100-300) and late(300-500) channel
        for a,b,tag in [(100,300,'early'),(300,500,'late')]:
            w=(t_ms>=a)&(t_ms<=b); sub=tmap[:,w]
            ci2=np.unravel_index(np.argmax(np.abs(sub)),sub.shape)[0]
            tvals=sub[ci2]; li2=np.argmax(np.abs(tvals))
            print(f'    {tag} peak: {LAB[ci2]} t={tvals[li2]:+.2f} @ {t_ms[w][li2]:.0f}ms')
