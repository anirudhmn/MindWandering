#!/usr/bin/env python
"""
Overlap-corrected regression-ERP (rERP / deconvolution, unfold-style) for ZuCo.
Naive fixation-locked averages are contaminated by overlapping adjacent fixations (median next-fix
~206 ms), and the overlap timing itself covaries with surprisal (surprising words fixated longer),
so the N400 is unrecoverable without deconvolution (ROAMM's finding).

Per subject x task: model the sentence-continuous EEG as a sum of time-expanded responses to EVERY
fixation. Design columns = {intercept, z(zipf), z(surprisal), z(wlen)} x FIR lags (-100..+500 ms).
Since the design is shared across channels, solve (X'X+lambdaI)^-1 X'Y once for all 105 channels.
Output kernels beta[pred, lag, ch] per subject -> artifacts/rerp/rerp_<SUBJ>_<TASK>.npz
Run: python zuco/scripts/rerp.py NR   (or TSR / ALL)
"""
import sys, os, glob, time, warnings, numpy as np, pandas as pd, scipy.io as sio
from scipy import sparse
warnings.filterwarnings('ignore')
from pathlib import Path
ROOT=str(Path(__file__).resolve().parents[1]); A=f'{ROOT}/artifacts'
OUT=f'{A}/rerp'; os.makedirs(OUT,exist_ok=True)
FS=500; NCH=105
LAG0_MS,LAG1_MS=-100,500
L0,L1=int(LAG0_MS/1000*FS),int(LAG1_MS/1000*FS)      # -50 .. +250
LAGS=np.arange(L0,L1)                                 # 300 lags
NLAG=len(LAGS)
PREDS=['intercept','zipf','surprisal','wlen']
NP_=len(PREDS)

def as_segs(re):
    if np.size(re)==0: return []
    raw=list(re) if (hasattr(re,'dtype') and re.dtype==object) else [re]
    out=[]
    for x in raw:
        a=np.asarray(x,dtype=np.float64)
        if a.ndim==2 and a.shape[0]==NCH and a.shape[1]>0: out.append(a)
    return out

def find_onset(seg,rd,atol=1e-3):
    N=seg.shape[1]; T=rd.shape[1]
    d=np.abs(rd-seg[:,:1]).sum(0); cands=np.where(d<atol)[0]
    hits=[int(j) for j in cands if j+N<=T and np.allclose(rd[:,j:j+N],seg,atol=atol)]
    return hits[0] if len(hits)==1 else -1

def zscore(x):
    x=np.asarray(x,float); m=np.nanmean(x); s=np.nanstd(x)
    return (x-m)/(s+1e-12)

def collect_fixations(path,task,ling):
    """Return per-sentence: rawData + list of (onset, zipf, surprisal, wlen) for every matched fixation."""
    m=sio.loadmat(path,squeeze_me=True,struct_as_record=False)
    sd=np.atleast_1d(m['sentenceData'])
    L=ling[ling.task==task]
    lk={(r.sent_idx,r.word_idx):(r.zipf,r.surprisal,r.wlen) for r in L.itertuples()}
    out=[]
    for si,s in enumerate(sd):
        rd=np.asarray(s.rawData,dtype=np.float64)
        if rd.ndim!=2 or rd.shape[0]!=NCH: continue
        fix=[]
        for wi,w in enumerate(np.atleast_1d(s.word)):
            if not hasattr(w,'rawEEG'): continue
            segs=as_segs(w.rawEEG)
            if not segs: continue
            z,sp,wl=lk.get((si,wi),(np.nan,np.nan,np.nan))
            for seg in segs:
                on=find_onset(seg,rd)
                if on>=0: fix.append((on,z,sp,wl))
        if fix: out.append((rd,fix))
    return out

def build_and_solve(sentences, lam_frac=1e-2):
    """Accumulate normal equations across sentences; z-score predictors across all fixations first."""
    # gather predictor stats across subject
    allp=np.array([[f[1],f[2],f[3]] for _,fl in sentences for f in fl],float)  # zipf,surp,wlen
    mu=np.nanmean(allp,0); sd=np.nanstd(allp,0)+1e-12
    XtX=np.zeros((NP_*NLAG,NP_*NLAG)); XtY=np.zeros((NP_*NLAG,NCH))
    for rd,fl in sentences:
        T=rd.shape[1]
        rows=[]; cols=[]; vals=[]
        for (on,z,sp,wl) in fl:
            pv=[1.0, (z-mu[0])/sd[0] if np.isfinite(z) else 0.0,
                     (sp-mu[1])/sd[1] if np.isfinite(sp) else 0.0,
                     (wl-mu[2])/sd[2] if np.isfinite(wl) else 0.0]
            for li,lag in enumerate(LAGS):
                t=on+lag
                if 0<=t<T:
                    for pi in range(NP_):
                        rows.append(t); cols.append(pi*NLAG+li); vals.append(pv[pi])
        if not rows: continue
        X=sparse.csr_matrix((vals,(rows,cols)),shape=(T,NP_*NLAG))
        XtX+=(X.T@X).toarray()
        XtY+=X.T@rd.T                         # (P*L x T)@(T x NCH)
    lam=lam_frac*np.trace(XtX)/XtX.shape[0]
    beta=np.linalg.solve(XtX+lam*np.eye(XtX.shape[0]), XtY)   # (P*L x NCH)
    return beta.reshape(NP_,NLAG,NCH), mu, sd

def main():
    which=sys.argv[1] if len(sys.argv)>1 else 'ALL'
    tasks={'NR':'task2_NR_matlab','TSR':'task3_TSR_matlab'}
    if which!='ALL': tasks={which:tasks[which]}
    ling=pd.concat([pd.read_parquet(f'{A}/linguistic_NR.parquet'),
                    pd.read_parquet(f'{A}/linguistic_TSR.parquet')],ignore_index=True)
    for task,folder in tasks.items():
        for path in sorted(glob.glob(f'{ROOT}/{folder}/results*_{task}.mat')):
            subj=os.path.basename(path).replace('results','').replace(f'_{task}.mat','')
            t0=time.time()
            sents=collect_fixations(path,task,ling)
            nfix=sum(len(fl) for _,fl in sents)
            beta,mu,sd=build_and_solve(sents)
            np.savez(f'{OUT}/rerp_{subj}_{task}.npz',beta=beta.astype(np.float32),
                     preds=PREDS,lags=LAGS,mu=mu,sd=sd,nfix=nfix)
            print(f'{subj} {task}: {len(sents)} sents, {nfix} fixations, beta{beta.shape}, {time.time()-t0:.0f}s',flush=True)

if __name__=='__main__':
    main()
