#!/usr/bin/env python3
"""Peri-MW dynamics: align continuous signals to MW span onset and offset.

MW is a slow state with gradual transitions; static windows average that away. Here we
extract, time-locked to each MW-span onset and offset (+/- 8 s), the trajectories of
pupil, posterior alpha/beta power, blink rate, saccade rate, and instantaneous fixation
duration, z-scored within run, binned to 0.5 s. Averaged per subject then across subjects
this reveals whether the brain/eye state ramps into and out of mind-wandering.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert

OUT = Path("roamm/artifacts/coupling")
SF = 256.0
HALF = int(8*SF); BIN = int(0.5*SF); NBIN = 2*HALF//BIN
EEG = ['Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1','C1','C3','C5','T7','TP7','CP5','CP3','CP1','P1','P3','P5','P7','P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz','Fpz','Fp2','AF8','AF4','Afz','Fz','F2','F4','F6','F8','FT8','FC6','FC4','FC2','FCz','Cz','C2','C4','C6','T8','TP8','CP6','CP4','CP2','P2','P4','P6','P8','P10','PO8','PO4','O2']
CHI={c:i for i,c in enumerate(EEG)}
POST=['PO7','PO3','POz','PO4','PO8','O1','Oz','O2','Iz','P1','P3','Pz','P2','P4']
CENT=['C1','Cz','C2','FC1','FCz','FC2','C3','C4','FC4','CP2']
POSTi=[CHI[c] for c in POST]; CENTi=[CHI[c] for c in CENT]

def zrun(x):
    m,s=np.nanmean(x),np.nanstd(x); return (x-m)/(s+1e-9)

def binned(sig, center):
    seg=sig[center-HALF:center+HALF]
    return seg[:NBIN*BIN].reshape(NBIN,BIN).mean(axis=1)

def main():
    print("loading pickle...", flush=True)
    raw=pd.read_pickle("data/derivatives/features_df.pkl")
    n=len(raw)
    eeg=np.array(raw[EEG].to_numpy(dtype=np.float32),copy=True)
    time=pd.to_numeric(raw["time"],errors="coerce").to_numpy()
    is_mw=raw["is_mw"].fillna(False).to_numpy(bool)
    fp=raw["first_pass_reading"].fillna(False).to_numpy(bool)
    is_blink=raw["is_blink"].fillna(False).to_numpy(float)
    is_sacc=raw["is_sacc"].fillna(False).to_numpy(float)
    pupil=pd.to_numeric(raw["blink_interp_LPupil"],errors="coerce").to_numpy()
    fixdur=pd.to_numeric(raw["fix_L_duration"],errors="coerce").to_numpy()
    del raw
    boundaries=np.empty(n,bool); boundaries[0]=False; boundaries[1:]=time[1:]<time[:-1]
    subject=(np.cumsum(boundaries)//5).astype(int); run_id=np.cumsum(boundaries)
    uniq,starts=np.unique(run_id,return_index=True); starts=list(starts)+[n]
    bhp=butter(4,[8/(SF/2),12/(SF/2)],btype="band"); bbeta=butter(4,[15/(SF/2),25/(SF/2)],btype="band")

    onset_epochs={k:[] for k in ["pupil","alpha","beta","blink","sacc","fixdur"]}
    offset_epochs={k:[] for k in onset_epochs}
    pseudo_epochs={k:[] for k in onset_epochs}
    meta_on=[]; meta_off=[]; meta_pseudo=[]
    rng=np.random.default_rng(123)
    for k in range(len(uniq)):
        s,e=starts[k],starts[k+1]; L=e-s
        blk=eeg[s:e].astype(np.float64); blk-=blk.mean(axis=1,keepdims=True)
        alpha=zrun(np.log(np.abs(hilbert(filtfilt(*bhp,blk[:,POSTi],axis=0),axis=0))**2+1e-30).mean(axis=1))
        beta=zrun(np.log(np.abs(hilbert(filtfilt(*bbeta,blk[:,CENTi],axis=0),axis=0))**2+1e-30).mean(axis=1))
        sig={"pupil":zrun(pupil[s:e]),"alpha":alpha,"beta":beta,
             "blink":is_blink[s:e],"sacc":is_sacc[s:e],"fixdur":zrun(fixdur[s:e])}
        mw=is_mw[s:e]; fpr=fp[s:e]
        d=np.diff(mw.astype(int))
        onsets=np.flatnonzero(d==1)+1; offsets=np.flatnonzero(d==-1)+1
        subj=int(subject[s])
        for o in onsets:
            if o-HALF<0 or o+HALF>=L or not fpr[o]: continue
            for kk in sig: onset_epochs[kk].append(binned(sig[kk],o))
            meta_on.append(subj)
        for o in offsets:
            if o-HALF<0 or o+HALF>=L: continue
            for kk in sig: offset_epochs[kk].append(binned(sig[kk],o))
            meta_off.append(subj)
        # pseudo-onsets: on-task first-pass samples >=10 s from any MW, matched count
        mw_dil = np.convolve(mw.astype(int), np.ones(int(10*SF)*2+1,int), mode="same") > 0
        cand = np.flatnonzero(fpr & ~mw_dil & (np.arange(L) >= HALF) & (np.arange(L) < L-HALF))
        if len(cand) and len(onsets):
            for o in rng.choice(cand, size=min(len(onsets), len(cand)), replace=False):
                for kk in sig: pseudo_epochs[kk].append(binned(sig[kk],int(o)))
                meta_pseudo.append(subj)
        if (k+1)%40==0: print("  run",k+1,"onsets",len(meta_on),flush=True)

    t=(np.arange(NBIN)*BIN - HALF)/SF + 0.25
    np.savez(OUT/"dynamics.npz", t=t,
             meta_on=np.array(meta_on), meta_off=np.array(meta_off), meta_pseudo=np.array(meta_pseudo),
             **{f"on_{k}":np.array(v) for k,v in onset_epochs.items()},
             **{f"off_{k}":np.array(v) for k,v in offset_epochs.items()},
             **{f"pseudo_{k}":np.array(v) for k,v in pseudo_epochs.items()})
    print("saved dynamics.npz | onsets:",len(meta_on),"offsets:",len(meta_off),
          "pseudo:",len(meta_pseudo),"| t:",t[0],"..",t[-1])

if __name__=="__main__":
    main()
