#!/usr/bin/env python3
"""Overlap-corrected regression ERP (rERP / linear deconvolution), per subject.

Naive FRP averaging is contaminated by the next fixation (~230 ms away), which lands
squarely on the N400 window. We instead model the continuous EEG as a sum of overlapping
fixation responses via a time-expanded design matrix (the 'unfold' approach) and solve
for isolated response kernels.

Model (per subject, per channel): y(t) = sum over predictors p of (X_p * beta_p)(t)
with a lagged basis spanning [-100, +500] ms around every fixation onset. Predictors
(z-scored within subject unless noted):
  0 intercept   1 zipf   2 length   3 surprisal   4 logdur
  5 mw (0/1)    6 zipf:mw   7 surprisal:mw
Kernels of interest: beta_surprisal(tau) = overlap-corrected surprisal rERP (the N400),
beta_surprisal:mw(tau) = overlap-corrected change during MW (the decoupling estimate).

Ridge-regularized normal equations; 64 channels share the design. Output:
  rerp_betas.npy  [n_subj, n_pred, n_lag, 64]
  rerp_meta.json  {lags_ms, pred_names, channels, subjects}
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.signal import butter, filtfilt

OUT = Path("roamm/artifacts/coupling")
SF = 256.0
EEG = ['Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1','C1','C3','C5','T7','TP7','CP5','CP3','CP1','P1','P3','P5','P7','P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz','Fpz','Fp2','AF8','AF4','Afz','Fz','F2','F4','F6','F8','FT8','FC6','FC4','FC2','FCz','Cz','C2','C4','C6','T8','TP8','CP6','CP4','CP2','P2','P4','P6','P8','P10','PO8','PO4','O2']
PRED = ["intercept","zipf","length","surprisal","logdur","mw","zipf:mw","surprisal:mw"]
LAGS = np.arange(int(round(-0.100*SF)), int(round(0.500*SF)) + 1)   # -26..128
NL = len(LAGS)
NP = len(PRED)

def main():
    print("loading pickle...", flush=True)
    raw = pd.read_pickle("data/derivatives/features_df.pkl")
    n = len(raw)
    eeg = np.array(raw[EEG].to_numpy(dtype=np.float32), copy=True)
    time = pd.to_numeric(raw["time"], errors="coerce").to_numpy()
    boundaries = np.empty(n, bool); boundaries[0] = False
    boundaries[1:] = time[1:] < time[:-1]
    subject = (np.cumsum(boundaries) // 5).astype(int)
    run_id = np.cumsum(boundaries)
    del raw

    # preprocess EEG per run (avg ref + 0.1-30 Hz), same as FRP pipeline
    bhp, ahp = butter(2, 0.1/(SF/2), btype="high")
    blp, alp = butter(4, 30.0/(SF/2), btype="low")
    uniq, starts = np.unique(run_id, return_index=True)
    starts = list(starts) + [n]
    print("preprocessing", len(uniq), "runs...", flush=True)
    for k in range(len(uniq)):
        s, e = starts[k], starts[k+1]
        blk = eeg[s:e].astype(np.float64, copy=True)
        blk -= blk.mean(axis=1, keepdims=True)
        blk = filtfilt(bhp, ahp, blk, axis=0)
        blk = filtfilt(blp, alp, blk, axis=0)
        eeg[s:e] = (blk * 1e6).astype(np.float32)   # to uV
        if (k+1) % 60 == 0: print("  filtered", k+1, flush=True)

    # per-fixation predictors
    fix = pd.read_parquet(OUT/"fixations.parquet")
    wf = pd.read_parquet(OUT/"word_features.parquet")[["word_key","length","zipf","surprisal","clean"]]
    fix = fix.merge(wf, on="word_key", how="left")
    ok = (fix["clean"].str.len()>=1)&(fix["zipf"]>0)&fix["surprisal"].notna()&fix["fix_dur"].between(50,1000)
    fix = fix[ok].copy()
    fix["logdur"] = np.log(fix["fix_dur"].to_numpy())
    for c in ["zipf","length","surprisal","logdur"]:
        fix[c+"_z"] = fix.groupby("subject")[c].transform(lambda s:(s-s.mean())/(s.std()+1e-9))
    fix["mw"] = fix["is_mw"].astype(float)

    # run sample ranges
    run_start = {int(u): int(starts[i]) for i, u in enumerate(uniq)}
    run_end = {int(u): int(starts[i+1]) for i, u in enumerate(uniq)}
    # map each fixation to its run_id via onset abs idx
    fix["rid"] = run_id[fix["onset_abs_idx"].to_numpy()]

    subj_ids = sorted(fix["subject"].unique())
    betas = np.zeros((len(subj_ids), NP, NL, 64), dtype=np.float32)
    lag0 = -LAGS[0]  # index offset so lag 0 aligns
    for si, subj in enumerate(subj_ids):
        fs = fix[fix["subject"] == subj]
        XtX = np.zeros((NP*NL, NP*NL)); XtY = np.zeros((NP*NL, 64))
        ntot = 0
        for rid, fr in fs.groupby("rid"):
            rs, re = run_start[int(rid)], run_end[int(rid)]
            runlen = re - rs
            Y = eeg[rs:re].astype(np.float64)             # [runlen,64]
            rel = fr["onset_abs_idx"].to_numpy() - rs      # onset within run
            Pv = np.column_stack([
                np.ones(len(fr)), fr["zipf_z"], fr["length_z"], fr["surprisal_z"],
                fr["logdur_z"], fr["mw"], fr["zipf_z"]*fr["mw"], fr["surprisal_z"]*fr["mw"],
            ]).astype(np.float64)                          # [k, NP]
            # time-expanded sparse design [runlen, NP*NL]
            rows = (rel[:, None] + LAGS[None, :])          # [k, NL]
            valid = (rows >= 0) & (rows < runlen)
            blocks = []
            for p in range(NP):
                rr = rows[valid]
                cc = (np.broadcast_to(np.arange(NL), rows.shape))[valid] + p*NL
                dd = np.broadcast_to(Pv[:, p][:, None], rows.shape)[valid]
                blocks.append((rr, cc, dd))
            rr = np.concatenate([b[0] for b in blocks])
            cc = np.concatenate([b[1] for b in blocks])
            dd = np.concatenate([b[2] for b in blocks])
            X = sparse.csr_matrix((dd, (rr, cc)), shape=(runlen, NP*NL))
            XtX += (X.T @ X).toarray()
            XtY += X.T @ Y
            ntot += len(fr)
        lam = 1e-2 * np.trace(XtX) / (NP*NL)
        beta = np.linalg.solve(XtX + lam*np.eye(NP*NL), XtY)   # [NP*NL, 64]
        betas[si] = beta.reshape(NP, NL, 64)
        print(f"  subj {subj:2d}  fixations={ntot:5d}  done ({si+1}/{len(subj_ids)})", flush=True)

    np.save(OUT/"rerp_betas.npy", betas)
    (OUT/"rerp_meta.json").write_text(json.dumps({
        "lags_ms": (LAGS/SF*1000).round(2).tolist(), "pred_names": PRED,
        "channels": EEG, "subjects": [int(s) for s in subj_ids]}, indent=2))
    print("saved rerp_betas.npy", betas.shape, flush=True)

if __name__ == "__main__":
    main()
