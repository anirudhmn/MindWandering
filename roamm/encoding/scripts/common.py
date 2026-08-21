#!/usr/bin/env python3
"""Shared machinery for the pooled deconvolutional encoding model.

Fast overlap-corrected deconvolution. The time-expanded normal equations have Toeplitz block
structure: writing d for the difference in sample onset between two fixations,

    XtX[(p,l),(q,m)] = G_pq(l - m),   G_pq(d) = sum over event pairs at offset d of x_ep x_e'q

so XtX is assembled from the few neighbouring events within a kernel length of each fixation
rather than from a sparse matrix product over the whole recording, and
XtY[(p,l),c] = sum_e x_ep Y[onset_e + l, c] is a gather. This is algebraically the same solve as
`roamm/build/build_rerp.py` performs; `00_validate_solver.py` checks that it reproduces the
frozen kernels. The speed is what makes a pooled fit over 220 held-out folds affordable.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
COUP = ROOT / "roamm/artifacts/coupling"
IT = ROOT / "roamm/encoding"
ART = IT / "artifacts"
RES = IT / "results"
for d in (ART, RES):
    d.mkdir(parents=True, exist_ok=True)

SF = 256.0
EEG_CH = ['Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1','C1','C3','C5','T7','TP7',
          'CP5','CP3','CP1','P1','P3','P5','P7','P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz',
          'Fpz','Fp2','AF8','AF4','Afz','Fz','F2','F4','F6','F8','FT8','FC6','FC4','FC2','FCz','Cz',
          'C2','C4','C6','T8','TP8','CP6','CP4','CP2','P2','P4','P6','P8','P10','PO8','PO4','O2']
OCC = ['PO7','PO8','PO3','PO4','O1','O2','Oz','POz','P7','P8','P9','P10','Iz']
CP = ['Cz','CPz','Pz','CP1','CP2','C1','C2','P1','P2']
OCC_I = [EEG_CH.index(c) for c in OCC]
CP_I = [EEG_CH.index(c) for c in CP]

TEXT_BASE = ["zipf", "length", "s_local", "gain_long", "gain_shuf"]


def lag_grid(t0_ms=-100.0, t1_ms=500.0):
    lags = np.arange(int(round(t0_ms / 1000 * SF)), int(round(t1_ms / 1000 * SF)) + 1)
    return lags, lags / SF * 1000.0


def build_XtX(onsets, X, lags):
    """Toeplitz-block normal matrix from event-pair offsets. `onsets` must be sorted."""
    NP, NL = X.shape[1], len(lags)
    Wn = NL - 1
    G = np.zeros((NP, NP, 2 * NL - 1))
    lo = np.searchsorted(onsets, onsets - Wn, side="left")
    hi = np.searchsorted(onsets, onsets + Wn, side="right")
    ii = np.repeat(np.arange(len(onsets)), hi - lo)
    jj = (np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)])
          if len(onsets) else np.array([], int))
    idx = (onsets[jj] - onsets[ii] + Wn).astype(np.int64)
    Xi, Xj = X[ii], X[jj]
    for p in range(NP):
        xp = Xi[:, p]
        for q in range(NP):
            G[p, q] = np.bincount(idx, weights=xp * Xj[:, q], minlength=2 * NL - 1)
    L = np.arange(NL)
    T = (L[:, None] - L[None, :]) + Wn
    XtX = np.empty((NP * NL, NP * NL))
    for p in range(NP):
        for q in range(NP):
            XtX[p * NL:(p + 1) * NL, q * NL:(q + 1) * NL] = G[p, q][T]
    return XtX


def build_XtY(eeg, run_start, onsets, X, lags, chunk=4096):
    NP, NL, NCH = X.shape[1], len(lags), eeg.shape[1]
    XtY = np.zeros((NP, NL, NCH))
    for s in range(0, len(onsets), chunk):
        o = onsets[s:s + chunk] + run_start
        rows = (o[:, None] + lags[None, :]).ravel()
        E = np.asarray(eeg[rows], dtype=np.float64).reshape(len(o), NL, NCH)
        XtY += np.einsum("ep,elc->plc", X[s:s + chunk], E, optimize=True)
    return XtY.reshape(NP * NL, NCH)


def fit_ridge(XtX, XtY, NP, NL, lam_scale=1e-2):
    """Ridge solve with the regularisation of the frozen rERP pipeline."""
    lam = lam_scale * np.trace(XtX) / (NP * NL)
    beta = np.linalg.solve(XtX + lam * np.eye(NP * NL), XtY)
    return beta.reshape(NP, NL, -1), lam


def predict_run(runlen, onsets, X, beta, lags):
    """Continuous prediction over one run, summing the overlapping kernels."""
    NCH = beta.shape[2]
    pred = np.zeros((runlen, NCH), np.float64)
    for li in range(len(lags)):
        pred[onsets + lags[li]] += X @ beta[:, li, :]
    return pred


def boot_ci(v, n=10000, seed=66):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    idx = rng.integers(0, len(v), size=(n, len(v)))
    bm = v[idx].mean(axis=1)
    t, p = stats.ttest_1samp(v, 0)
    return dict(mean=float(v.mean()),
                ci=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                t=float(t), p=float(p), n=int(len(v)), n_pos=int((v > 0).sum()),
                sd=float(bm.std()))


def open_eeg():
    """Preprocessed continuous recording, written by 01_cache_eeg.py (not redistributed)."""
    meta = json.loads((ART / "eeg_cache_meta.json").read_text())
    n_samp, n_ch = meta["shape"]
    eeg = np.memmap(ART / "eeg_pp.f32", dtype=np.float32, mode="r", shape=(n_samp, n_ch))
    return eeg, np.load(ART / "run_bounds.npy")
