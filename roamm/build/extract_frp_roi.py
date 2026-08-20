#!/usr/bin/env python3
"""Fixation-related potentials (FRPs): reduce each first-pass fixation to scalar
EEG amplitudes in a-priori spatiotemporal ROIs, so we can test whether the neural
response to word properties attenuates during mind-wandering — in exact parallel to
the behavioral coupling analysis.

Preprocessing (per run): average reference, 0.1-30 Hz zero-phase Butterworth.
Epoch [-100, +400] ms around fixation onset, baseline [-100, 0] ms.
Overlap caveat: mean saccade interval ~230 ms, so windows beyond ~230 ms overlap the
next fixation; the early occipitotemporal window (primary) is overlap-clean, later
windows are reported with that caveat and revisited by deconvolution if warranted.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

OUT = Path("roamm/artifacts/coupling")
SF = 256.0
EEG_COLS = ['Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1','C1','C3','C5','T7','TP7','CP5','CP3','CP1','P1','P3','P5','P7','P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz','Fpz','Fp2','AF8','AF4','Afz','Fz','F2','F4','F6','F8','FT8','FC6','FC4','FC2','FCz','Cz','C2','C4','C6','T8','TP8','CP6','CP4','CP2','P2','P4','P6','P8','P10','PO8','PO4','O2']
CH = {c: i for i, c in enumerate(EEG_COLS)}

OCC = ['PO7','PO8','PO3','PO4','O1','O2','Oz','POz','P7','P8','P9','P10','Iz']
CP  = ['Cz','CPz','Pz','CP1','CP2','C1','C2','P1','P2','Pz']
FRONT = ['Fz','FCz','Cz','F1','F2','FC1','FC2','AFz','Afz']
ROIS = {
    "occ_P1":  (OCC, 0.080, 0.130),
    "occ_N1":  (OCC, 0.150, 0.220),   # primary: overlap-clean lexical/frequency window
    "occ_P2":  (OCC, 0.220, 0.290),
    "cp_mid":  (CP,  0.220, 0.300),
    "cp_N400": (CP,  0.300, 0.420),   # overlap-contaminated; caveat
    "front_late": (FRONT, 0.300, 0.420),
}

def roi_idx(names):
    return np.array(sorted({CH[c] for c in names if c in CH}), dtype=int)

def main():
    print("loading pickle...", flush=True)
    raw = pd.read_pickle("data/derivatives/features_df.pkl")
    n = len(raw)
    eeg = np.array(raw[EEG_COLS].to_numpy(dtype=np.float32), copy=True)  # writable [n, 64]
    mid = eeg[n // 3: n // 3 + 500000]
    print("EEG scale (mid): median|x|=%.4g  p99|x|=%.4g  frac_zero=%.3f" % (
        np.median(np.abs(mid)), np.percentile(np.abs(mid), 99),
        float((mid == 0).mean())), flush=True)

    time = pd.to_numeric(raw["time"], errors="coerce").to_numpy()
    run_num = pd.to_numeric(raw["run_num"], errors="coerce").to_numpy()
    boundaries = np.empty(n, dtype=bool); boundaries[0] = False
    boundaries[1:] = time[1:] < time[:-1]
    run_id = np.cumsum(boundaries)  # unique per subject-run
    del raw

    fix = pd.read_parquet(OUT / "fixations.parquet")
    onset = fix["onset_abs_idx"].to_numpy()

    # epoch sample offsets
    pre = int(round(0.100 * SF)); post = int(round(0.400 * SF))
    t = (np.arange(-pre, post + 1) / SF)
    base_mask = (t >= -0.100) & (t <= 0.0)
    roi_windows = {name: (roi_idx(ch), (t >= a) & (t <= b)) for name, (ch, a, b) in ROIS.items()}

    # --- per-run preprocessing: average reference + 0.1-30 Hz bandpass ---
    bhp, ahp = butter(2, 0.1 / (SF / 2), btype="high")
    blp, alp = butter(4, 30.0 / (SF / 2), btype="low")
    print("preprocessing %d runs..." % run_id.max(), flush=True)
    order = np.argsort(run_id, kind="stable")
    # process contiguous run blocks in place
    uniq, starts = np.unique(run_id, return_index=True)
    starts = list(starts) + [n]
    for k in range(len(uniq)):
        s, e = starts[k], starts[k + 1]
        block = eeg[s:e].astype(np.float64, copy=True)
        block -= block.mean(axis=1, keepdims=True)          # average reference
        block = filtfilt(bhp, ahp, block, axis=0)
        block = filtfilt(blp, alp, block, axis=0)
        eeg[s:e] = block.astype(np.float32)
        if (k + 1) % 50 == 0:
            print("  filtered run", k + 1, flush=True)

    # --- epoch + ROI reduction ---
    nfix = len(onset)
    valid = (onset - pre >= 0) & (onset + post < n) & \
            (run_id[np.clip(onset - pre, 0, n - 1)] == run_id[np.clip(onset + post, 0, n - 1)])
    print("epochable fixations:", int(valid.sum()), "/", nfix, flush=True)

    roi_vals = {name: np.full(nfix, np.nan, np.float32) for name in ROIS}
    peak2peak = np.full(nfix, np.nan, np.float32)  # for artifact rejection
    idxs = np.flatnonzero(valid)
    CHUNK = 20000
    for c0 in range(0, len(idxs), CHUNK):
        ci = idxs[c0:c0 + CHUNK]
        # gather epochs [m, T, 64]
        rel = np.arange(-pre, post + 1)
        rows = onset[ci][:, None] + rel[None, :]         # [m, T]
        ep = eeg[rows]                                   # [m, T, 64]
        ep = ep - ep[:, base_mask, :].mean(axis=1, keepdims=True)  # baseline correct
        peak2peak[ci] = (ep.max(axis=1) - ep.min(axis=1)).max(axis=1)
        for name, (chs, wmask) in roi_windows.items():
            roi_vals[name][ci] = ep[:, wmask, :][:, :, chs].mean(axis=(1, 2))
        if (c0 // CHUNK) % 5 == 0:
            print("  epoched", c0 + len(ci), "/", len(idxs), flush=True)

    for name in ROIS:
        fix["frp_" + name] = roi_vals[name]
    fix["frp_p2p"] = peak2peak
    fix["frp_valid"] = valid
    fix.to_parquet(OUT / "fixations_frp.parquet", index=False)
    print("wrote fixations_frp.parquet", fix.shape, flush=True)
    good = valid & np.isfinite(peak2peak)
    print("p2p percentiles (uV-ish):", np.percentile(peak2peak[good], [50, 90, 99]).round(1))

if __name__ == "__main__":
    main()
