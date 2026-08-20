#!/usr/bin/env python3
"""Extract compact single-trial FRP epochs for multivariate word decoding.

Same preprocessing as extract_frp_roi (per-run average reference + 0.1-30 Hz),
epoch [-100, +400] ms, baseline [-100, 0] ms, downsample x2 to 128 Hz. Save a
float16 tensor [n_fixations, 64ch, T] aligned row-for-row with fixations.parquet.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

OUT = Path("roamm/artifacts/coupling")
SF = 256.0
EEG_COLS = ['Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1','C1','C3','C5','T7','TP7','CP5','CP3','CP1','P1','P3','P5','P7','P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz','Fpz','Fp2','AF8','AF4','Afz','Fz','F2','F4','F6','F8','FT8','FC6','FC4','FC2','FCz','Cz','C2','C4','C6','T8','TP8','CP6','CP4','CP2','P2','P4','P6','P8','P10','PO8','PO4','O2']

def main():
    print("loading pickle...", flush=True)
    raw = pd.read_pickle("data/derivatives/features_df.pkl")
    n = len(raw)
    eeg = np.array(raw[EEG_COLS].to_numpy(dtype=np.float32), copy=True)
    time = pd.to_numeric(raw["time"], errors="coerce").to_numpy()
    boundaries = np.empty(n, dtype=bool); boundaries[0] = False
    boundaries[1:] = time[1:] < time[:-1]
    run_id = np.cumsum(boundaries)
    del raw

    bhp, ahp = butter(2, 0.1 / (SF / 2), btype="high")
    blp, alp = butter(4, 30.0 / (SF / 2), btype="low")
    uniq, starts = np.unique(run_id, return_index=True)
    starts = list(starts) + [n]
    print("preprocessing %d runs..." % len(uniq), flush=True)
    for k in range(len(uniq)):
        s, e = starts[k], starts[k + 1]
        block = eeg[s:e].astype(np.float64, copy=True)
        block -= block.mean(axis=1, keepdims=True)
        block = filtfilt(bhp, ahp, block, axis=0)
        block = filtfilt(blp, alp, block, axis=0)
        eeg[s:e] = block.astype(np.float32)
        if (k + 1) % 50 == 0:
            print("  filtered run", k + 1, flush=True)

    fix = pd.read_parquet(OUT / "fixations.parquet")
    onset = fix["onset_abs_idx"].to_numpy()
    pre = int(round(0.100 * SF)); post = int(round(0.400 * SF))
    rel = np.arange(-pre, post + 1)                       # 129 samples
    t = rel / SF
    base_mask = (t >= -0.100) & (t <= 0.0)
    ds = 2                                                 # -> 128 Hz
    keep = np.arange(0, len(rel), ds)
    t_ds = t[keep]
    nfix = len(onset)
    epochs = np.zeros((nfix, 64, len(keep)), dtype=np.float16)
    valid = (onset - pre >= 0) & (onset + post < n) & \
            (run_id[np.clip(onset - pre, 0, n-1)] == run_id[np.clip(onset + post, 0, n-1)])
    idxs = np.flatnonzero(valid)
    CH = 20000
    for c0 in range(0, len(idxs), CH):
        ci = idxs[c0:c0 + CH]
        rows = onset[ci][:, None] + rel[None, :]          # [m,129]
        ep = eeg[rows]                                     # [m,129,64]
        ep = ep - ep[:, base_mask, :].mean(axis=1, keepdims=True)
        ep = ep[:, keep, :].transpose(0, 2, 1)            # [m,64,T]
        epochs[ci] = (ep * 1e6).astype(np.float16)        # store in uV
        if (c0 // CH) % 5 == 0:
            print("  epoched", c0 + len(ci), "/", len(idxs), flush=True)

    np.save(OUT / "frp_epochs.npy", epochs)
    np.save(OUT / "frp_epochs_time.npy", t_ds)
    np.save(OUT / "frp_epochs_valid.npy", valid)
    print("saved frp_epochs.npy", epochs.shape, "time", t_ds.round(3), flush=True)

if __name__ == "__main__":
    main()
