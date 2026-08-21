#!/usr/bin/env python3
"""Cache the preprocessed continuous recording once, so later stages never touch the raw frame.

Preprocessing is the established pipeline: per-run average reference, second-order 0.1 Hz
zero-phase high pass, fourth-order 30 Hz low pass, scaled to microvolts. Writes a float32 memmap
and the run boundary index, and verifies that `fixations.parquet`'s sample index still addresses
the same frame by comparing fixation onset times.

Needs the raw dataset (see README): reads `data/derivatives/features_df.pkl`, about 47 GB. The
memmap it writes is about 12 GB and is not redistributed.
"""
from __future__ import annotations
import gc, json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ART, COUP, ROOT, SF, EEG_CH

print("loading the synchronised frame (about 47 GB)...", flush=True)
raw = pd.read_pickle(ROOT / "data/derivatives/features_df.pkl")
n = len(raw)
missing = [c for c in EEG_CH if c not in raw.columns]
if missing:
    raise SystemExit(f"missing channels: {missing}")

fix = pd.read_parquet(COUP / "fixations.parquet")
idx = fix["onset_abs_idx"].to_numpy()
val = {"n_rows": int(n), "max_onset_abs_idx": int(idx.max())}
ts = pd.to_numeric(raw["fix_L_tStart"].iloc[idx], errors="coerce").to_numpy()
d = np.abs(ts - fix["tStart"].to_numpy())
val.update(align_max_abs_diff=float(np.nanmax(d)), align_frac_exact=float(np.mean(d < 1e-6)))
print("alignment: max|dt|=%.3g  exact=%.6f" % (val["align_max_abs_diff"], val["align_frac_exact"]),
      flush=True)

time = pd.to_numeric(raw["time"], errors="coerce").to_numpy()
b = np.empty(n, bool)
b[0] = False
b[1:] = time[1:] < time[:-1]
bounds = np.array(list(np.unique(np.cumsum(b), return_index=True)[1]) + [n], dtype=np.int64)
eeg = np.array(raw[EEG_CH].to_numpy(dtype=np.float32), copy=True)
del raw
gc.collect()

bhp, ahp = butter(2, 0.1 / (SF / 2), btype="high")
blp, alp = butter(4, 30.0 / (SF / 2), btype="low")
for k in range(len(bounds) - 1):
    s, e = int(bounds[k]), int(bounds[k + 1])
    blk = eeg[s:e].astype(np.float64, copy=True)
    blk -= blk.mean(1, keepdims=True)
    blk = filtfilt(bhp, ahp, blk, 0)
    blk = filtfilt(blp, alp, blk, 0)
    eeg[s:e] = (blk * 1e6).astype(np.float32)

mm = np.memmap(ART / "eeg_pp.f32", dtype=np.float32, mode="w+", shape=eeg.shape)
mm[:] = eeg
mm.flush()
del mm
np.save(ART / "run_bounds.npy", bounds)
val.update(shape=list(eeg.shape), n_runs=int(len(bounds) - 1), channels=EEG_CH, sfreq=SF)
(ART / "eeg_cache_meta.json").write_text(json.dumps(val, indent=2))
print("wrote eeg_pp.f32", eeg.shape, flush=True)
