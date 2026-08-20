#!/usr/bin/env python
"""
Extraction 1 (Part A): per-word FIRST-FIXATION fixation-related potentials (FRPs) from ZuCo.

Per subject x task: parse results<ZID>_<TASK>.mat, and for every word's FIRST fixation, recover the
fixation-onset sample index in the sentence-continuous `rawData` (per-fixation rawEEG segments are
EXACT slices of rawData), then re-epoch -100..+500 ms (fs=500Hz -> -50..+250 samples = 300 samples,
105 channels) with proper pre-fixation baseline. The dataset's per-fixation rawEEG is only fixation-
length, too short for the 300-450 ms N400 window -> we MUST re-epoch from rawData.

Outputs (per subject x task) in analysis/artifacts/frp/:
  meta_<ZID>_<TASK>.parquet   word-level table (one row per first-fixation kept)
  frp_<ZID>_<TASK>.npy        float32 [n_rows, 105, 300], -100..+500 ms, raw (no baseline corr)
Run:  python zuco/scripts/parse_frp.py NR    (or TSR, or ALL)
"""
import sys, os, glob, time
import numpy as np, pandas as pd
import scipy.io as sio

FS = 500                      # Hz (EEG.srate, confirmed)
PRE_MS, POST_MS = 100, 500
PRE, POST = int(PRE_MS*FS/1000), int(POST_MS*FS/1000)   # 50, 250
WIN = PRE + POST              # 300 samples
NCH = 105
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
OUT = f'{ROOT}/analysis/artifacts/frp'
os.makedirs(OUT, exist_ok=True)

def find_onset(seg, rd, atol=1e-3):
    """Return unique onset column of exact slice `seg` (105xN) in `rd` (105xT), else -1."""
    N = seg.shape[1]; T = rd.shape[1]
    diffs = np.abs(rd - seg[:, :1]).sum(axis=0)      # match first column signature
    cands = np.where(diffs < atol)[0]
    hits = [int(j) for j in cands if j+N <= T and np.allclose(rd[:, j:j+N], seg, atol=atol)]
    return hits[0] if len(hits) == 1 else -1

def as_segs(re):
    if np.size(re) == 0: return []
    raw = list(re) if (hasattr(re, 'dtype') and re.dtype == object) else [re]
    segs = []
    for x in raw:
        a = np.asarray(x, dtype=np.float64)
        if a.ndim == 2 and a.shape[0] == NCH and a.shape[1] > 0:
            segs.append(a)
    return segs

def scalar(x, default=np.nan):
    a = np.asarray(x).ravel()
    return float(a[0]) if a.size else default

def parse_file(path, subj, task):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    sd = m['sentenceData']
    rows, epochs = [], []
    n_multi = n_short = 0
    for si, s in enumerate(np.atleast_1d(sd)):
        rd = np.asarray(s.rawData, dtype=np.float64)
        if rd.ndim != 2 or rd.shape[0] != NCH: continue
        T = rd.shape[1]
        words = np.atleast_1d(s.word)
        nw = len(words)
        for wi, w in enumerate(words):
            if not hasattr(w, 'rawEEG'): continue
            segs = as_segs(w.rawEEG)
            if not segs: continue
            seg0 = segs[0]
            if seg0.shape[0] != NCH: continue
            onset = find_onset(seg0, rd)
            if onset < 0:
                n_multi += 1; continue
            lo, hi = onset - PRE, onset + POST
            if lo < 0 or hi > T:
                n_short += 1
                ep = np.full((NCH, WIN), np.nan, np.float32)
                a, b = max(lo, 0), min(hi, T)
                ep[:, a-lo:b-lo] = rd[:, a:b].astype(np.float32)
            else:
                ep = rd[:, lo:hi].astype(np.float32)
            rows.append(dict(subject=subj, task=task, sent_idx=si, word_idx=wi,
                             word=str(getattr(w, 'content', '')), sent_nwords=nw,
                             FFD=scalar(getattr(w, 'FFD', np.nan)),
                             GD=scalar(getattr(w, 'GD', np.nan)),
                             TRT=scalar(getattr(w, 'TRT', np.nan)),
                             nFix=scalar(getattr(w, 'nFixations', np.nan)),
                             onset=onset, sent_T=T, full_window=int(lo >= 0 and hi <= T)))
            epochs.append(ep)
    meta = pd.DataFrame(rows)
    frp = np.stack(epochs).astype(np.float32) if epochs else np.zeros((0, NCH, WIN), np.float32)
    return meta, frp, n_multi, n_short

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'ALL'
    tasks = {'NR': 'task2_NR_matlab', 'TSR': 'task3_TSR_matlab', 'SR': 'task1_SR_matlab'}
    if which != 'ALL': tasks = {which: tasks[which]}
    for task, folder in tasks.items():
        files = sorted(glob.glob(f'{ROOT}/{folder}/results*_{task}.mat'))
        for path in files:
            subj = os.path.basename(path).replace('results', '').replace(f'_{task}.mat', '')
            t0 = time.time()
            meta, frp, nm, ns = parse_file(path, subj, task)
            meta.to_parquet(f'{OUT}/meta_{subj}_{task}.parquet')
            np.save(f'{OUT}/frp_{subj}_{task}.npy', frp)
            print(f'{subj} {task}: {len(meta)} first-fix words, frp {frp.shape}, '
                  f'unmatched={nm}, edge-clipped={ns}, {time.time()-t0:.0f}s', flush=True)

if __name__ == '__main__':
    main()
