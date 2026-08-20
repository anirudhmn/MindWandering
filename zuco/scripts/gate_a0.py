#!/usr/bin/env python
"""
GATE A0 — replicate ROAMM's on-task word->brain coupling in ZuCo NR (deep reading).
  A0a Frequency FRP: Zipf modulates occipitotemporal amplitude 150-290 ms.
  A0b Surprisal N400: surprisal modulates centroparietal amplitude 300-450 ms (negative-going).
  A0c Behavior: log FFD ~ Zipf(-) + surprisal(+).
Method: single-trial ROI-window mean amplitude (baseline -100..0 ms) -> per-subject multiple
regression on z(zipf), z(surprisal), z(wlen); group t-test on slopes (subject = unit) + within-
subject predictor-shuffle null (1000x). PASS = A0a & A0b CI-clear of shuffle.
"""
import os, glob, json, numpy as np, pandas as pd
from scipy import stats

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1]); A = f'{ROOT}/artifacts'
FS, PRE = 500, 50                       # onset at sample 50 (-100..+500 ms)
def ms2i(ms): return PRE + int(round(ms/1000*FS))
OCC_W = slice(ms2i(150), ms2i(290)); CP_W = slice(ms2i(300), ms2i(450)); BASE = slice(0, PRE)

def rois():
    r = json.load(open(f'{A}/chanlocs_105.json')); lab=[x['label'] for x in r]
    X=np.array([x['X'] for x in r]); Y=np.array([x['Y'] for x in r]); Z=np.array([x['Z'] for x in r])
    occ=[i for i in range(len(lab)) if X[i]<-5 and Z[i]<4]
    cp =[i for i in range(len(lab)) if -7<X[i]<-1 and Z[i]>4.5 and abs(Y[i])<5]
    return occ, cp

def load_task(task):
    occ, cp = rois()
    ling = pd.read_parquet(f'{A}/linguistic_{task}.parquet')
    recs = []
    for mp in sorted(glob.glob(f'{A}/frp/meta_*_{task}.parquet')):
        subj = os.path.basename(mp).split('_')[1]
        meta = pd.read_parquet(mp)
        frp = np.load(mp.replace('meta_', 'frp_').replace('.parquet', '.npy'))  # [n,105,300]
        base = np.nanmean(frp[:, :, BASE], axis=2, keepdims=True)
        frp = frp - base                                    # baseline correct
        occ_amp = np.nanmean(frp[:, occ, :][:, :, OCC_W], axis=(1, 2))
        cp_amp  = np.nanmean(frp[:, cp,  :][:, :, CP_W],  axis=(1, 2))
        m = meta.copy(); m['occ_amp'] = occ_amp; m['cp_amp'] = cp_amp
        recs.append(m.merge(ling, on=['task','sent_idx','word_idx'], how='left', suffixes=('','_l')))
    df = pd.concat(recs, ignore_index=True)
    return df

def zscore(x):
    x = np.asarray(x, float); return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

def per_subject_slopes(df, ycol, predictors, key):
    """Per-subject multiple OLS; return dict predictor->array of subject slopes for `key`."""
    slopes = {p: [] for p in predictors}
    for s, g in df.groupby('subject'):
        g = g.dropna(subset=[ycol]+predictors)
        if len(g) < 100: continue
        Xz = np.column_stack([zscore(g[p]) for p in predictors] + [np.ones(len(g))])
        y = np.asarray(g[ycol], float)
        beta, *_ = np.linalg.lstsq(Xz, y, rcond=None)
        for j, p in enumerate(predictors): slopes[p].append(beta[j])
    return {p: np.array(v) for p, v in slopes.items()}

def shuffle_null(df, ycol, predictors, target, n=1000, seed=0):
    rng = np.random.default_rng(seed); null = []
    for _ in range(n):
        sl = []
        for s, g in df.groupby('subject'):
            g = g.dropna(subset=[ycol]+predictors)
            if len(g) < 100: continue
            perm = rng.permutation(len(g))
            cols = [zscore(g[p].values[perm]) if p == target else zscore(g[p]) for p in predictors]
            Xz = np.column_stack(cols + [np.ones(len(g))]); y = np.asarray(g[ycol], float)
            beta, *_ = np.linalg.lstsq(Xz, y, rcond=None)
            sl.append(beta[predictors.index(target)])
        null.append(np.mean(sl))
    return np.array(null)

def report(name, slopes, target, null=None):
    v = slopes[target]; t, p = stats.ttest_1samp(v, 0)
    ci = np.percentile(v, [2.5, 97.5]) if len(v) > 1 else [np.nan, np.nan]
    line = f'{name}: slope={v.mean():+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] t({len(v)-1})={t:.2f} p={p:.2e} (n={len(v)} subj, {np.mean(v>0)*100:.0f}% +)'
    if null is not None:
        pnull = (np.sum(np.abs(null) >= abs(v.mean())) + 1) / (len(null) + 1)
        line += f' | shuffle mean={null.mean():+.4f} sd={null.std():.4f} p_perm={pnull:.4f}'
    print(line); return v

if __name__ == '__main__':
    df = load_task('NR')
    print(f'NR single-trial words with linguistics: {df.occ_amp.notna().sum()}, subjects={df.subject.nunique()}')
    print(f'windows: OCC {OCC_W.start}:{OCC_W.stop} (150-290ms), CP {CP_W.start}:{CP_W.stop} (300-450ms)\n')

    print('--- A0a  Frequency FRP (occipitotemporal 150-290 ms) ---')
    sl = per_subject_slopes(df, 'occ_amp', ['zipf','surprisal','wlen'], 'occ')
    nl = shuffle_null(df, 'occ_amp', ['zipf','surprisal','wlen'], 'zipf')
    report('  zipf->OCC', sl, 'zipf', nl)

    print('\n--- A0b  Surprisal N400 (centroparietal 300-450 ms, expect NEGATIVE) ---')
    sl = per_subject_slopes(df, 'cp_amp', ['surprisal','zipf','wlen'], 'cp')
    nl = shuffle_null(df, 'cp_amp', ['surprisal','zipf','wlen'], 'surprisal')
    report('  surprisal->CP', sl, 'surprisal', nl)

    print('\n--- A0c  Behavior: log FFD (ms) coupling ---')
    df['logFFD'] = np.log(df['FFD'].clip(lower=1) / FS * 1000)
    slb = per_subject_slopes(df, 'logFFD', ['zipf','surprisal','wlen'], 'beh')
    report('  zipf->logFFD (expect -)', slb, 'zipf')
    report('  surprisal->logFFD (expect +)', slb, 'surprisal')
