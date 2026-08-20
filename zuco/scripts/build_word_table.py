"""Build a compact per-word neural feature table for NR and TSR (all 12 subjects).

Reduces each fixation-locked FRP (n_word, 105ch, 300samp @ 500Hz, -100..+500ms)
to scalar ROI amplitudes in canonical windows, then merges linguistic features
(surprisal/zipf/wlen) and per-sentence TSR relation labels.

Windows (t=0 at sample 50): baseline 0:50 (-100..0); N1 125:175 (150-250ms);
P3/LPC-central 175:275 (250-450ms); late/N400-LPC 200:300 (300-500ms).
"""
import json, csv, collections
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRP = ROOT/'analysis/artifacts/frp'
ART = ROOT/'analysis/artifacts'
OUT = ART/'tsr'; OUT.mkdir(exist_ok=True)

SUBJS = ['ZAB','ZDM','ZDN','ZGW','ZJM','ZJN','ZJS','ZKB','ZKH','ZKW','ZMG','ZPH']
TASKS = ['NR','TSR']

# --- ROIs from chanlocs ---
cl = json.load(open(ART/'chanlocs_105.json'))
X=np.array([c['X'] for c in cl]);Y=np.array([c['Y'] for c in cl]);Z=np.array([c['Z'] for c in cl])
OCC = np.where((X<-5)&(Z<4))[0]
CP  = np.where((X>-7)&(X<-1)&(Z>4.5)&(np.abs(Y)<5))[0]
PZ  = np.where((X>-9)&(X<-2)&(Z>4)&(np.abs(Y)<4))[0]

BASE=slice(0,50); N1=slice(125,175); P3=slice(175,275); LATE=slice(200,300)

def roi_scalars(frp):
    # frp: (n,105,300). baseline-correct per ch, then ROI-mean waveform, window-mean.
    bl = frp[:,:,BASE].mean(2, keepdims=True)
    x = frp - bl
    occ = x[:,OCC,:].mean(1)   # (n,300)
    cp  = x[:,CP,:].mean(1)
    pz  = x[:,PZ,:].mean(1)
    return dict(
        occ_N1 = occ[:,N1].mean(1),
        cp_late= cp[:,LATE].mean(1),
        cp_p3  = cp[:,P3].mean(1),
        pz_p3  = pz[:,P3].mean(1),
        pz_late= pz[:,LATE].mean(1),
    )

# --- TSR relation labels (aligned to .mat sentence order = sent_idx) ---
def load_rel():
    rows=[]
    with open(ROOT/'analysis/materials/relations_labels_task3.csv',encoding='utf-8') as f:
        f.readline()
        for line in f:
            p=line.rstrip('\n').split(';')
            rel=p[-1]; rows.append(rel)
    return rows  # index = sent_idx (0..406)
REL = load_rel()
rel_map = {i:r for i,r in enumerate(REL)}

# --- linguistic features per task ---
lingNR = pd.read_parquet(ART/'linguistic_NR.parquet')
lingTSR= pd.read_parquet(ART/'linguistic_TSR.parquet')
LING={'NR':lingNR,'TSR':lingTSR}

frames=[]
for subj in SUBJS:
    for task in TASKS:
        meta = pd.read_parquet(FRP/f'meta_{subj}_{task}.parquet').reset_index(drop=True)
        frp  = np.load(FRP/f'frp_{subj}_{task}.npy', mmap_mode='r')
        assert len(meta)==frp.shape[0], (subj,task,len(meta),frp.shape)
        sc = roi_scalars(np.asarray(frp))
        for k,v in sc.items(): meta[k]=v
        # merge linguistic on (sent_idx, word_idx)
        L = LING[task][['sent_idx','word_idx','surprisal','zipf','wlen','is_content']]
        meta = meta.merge(L, on=['sent_idx','word_idx'], how='left')
        if task=='TSR':
            meta['relation']=meta['sent_idx'].map(rel_map)
        else:
            meta['relation']=np.nan
        frames.append(meta)
        print(subj,task,len(meta),'ling-cov',meta['surprisal'].notna().mean().round(3))

df = pd.concat(frames, ignore_index=True)
df.to_parquet(OUT/'word_neural.parquet')
print('\nTOTAL rows', len(df))
print('by task', df.task.value_counts().to_dict())
print('TSR relation coverage:', df[df.task=='TSR'].relation.notna().mean())
print(df[df.task=='TSR'].relation.value_counts().to_dict())
