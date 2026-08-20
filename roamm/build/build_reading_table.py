#!/usr/bin/env python3
"""Build word-level reading tables to test whether mind-wandering dismantles PREDICTIVE
reading (skipping, regressions, parafoveal preview) while sparing reactive processing.

From the ordered stimulus words (reading order) + the first-pass fixation sequence:
  reading_fixations.parquet — one row per first-pass fixation: reading position, word
    features (N, N-1, N+1), MW, fixation duration, and whether the NEXT saccade is a
    regression (backward move).
  reading_words.parquet — one row per word within each subject-run reading span: skipped
    (no first-pass fixation) or not, MW state (from nearest fixated word), word features.
"""
from __future__ import annotations
from pathlib import Path
import glob, os
import numpy as np
import pandas as pd

OUT = Path("roamm/artifacts/coupling")
STIM = Path("data/derivatives/stimuli/wiki_stories")

# ---- ordered words per story + features ----
wf = pd.read_parquet(OUT/"word_features.parquet")[["word_key","length","zipf","surprisal","clean"]]
coord_rows=[]
for csv in glob.glob(str(STIM/"*_coordinates.csv")):
    story=os.path.basename(csv).replace("_coordinates.csv","")
    d=pd.read_csv(csv)
    d=d.reset_index(drop=True)
    d["story"]=story; d["pos"]=np.arange(len(d))     # reading-order position within story
    coord_rows.append(d[["word_key","story","pos","page"]])
words=pd.concat(coord_rows,ignore_index=True).drop_duplicates("word_key")
words=words.merge(wf,on="word_key",how="left")
# position -> features maps per story for neighbor lookups
words=words.sort_values(["story","pos"]).reset_index(drop=True)
key2=words.set_index(["story","pos"])

def neighbor(df, dcol, off):
    idx=list(zip(df["story"], df["pos"]+off))
    sub=key2.reindex(idx)[dcol].to_numpy()
    return sub

# ---- fixations -> positions, neighbors, regression ----
fix=pd.read_parquet(OUT/"fixations.parquet")
fix=fix.merge(words[["word_key","story","pos","zipf","surprisal","length"]],
              on="word_key",how="left",suffixes=("","_w"))
fix=fix.dropna(subset=["pos"]).copy()
fix["pos"]=fix["pos"].astype(int)
fix=fix.sort_values(["subject","run","tStart"]).reset_index(drop=True)
# next fixation position within subject-run -> regression flag (backward move)
g=fix.groupby(["subject","run"],sort=False)
fix["next_pos"]=g["pos"].shift(-1)
fix["prev_pos"]=g["pos"].shift(1)
fix["regression_out"]=(fix["next_pos"]<fix["pos"]).astype("float")
fix.loc[fix["next_pos"].isna(),"regression_out"]=np.nan
# parafoveal (N+1) and previous (N-1) word features by TEXT position
for col in ["zipf","surprisal","length"]:
    fix[f"{col}_next"]=neighbor(fix,col,+1)
    fix[f"{col}_prev"]=neighbor(fix,col,-1)
fix.to_parquet(OUT/"reading_fixations.parquet",index=False)
print("reading_fixations:",fix.shape,"| regression-out rate:",round(float(np.nanmean(fix["regression_out"])),3))

# ---- word-level skipping table ----
rows=[]
for (subj,run),gg in fix.groupby(["subject","run"],sort=False):
    story=gg["story"].iloc[0]
    fixated=set(gg["pos"].tolist())
    lo,hi=gg["pos"].min(),gg["pos"].max()
    span=words[(words.story==story)&(words.pos>=lo)&(words.pos<=hi)].sort_values("pos")
    # MW state along positions from fixated words (majority is_mw per fixated pos)
    mwmap=gg.groupby("pos")["is_mw"].mean()
    posarr=span["pos"].to_numpy()
    fixpos=np.array(sorted(mwmap.index)); fixmw=mwmap.loc[fixpos].to_numpy()
    # nearest fixated position's MW for each position
    nearest=fixpos[np.clip(np.searchsorted(fixpos,posarr),0,len(fixpos)-1)]
    # refine: choose closer of left/right neighbor
    left=fixpos[np.clip(np.searchsorted(fixpos,posarr)-1,0,len(fixpos)-1)]
    choose_left=np.abs(posarr-left)<np.abs(posarr-nearest)
    nn=np.where(choose_left,left,nearest)
    mw_state=(pd.Series(nn).map(dict(zip(fixpos,fixmw))).to_numpy()>=0.5).astype(int)
    sk=(~np.isin(posarr,list(fixated))).astype(int)
    sub=span.copy()
    sub["subject"]=subj; sub["run"]=run; sub["skipped"]=sk; sub["is_mw"]=mw_state
    rows.append(sub[["subject","run","story","pos","page","word_key","zipf","surprisal","length","skipped","is_mw"]])
wdf=pd.concat(rows,ignore_index=True)
wdf.to_parquet(OUT/"reading_words.parquet",index=False)
print("reading_words:",wdf.shape,"| skip rate:",round(float(wdf["skipped"].mean()),3),
      "| MW rate:",round(float(wdf["is_mw"].mean()),3))
