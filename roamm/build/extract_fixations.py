#!/usr/bin/env python3
"""Extract a compact per-fixation table from the raw 256 Hz frame (one load).

Each fixation in the raw frame is a span of consecutive samples sharing the same
fix_L_tStart / fix_L_duration / fix_L_fixed_word_key. We collapse each first-pass
left-eye fixation to a single record, tagging it with the word it landed on, the
mind-wandering label, timing/order covariates, and the ABSOLUTE row index of the
fixation onset (stable across reloads of the same pickle) so that EEG epochs can be
cut later without recomputing fixation boundaries.

Output: roamm/artifacts/coupling/fixations.parquet
"""
from __future__ import annotations
import gc
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("roamm/artifacts/coupling")
OUT.mkdir(parents=True, exist_ok=True)

print("loading pickle...", flush=True)
raw = pd.read_pickle("data/derivatives/features_df.pkl")
n = len(raw)
print("loaded", raw.shape, flush=True)

# --- subject id: time resets at each run start; 5 runs per subject (as in build_corrected_labels) ---
time = pd.to_numeric(raw["time"], errors="coerce").to_numpy()
boundaries = np.empty(n, dtype=bool)
boundaries[0] = False
boundaries[1:] = time[1:] < time[:-1]
subject = (np.cumsum(boundaries, dtype=np.int32) // 5).astype(np.int32)
run = pd.to_numeric(raw["run_num"], errors="coerce").to_numpy()
print("n subjects:", subject.max() + 1, "n runs total:", int(np.sum(boundaries)) + 1, flush=True)

# --- left-eye fixation onsets ---
fx_start = pd.to_numeric(raw["fix_L_tStart"], errors="coerce").to_numpy()
word_key = raw["fix_L_fixed_word_key"].astype("string").to_numpy()  # <NA> where none
valid = pd.notna(raw["fix_L_fixed_word_key"].to_numpy()) & np.isfinite(fx_start)

# boundary between fixations: valid sample whose (tStart, subject, run) differs from prev
prev_fx = np.empty(n); prev_fx[0] = np.nan; prev_fx[1:] = fx_start[:-1]
prev_subj = np.empty(n, dtype=np.int64); prev_subj[0] = -1; prev_subj[1:] = subject[:-1]
prev_run = np.empty(n); prev_run[0] = np.nan; prev_run[1:] = run[:-1]
changed = (fx_start != prev_fx) | (subject != prev_subj) | (run != prev_run)
new_fix = valid & (changed | ~np.r_[False, valid[:-1]])

fix_id = np.full(n, -1, dtype=np.int64)
fix_id[valid] = np.cumsum(new_fix)[valid] - 1  # 0-based ids over valid samples
print("n fixation samples:", int(valid.sum()), "n fixations:", int(new_fix.sum()), flush=True)

# --- assemble per-sample fields we need to aggregate ---
is_mw = raw["is_mw"].fillna(False).to_numpy(dtype=np.uint8)
first_pass = raw["first_pass_reading"].fillna(False).to_numpy(dtype=np.uint8)
page_num = pd.to_numeric(raw["page_num"], errors="coerce").to_numpy()
story = raw["story_name"].astype("string").to_numpy()
fix_dur = pd.to_numeric(raw["fix_L_duration"], errors="coerce").to_numpy()
fix_dur_r = pd.to_numeric(raw["fix_R_duration"], errors="coerce").to_numpy()
tsample = pd.to_numeric(raw["tSample"], errors="coerce").to_numpy()
abs_idx = np.arange(n, dtype=np.int64)

sub = valid
df = pd.DataFrame({
    "fix_id": fix_id[sub],
    "abs_idx": abs_idx[sub],
    "subject": subject[sub],
    "run": run[sub],
    "story": story[sub],
    "page": page_num[sub],
    "word_key": word_key[sub],
    "tStart": fx_start[sub],
    "tSample": tsample[sub],
    "fix_dur": fix_dur[sub],
    "fix_dur_r": fix_dur_r[sub],
    "is_mw": is_mw[sub],
    "first_pass": first_pass[sub],
})
del raw; gc.collect()

g = df.groupby("fix_id", sort=True)
fix = pd.DataFrame({
    "onset_abs_idx": g["abs_idx"].first(),
    "subject": g["subject"].first(),
    "run": g["run"].first(),
    "story": g["story"].first(),
    "page": g["page"].first(),
    "word_key": g["word_key"].first(),
    "tStart": g["tStart"].first(),
    "tSample_onset": g["tSample"].first(),
    "fix_dur": g["fix_dur"].first(),         # ms, constant within fixation
    "fix_dur_r": g["fix_dur_r"].first(),
    "n_samples": g.size(),
    "mw_frac": g["is_mw"].mean(),
    "firstpass_frac": g["first_pass"].mean(),
}).reset_index(drop=True)

# fixation order within subject-run (time-on-task proxy)
fix = fix.sort_values(["subject", "run", "tStart"]).reset_index(drop=True)
fix["fix_order"] = fix.groupby(["subject", "run"]).cumcount()
fix["is_mw"] = (fix["mw_frac"] >= 0.5).astype(np.int8)
fix["is_firstpass"] = (fix["firstpass_frac"] >= 0.9)

fix.to_parquet(OUT / "fixations.parquet", index=False)
print("wrote", OUT / "fixations.parquet", fix.shape, flush=True)
print("first-pass fixations:", int(fix["is_firstpass"].sum()),
      "| MW rate among first-pass:", round(float(fix.loc[fix.is_firstpass, "is_mw"].mean()), 4), flush=True)
print("median fix_dur (ms):", float(fix["fix_dur"].median()),
      "| subjects:", fix["subject"].nunique(), flush=True)
