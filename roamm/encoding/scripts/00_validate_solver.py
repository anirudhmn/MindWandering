#!/usr/bin/env python3
"""Check the fast solver against the frozen rERP kernels.

Refits the eight-predictor model of `roamm/build/build_rerp.py` for a few readers through the
Toeplitz path and correlates the kernels with `rerp_betas.npy`. Any real discrepancy would
invalidate everything downstream, so this runs first.

Needs the cached continuous recording from 01_cache_eeg.py.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import COUP, RES, lag_grid, build_XtX, build_XtY, fit_ridge, open_eeg

PRED = ["intercept", "zipf", "length", "surprisal", "logdur", "mw", "zipf:mw", "surprisal:mw"]
LAGS, _ = lag_grid(-100, 500)
NL, NP = len(LAGS), len(PRED)
READERS = [0, 7, 21]

eeg, bounds = open_eeg()
fix = pd.read_parquet(COUP / "fixations.parquet")
wf = pd.read_parquet(COUP / "word_features.parquet")[["word_key", "length", "zipf", "surprisal", "clean"]]
fix = fix.merge(wf, on="word_key", how="left")
fix = fix[(fix["clean"].str.len() >= 1) & (fix["zipf"] > 0) & fix["surprisal"].notna()
          & fix["fix_dur"].between(50, 1000)].copy()
fix["logdur"] = np.log(fix["fix_dur"].to_numpy())
for c in ["zipf", "length", "surprisal", "logdur"]:
    fix[c + "_z"] = fix.groupby("subject")[c].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
fix["mw"] = fix["is_mw"].astype(float)
run_of = np.zeros(int(bounds[-1]), np.int32)
for k in range(len(bounds) - 1):
    run_of[bounds[k]:bounds[k + 1]] = k
fix["rid"] = run_of[fix["onset_abs_idx"].to_numpy()]

ref = np.load(COUP / "rerp_betas.npy")
report = {}
for subj in READERS:
    fs = fix[fix.subject == subj]
    XtX = np.zeros((NP * NL, NP * NL))
    XtY = np.zeros((NP * NL, 64))
    for rid, fr in fs.groupby("rid"):
        rs, re = int(bounds[int(rid)]), int(bounds[int(rid) + 1])
        rel = fr["onset_abs_idx"].to_numpy() - rs
        keep = (rel + LAGS[0] >= 0) & (rel + LAGS[-1] < re - rs)
        fr, rel = fr[keep], rel[keep]
        o = np.argsort(rel)
        rel, fr = rel[o], fr.iloc[o]
        X = np.column_stack([np.ones(len(fr)), fr["zipf_z"], fr["length_z"], fr["surprisal_z"],
                             fr["logdur_z"], fr["mw"], fr["zipf_z"] * fr["mw"],
                             fr["surprisal_z"] * fr["mw"]]).astype(np.float64)
        XtX += build_XtX(rel, X, LAGS)
        XtY += build_XtY(eeg, rs, rel, X, LAGS)
    beta, _ = fit_ridge(XtX, XtY, NP, NL)
    report[f"reader{subj}"] = {
        nm: dict(corr=float(np.corrcoef(beta[p].ravel(), ref[subj, p].ravel())[0, 1]),
                 max_abs_diff_uV=float(np.abs(beta[p] - ref[subj, p]).max()))
        for p, nm in enumerate(PRED)}
    print(f"reader {subj}: " + "  ".join(f"{k} r={v['corr']:.4f}"
                                         for k, v in report[f"reader{subj}"].items()), flush=True)

(RES / "solver_validation.json").write_text(json.dumps(report, indent=2))
print("wrote", RES / "solver_validation.json")
