#!/usr/bin/env python3
"""Per-reader nuisance model, and the held-out residual recording it leaves behind.

The nuisance kernels

    intercept, log fixation duration, fixation order, incoming and outgoing saccade amplitude,
    page progress, mind-wandering, mind-wandering x log duration

are fitted per reader on that reader's other runs and used to predict the held-out run, so the
additive state change is absorbed into the baseline before any word property is looked at. What
is stored for each held-out fixation is the residual over its 0 to 500 ms window, together with
the text design and the onsets, which is everything the pooled fit in 03 needs.

Per-reader TEXT kernels are not fitted here. They do not generalise: on a three-reader check the
held-out improvement was -0.194, -0.110 and +0.005 uV^2, the text kernels making prediction
worse. That is what the group-level character of these coupling estimates implies, and it is why
the text kernel is pooled in 03.

Needs the cached recording from 01_cache_eeg.py. The residual cache is about 6 GB and is not
redistributed.
"""
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import (ART, COUP, ROOT, TEXT_BASE, lag_grid, build_XtX, build_XtY, fit_ridge,
                    predict_run, open_eeg)

NUIS = ["intercept", "logdur_z", "order_z", "log_in_amp_z", "log_out_amp_z", "page_prog_z",
        "mw", "mw:logdur_z"]
NPn = len(NUIS)
LAGS, LAGS_MS = lag_grid(-100, 500)
NL = len(LAGS)
EVAL = np.where((LAGS_MS >= 0) & (LAGS_MS <= 500))[0]
RESID = ART / "resid"


def build_events():
    """One row per usable fixation, with the inclusion rule of the frozen rERP pipeline."""
    fix = pd.read_parquet(COUP / "fixations.parquet")
    wf = pd.read_parquet(COUP / "word_features.parquet")[["word_key", "clean", "length", "zipf"]]
    ms = pd.read_parquet(COUP / "word_multiscale.parquet")[
        ["word_key", "gpt2_s_sent", "gpt2_gain_long", "gpt2_gain_shuf"]]
    lay = pd.read_parquet(COUP / "words_layout.parquet")[["word_key", "pos", "story", "page"]]
    sac = pd.read_parquet(COUP / "saccades.parquet")[["subject", "onset_abs_idx", "in_amp_px", "amp_px"]]
    fix = (fix.merge(wf, on="word_key", how="left").merge(ms, on="word_key", how="left")
              .merge(lay.drop(columns=["story", "page"]), on="word_key", how="left")
              .merge(sac, on=["subject", "onset_abs_idx"], how="left"))
    fix = fix.rename(columns={"gpt2_s_sent": "s_local", "gpt2_gain_long": "gain_long",
                              "gpt2_gain_shuf": "gain_shuf"})
    ok = ((fix["clean"].str.len() >= 1) & (fix["zipf"] > 0) & fix["s_local"].notna()
          & fix["gain_long"].notna() & fix["gain_shuf"].notna() & fix["fix_dur"].between(50, 1000))
    fix = fix[ok].copy()
    fix["logdur"] = np.log(fix["fix_dur"].to_numpy())
    fix["order"] = fix["fix_order"].astype(float)
    fix["mw"] = fix["is_mw"].astype(float)
    # a missing amplitude is the first fixation of a page or an unmapped neighbour; dropping
    # those rows would delete episode onsets, so they take the reader's median instead
    for c in ["in_amp_px", "amp_px"]:
        fix[c] = fix.groupby("subject")[c].transform(lambda s: s.fillna(s.median()))
        fix[c] = fix[c].fillna(fix[c].median())
    fix["log_in_amp"] = np.log1p(fix["in_amp_px"])
    fix["log_out_amp"] = np.log1p(fix["amp_px"])
    pagelen = pd.read_parquet(COUP / "words_layout.parquet").groupby(
        ["story", "page"])["pos"].max().rename("page_max_pos").reset_index()
    fix = fix.merge(pagelen, on=["story", "page"], how="left")
    fix["page_prog"] = fix["pos"] / fix["page_max_pos"].clip(lower=1)
    for c in ["logdur", "order", "log_in_amp", "log_out_amp", "page_prog"] + TEXT_BASE:
        fix[c + "_z"] = fix.groupby("subject")[c].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-9))
    return fix


def design(fr):
    mw = fr["mw"].to_numpy()
    cols = [np.ones(len(fr))]
    for c in NUIS[1:]:
        cols.append(fr[c.split(":")[1]].to_numpy() * mw if c.startswith("mw:") else fr[c].to_numpy())
    return np.column_stack(cols).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    RESID.mkdir(parents=True, exist_ok=True)

    eeg, bounds = open_eeg()
    run_of = np.zeros(int(bounds[-1]), np.int32)
    for k in range(len(bounds) - 1):
        run_of[bounds[k]:bounds[k + 1]] = k
    ev = build_events()
    ev["rid"] = run_of[ev["onset_abs_idx"].to_numpy()]
    ev.to_parquet(ART / "events.parquet", index=False)
    print(f"{len(ev)} fixations, {ev.subject.nunique()} readers, "
          f"{100*ev['mw'].mean():.1f}% mind-wandering", flush=True)

    for subj in sorted(ev.subject.unique()):
        if (RESID / f"nuisbeta_s{int(subj):02d}.npy").exists():
            continue
        es = ev[ev.subject == subj]
        rids = sorted(es.rid.unique())
        runs = {int(r): np.asarray(eeg[int(bounds[int(r)]):int(bounds[int(r) + 1])], np.float32)
                for r in rids}
        bsum, nb = np.zeros((NPn, NL, 64)), 0
        for held in rids:
            XtX = np.zeros((NPn * NL, NPn * NL))
            XtY = np.zeros((NPn * NL, 64))
            for rid, fr in es[es.rid != held].groupby("rid"):
                Y = runs[int(rid)]
                rel = fr["onset_abs_idx"].to_numpy() - int(bounds[int(rid)])
                k = (rel + LAGS[0] >= 0) & (rel + LAGS[-1] < Y.shape[0])
                fr, rel = fr[k], rel[k]
                o = np.argsort(rel)
                rel, fr = rel[o], fr.iloc[o]
                X = design(fr)
                XtX += build_XtX(rel, X, LAGS)
                XtY += build_XtY(Y, 0, rel, X, LAGS)
            beta, _ = fit_ridge(XtX, XtY, NPn, NL)
            bsum += beta
            nb += 1

            te = es[es.rid == held]
            Y = runs[int(held)]
            rel = te["onset_abs_idx"].to_numpy() - int(bounds[int(held)])
            k = (rel + LAGS[0] >= 0) & (rel + LAGS[-1] < Y.shape[0])
            te, rel = te[k].copy(), rel[k]
            o = np.argsort(rel)
            rel, te = rel[o], te.iloc[o]
            if len(te) == 0:
                continue
            pred = predict_run(Y.shape[0], rel, design(te), beta, LAGS)
            rows = rel[:, None] + LAGS[None, EVAL]
            np.savez(RESID / f"s{int(subj):02d}_r{int(held)}.npz",
                     resid=(Y[rows].astype(np.float64) - pred[rows]).astype(np.float16),
                     onset_rel=rel.astype(np.int64), runlen=np.int64(Y.shape[0]),
                     mw=te["mw"].to_numpy().astype(np.float32),
                     onset_abs_idx=te["onset_abs_idx"].to_numpy(),
                     story=np.array(te["story"].iloc[0]),
                     text=te[TEXT_BASE].to_numpy().astype(np.float64),
                     logdur=te["logdur"].to_numpy(), order=te["order"].to_numpy(),
                     page_prog=te["page_prog"].to_numpy(),
                     log_in_amp=te["log_in_amp"].to_numpy())
        np.save(RESID / f"nuisbeta_s{int(subj):02d}.npy", (bsum / max(nb, 1)).astype(np.float32))
        print(f"  reader {subj}: {nb} folds", flush=True)
    print("residual cache complete")


if __name__ == "__main__":
    main()
