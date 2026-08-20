#!/usr/bin/env python3
"""Behavioral decoupling test: does oculomotor sensitivity to word frequency (and
length) collapse during mind-wandering?

Classic 'mindless reading' prediction (Reichle et al. 2010): during MW the eyes
keep moving but stop responding to what the words are, so the word-frequency effect
on fixation duration attenuates toward zero.

Per subject we fit
    log(fix_dur) ~ zipf_c + length_c + order_z
                   + is_mw + zipf_c:is_mw + length_c:is_mw + order_z:is_mw
(predictors centered within subject). The interaction zipf_c:is_mw is the change in
the frequency slope during MW. Decoupling predicts: on-task zipf slope < 0, and the
interaction > 0 (slope pushed toward zero). Group inference is a subject-level
one-sample test on the interaction coefficients (+ a label-shuffle control).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path("roamm/artifacts/coupling")
RNG = np.random.default_rng(20260718)

fix = pd.read_parquet(OUT / "fixations.parquet")
wf = pd.read_parquet(OUT / "word_features.parquet")[["word_key", "length", "zipf", "clean"]]
df = fix.merge(wf, on="word_key", how="left")

# quality filters: real words, plausible reading fixations
df = df[(df["clean"].str.len() >= 1) & (df["zipf"] > 0)]
df = df[(df["fix_dur"] >= 50) & (df["fix_dur"] <= 1000)]
df["log_dur"] = np.log(df["fix_dur"].to_numpy())
print("fixations after filtering:", len(df), "| MW rate:", round(df["is_mw"].mean(), 4))

TERMS = ["intercept", "zipf_c", "length_c", "order_z",
         "is_mw", "zipf_x_mw", "length_x_mw", "order_x_mw"]

def fit_subject(sub: pd.DataFrame, mw: np.ndarray) -> np.ndarray | None:
    if mw.sum() < 50 or (1 - mw).sum() < 200:
        return None
    zipf_c = sub["zipf"].to_numpy() - sub["zipf"].mean()
    length_c = sub["length"].to_numpy() - sub["length"].mean()
    order = sub["fix_order"].to_numpy().astype(float)
    order_z = (order - order.mean()) / (order.std() + 1e-9)
    y = sub["log_dur"].to_numpy()
    X = np.column_stack([
        np.ones(len(sub)), zipf_c, length_c, order_z,
        mw, zipf_c * mw, length_c * mw, order_z * mw,
    ])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta

def run(shuffle: bool) -> pd.DataFrame:
    recs = []
    for subj, sub in df.groupby("subject"):
        mw = sub["is_mw"].to_numpy().astype(float)
        if shuffle:
            mw = RNG.permutation(mw)
        beta = fit_subject(sub, mw)
        if beta is None:
            continue
        rec = {"subject": subj, "n": len(sub), "n_mw": int(mw.sum())}
        rec.update(dict(zip(TERMS, beta)))
        recs.append(rec)
    return pd.DataFrame(recs)

def group_report(res: pd.DataFrame, name: str) -> dict:
    out = {"analysis": name, "n_subjects": len(res)}
    for term in ["zipf_c", "length_c", "zipf_x_mw", "length_x_mw", "is_mw"]:
        vals = res[term].to_numpy()
        t, p = stats.ttest_1samp(vals, 0.0)
        # subject bootstrap CI
        boot = [RNG.choice(vals, len(vals), replace=True).mean() for _ in range(10000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[term] = {
            "mean": float(vals.mean()), "ci": [float(lo), float(hi)],
            "t": float(t), "p": float(p),
            "frac_positive": float((vals > 0).mean()),
        }
    return out

real = run(shuffle=False)
shuf = run(shuffle=True)
real.to_csv(OUT / "behavioral_subject_slopes.csv", index=False)

report = {
    "real": group_report(real, "real"),
    "shuffle": group_report(shuf, "label_shuffle_control"),
}
(OUT / "behavioral_coupling_report.json").write_text(json.dumps(report, indent=2) + "\n")

print("\n=== ON-TASK slopes (sanity: zipf<0 shorter fix for frequent words, length>0) ===")
for term in ["zipf_c", "length_c"]:
    r = report["real"][term]
    print(f"  {term:12} mean={r['mean']:+.4f} CI=[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] t={r['t']:+.2f} p={r['p']:.2e}")
print("\n=== DECOUPLING (interaction with MW; zipf_x_mw>0 and length_x_mw<0 = attenuation) ===")
for term in ["zipf_x_mw", "length_x_mw"]:
    r = report["real"][term]
    print(f"  {term:12} mean={r['mean']:+.4f} CI=[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] t={r['t']:+.2f} p={r['p']:.2e} frac+={r['frac_positive']:.2f}")
print("\n=== SHUFFLE control (should be ~0) ===")
for term in ["zipf_x_mw", "length_x_mw"]:
    r = report["shuffle"][term]
    print(f"  {term:12} mean={r['mean']:+.4f} CI=[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] p={r['p']:.2e}")
print(f"\nsubjects included: {report['real']['n_subjects']} / 44")
