#!/usr/bin/env python3
"""Neural decoupling test: does the fixation-related potential's sensitivity to word
frequency attenuate during mind-wandering?

Mirror of coupling_behavioral.py, but the response is FRP ROI amplitude (uV) instead
of log fixation duration. Per subject:
    frp_roi ~ zipf_c + length_c + logdur_c + order_z
              + is_mw + zipf_c:is_mw + length_c:is_mw + logdur_c:is_mw + order_z:is_mw
The zipf_c:is_mw term is the change in the neural frequency slope during MW.
Group inference = subject-level one-sample tests + subject bootstrap; FDR across ROIs;
label-shuffle control.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path("roamm/artifacts/coupling")
RNG = np.random.default_rng(20260718)
ROIS = ["occ_P1", "occ_N1", "occ_P2", "cp_mid", "cp_N400", "front_late"]
PRIMARY = "occ_N1"

fix = pd.read_parquet(OUT / "fixations_frp.parquet")
wf = pd.read_parquet(OUT / "word_features.parquet")[["word_key", "length", "zipf", "clean"]]
df = fix.merge(wf, on="word_key", how="left")

# quality filters
df = df[df["frp_valid"] & (df["clean"].str.len() >= 1) & (df["zipf"] > 0)]
df = df[(df["fix_dur"] >= 50) & (df["fix_dur"] <= 1000)]
# artifact rejection on peak-to-peak (volts). Convert ROI amps + p2p to uV.
for r in ROIS:
    df["frp_" + r] = df["frp_" + r] * 1e6
df["p2p_uV"] = df["frp_p2p"] * 1e6
p2p_thresh = 150.0
n_before = len(df)
df = df[df["p2p_uV"] <= p2p_thresh]
print(f"epochs: {n_before} -> {len(df)} after p2p<={p2p_thresh}uV "
      f"({100*len(df)/n_before:.1f}% kept) | MW rate {df['is_mw'].mean():.4f}")
df["logdur"] = np.log(df["fix_dur"].to_numpy())

def fit_subject(sub, resp, mw):
    if mw.sum() < 40 or (1 - mw).sum() < 200:
        return None
    def c(x):
        x = x.to_numpy().astype(float); return x - x.mean()
    zipf_c, length_c, logdur_c = c(sub["zipf"]), c(sub["length"]), c(sub["logdur"])
    order = sub["fix_order"].to_numpy().astype(float)
    order_z = (order - order.mean()) / (order.std() + 1e-9)
    y = sub[resp].to_numpy().astype(float)
    ok = np.isfinite(y)
    X = np.column_stack([np.ones(len(sub)), zipf_c, length_c, logdur_c, order_z,
                         mw, zipf_c*mw, length_c*mw, logdur_c*mw, order_z*mw])
    if ok.sum() < 240:
        return None
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    names = ["intercept","zipf","length","logdur","order",
             "is_mw","zipf_x_mw","length_x_mw","logdur_x_mw","order_x_mw"]
    return dict(zip(names, beta))

def analyze(resp, shuffle=False):
    recs = []
    for subj, sub in df.groupby("subject"):
        mw = sub["is_mw"].to_numpy().astype(float)
        if shuffle:
            mw = RNG.permutation(mw)
        b = fit_subject(sub, resp, mw)
        if b is None:
            continue
        b["subject"] = subj
        recs.append(b)
    return pd.DataFrame(recs)

def group(vals):
    vals = vals[np.isfinite(vals)]
    t, p = stats.ttest_1samp(vals, 0.0)
    boot = np.array([RNG.choice(vals, len(vals), replace=True).mean() for _ in range(5000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(vals.mean()), "ci": [float(lo), float(hi)],
            "t": float(t), "p": float(p), "n": int(len(vals)),
            "frac_neg": float((vals < 0).mean())}

def fdr(pvals):
    p = np.asarray(pvals); order = np.argsort(p); m = len(p)
    q = np.empty(m); q[order] = (p[order] * m / (np.arange(m) + 1))
    # enforce monotonicity
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    out = np.empty(m); out[order] = np.clip(q_sorted, 0, 1)
    return out

report = {}
print("\n%-10s | on-task zipf slope (uV/zipf)        | MW interaction zipf_x_mw" % "ROI")
print("-" * 92)
rows = []
for roi in ROIS:
    res = analyze("frp_" + roi)
    ont = group(res["zipf"].to_numpy())
    inter = group(res["zipf_x_mw"].to_numpy())
    report[roi] = {"on_task_zipf": ont, "zipf_x_mw": inter, "n_subjects": len(res)}
    rows.append((roi, ont, inter))
    print(f"{roi:10} | mean={ont['mean']:+.4f} CI[{ont['ci'][0]:+.3f},{ont['ci'][1]:+.3f}] p={ont['p']:.1e} "
          f"| mean={inter['mean']:+.4f} CI[{inter['ci'][0]:+.3f},{inter['ci'][1]:+.3f}] p={inter['p']:.1e}")

# FDR across ROIs on the interaction
inter_p = [report[r]["zipf_x_mw"]["p"] for r in ROIS]
inter_q = fdr(inter_p)
for r, q in zip(ROIS, inter_q):
    report[r]["zipf_x_mw"]["q_fdr"] = float(q)

# shuffle control on primary ROI
sh = analyze("frp_" + PRIMARY, shuffle=True)
report["shuffle_primary"] = {"roi": PRIMARY, "zipf_x_mw": group(sh["zipf_x_mw"].to_numpy())}

print(f"\nPrimary ROI = {PRIMARY}")
print(f"  on-task zipf slope: {report[PRIMARY]['on_task_zipf']['mean']:+.4f} uV/zipf, "
      f"p={report[PRIMARY]['on_task_zipf']['p']:.2e}")
i = report[PRIMARY]["zipf_x_mw"]
print(f"  MW interaction:     {i['mean']:+.4f}, p={i['p']:.2e}, q_FDR={i['q_fdr']:.3f}, "
      f"frac subjects neg={i['frac_neg']:.2f}")
print(f"  shuffle interaction:{report['shuffle_primary']['zipf_x_mw']['mean']:+.4f}, "
      f"p={report['shuffle_primary']['zipf_x_mw']['p']:.2e}")

(OUT / "neural_coupling_report.json").write_text(json.dumps(report, indent=2) + "\n")
print("\nwrote neural_coupling_report.json")
