#!/usr/bin/env python3
"""Specificity controls for the two positive mechanism results.

C-control: regressions accumulate across a mind-wandering episode. Is that specific to
           mind-wandering, or does any stretch of reading show the same drift? Matched
           on-task pseudo-episodes provide the null. Time on task is also controlled.
D-control: the inter-subject alignment drop appears only in the component of gaze duration
           that word properties do not explain. Is that specific to the reported spans?
           Labels are shuffled within reader at matched count.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RES, COUP, boot_ci, fmt

RNG = np.random.default_rng(59)
BINS = [0, 2, 5, 10, 1e9]
LABS = ["0-2 s", "2-5 s", "5-10 s", ">10 s"]
rep = {}

f = pd.read_parquet(COUP / "reading_fixations.parquet").sort_values(["subject", "run", "tStart"])
g = f.groupby(["subject", "run"], sort=False)
f["refix"] = (g["pos"].shift(-1) == f["pos"]).astype(float)
f.loc[g["pos"].shift(-1).isna(), "refix"] = np.nan
f["logdur"] = np.log(f.fix_dur.clip(lower=1))
f["run_pos"] = g.cumcount() / g["pos"].transform("size")


def episode_trend(df, meas):
    """Per-reader linear trend of `meas` across time-since-segment-start bins."""
    d = df.dropna(subset=[meas]).copy()
    d["tb"] = pd.cut(d.t_since, BINS, labels=LABS, right=False)
    piv = d.groupby(["subject", "tb"], observed=True)[meas].mean().unstack().dropna()
    if piv.shape[0] < 10 or piv.shape[1] < 4:
        return None, None
    tr = np.array([np.polyfit(np.arange(piv.shape[1]), r.to_numpy(), 1)[0] for _, r in piv.iterrows()])
    return tr, piv


# --------------------------------------------------- real episodes
segs, lens = [], []
for (s, r), gg in f.groupby(["subject", "run"], sort=False):
    mw = gg.is_mw.to_numpy().astype(int)
    t = gg.tStart.to_numpy()
    i = 0
    while i < len(mw):
        if mw[i] == 1:
            j = i
            while j + 1 < len(mw) and mw[j + 1] == 1:
                j += 1
            h = gg.iloc[i:j + 1].copy()
            h["t_since"] = t[i:j + 1] - t[i]
            segs.append(h)
            lens.append(j - i + 1)
            i = j + 1
        else:
            i += 1
REAL = pd.concat(segs, ignore_index=True)
lens = np.array(lens)
print(f"{len(lens)} mind-wandering episodes, median {int(np.median(lens))} fixations, "
      f"median duration {REAL.groupby(['subject','run']).t_since.max().median():.1f} s")

print("\n=== C. Observed trends across the episode ===")
rep["C_observed"] = {}
for meas, lab in [("regression_out", "regression rate"), ("logdur", "fixation duration (log)"),
                  ("refix", "refixation rate")]:
    tr, piv = episode_trend(REAL, meas)
    r = boot_ci(tr)
    rep["C_observed"][meas] = {"trend": r, "bin_means": [float(piv[c].mean()) for c in piv.columns],
                               "n_readers": int(piv.shape[0])}
    print(f"  {lab:26s} " + "  ".join(f"{c}: {piv[c].mean():+.4f}" for c in piv.columns))
    print(fmt("     trend", r))

print("\n=== C-control. Matched on-task pseudo-episodes (300 draws) ===")
onmask = {k: np.where(v.is_mw.to_numpy() == 0)[0] for k, v in f.groupby(["subject", "run"], sort=False)}
groups = {k: v for k, v in f.groupby(["subject", "run"], sort=False)}
n_per_run = REAL.groupby(["subject", "run"]).size().index
null = {m: [] for m in ["regression_out", "logdur", "refix"]}
for it in range(300):
    rows = []
    for (s, r), gg in groups.items():
        idx = onmask[(s, r)]
        if len(idx) < 80:
            continue
        for _ in range(2):
            L = int(RNG.choice(lens))
            if len(idx) <= L + 1:
                continue
            st = int(RNG.integers(0, len(idx) - L))
            sel = idx[st:st + L]
            h = gg.iloc[sel].copy()
            tt = h.tStart.to_numpy()
            h["t_since"] = tt - tt[0]
            rows.append(h)
    if not rows:
        continue
    P = pd.concat(rows, ignore_index=True)
    for m in null:
        tr, _ = episode_trend(P, m)
        if tr is not None:
            null[m].append(float(tr.mean()))
rep["C_control"] = {}
for m, lab in [("regression_out", "regression rate"), ("logdur", "fixation duration"),
               ("refix", "refixation rate")]:
    v = np.array(null[m])
    obs = rep["C_observed"][m]["trend"]["mean"]
    p = float((np.abs(v) >= abs(obs)).mean())
    rep["C_control"][m] = {"observed": obs, "null_mean": float(v.mean()), "null_sd": float(v.std()),
                           "p_perm": p, "n_iter": int(len(v))}
    print(f"  {lab:26s} observed {obs:+.4f}  pseudo {v.mean():+.4f} +/- {v.std():.4f}  "
          f"p_perm = {p:.4f}")

print("\n=== C. Time-on-task control ===")
d = REAL.dropna(subset=["regression_out"]).copy()
d["tb"] = pd.cut(d.t_since, BINS, labels=LABS, right=False)
rp = d.groupby(["subject", "tb"], observed=True).run_pos.mean().unstack().dropna()
print("  mean position through the run by bin: " + "  ".join(f"{c}: {rp[c].mean():.3f}" for c in rp.columns))
resid = []
for s, gg in d.groupby("subject"):
    if gg.tb.nunique() < 4:
        continue
    X = np.column_stack([np.ones(len(gg)), gg.run_pos.to_numpy()])
    b = np.linalg.lstsq(X, gg.regression_out.to_numpy(), rcond=None)[0]
    gg = gg.assign(res=gg.regression_out.to_numpy() - X @ b)
    m = gg.groupby("tb", observed=True).res.mean()
    if len(m) == 4:
        resid.append(np.polyfit(np.arange(4), m.to_numpy(), 1)[0])
rep["C_timeontask_residual_trend"] = boot_ci(np.array(resid))
print(fmt("  trend after removing run position", rep["C_timeontask_residual_trend"]))

# --------------------------------------------------- D control
print("\n=== D-control. Shuffled labels for the residual alignment drop (300 draws) ===")
W = (f[f.is_firstpass == 1].groupby(["subject", "word_key"])
     .agg(GD=("fix_dur", "sum"), is_mw=("is_mw", "first")).reset_index())
wf = pd.read_parquet(COUP / "word_features.parquet")[["word_key", "zipf", "surprisal", "length"]]
W = W.merge(wf, on="word_key", how="left").dropna(subset=["zipf", "surprisal", "length"])
W["y"] = np.log(W.GD.clip(lower=1))
W["y"] = W.groupby("subject").y.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
# residualise on word properties within reader, pooling states so the definition is fixed
out = []
for s, gg in W.groupby("subject"):
    X = np.column_stack([np.ones(len(gg))] + [gg[c].to_numpy() for c in ["zipf", "surprisal", "length"]])
    b = np.linalg.lstsq(X, gg.y.to_numpy(), rcond=None)[0]
    out.append(pd.Series(gg.y.to_numpy() - X @ b, index=gg.index))
W["res"] = pd.concat(out).sort_index()

on = W[W.is_mw == 0]
tot = on.groupby("word_key").res.agg(["sum", "count"])
loo_by_subj = {}
for s, gg in W.groupby("subject"):
    own = gg[gg.is_mw == 0].groupby("word_key").res.agg(["sum", "count"])
    t = tot.join(own, rsuffix="_own", how="left").fillna(0)
    n_other = t["count"] - t["count_own"]
    loo = ((t["sum"] - t["sum_own"]) / n_other.replace(0, np.nan))[n_other >= 5]
    loo_by_subj[s] = loo


def isc_diff(labels_by_subj):
    diffs = []
    for s, gg in W.groupby("subject"):
        loo = loo_by_subj[s]
        sub = gg[gg.word_key.isin(loo.index)]
        lab = labels_by_subj[s]
        a = sub[lab == 1]
        b = sub[lab == 0]
        if len(a) < 60 or len(b) < 60:
            continue
        rs = []
        for _ in range(20):
            bb = b.iloc[RNG.choice(len(b), size=len(a), replace=False)]
            rs.append(np.corrcoef(bb.res, loo.loc[bb.word_key])[0, 1])
        diffs.append(np.corrcoef(a.res, loo.loc[a.word_key])[0, 1] - np.mean(rs))
    return np.array(diffs)


true_lab = {s: gg[gg.word_key.isin(loo_by_subj[s].index)].is_mw.to_numpy()
            for s, gg in W.groupby("subject")}
obs = isc_diff(true_lab)
r = boot_ci(obs)
rep["D_observed"] = {"n_readers": int(len(obs)), "diff": r}
print(f"  residual alignment, MW minus on-task, n={len(obs)}")
print(fmt("     difference", r))

null_d = []
for it in range(300):
    lab = {s: RNG.permutation(v) for s, v in true_lab.items()}
    null_d.append(float(isc_diff(lab).mean()))
nl = np.array(null_d)
p = float((np.abs(nl) >= abs(r["mean"])).mean())
rep["D_control"] = {"observed": r["mean"], "null_mean": float(nl.mean()),
                    "null_sd": float(nl.std()), "p_perm": p, "n_iter": int(len(nl))}
print(f"  observed {r['mean']:+.4f}  shuffled {nl.mean():+.4f} +/- {nl.std():.4f}  p_perm = {p:.4f}")

json.dump(rep, open(RES / "mechanism_controls.json", "w"), indent=2, default=float)
print(f"\nwrote {RES/'mechanism_controls.json'}")
