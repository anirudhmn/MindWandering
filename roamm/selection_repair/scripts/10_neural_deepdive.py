#!/usr/bin/env python3
"""Is the neural null real, or is the neural test simply uninformative?

Works directly on the deconvolved regression-ERP betas
(44 subjects x 8 predictors x 155 lags x 64 channels) rather than on the ROI summaries.

1. Time-resolved interaction kernels with cluster permutation over the whole epoch.
2. Sensitivity: what interaction magnitude could this design have detected?
3. Overlap-corrected versus single-trial estimates, to locate the disagreement.
4. Item-level analysis, which sidesteps the per-reader trial-count problem.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy import stats
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ART, RES, COUP, boot_ci, fmt

RNG = np.random.default_rng(59)
B = np.load(COUP / "rerp_betas.npy")            # subj x pred x lag x chan
meta = json.load(open(COUP / "rerp_meta.json"))
PRED = meta["pred_names"]; CH = meta["channels"]; T = np.array(meta["lags_ms"])
OCC = ['PO7','PO8','PO3','PO4','O1','O2','Oz','POz','P7','P8','P9','P10','Iz']
CP  = ['Cz','CPz','Pz','CP1','CP2','C1','C2','P1','P2']
occ = [CH.index(c) for c in OCC]; cp = [CH.index(c) for c in CP]
ip = {p: PRED.index(p) for p in PRED}
rep = {}
print(f"betas {B.shape}; predictors {PRED}")


def roi_ts(pred, chans):
    return B[:, ip[pred], :, :][:, :, chans].mean(axis=2)      # subj x lag


def cluster_test(X, tmask=None, n=5000):
    """One-sample cluster test over time with sign flipping. X = subj x lag."""
    t = stats.ttest_1samp(X, 0, axis=0).statistic
    thr = stats.t.ppf(0.975, X.shape[0] - 1)
    m = np.abs(t) > thr
    if tmask is not None:
        m = m & tmask
    clusters, cur = [], []
    for i, v in enumerate(m):
        if v:
            cur.append(i)
        elif cur:
            clusters.append(cur); cur = []
    if cur:
        clusters.append(cur)
    if not clusters:
        return 1.0, None, 0.0
    masses = [abs(t[c].sum()) for c in clusters]
    obs = max(masses); best = clusters[int(np.argmax(masses))]
    null = np.empty(n)
    for k in range(n):
        s = RNG.choice([-1, 1], size=(X.shape[0], 1))
        tt = stats.ttest_1samp(X * s, 0, axis=0).statistic
        mm = np.abs(tt) > thr
        if tmask is not None:
            mm = mm & tmask
        cc, cu, best_m = [], [], 0.0
        for i, v in enumerate(mm):
            if v:
                cu.append(i)
            elif cu:
                best_m = max(best_m, abs(tt[cu].sum())); cu = []
        if cu:
            best_m = max(best_m, abs(tt[cu].sum()))
        null[k] = best_m
    return float((null >= obs).mean()), (T[best[0]], T[best[-1]]), float(obs)


print("\n=== 1. Time-resolved interaction kernels, whole epoch ===")
rep["timeresolved"] = {}
for pred, chans, lab, win in [("zipf:mw", occ, "frequency x MW (occipital)", (150, 290)),
                              ("surprisal:mw", cp, "surprisal x MW (centroparietal)", (300, 450))]:
    X = roi_ts(pred, chans) * (1e6 if np.nanmax(np.abs(roi_ts(pred, chans))) < 1e-3 else 1)
    p_all, win_all, mass = cluster_test(X)
    tm = (T >= win[0]) & (T <= win[1])
    p_win, win_w, _ = cluster_test(X, tmask=tm)
    rep["timeresolved"][pred] = {"cluster_p_whole_epoch": p_all, "cluster_window": win_all,
                                 "cluster_p_within_apriori_window": p_win}
    print(f"  {lab}: whole-epoch cluster p={p_all:.3f} (window {win_all}); "
          f"within a-priori window p={p_win:.3f}")

print("\n=== 2. Sensitivity of the neural interaction test ===")
rep["sensitivity"] = {}
for base_p, int_p, chans, win, lab in [
        ("zipf", "zipf:mw", occ, (150, 290), "frequency, occipital"),
        ("surprisal", "surprisal:mw", cp, (300, 450), "surprisal, centroparietal")]:
    tm = (T >= win[0]) & (T <= win[1])
    sc = 1e6 if np.nanmax(np.abs(B[:, ip[base_p]])) < 1e-3 else 1
    b = roi_ts(base_p, chans)[:, tm].mean(axis=1) * sc
    i = roi_ts(int_p, chans)[:, tm].mean(axis=1) * sc
    n = len(b)
    se = i.std(ddof=1) / np.sqrt(n)
    mde = 2.87 * se                                  # ~80% power, two-sided alpha .05
    ratio = i / abs(b.mean())
    r = boot_ci(ratio * 100)
    rep["sensitivity"][lab] = {"base_uV_per_SD": float(b.mean()),
                               "interaction_uV": float(i.mean()), "se": float(se),
                               "MDE_uV": float(mde), "MDE_pct_of_base": float(mde / abs(b.mean()) * 100),
                               "interaction_pct_of_base": r,
                               "p": float(stats.ttest_1samp(i, 0).pvalue)}
    print(f"  {lab}: base {b.mean():+.4f} uV/SD; interaction {i.mean():+.4f} uV "
          f"(p={stats.ttest_1samp(i,0).pvalue:.3f})")
    print(f"    interaction as % of base: {r['mean']:+.1f}% [{r['ci'][0]:+.1f},{r['ci'][1]:+.1f}]")
    print(f"    smallest detectable interaction at 80% power: {mde:.4f} uV = "
          f"{mde/abs(b.mean())*100:.0f}% of the base effect")

print("\n=== 3. Overlap-corrected versus single-trial estimates ===")
fx = pd.read_parquet(COUP / "fixations_frp.parquet")
wf = pd.read_parquet(COUP / "word_features.parquet")[["word_key", "zipf", "surprisal", "length"]]
d = fx[(fx.is_firstpass == 1) & (fx.frp_valid)].merge(wf, on="word_key", how="left").dropna(
    subset=["zipf", "surprisal", "frp_occ_N1", "frp_cp_N400"])
for c in ["frp_occ_N1", "frp_cp_N400"]:
    d[c] = d[c] * 1e6
d = d[(d.frp_occ_N1.abs() < 50) & (d.frp_cp_N400.abs() < 50)]
rep["single_trial"] = {}
for ycol, xcol, lab in [("frp_occ_N1", "zipf", "occ ~ frequency"),
                        ("frp_cp_N400", "surprisal", "N400 ~ surprisal")]:
    rows = []
    for s, g in d.groupby("subject"):
        ge, gd = g[g.is_mw == 0], g[g.is_mw == 1]
        if len(ge) < 300 or len(gd) < 100:
            continue
        z = lambda x: (x - x.mean()) / x.std()
        rows.append((np.polyfit(z(ge[xcol]), ge[ycol], 1)[0],
                     np.polyfit(z(gd[xcol]), gd[ycol], 1)[0], len(gd)))
    a = np.array(rows)
    ratio = a[:, 1] / abs(a[:, 0].mean())
    r = boot_ci((a[:, 1] - a[:, 0]) / abs(a[:, 0].mean()) * 100)
    rep["single_trial"][lab] = {"n_readers": int(len(a)), "on": float(a[:, 0].mean()),
                                "mw": float(a[:, 1].mean()), "change_pct": r,
                                "median_mw_trials": float(np.median(a[:, 2]))}
    print(f"  {lab}: {len(a)} readers, median {int(np.median(a[:,2]))} MW trials; "
          f"on {a[:,0].mean():+.4f} mw {a[:,1].mean():+.4f}")
    print(fmt("    change as % of base", r))

print("\n=== 4. Item-level test (averages over readers, sidesteps per-reader trial counts) ===")
rep["item_level"] = {}
for ycol, xcol, lab in [("frp_occ_N1", "zipf", "occ ~ frequency"),
                        ("frp_cp_N400", "surprisal", "N400 ~ surprisal")]:
    it = d.groupby(["word_key", "is_mw"]).agg(y=(ycol, "mean"), n=(ycol, "size"),
                                              x=(xcol, "first")).reset_index()
    it = it[it.n >= 5]
    piv = it.pivot(index="word_key", columns="is_mw", values="y")
    xs = it.groupby("word_key").x.first()
    both = piv.dropna()
    print(f"  {lab}: {len(both)} word tokens with >=5 readers in BOTH states")
    out = {}
    for st, tag in [(0, "on"), (1, "mw")]:
        sub = it[(it.is_mw == st)]
        sl, _ = np.polyfit((sub.x - sub.x.mean()) / sub.x.std(), sub.y, 1)
        out[tag] = float(sl)
    # bootstrap over items for the difference
    bs = []
    for _ in range(2000):
        idx = RNG.integers(0, len(both), len(both))
        w = both.index.to_numpy()[idx]
        xv = xs.loc[w].to_numpy(); xz = (xv - xv.mean()) / xv.std()
        s0 = np.polyfit(xz, both[0].to_numpy()[idx], 1)[0]
        s1 = np.polyfit(xz, both[1].to_numpy()[idx], 1)[0]
        bs.append((s1 - s0) / abs(s0) * 100)
    bs = np.array(bs)
    xv = xs.loc[both.index].to_numpy(); xz = (xv - xv.mean()) / xv.std()
    s0 = np.polyfit(xz, both[0].to_numpy(), 1)[0]; s1 = np.polyfit(xz, both[1].to_numpy(), 1)[0]
    rep["item_level"][lab] = {"n_items": int(len(both)), "slope_on": float(s0), "slope_mw": float(s1),
                              "change_pct": float((s1 - s0) / abs(s0) * 100),
                              "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                              "p_boot": float(2 * min((bs > 0).mean(), (bs < 0).mean()))}
    q = rep["item_level"][lab]
    print(f"    matched-item slopes: on {s0:+.4f}  MW {s1:+.4f}  change {q['change_pct']:+.1f}% "
          f"[{q['ci'][0]:+.1f},{q['ci'][1]:+.1f}]  p={q['p_boot']:.3f}")

json.dump(rep, open(RES / "neural_deepdive.json", "w"), indent=2, default=float)
print(f"\nwrote {RES/'neural_deepdive.json'}")
