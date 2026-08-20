#!/usr/bin/env python3
"""Neural coupling under the same within-token identification, plus formal equivalence
(TOST) and Bayes factors for every null this analysis relies on."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy import stats, integrate
from common import ART, RES, COUP, boot_ci

PROPS = ["zipf", "surprisal", "length"]
rep = {}


def bf01_ttest(d, r=0.7071):
    """JZS Bayes factor in favour of H0 for a one-sample t test."""
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    n = len(d); t = d.mean() / (d.std(ddof=1) / np.sqrt(n)); v = n - 1

    def num(g):
        return ((1 + n * g) ** -0.5
                * (1 + t ** 2 / ((1 + n * g) * v)) ** (-(v + 1) / 2)
                * (r ** 2 / (2 * np.pi)) ** 0.5 * g ** -1.5 * np.exp(-r ** 2 / (2 * g)))

    bf10 = integrate.quad(num, 1e-12, np.inf, limit=300)[0] / (1 + t ** 2 / v) ** (-(v + 1) / 2)
    return dict(t=float(t), df=int(v), BF10=float(bf10), BF01=float(1 / bf10))


def pct_ci(d, base, n=10000):
    """Bootstrap CI of the MW-minus-on-task difference, expressed as % of the on-task effect."""
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    rng = np.random.default_rng(59)
    bm = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1) / abs(base) * 100
    return dict(mean=float(d.mean() / abs(base) * 100),
                ci=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))], n=int(len(d)))


def tost(d, bound):
    """Two one-sided tests against +/-bound. Returns the larger (governing) p."""
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    n = len(d); se = d.std(ddof=1) / np.sqrt(n)
    t_lo = (d.mean() + bound) / se; t_hi = (d.mean() - bound) / se
    p_lo = stats.t.sf(t_lo, n - 1); p_hi = stats.t.cdf(t_hi, n - 1)
    return dict(bound=float(bound), mean=float(d.mean()),
                p_lower=float(p_lo), p_upper=float(p_hi), p_tost=float(max(p_lo, p_hi)),
                equivalent=bool(max(p_lo, p_hi) < 0.05))


# ------------------------------------------------------------------ neural within-token
fx = pd.read_parquet(COUP / "fixations_frp.parquet")
wf = pd.read_parquet(COUP / "word_features.parquet")[["word_key", "zipf", "surprisal", "length"]]
fp = fx[(fx.is_firstpass == 1) & (fx.frp_valid)]
W = (fp.sort_values(["subject", "word_key", "fix_order"])
       .groupby(["subject", "word_key"])
       .agg(occ_N1=("frp_occ_N1", "first"), cp_N400=("frp_cp_N400", "first"),
            is_mw=("is_mw", "first"), GD=("fix_dur", "sum"))
       .reset_index().merge(wf, on="word_key", how="left").dropna(subset=PROPS))
for c in ["occ_N1", "cp_N400"]:
    W[c] = W[c] * 1e6                      # volts -> microvolts
    W = W[np.abs(W[c]) < 50]               # drop gross artefact epochs
for p in PROPS:
    W[f"z{p}"] = (W[p] - W[p].mean()) / W[p].std()
W["logGD"] = np.log(W.GD.clip(lower=1))
print(f"neural word rows {len(W)}, tokens {W.word_key.nunique()}, subjects {W.subject.nunique()}, "
      f"MW {W.is_mw.mean():.4f}")

import sys
sys.path.insert(0, ".")
from importlib import import_module
twoway_fe = import_module("04_g5_g6_duration".replace("04_", "_04_")) if False else None


def _twoway_fe(df, y, xcols, fe1="word_key", fe2="subject", tol=1e-9, maxit=200):
    cols = [y] + xcols
    M = df[cols].to_numpy(float).copy()
    g1 = df[fe1].astype("category").cat.codes.to_numpy()
    g2 = df[fe2].astype("category").cat.codes.to_numpy()
    n1, n2 = g1.max() + 1, g2.max() + 1
    c1 = np.bincount(g1, minlength=n1).astype(float)
    c2 = np.bincount(g2, minlength=n2).astype(float)
    for _ in range(maxit):
        prev = M[:, 0].copy()
        for gi, ci, ni in ((g1, c1, n1), (g2, c2, n2)):
            for j in range(M.shape[1]):
                s = np.bincount(gi, weights=M[:, j], minlength=ni)
                M[:, j] -= (s / ci)[gi]
        if np.max(np.abs(M[:, 0] - prev)) < tol:
            break
    Y, X = M[:, 0], M[:, 1:]
    XtX = X.T @ X; XtXi = np.linalg.inv(XtX)
    beta = XtXi @ (X.T @ Y); resid = Y - X @ beta
    meat = np.zeros_like(XtX)
    for s in np.unique(g2):
        m = g2 == s
        u = (X[m] * resid[m][:, None]).sum(axis=0)
        meat += np.outer(u, u)
    nc = len(np.unique(g2))
    V = XtXi @ meat @ XtXi * nc / (nc - 1)
    se = np.sqrt(np.diag(V)); t = beta / se
    return {c: dict(beta=float(b), se=float(s), t=float(tt),
                    p=float(2 * stats.t.sf(abs(tt), nc - 1)))
            for c, b, s, tt in zip(xcols, beta, se, t)}


print("\n=== Neural coupling, two-way (token x subject) FE ===")
rep["neural_FE"] = {}
for ycol, prop in [("occ_N1", "zipf"), ("cp_N400", "surprisal")]:
    d = W.dropna(subset=[ycol]).copy()
    d["mw"] = d.is_mw.astype(float)
    d[f"mw_x_{prop}"] = d["mw"] * d[f"z{prop}"]
    r = _twoway_fe(d, ycol, ["mw", f"mw_x_{prop}"])
    rep["neural_FE"][ycol] = r
    print(f"  {ycol:8s} (n={len(d)})  " +
          "  ".join(f"{k}: beta={v['beta']:+.4f} uV se={v['se']:.4f} p={v['p']:.3g}" for k, v in r.items()))

# per-subject neural slopes for the equivalence test
rows = []
for s, g in W.groupby("subject"):
    rec = {"subject": s}; ok = True
    for st, tag in [(0, "on"), (1, "mw")]:
        gs = g[g.is_mw == st]
        if len(gs) < 300:
            ok = False; break
        rec[f"occ_{tag}"] = float(np.polyfit(gs.zzipf, gs.occ_N1, 1)[0])
        rec[f"n400_{tag}"] = float(np.polyfit(gs.zsurprisal, gs.cp_N400, 1)[0])
    if ok:
        rows.append(rec)
N = pd.DataFrame(rows)
print(f"\n  per-subject neural slopes, n={len(N)}: "
      f"occ_N1~zipf on={N.occ_on.mean():+.4f} mw={N.occ_mw.mean():+.4f} | "
      f"cp_N400~surp on={N.n400_on.mean():+.4f} mw={N.n400_mw.mean():+.4f}")
rep["neural_persubject"] = {
    "occ_zipf": {"on": float(N.occ_on.mean()), "mw": float(N.occ_mw.mean()),
                 "diff": boot_ci((N.occ_mw - N.occ_on).to_numpy())},
    "n400_surp": {"on": float(N.n400_on.mean()), "mw": float(N.n400_mw.mean()),
                  "diff": boot_ci((N.n400_mw - N.n400_on).to_numpy())}}

# ------------------------------------------------------------------ equivalence + BF
print("\n=== Equivalence (TOST, SESOI = 20% of the on-task effect) and Bayes factors ===")
rep["equivalence"] = {}

D = pd.read_csv(ART / "somersD_primary.csv")
for p in PROPS:
    d = (D[f"{p}_mw"] - D[f"{p}_on"]).to_numpy()
    base = abs(D[f"{p}_on"].mean())
    rep["equivalence"][f"skip_selectivity_{p}"] = {
        "on_task_effect": float(D[f"{p}_on"].mean()),
        "pct": pct_ci(d, D[f"{p}_on"].mean()),
        "tost_20pct": tost(d, 0.20 * base), "tost_10pct": tost(d, 0.10 * base),
        "bf": bf01_ttest(d)}

W2 = W.copy()
for p in ["zipf", "surprisal"]:
    rows = []
    for s, g in W2.groupby("subject"):
        ge, gd = g[g.is_mw == 0], g[g.is_mw == 1]
        if len(ge) < 200 or len(gd) < 80:
            continue
        rows.append((np.polyfit(ge[f"z{p}"], ge.logGD, 1)[0], np.polyfit(gd[f"z{p}"], gd.logGD, 1)[0]))
    a = np.array(rows)
    d = a[:, 1] - a[:, 0]
    rep["equivalence"][f"duration_{p}"] = {
        "on_task_effect": float(a[:, 0].mean()),
        "pct": pct_ci(d, a[:, 0].mean()),
        "tost_20pct": tost(d, 0.20 * abs(a[:, 0].mean())), "bf": bf01_ttest(d)}

for k in ["occ_zipf", "n400_surp"]:
    on = N["occ_on" if k == "occ_zipf" else "n400_on"].to_numpy()
    mw = N["occ_mw" if k == "occ_zipf" else "n400_mw"].to_numpy()
    rep["equivalence"][f"neural_{k}"] = {
        "on_task_effect": float(on.mean()),
        "pct": pct_ci(mw - on, on.mean()),
        "tost_20pct": tost(mw - on, 0.20 * abs(on.mean())),
        "tost_50pct": tost(mw - on, 0.50 * abs(on.mean())), "bf": bf01_ttest(mw - on)}

for k, v in rep["equivalence"].items():
    t20 = v["tost_20pct"]
    pc = v["pct"]
    print(f"  {k:26s} on-task={v['on_task_effect']:+.4f}  Δ={pc['mean']:+.1f}% "
          f"[{pc['ci'][0]:+.1f},{pc['ci'][1]:+.1f}]  TOST+/-20%: p={t20['p_tost']:.3g} "
          f"{'EQUIVALENT' if t20['equivalent'] else 'not established'}   BF01={v['bf']['BF01']:.2f}")

json.dump(rep, open(RES / "neural_equivalence.json", "w"), indent=2, default=float)
print(f"\nwrote {RES/'neural_equivalence.json'}")
