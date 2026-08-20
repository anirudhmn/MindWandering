#!/usr/bin/env python3
"""Shared helpers for the localisation analysis (semantic importance tracking)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
COUP = ROOT / "roamm/artifacts/coupling"
IT = ROOT / "roamm/localisation"
ART, RES, FIG = IT / "artifacts", IT / "results", IT / "figures"
for d in (ART, RES, FIG):
    d.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(60)


def boot_ci(v, n=10000, rng=None):
    rng = rng or RNG
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return dict(mean=float(np.mean(v)) if len(v) else np.nan, ci=[np.nan, np.nan],
                    t=np.nan, p=np.nan, n=int(len(v)), n_pos=0, sd=np.nan)
    bm = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    t, p = stats.ttest_1samp(v, 0)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                ci=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                t=float(t), p=float(p), n=int(len(v)), n_pos=int((v > 0).sum()))


def fmt(name, r, w=36):
    return (f"  {name:<{w}} {r['mean']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] "
            f"t={r['t']:+.2f} p={r['p']:.3g} {r['n_pos']}/{r['n']}")


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); adj = np.empty_like(p); run = 0.0; m = len(p)
    for i, o in enumerate(order):
        run = max(run, (m - i) * p[o]); adj[o] = min(1.0, run)
    return adj


def absorb(M: np.ndarray, groups: list[np.ndarray], tol=1e-9, maxit=200) -> np.ndarray:
    """Within-transform M by alternating projections on several fixed-effect dimensions."""
    M = np.asarray(M, float).copy()
    if M.ndim == 1:
        M = M[:, None]
    codes = []
    for g in groups:
        _, c = np.unique(g, return_inverse=True)
        codes.append((c, c.max() + 1))
    if len(codes) == 1:
        c, k = codes[0]
        cnt = np.bincount(c, minlength=k).astype(float)
        for j in range(M.shape[1]):
            M[:, j] -= (np.bincount(c, weights=M[:, j], minlength=k) / cnt)[c]
        return M
    for _ in range(maxit):
        prev = M.copy()
        for c, k in codes:
            cnt = np.bincount(c, minlength=k).astype(float)
            for j in range(M.shape[1]):
                M[:, j] -= (np.bincount(c, weights=M[:, j], minlength=k) / cnt)[c]
        if np.max(np.abs(M - prev)) < tol:
            break
    return M


def ols_cluster(y, X, cluster, names=None):
    """OLS with cluster-robust (CR0) SE. y,X already within-transformed if FE are used."""
    y = np.asarray(y, float); X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    _, cc = np.unique(cluster, return_inverse=True)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in range(cc.max() + 1):
        m = cc == g
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    names = names or [f"x{i}" for i in range(X.shape[1])]
    out = {}
    for i, nm in enumerate(names):
        z = beta[i] / se[i] if se[i] > 0 else np.nan
        out[nm] = dict(beta=float(beta[i]), se=float(se[i]), z=float(z),
                       p=float(2 * stats.norm.sf(abs(z))),
                       ci=[float(beta[i] - 1.96 * se[i]), float(beta[i] + 1.96 * se[i])])
    out["_n"] = int(len(y)); out["_n_clusters"] = int(cc.max() + 1)
    return out


def z(s):
    s = np.asarray(s, float)
    return (s - np.nanmean(s)) / np.nanstd(s)


def mde(se, power=0.80, alpha=0.05):
    """Minimum detectable effect at two-sided alpha and given power."""
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se)


def load_word_measures():
    """(subject, word_key) reading measures joined to word-level importance + covariates."""
    W = pd.read_parquet(ART / "word_importance.parquet")
    fx = pd.read_parquet(COUP / "reading_fixations.parquet")
    fx = fx[fx["fix_dur"].between(50, 1000)]
    fp = fx[fx["is_firstpass"]]
    agg = fp.groupby(["subject", "word_key"], observed=True).agg(
        gaze_dur=("fix_dur", "sum"), n_fix_fp=("fix_dur", "size"),
        ffd=("fix_dur", "first"), mw_frac=("mw_frac", "mean"),
        is_mw=("is_mw", "max"), run=("run", "first"), pos=("pos", "first")).reset_index()
    # NB reading_fixations.parquet contains FIRST-PASS fixations only (is_firstpass is True on
    # every row, verified), so gaze duration here is the standard first-pass gaze duration and
    # no later-rereading measure can be built from it. Regressions are taken from the
    # regression_out flag (the saccade leaving this word went backwards).
    reg = fp.groupby(["subject", "word_key"], observed=True)["regression_out"].max().rename("regression_out").reset_index()
    D = agg.merge(reg, on=["subject", "word_key"], how="left")
    D["n_refix"] = D["n_fix_fp"] - 1
    D = D.merge(W, on="word_key", how="inner", validate="m:1")
    return D
