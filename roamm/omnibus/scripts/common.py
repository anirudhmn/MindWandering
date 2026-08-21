#!/usr/bin/env python3
"""Shared helpers for the omnibus model-based coupling test."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
COUP = ROOT / "roamm/artifacts/coupling"
IT = ROOT / "roamm/omnibus"
ART = IT / "artifacts"
RES = IT / "results"
for d in (ART, RES):
    d.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(66)

# candidate window: words at page positions pos-W .. pos+W around the fixated word
W = 20
NC = 2 * W + 1

TEXT = ["zipf", "length", "s_local", "gain_long", "gain_shuf"]


def boot_ci(v, n=10000, rng=None):
    """Reader-level bootstrap mean CI."""
    rng = rng or np.random.default_rng(66)
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return dict(mean=np.nan, ci=[np.nan, np.nan], t=np.nan, p=np.nan, n=len(v), n_pos=0,
                    sd=np.nan)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    bm = v[idx].mean(axis=1)
    t, p = stats.ttest_1samp(v, 0)
    return dict(mean=float(v.mean()),
                ci=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                t=float(t), p=float(p), n=int(len(v)), n_pos=int((v > 0).sum()),
                sd=float(bm.std()))


def boot_ratio(num, den, n=10000, rng=None):
    """Paired reader-level bootstrap of a ratio of means (retention)."""
    rng = rng or np.random.default_rng(66)
    num, den = np.asarray(num, float), np.asarray(den, float)
    ok = np.isfinite(num) & np.isfinite(den)
    num, den = num[ok], den[ok]
    k = len(num)
    draws = np.empty(n)
    for b in range(n):
        i = rng.integers(0, k, k)
        draws[b] = num[i].mean() / den[i].mean()
    return dict(retention=float(num.mean() / den.mean()),
                ci=[float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                one_sided_lower_95=float(np.percentile(draws, 5)),
                mde80_pct=float(100 * 2.802 * draws.std()), n=int(k))


def per_reader(v, subject, keep, subs, minn=30):
    """Mean of v within each reader over the rows in `keep`; NaN if too few rows."""
    out = []
    for s in subs:
        m = (subject == s) & keep & np.isfinite(v)
        out.append(v[m].mean() if m.sum() >= minn else np.nan)
    return np.array(out)


def group_means(v, codes, keep, ngroups, minn=30):
    """Per-reader means by bincount; fast enough to sit inside a permutation loop."""
    w = keep.astype(np.float64)
    c = np.bincount(codes, weights=w, minlength=ngroups)
    t = np.bincount(codes, weights=np.where(np.isfinite(v), v, 0.0) * w, minlength=ngroups)
    return np.where(c >= minn, t / np.maximum(c, 1), np.nan)


def demean(v, codes):
    m = np.bincount(codes, weights=v) / np.maximum(np.bincount(codes), 1)
    return v - m[codes]


def fe_ols(y, X, fes, cluster, iters=12):
    """OLS of y on X after absorbing the fixed effects in `fes`, cluster-robust SE.

    Fixed effects are absorbed by alternating within-group demeaning rather than by
    constructing dummies, which is what makes a word-level absorber affordable here.
    """
    yy = np.asarray(y, float).copy()
    XX = np.asarray(X, float).copy()
    for _ in range(iters):
        for fe in fes:
            yy = demean(yy, fe)
            for j in range(XX.shape[1]):
                XX[:, j] = demean(XX[:, j], fe)
    XtXi = np.linalg.pinv(XX.T @ XX)
    b = XtXi @ (XX.T @ yy)
    r = yy - XX @ b
    o = np.argsort(cluster)
    cs, Xs, rs = cluster[o], XX[o], r[o]
    bnd = np.r_[0, np.flatnonzero(np.diff(cs)) + 1, len(cs)]
    meat = np.zeros((XX.shape[1],) * 2)
    for k in range(len(bnd) - 1):
        g = Xs[bnd[k]:bnd[k + 1]].T @ rs[bnd[k]:bnd[k + 1]]
        meat += np.outer(g, g)
    se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    return float(b[0]), float(se[0]), float(2 * stats.norm.sf(abs(b[0] / se[0])))
