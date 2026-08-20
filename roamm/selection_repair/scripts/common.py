#!/usr/bin/env python3
"""Shared helpers for the selection and repair analysis."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
COUP = ROOT / "roamm/artifacts/coupling"
IT = ROOT / "roamm/selection_repair"
ART = IT / "artifacts"
RES = IT / "results"
FIG = IT / "figures"
for d in (ART, RES, FIG):
    d.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(59)


def boot_ci(v, n=10000, rng=None):
    """Subject-level bootstrap mean CI."""
    rng = rng or RNG
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return dict(mean=float(np.mean(v)) if len(v) else np.nan, ci=[np.nan, np.nan],
                    t=np.nan, p=np.nan, n=int(len(v)), n_pos=0)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    bm = v[idx].mean(axis=1)
    t, p = stats.ttest_1samp(v, 0)
    return dict(mean=float(v.mean()),
                ci=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                t=float(t), p=float(p), n=int(len(v)), n_pos=int((v > 0).sum()))


def fmt(name, r, width=34):
    return (f"  {name:<{width}} {r['mean']:+.4f} "
            f"[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] t={r['t']:+.2f} "
            f"p={r['p']:.2g} {r['n_pos']}/{r['n']}")


def somers_d(x, y):
    """Somers' D = 2*AUC-1 for continuous x predicting binary y.

    Rank-based, hence invariant to monotone transforms of x and to the base rate of y.
    Positive => higher x associated with y=1.
    """
    x = np.asarray(x, float)
    y = np.asarray(y).astype(int)
    m = np.isfinite(x)
    x, y = x[m], y[m]
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 < 20 or n0 < 20:
        return np.nan
    r = stats.rankdata(x)
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return 2 * auc - 1


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for i, o in enumerate(order):
        val = (m - i) * p[o]
        running = max(running, val)
        adj[o] = min(1.0, running)
    return adj


def load_words():
    return pd.read_parquet(COUP / "reading_words.parquet")


def load_fix():
    return pd.read_parquet(COUP / "reading_fixations.parquet")


def add_bracket(words: pd.DataFrame, fix: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Flag words whose local region was demonstrably traversed.

    A word at reading position p is 'bracketed' if a mapped first-pass fixation exists at
    some position in [p-k, p-1] AND in [p+1, p+k] within the same subject-run. This removes
    tracking/mapping blackouts, which would otherwise be scored as skips.
    """
    out = []
    fixpos = {kk: set(g["pos"].to_numpy()) for kk, g in fix.groupby(["subject", "run"], sort=False)}
    for kk, g in words.groupby(["subject", "run"], sort=False):
        fp = fixpos.get(kk, set())
        if not fp:
            continue
        pos = g["pos"].to_numpy()
        mx = pos.max() + k + 2
        occ = np.zeros(mx + k + 2, bool)
        fpa = np.fromiter((p for p in fp if 0 <= p < len(occ)), int)
        occ[fpa] = True
        cum = np.concatenate([[0], np.cumsum(occ)])

        def any_in(lo, hi):  # inclusive counts on [lo,hi]
            lo = np.clip(lo, 0, len(occ) - 1)
            hi = np.clip(hi, 0, len(occ) - 1)
            return (cum[hi + 1] - cum[lo]) > 0

        left = any_in(pos - k, pos - 1)
        right = any_in(pos + 1, pos + k)
        gg = g.copy()
        gg["bracketed"] = left & right
        out.append(gg)
    return pd.concat(out, ignore_index=True)
