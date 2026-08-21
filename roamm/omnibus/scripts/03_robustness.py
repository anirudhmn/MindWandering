#!/usr/bin/env python3
"""Robustness of the omnibus retention (supplementary section S10).

The per-transition target read-out is heavy-tailed and retention is a ratio of means, so a
handful of extreme transitions could in principle set it. Retention is re-derived under eleven
exclusions and two robust estimators. Nothing is refitted; every variant reads the same frozen
per-transition values.

Also decomposes the read-out by kind of movement. Retention looks higher for forward saccades
than for regressions and refixations, but that is a difference in the denominator: forward
moves have more text-driven signal to begin with. The within-reader difference of differences
is the test that settles it.

Reads policy_D_real.npz; writes robustness.json.
"""
from __future__ import annotations
import json
import numpy as np
from scipy import stats
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ART, RES, boot_ci, boot_ratio

z = np.load(ART / "policy_D_real.npz", allow_pickle=True)
sj, mw = z["subject"], z["mw"]
subs = np.unique(sj)
Dt = np.asarray(z["Dtgt"], float)
Dd = np.asarray(z["Ddur"], float)
kind = z["kind"].astype(str)
delta = z["target_idx"] - 20
fixdur = z["fix_dur"]
blk = z["block_id"]

# episode interiors: drop the first and last two transitions of every contiguous state block
interior = np.ones(len(blk), bool)
for b in np.unique(blk):
    i = np.flatnonzero(blk == b)
    if len(i) <= 6:
        interior[i] = False
    else:
        interior[i[:2]] = False
        interior[i[-2:]] = False


def within_reader(v, lo, hi, mode):
    """Trim or winsorize at within-reader percentiles."""
    out = v.copy()
    keep = np.ones(len(v), bool)
    for s in subs:
        i = np.flatnonzero(sj == s)
        a, b = np.percentile(v[i], [lo, hi])
        if mode == "trim":
            keep[i] = (v[i] >= a) & (v[i] <= b)
        else:
            out[i] = np.clip(v[i], a, b)
    return out, keep


def summarise(v, keep, est="mean", minn=30):
    f = np.isfinite(v) & keep
    agg = np.mean if est == "mean" else np.median
    don, dmw = [], []
    for s in subs:
        a = v[(sj == s) & (mw == 0) & f]
        b = v[(sj == s) & (mw == 1) & f]
        don.append(agg(a) if len(a) >= minn else np.nan)
        dmw.append(agg(b) if len(b) >= minn else np.nan)
    don, dmw = np.array(don), np.array(dmw)
    ok = np.isfinite(don) & np.isfinite(dmw)
    r = boot_ratio(dmw[ok], don[ok])
    r.update(n=int(f.sum()), n_readers=int(ok.sum()),
             D_on=float(don[ok].mean()), D_mw=float(dmw[ok].mean()))
    return r


ALL = np.ones(len(sj), bool)
tr1, k1 = within_reader(Dt, 1, 99, "trim")
tr5, k5 = within_reader(Dt, 5, 95, "trim")
w1, _ = within_reader(Dt, 1, 99, "winsor")
w5, _ = within_reader(Dt, 5, 95, "winsor")

VARIANTS = [
    ("all transitions", Dt, ALL, "mean"),
    ("trimmed outside 1-99 pct", Dt, k1, "mean"),
    ("trimmed outside 5-95 pct", Dt, k5, "mean"),
    ("winsorized at 1/99 pct", w1, ALL, "mean"),
    ("winsorized at 5/95 pct", w5, ALL, "mean"),
    ("median per reader", Dt, ALL, "median"),
    ("saccades <= 10 words", Dt, np.abs(delta) <= 10, "mean"),
    ("saccades <= 5 words", Dt, np.abs(delta) <= 5, "mean"),
    ("fixations 80-800 ms", Dt, (fixdur >= 80) & (fixdur <= 800), "mean"),
    ("episode interiors only", Dt, interior, "mean"),
    ("forward saccades only", Dt, kind == "forward", "mean"),
]

out = {"variants": {}, "distribution": dict(
    sd=float(np.nanstd(Dt)), mean=float(np.nanmean(Dt)), median=float(np.nanmedian(Dt)))}
print(f"{'variant':28s} {'n':>8s} {'D_on':>8s} {'retention':>10s} {'95% CI':>18s}")
for name, v, keep, est in VARIANTS:
    r = summarise(v, keep, est)
    out["variants"][name] = r
    print(f"{name:28s} {r['n']:8d} {r['D_on']:+8.4f} {r['retention']:10.3f} "
          f"[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]")

out["duration_head"] = {"all transitions": summarise(Dd, ALL, "mean"),
                        "trimmed outside 1-99 pct": summarise(Dd, within_reader(Dd, 1, 99, "trim")[1], "mean")}

# by kind of movement, and the difference of differences that interprets it
by_kind = {}
for lab, keep in [("forward", kind == "forward"), ("regression", kind == "regression"),
                  ("refixation", kind == "refixation"), ("non_forward", kind != "forward")]:
    by_kind[lab] = summarise(Dt, keep, "mean")
    r = by_kind[lab]
    print(f"{lab:28s} {r['n']:8d} {r['D_on']:+8.4f} {r['retention']:10.3f} "
          f"[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]")

fw = kind == "forward"
rows = []
for s in subs:
    m = sj == s
    cells = {}
    for lab, k in [("fw", fw), ("nf", ~fw)]:
        for st in (0, 1):
            v = Dt[m & k & (mw == st)]
            cells[(lab, st)] = v.mean() if len(v) >= 30 else np.nan
    if all(np.isfinite(list(cells.values()))):
        rows.append([cells[("fw", 1)] - cells[("fw", 0)], cells[("nf", 1)] - cells[("nf", 0)]])
A = np.array(rows)
dd = A[:, 1] - A[:, 0]
b = boot_ci(dd)
t, p = stats.ttest_1samp(dd, 0)
out["by_kind"] = by_kind
out["difference_of_differences"] = dict(
    shortfall_forward=float(A[:, 0].mean()), shortfall_non_forward=float(A[:, 1].mean()),
    delta=b["mean"], ci=b["ci"], t=float(t), p=float(p), n_readers=int(len(A)),
    n_larger_non_forward=int((dd < 0).sum()), mde80=float(2.802 * b["sd"]))
print(f"\nshortfall in bits: forward {A[:,0].mean():+.4f}  non-forward {A[:,1].mean():+.4f}")
print(f"difference of differences {b['mean']:+.4f} [{b['ci'][0]:+.4f},{b['ci'][1]:+.4f}] "
      f"t={t:.2f} p={p:.4f}  {int((dd<0).sum())}/{len(dd)} readers")

(RES / "robustness.json").write_text(json.dumps(out, indent=2))
print("wrote", RES / "robustness.json")
