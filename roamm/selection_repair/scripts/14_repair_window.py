#!/usr/bin/env python3
"""G4 addendum — is the five-fixation lookahead the right window for a corrective return?

Script 03 counts a corrective return if any of the next five fixations lands on the word the
reader stepped over. Three questions that choice raises, answered here:

  1. Where do returns actually land?  (the empirical justification for K=5)
  2. Does the MW contrast depend on K?  (it does)
  3. Is a window counted in FIXATIONS matched across states?  (it is not, in two opposite
     directions — so the K dependence is exposure, not repair)

The fixation-count window is not opportunity-matched: the nominal five fixations spans about
twice as much wall-clock time during mind-wandering, while MW single-skips sit later on the
page and so have less page left. A window defined in SECONDS, restricted to events with at
least that much page remaining, controls both. Under it the contrast is null at every width.

Writes `results/repair_window.json`.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RES, COUP, boot_ci, fmt

K_PUBLISHED = 5
K_GRID = [1, 2, 3, 5, 8, 10, 20, 40, None]        # None = unbounded, same page
T_GRID = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]          # seconds

rep = {"K_published": K_PUBLISHED}

# `saccades.parquet` is the fully word-mapped fixation sequence (402,082 fixations, including
# the 69,445 refixations and 78,417 regressions that `reading_fixations.parquet` cannot
# express). Returns to a skipped word are visible in both, but only here is "the next five
# fixations" five actual fixations.
s = pd.read_parquet(COUP / "saccades.parquet").sort_values(["subject", "run", "tStart"])

rows = []
for (sub, run), g in s.groupby(["subject", "run"], sort=False):
    pos = g["pos"].to_numpy()
    page = g["page"].to_numpy()
    mw = g["is_mw"].to_numpy().astype(int)
    t = g["tStart"].to_numpy(float)
    n = len(pos)
    for i in range(n - 1):
        if pos[i + 1] != pos[i] + 2:          # exactly one word stepped over
            continue
        if mw[i] != mw[i + 1]:
            continue
        if page[i] != page[i + 1]:            # the step itself must stay on one page
            continue
        skipped = pos[i] + 1
        rest = np.arange(i + 2, n)
        rest = rest[page[rest] == page[i + 1]]
        if len(rest) == 0:
            continue
        hit = rest[pos[rest] == skipped]
        lag = int(np.flatnonzero(rest == hit[0])[0]) + 1 if len(hit) else np.inf
        dt = float(t[hit[0]] - t[i + 1]) if len(hit) else np.inf
        rows.append((sub, mw[i], lag, dt, len(rest), float(t[rest[-1]] - t[i + 1])))

C = pd.DataFrame(rows, columns=["subject", "is_mw", "lag", "dt", "n_left", "t_left"])
print(f"single-skip events: {len(C)}  (MW {int(C.is_mw.sum())})  subjects {C.subject.nunique()}")


def contrast(hit, keep=None):
    """Reader-level paired contrast, MW minus on-task, on a 0/1 event outcome."""
    d = C.assign(h=np.asarray(hit, float))
    if keep is not None:
        d = d[keep]
    per = d.groupby(["subject", "is_mw"]).h.mean().unstack().dropna()
    r = boot_ci((per[1] - per[0]).to_numpy())
    r["on_task"] = float(per[0].mean())
    r["mw"] = float(per[1].mean())
    r["rel_pct"] = float(100 * (per[1].mean() - per[0].mean()) / per[0].mean())
    r["n_events"] = int(len(d))
    return r


# ---------------- 1. where returns land ----------------
ret = C[np.isfinite(C.lag)]
lag = ret.lag.to_numpy()
rep["latency"] = {
    "ever_returns": float(len(ret) / len(C)),
    "median_lag_fixations": float(np.median(lag)),
    "median_latency_s": float(np.median(ret.dt)),
    "captured_by_K": {str(k): float(np.mean(lag <= k)) for k in [1, 2, 3, 5, 8, 10, 20, 40]},
}
print(f"\nreturns to the skipped word: {rep['latency']['ever_returns']:.3f} of events ever "
      f"return on the same page")
print(f"  median latency {rep['latency']['median_lag_fixations']:.0f} fixations / "
      f"{rep['latency']['median_latency_s']:.2f} s")
for k, v in rep["latency"]["captured_by_K"].items():
    print(f"  K={k:>3}: captures {v:.3f} of all returns")

# ---------------- 2. the K dependence ----------------
lag_all = C.lag.to_numpy()
rep["K_sweep"] = {}
print(f"\nMW contrast by fixation-count window:")
for k in K_GRID:
    key = "unbounded" if k is None else str(k)
    r = contrast(np.isfinite(lag_all) if k is None else (lag_all <= k))
    rep["K_sweep"][key] = r
    print(f"  K={key:>9} on {r['on_task']:.4f} MW {r['mw']:.4f} rel {r['rel_pct']:+6.1f}% "
          f"{fmt('', r, width=0)}")

# ---------------- 3. the window is not opportunity-matched ----------------
# (a) real duration of the nominal K=5 window, counting the fixations that carry no word
#     identity (off-text gaze and re-reading of already-visited words) back in.
a = pd.read_parquet(COUP / "all_fixations.parquet").sort_values(["subject", "run", "tStart"])
w = pd.read_parquet(COUP / "reading_words.parquet")
a["pos"] = (w.drop_duplicates("word_key").set_index("word_key")["pos"]
            .reindex(a.word_key).to_numpy())
span = []
for (sub, run), g in a.groupby(["subject", "run"], sort=False):
    pos = g["pos"].to_numpy()
    page = g["page"].to_numpy()
    mw = g["is_mw"].to_numpy().astype(int)
    t = g["tStart"].to_numpy(float)
    n = len(pos)
    mapped = np.flatnonzero(~np.isnan(pos))
    mp, mpg, mmw = pos[mapped], page[mapped], mw[mapped]
    for j in range(len(mapped) - 1):
        if mp[j + 1] != mp[j] + 2 or mmw[j] != mmw[j + 1] or mpg[j] != mpg[j + 1]:
            continue
        i_land = mapped[j + 1]
        i_end = mapped[j + 1 + K_PUBLISHED] if j + 1 + K_PUBLISHED < len(mapped) else n - 1
        span.append((sub, mmw[j], i_end - i_land, t[i_end] - t[i_land]))
S = pd.DataFrame(span, columns=["subject", "is_mw", "span_fix", "span_s"])
per = S.groupby(["subject", "is_mw"]).span_s.mean().unstack().dropna()
rep["window_width"] = {
    "span_s_on_task": float(S.loc[S.is_mw == 0, "span_s"].mean()),
    "span_s_mw": float(S.loc[S.is_mw == 1, "span_s"].mean()),
    "span_fix_on_task": float(S.loc[S.is_mw == 0, "span_fix"].mean()),
    "span_fix_mw": float(S.loc[S.is_mw == 1, "span_fix"].mean()),
    "diff": boot_ci((per[1] - per[0]).to_numpy()),
}
ww = rep["window_width"]
print(f"\nreal extent of the nominal {K_PUBLISHED}-fixation window:")
print(f"  on-task {ww['span_fix_on_task']:.2f} fixations / {ww['span_s_on_task']:.2f} s")
print(f"  MW      {ww['span_fix_mw']:.2f} fixations / {ww['span_s_mw']:.2f} s")
print(fmt("  window duration (MW-on), s", ww["diff"]))

# (b) opportunity remaining after the skip
rep["opportunity"] = {
    "n_left_median_on_task": float(C.loc[C.is_mw == 0, "n_left"].median()),
    "n_left_median_mw": float(C.loc[C.is_mw == 1, "n_left"].median()),
    "t_left_median_on_task": float(C.loc[C.is_mw == 0, "t_left"].median()),
    "t_left_median_mw": float(C.loc[C.is_mw == 1, "t_left"].median()),
}
op = rep["opportunity"]
print(f"\npage remaining after the skip (median): on-task {op['n_left_median_on_task']:.0f} "
      f"fixations / {op['t_left_median_on_task']:.1f} s   "
      f"MW {op['n_left_median_mw']:.0f} / {op['t_left_median_mw']:.1f} s")

# ---------------- 4. the matched test ----------------
dt_all = C.dt.to_numpy()
rep["time_matched"] = {}
print(f"\nMW contrast by TIME window, opportunity matched (event kept only if the page has at "
      f"least T s left):")
for T in T_GRID:
    r = contrast(dt_all <= T, keep=(C.t_left >= T))
    rep["time_matched"][str(T)] = r
    print(f"  T={T:5.1f}s on {r['on_task']:.4f} MW {r['mw']:.4f} rel {r['rel_pct']:+6.1f}% "
          f"{fmt('', r, width=0)}  kept {r['n_events'] / len(C):.3f}")

json.dump(rep, open(RES / "repair_window.json", "w"), indent=2, default=float)
print(f"\nwrote {RES / 'repair_window.json'}")
