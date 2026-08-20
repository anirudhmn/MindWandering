#!/usr/bin/env python3
"""G0b — rebuild skipping with the standard scan-path operationalisation.

A word counts as SKIPPED only if the reader's first-pass scan path stepped over it in a
single forward saccade of at most G intervening words. Words that fall inside a large
positional gap were never demonstrably scanned (gaze off text, mapping loss, or a line/page
transition) and are EXCLUDED rather than scored as skips.

Emits words_traversal.parquet with: skipped, is_mw (from the bounding fixations), gap size,
and the state agreement flag.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ART, RES, load_words, load_fix, boot_ci, fmt

GAPS = [1, 2, 3, 4, 6, 10, 10 ** 6]
G_PRIMARY = 4

w = load_words()
f = load_fix().sort_values(["subject", "run", "tStart"])

wf = w.set_index(["subject", "run", "pos"])[["zipf", "surprisal", "length", "page", "story", "word_key"]]

rows = []
gap_hist = {0: [], 1: []}
for (s, r), g in f.groupby(["subject", "run"], sort=False):
    pos = g["pos"].to_numpy()
    mw = g["is_mw"].to_numpy().astype(int)
    mwf = g["mw_frac"].to_numpy()
    # fixated words: state = state of first fixation on that word
    first = {}
    for p, m, mf in zip(pos, mw, mwf):
        if p not in first:
            first[p] = (m, mf)
    for p, (m, mf) in first.items():
        rows.append((s, r, p, 0, 0, m, mf, True))
    fixated = set(first)
    # forward steps -> stepped-over words
    for i in range(len(pos) - 1):
        a, b = pos[i], pos[i + 1]
        if b <= a + 1:
            continue
        gap = b - a - 1
        agree = mw[i] == mw[i + 1]
        st = mw[i]
        mf = 0.5 * (mwf[i] + mwf[i + 1])
        gap_hist[st].append(gap)
        for p in range(a + 1, b):
            if p in fixated:
                continue
            rows.append((s, r, p, 1, gap, st, mf, agree))

T = pd.DataFrame(rows, columns=["subject", "run", "pos", "skipped", "gap", "is_mw", "mw_frac", "state_agree"])
# a stepped-over word can appear under several steps (different runs of the same subject are
# distinct rows already); keep the smallest gap per subject-run-pos
T = T.sort_values(["subject", "run", "pos", "skipped", "gap"]).drop_duplicates(["subject", "run", "pos"], keep="first")
T = T.join(wf, on=["subject", "run", "pos"])
T = T.dropna(subset=["zipf", "length"])
T.to_parquet(ART / "words_traversal.parquet", index=False)

rep = {"n_rows": int(len(T)), "n_subjects": int(T.subject.nunique())}
print(f"traversal table: {T.shape}, subjects {T.subject.nunique()}")

print("\nGap-size distribution of forward steps (number of stepped-over words):")
for st, lab in [(0, "on-task"), (1, "MW")]:
    v = np.array(gap_hist[st])
    q = {f"<= {k}": float((v <= k).mean()) for k in [1, 2, 3, 4, 6, 10]}
    rep[f"gap_dist_{lab}"] = {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)), **q}
    print(f"  {lab:8s} n={len(v):7d} mean={v.mean():5.2f} median={np.median(v):.0f} " +
          "  ".join(f"{k}:{p:.3f}" for k, p in q.items()))

print("\nSkip rate by gap threshold G (analysis set = fixated + stepped-over with gap<=G):")
rep["skip_rate_by_G"] = {}
for G in GAPS:
    sub = T[(T.skipped == 0) | (T.gap <= G)]
    per = sub.groupby(["subject", "is_mw"]).skipped.mean().unstack()
    per = per.dropna()
    d = boot_ci((per[1] - per[0]).to_numpy())
    rep["skip_rate_by_G"][str(G)] = {"on_task": float(sub.loc[sub.is_mw == 0, "skipped"].mean()),
                                     "mw": float(sub.loc[sub.is_mw == 1, "skipped"].mean()),
                                     "diff": d, "n_rows": int(len(sub))}
    lab = "inf" if G > 1000 else str(G)
    print(f"  G={lab:>4s}  n={len(sub):7d}  on-task {sub.loc[sub.is_mw==0,'skipped'].mean():.3f} "
          f"MW {sub.loc[sub.is_mw==1,'skipped'].mean():.3f} " + fmt("diff", d, width=6))

# fraction of MW stepped-over words that sit in large (untraversed) gaps
sk = T[T.skipped == 1]
rep["frac_skips_in_large_gap"] = {int(k): float(v) for k, v in
                                  (sk.gap > G_PRIMARY).groupby(sk.is_mw).mean().to_dict().items()}
print(f"\nFraction of stepped-over words sitting in gaps > {G_PRIMARY} (untraversed/blackout): "
      f"on-task {rep['frac_skips_in_large_gap'][0]:.3f}  MW {rep['frac_skips_in_large_gap'][1]:.3f}")
rep["state_agree_rate"] = float(T.loc[T.skipped == 1, "state_agree"].mean())
print(f"Bounding-fixation MW-state agreement for stepped-over words: {rep['state_agree_rate']:.4f}")

json.dump(rep, open(RES / "g0b_traversal.json", "w"), indent=2, default=float)
print(f"\nwrote {ART/'words_traversal.parquet'}, {RES/'g0b_traversal.json'}")
