#!/usr/bin/env python3
"""Does the extra effort during mind-wandering scale with word difficulty?

S8 answers this by splitting words into quartiles of surprisal within reader and
fitting a trend across the four quartile differences. That is a gradient test, but it is
a gradient over bins, and Figure 4b showed what binning can hide. Here the same question
is asked continuously, and against three difficulty variables rather than one.

Per reader, per effort measure, per difficulty variable, by least squares on that
reader's fixations:

    measure ~ difficulty_z + is_mw + difficulty_z:is_mw

The interaction is what repair predicts to be non-zero. Group inference is the paper's
convention: a one-sample t test over readers with a 10,000-sample reader-level bootstrap
interval, and Holm across the nine interactions.

Effort measures follow the manuscript: log fixation duration on first-pass fixations, the
regression-out indicator, and refixation. Difficulty is surprisal (what S8 used), Zipf
frequency reversed so higher means harder, and length; taking all three means the answer
does not rest on how surprisal was computed.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import COUP, RES, boot_ci, holm  # noqa: E402

MIN_FIX = 200      # fixations per reader
MIN_STATE = 40     # per state, so a reader contributes only if both are estimable

MEASURES = [("log fixation duration", "log_dur", True),
            ("regressions out", "regression_out", False),
            ("refixations", "refix", False)]
DIFFS = [("surprisal", "hard_surprisal"), ("rarity", "hard_rarity"),
         ("length", "hard_length")]


def main():
    f = pd.read_parquet(COUP / "reading_fixations.parquet")
    f = f.sort_values(["subject", "run", "tStart"]).reset_index(drop=True)
    g = f.groupby(["subject", "run"], sort=False)
    f["refix"] = (g["pos"].shift(-1) == f["pos"]).astype(float)
    f.loc[g["pos"].shift(-1).isna(), "refix"] = np.nan
    f["log_dur"] = np.log(f.fix_dur.clip(lower=1))
    f["hard_surprisal"] = f.surprisal
    f["hard_rarity"] = -f.zipf
    f["hard_length"] = f.length

    rep = {"n_fixations": int(len(f)), "n_readers": int(f.subject.nunique()), "tests": {}}
    rows = []
    for mname, mcol, firstpass_only in MEASURES:
        for dname, dcol in DIFFS:
            betas, on_slopes = [], []
            for _, gs in f.groupby("subject"):
                d = gs[gs.is_firstpass == 1] if firstpass_only else gs
                d = d[[mcol, dcol, "is_mw"]].dropna()
                if (len(d) < MIN_FIX or d.is_mw.sum() < MIN_STATE
                        or (1 - d.is_mw).sum() < MIN_STATE):
                    continue
                x = ((d[dcol] - d[dcol].mean()) / d[dcol].std()).to_numpy()
                mw = d.is_mw.to_numpy(float)
                X = np.column_stack([np.ones(len(d)), x, mw, x * mw])
                b, *_ = np.linalg.lstsq(X, d[mcol].to_numpy(float), rcond=None)
                betas.append(b[3])
                on_slopes.append(b[1])
            r = boot_ci(betas)
            on = float(np.mean(on_slopes))
            # the interaction as a percentage of the on-task difficulty slope, so the
            # three measures can be read on one scale
            pct = [100 * v / on for v in (r["mean"], r["ci"][0], r["ci"][1])]
            if on < 0:
                pct = [pct[0], pct[2], pct[1]]
            r.update(on_task_slope=on, pct_of_on_task=pct[0], pct_ci=[pct[1], pct[2]])
            rep["tests"][f"{mname} x {dname}"] = r
            rows.append((mname, dname, r))
            print(f"{mname:<22} x {dname:<10} beta={r['mean']:+.5f} "
                  f"[{r['ci'][0]:+.5f},{r['ci'][1]:+.5f}] p={r['p']:.3f}  "
                  f"{pct[0]:+.1f}% of the on-task slope [{pct[1]:+.1f}, {pct[2]:+.1f}]")

    adj = holm([r["p"] for _, _, r in rows])
    for (m, d, r), a in zip(rows, adj):
        rep["tests"][f"{m} x {d}"]["p_holm"] = float(a)
    rep["n_surviving_holm"] = int(sum(a < 0.05 for a in adj))
    print(f"\n{rep['n_surviving_holm']} of {len(rows)} survive Holm")
    for (m, d, r), a in zip(rows, adj):
        if r["p"] < 0.05:
            print(f"  nominal: {m} x {d}  p={r['p']:.3f} p_holm={a:.3f} "
                  f"({r['pct_of_on_task']:+.1f}% of the on-task slope)")

    (RES / "difficulty_gradient.json").write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {RES / 'difficulty_gradient.json'}")


if __name__ == "__main__":
    main()
