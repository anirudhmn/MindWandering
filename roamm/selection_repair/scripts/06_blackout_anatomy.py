#!/usr/bin/env python3
"""Anatomy of the MW 'blackout': what is the apparent extra skipping actually made of?

For every step in the mapped first-pass scan path, measure the positional gap and the
elapsed off-word time between the end of one fixation and the start of the next. A genuine
skip costs one saccade (~20-60 ms). A gap that also costs hundreds of ms means gaze spent
real time away from mapped text. Page transitions are separated out, and the state-changing
steps (MW onset/offset moments) are examined separately rather than dropped.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RES, ART, COUP, boot_ci, fmt

f = pd.read_parquet(COUP / "reading_fixations.parquet").sort_values(["subject", "run", "tStart"])
rows = []
for (s, r), g in f.groupby(["subject", "run"], sort=False):
    pos = g["pos"].to_numpy(); t0 = g["tStart"].to_numpy()
    dur = g["fix_dur"].to_numpy() / 1000.0
    page = g["page"].to_numpy(); mw = g["is_mw"].to_numpy().astype(int)
    rows.append(pd.DataFrame({
        "subject": s, "run": r,
        "gap": pos[1:] - pos[:-1] - 1,
        "off_s": t0[1:] - (t0[:-1] + dur[:-1]),
        "fix_s": dur[:-1],
        "forward": pos[1:] > pos[:-1],
        "samepage": page[1:] == page[:-1],
        "mw_from": mw[:-1], "mw_to": mw[1:]}))
S = pd.concat(rows, ignore_index=True)
S["state"] = np.where(S.mw_from == S.mw_to, S.mw_from, -1)   # -1 = state-changing step
S["off_s"] = S.off_s.clip(lower=0)
rep = {}

print(f"all consecutive fixation pairs: {len(S)}")
print(f"  same page {S.samepage.mean():.3f} | forward {S.forward.mean():.3f} | "
      f"state-changing {(S.state==-1).mean():.4f}")

# ---- where do the stepped-over words live? ----
F = S[S.forward & (S.gap > 0)].copy()
F["words"] = F.gap
tot = F.groupby(F.state.map({0: "on-task", 1: "MW", -1: "transition"})).words.sum()
big_same = F[(F.gap > 4) & F.samepage].groupby(
    F.state.map({0: "on-task", 1: "MW", -1: "transition"})).words.sum()
big_cross = F[(F.gap > 4) & ~F.samepage].groupby(
    F.state.map({0: "on-task", 1: "MW", -1: "transition"})).words.sum()
comp = pd.DataFrame({"total_stepped_over": tot,
                     "in_gap<=4": tot - big_same.reindex(tot.index).fillna(0) - big_cross.reindex(tot.index).fillna(0),
                     "in_big_gap_same_page": big_same.reindex(tot.index).fillna(0),
                     "in_big_gap_page_change": big_cross.reindex(tot.index).fillna(0)})
comp_pct = comp.div(comp.total_stepped_over, axis=0).round(3)
print("\nComposition of 'skipped' words (share of stepped-over words):")
print(comp_pct.to_string())
rep["skip_composition"] = json.loads(comp_pct.to_json())

# ---- off-word time by gap, SAME PAGE only ----
print("\nOff-word interval by positional gap (SAME PAGE forward steps only):")
sp = S[S.samepage & S.forward]
b = pd.cut(sp.gap, [-0.5, 0.5, 1.5, 4.5, 10.5, 1e9], labels=["0", "1", "2-4", "5-10", ">10"])
tab = sp.groupby([b, sp.state.map({0: "on-task", 1: "MW", -1: "transition"})], observed=True).agg(
    n=("off_s", "size"), median_ms=("off_s", lambda x: np.median(x) * 1000),
    mean_ms=("off_s", lambda x: np.mean(x) * 1000),
    frac_gt500ms=("off_s", lambda x: float((x > 0.5).mean()))).round(2)
print(tab.to_string())
rep["gap_time_same_page"] = json.loads(tab.reset_index().to_json(orient="records"))

# ---- state-changing steps: is MW onset/offset marked by a long off-text interval? ----
tr = S[S.state == -1]
onset = tr[(tr.mw_from == 0) & (tr.mw_to == 1)]
offset = tr[(tr.mw_from == 1) & (tr.mw_to == 0)]
rep["transition_steps"] = {
    "onset": {"n": int(len(onset)), "median_off_ms": float(np.median(onset.off_s) * 1000),
              "mean_off_ms": float(onset.off_s.mean() * 1000),
              "frac_gt500ms": float((onset.off_s > 0.5).mean()), "mean_gap": float(onset.gap.mean())},
    "offset": {"n": int(len(offset)), "median_off_ms": float(np.median(offset.off_s) * 1000),
               "mean_off_ms": float(offset.off_s.mean() * 1000),
               "frac_gt500ms": float((offset.off_s > 0.5).mean()), "mean_gap": float(offset.gap.mean())},
    "within_state": {"median_off_ms": float(np.median(S.loc[S.state >= 0, "off_s"]) * 1000),
                     "frac_gt500ms": float((S.loc[S.state >= 0, "off_s"] > 0.5).mean())}}
print("\nMW onset/offset steps (do the eyes leave the text at the boundary?):")
for k in ["onset", "offset", "within_state"]:
    v = rep["transition_steps"][k]
    print(f"  {k:14s} " + "  ".join(f"{kk}={vv:.3f}" for kk, vv in v.items()))

# ---- clean off-word time budget, within page, per subject x state ----
sp2 = S[S.samepage & (S.state >= 0)]
bud = sp2.groupby(["subject", "state"]).agg(off=("off_s", "sum"), fix=("fix_s", "sum"),
                                            n=("off_s", "size")).reset_index()
bud["off_frac"] = bud.off / (bud.off + bud.fix)
piv = bud.pivot(index="subject", columns="state", values="off_frac").dropna()
rep["offword_fraction_same_page"] = {"on_task": float(piv[0].mean()), "mw": float(piv[1].mean()),
                                     "diff": boot_ci((piv[1] - piv[0]).to_numpy())}
print(f"\nWithin-page off-word time fraction: on-task {piv[0].mean():.4f}  MW {piv[1].mean():.4f}")
print(fmt("diff (MW-on)", rep["offword_fraction_same_page"]["diff"]))

# ---- large same-page forward jumps: rate per state ----
rate = sp2.assign(big=(sp2.gap > 4)).groupby(["subject", "state"]).big.mean().unstack().dropna()
rep["large_jump_rate_same_page"] = {"on_task": float(rate[0].mean()), "mw": float(rate[1].mean()),
                                    "diff": boot_ci((rate[1] - rate[0]).to_numpy())}
print(f"\nRate of large (>4 word) same-page forward jumps: on-task {rate[0].mean():.4f} MW {rate[1].mean():.4f}")
print(fmt("diff (MW-on)", rep["large_jump_rate_same_page"]["diff"]))

bud.to_csv(ART / "offword_budget.csv", index=False)
json.dump(rep, open(RES / "blackout_anatomy.json", "w"), indent=2, default=float)
print(f"\nwrote {RES/'blackout_anatomy.json'}")
