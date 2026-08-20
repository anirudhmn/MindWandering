#!/usr/bin/env python3
"""G0 — skip-measurement audit.

Is the MW skip-rate increase real skipping, or tracking/mapping blackout? Builds the
bracketed skip variable and reports the pass/fail gate.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ART, RES, load_words, load_fix, add_bracket, boot_ci, fmt

w = load_words()
f = load_fix()
print(f"words {w.shape}  fixations {f.shape}  subjects {w.subject.nunique()}")

rep = {}
rep["skip_rate"] = {"on_task": float(w.loc[w.is_mw == 0, "skipped"].mean()),
                    "mw": float(w.loc[w.is_mw == 1, "skipped"].mean())}
per = w.groupby(["subject", "is_mw"]).skipped.mean().unstack()
rep["skip_rate_diff_subject"] = boot_ci((per[1] - per[0]).to_numpy())
print("\nRaw skip rate  on-task %.3f  MW %.3f" % (rep["skip_rate"]["on_task"], rep["skip_rate"]["mw"]))
print(fmt("skip-rate diff (MW-on)", rep["skip_rate_diff_subject"]))

# --- structure of skips: run length of consecutive skipped words ---
runs = {0: [], 1: []}
for _, g in w.sort_values(["subject", "run", "pos"]).groupby(["subject", "run"], sort=False):
    sk = g.skipped.to_numpy().astype(bool)
    mw = g.is_mw.to_numpy().astype(int)
    i = 0
    while i < len(sk):
        if sk[i]:
            j = i
            while j + 1 < len(sk) and sk[j + 1]:
                j += 1
            runs[int(round(mw[i:j + 1].mean()))].append(j - i + 1)
            i = j + 1
        else:
            i += 1
rep["skip_runlength"] = {k: {"mean": float(np.mean(v)), "median": float(np.median(v)),
                             "p95": float(np.percentile(v, 95)), "frac_ge5": float(np.mean(np.array(v) >= 5)),
                             "n": len(v)} for k, v in runs.items()}
print("\nConsecutive-skip run lengths (blackout signature):")
for k, lab in [(0, "on-task"), (1, "MW")]:
    r = rep["skip_runlength"][k]
    print(f"  {lab:8s} mean {r['mean']:.2f} median {r['median']:.0f} p95 {r['p95']:.0f} "
          f"frac>=5 {r['frac_ge5']:.3f} (n={r['n']})")

# --- bracketed skip ---
wb = add_bracket(w, f, k=3)
rep["bracket_retention_overall"] = float(wb.bracketed.mean())
rep["bracket_retention_by_state"] = wb.groupby("is_mw").bracketed.mean().to_dict()
rep["bracket_retention_by_state"] = {int(k): float(v) for k, v in rep["bracket_retention_by_state"].items()}
print(f"\nBracketed (fixation within +/-3 on BOTH sides) retention: overall "
      f"{rep['bracket_retention_overall']:.3f}  on-task {rep['bracket_retention_by_state'][0]:.3f}  "
      f"MW {rep['bracket_retention_by_state'][1]:.3f}")

wbb = wb[wb.bracketed]
rep["skip_rate_bracketed"] = {"on_task": float(wbb.loc[wbb.is_mw == 0, "skipped"].mean()),
                              "mw": float(wbb.loc[wbb.is_mw == 1, "skipped"].mean())}
perb = wbb.groupby(["subject", "is_mw"]).skipped.mean().unstack()
rep["skip_rate_diff_bracketed"] = boot_ci((perb[1] - perb[0]).to_numpy())
print("Bracketed skip rate  on-task %.3f  MW %.3f" % (rep["skip_rate_bracketed"]["on_task"],
                                                      rep["skip_rate_bracketed"]["mw"]))
print(fmt("bracketed skip-rate diff", rep["skip_rate_diff_bracketed"]))

# --- how much of the raw MW skip excess is blackout? ---
raw_d = rep["skip_rate_diff_subject"]["mean"]
brk_d = rep["skip_rate_diff_bracketed"]["mean"]
rep["blackout_share_of_excess"] = float(1 - brk_d / raw_d) if raw_d else np.nan
print(f"\nShare of the MW skip excess attributable to un-traversed/blackout regions: "
      f"{rep['blackout_share_of_excess']*100:.1f}%")

# --- word-property composition by state (for later matching) ---
comp = wbb.groupby("is_mw")[["zipf", "length", "surprisal"]].agg(["mean", "std"])
rep["composition"] = json.loads(comp.to_json())
print("\nWord composition (bracketed):")
print(comp.round(3).to_string())

mw_ret = rep["bracket_retention_by_state"][1]
gate = (mw_ret >= 0.60) and (rep["skip_rate_diff_bracketed"]["ci"][0] > 0)
rep["GATE_G0_PASS"] = bool(gate)
print(f"\nGATE G0: {'PASS' if gate else 'FAIL'} "
      f"(MW retention {mw_ret:.3f} >= 0.60, bracketed diff CI excludes 0)")

wb.to_parquet(ART / "words_bracketed.parquet", index=False)
json.dump(rep, open(RES / "g0_skip_audit.json", "w"), indent=2, default=float)
print(f"\nwrote {ART/'words_bracketed.parquet'} and {RES/'g0_skip_audit.json'}")
