#!/usr/bin/env python3
"""G4 — the repair channel: corrective regressions to skipped words, regressions to
difficulty, and immediate refixation. Tests the second locus the MW literature names.

Primary: for each forward saccade that steps over exactly ONE word, does the reader return
to that word within the next 5 fixations?
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ART, RES, COUP, boot_ci, fmt, somers_d, holm

K_LOOKAHEAD = 5
PROPS = ["zipf", "length", "surprisal"]

f = pd.read_parquet(COUP / "reading_fixations.parquet").sort_values(["subject", "run", "tStart"])
w = pd.read_parquet(COUP / "reading_words.parquet")
wprop = w.drop_duplicates("word_key").set_index("word_key")[PROPS]
pos2key = w.drop_duplicates(["story", "pos"]).set_index(["story", "pos"])["word_key"]

rep = {}

# ---------------- corrective regression to a singly-skipped word ----------------
rows = []
for (s, r), g in f.groupby(["subject", "run"], sort=False):
    pos = g["pos"].to_numpy()
    mw = g["is_mw"].to_numpy().astype(int)
    mwf = g["mw_frac"].to_numpy()
    story = g["story"].iloc[0]
    n = len(pos)
    for i in range(n - 1):
        if pos[i + 1] != pos[i] + 2:      # exactly one word stepped over
            continue
        if mw[i] != mw[i + 1]:
            continue
        skipped = pos[i] + 1
        hit = int(np.any(pos[i + 2:i + 2 + K_LOOKAHEAD] == skipped))
        rows.append((s, r, story, skipped, hit, mw[i], 0.5 * (mwf[i] + mwf[i + 1])))
C = pd.DataFrame(rows, columns=["subject", "run", "story", "pos", "corrective", "is_mw", "mw_frac"])
C["word_key"] = pos2key.reindex(list(zip(C.story, C.pos))).to_numpy()
C = C.join(wprop, on="word_key").dropna(subset=PROPS)
C.to_parquet(ART / "corrective_regressions.parquet", index=False)
print(f"single-skip events: {len(C)}  (MW {int(C.is_mw.sum())})  subjects {C.subject.nunique()}")
print(f"corrective-return rate overall: {C.corrective.mean():.4f}")

per = C.groupby(["subject", "is_mw"]).corrective.mean().unstack().dropna()
rep["corrective_rate"] = {"on_task": float(C.loc[C.is_mw == 0, "corrective"].mean()),
                          "mw": float(C.loc[C.is_mw == 1, "corrective"].mean()),
                          "diff": boot_ci((per[1] - per[0]).to_numpy())}
print(f"\nG4a corrective-return RATE  on-task {rep['corrective_rate']['on_task']:.4f} "
      f"MW {rep['corrective_rate']['mw']:.4f}")
print(fmt("rate diff (MW-on)", rep["corrective_rate"]["diff"]))

# does the corrective return track word difficulty, and does MW change that?
rowsD = []
for s, g in C.groupby("subject"):
    rec = {"subject": s}
    ok = True
    for st, tag in [(0, "on"), (1, "mw")]:
        gs = g[g.is_mw == st]
        if len(gs) < 60 or gs.corrective.nunique() < 2:
            ok = False
            break
        for p in PROPS:
            rec[f"{p}_{tag}"] = somers_d(gs[p].to_numpy(), gs.corrective.to_numpy())
    if ok:
        rowsD.append(rec)
D = pd.DataFrame(rowsD)
rep["corrective_selectivity"] = {}
ps = []
print(f"\nG4b corrective-return SELECTIVITY (Somers' D), n={len(D)} subjects:")
for p in PROPS:
    r = boot_ci((D[f"{p}_mw"] - D[f"{p}_on"]).to_numpy())
    r["D_on"] = float(D[f"{p}_on"].mean())
    r["D_mw"] = float(D[f"{p}_mw"].mean())
    r["D_on_test"] = boot_ci(D[f"{p}_on"].to_numpy())
    rep["corrective_selectivity"][p] = r
    ps.append(r["p"])
    print(f"  {p:10s} D_on={r['D_on']:+.4f} (p={r['D_on_test']['p']:.2g}) D_mw={r['D_mw']:+.4f} "
          f"Δ={r['mean']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] p={r['p']:.2g} {r['n_pos']}/{r['n']}")
for p, a in zip(PROPS, holm(ps)):
    rep["corrective_selectivity"][p]["p_holm"] = float(a)

# ---------------- overall regression rate + regression to difficulty ----------------
fr = f.dropna(subset=["regression_out"]).copy()
per = fr.groupby(["subject", "is_mw"]).regression_out.mean().unstack().dropna()
rep["regression_rate"] = {"on_task": float(fr.loc[fr.is_mw == 0, "regression_out"].mean()),
                          "mw": float(fr.loc[fr.is_mw == 1, "regression_out"].mean()),
                          "diff": boot_ci((per[1] - per[0]).to_numpy())}
print(f"\nG4c overall regression-out RATE  on-task {rep['regression_rate']['on_task']:.4f} "
      f"MW {rep['regression_rate']['mw']:.4f}")
print(fmt("rate diff (MW-on)", rep["regression_rate"]["diff"]))

rowsR = []
for s, g in fr.groupby("subject"):
    rec = {"subject": s}
    ok = True
    for st, tag in [(0, "on"), (1, "mw")]:
        gs = g[g.is_mw == st].dropna(subset=PROPS)
        if len(gs) < 150 or gs.regression_out.nunique() < 2:
            ok = False
            break
        for p in PROPS:
            rec[f"{p}_{tag}"] = somers_d(gs[p].to_numpy(), gs.regression_out.to_numpy())
    if ok:
        rowsR.append(rec)
R = pd.DataFrame(rowsR)
rep["regression_to_difficulty"] = {}
print(f"\nG4d regression-to-difficulty SELECTIVITY (Somers' D), n={len(R)} subjects:")
for p in PROPS:
    r = boot_ci((R[f"{p}_mw"] - R[f"{p}_on"]).to_numpy())
    r["D_on"] = float(R[f"{p}_on"].mean())
    r["D_mw"] = float(R[f"{p}_mw"].mean())
    r["D_on_test"] = boot_ci(R[f"{p}_on"].to_numpy())
    r["retention_pct"] = float(r["D_mw"] / r["D_on"] * 100) if r["D_on"] else np.nan
    rep["regression_to_difficulty"][p] = r
    print(f"  {p:10s} D_on={r['D_on']:+.4f} (p={r['D_on_test']['p']:.2g}) D_mw={r['D_mw']:+.4f} "
          f"ret={r['retention_pct']:5.1f}% Δ={r['mean']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] "
          f"p={r['p']:.2g} {r['n_pos']}/{r['n']}")

# ---------------- immediate refixation ----------------
f2 = f.copy()
g = f2.groupby(["subject", "run"], sort=False)
f2["refix"] = (g["pos"].shift(-1) == f2["pos"]).astype(float)
f2.loc[g["pos"].shift(-1).isna(), "refix"] = np.nan
fr2 = f2.dropna(subset=["refix"])
per = fr2.groupby(["subject", "is_mw"]).refix.mean().unstack().dropna()
rep["refixation_rate"] = {"on_task": float(fr2.loc[fr2.is_mw == 0, "refix"].mean()),
                          "mw": float(fr2.loc[fr2.is_mw == 1, "refix"].mean()),
                          "diff": boot_ci((per[1] - per[0]).to_numpy())}
print(f"\nG4e immediate-refixation RATE  on-task {rep['refixation_rate']['on_task']:.4f} "
      f"MW {rep['refixation_rate']['mw']:.4f}")
print(fmt("rate diff (MW-on)", rep["refixation_rate"]["diff"]))

rowsF = []
for s, gg in fr2.groupby("subject"):
    rec = {"subject": s}
    ok = True
    for st, tag in [(0, "on"), (1, "mw")]:
        gs = gg[gg.is_mw == st].dropna(subset=PROPS)
        if len(gs) < 150 or gs.refix.nunique() < 2:
            ok = False
            break
        for p in PROPS:
            rec[f"{p}_{tag}"] = somers_d(gs[p].to_numpy(), gs.refix.to_numpy())
    if ok:
        rowsF.append(rec)
F = pd.DataFrame(rowsF)
rep["refixation_selectivity"] = {}
print(f"\nG4f refixation SELECTIVITY (Somers' D), n={len(F)} subjects:")
for p in PROPS:
    r = boot_ci((F[f"{p}_mw"] - F[f"{p}_on"]).to_numpy())
    r["D_on"] = float(F[f"{p}_on"].mean())
    r["D_mw"] = float(F[f"{p}_mw"].mean())
    r["retention_pct"] = float(r["D_mw"] / r["D_on"] * 100) if r["D_on"] else np.nan
    rep["refixation_selectivity"][p] = r
    print(f"  {p:10s} D_on={r['D_on']:+.4f} D_mw={r['D_mw']:+.4f} ret={r['retention_pct']:5.1f}% "
          f"Δ={r['mean']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] p={r['p']:.2g} {r['n_pos']}/{r['n']}")

json.dump(rep, open(RES / "g4_repair.json", "w"), indent=2, default=float)
print(f"\nwrote {RES/'g4_repair.json'}")
