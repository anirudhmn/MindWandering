#!/usr/bin/env python3
"""Reviewer-driven checks on the selection analysis.

R1. What are the large positional gaps? Test the hypothesis that they are line transitions,
    using the stimulus layout (top coordinate defines the text line).
R2. Are the EXCLUDED words comparable across states? Report their word-property composition.
R3. Replace the gap-size criterion with a layout-based one (exclude words at line boundaries)
    and rerun the primary test.
R4. Match the exclusion rate across states and rerun.
R5. Label validation: do the canonical mind-wandering markers appear?
"""
from __future__ import annotations
import json, glob, os
import numpy as np
import pandas as pd
from scipy import stats
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ART, RES, COUP, ROOT, boot_ci, fmt, somers_d, RNG

PROPS = ["zipf", "length", "surprisal"]
STIM = ROOT / "data/derivatives/stimuli/wiki_stories"
rep = {}

# ---- stimulus layout: line index per word ----
rows = []
for csv in sorted(glob.glob(str(STIM / "*_coordinates.csv"))):
    story = os.path.basename(csv).replace("_coordinates.csv", "")
    d = pd.read_csv(csv).reset_index(drop=True)
    d["story"] = story
    d["pos"] = np.arange(len(d))
    # Words on one visual line share a baseline but their bbox tops jitter by glyph height,
    # so cluster tops within page with a tolerance well below the line spacing.
    def _lines(g):
        t = np.sort(g["top"].unique())
        cuts = np.concatenate([[0], np.cumsum(np.diff(t) > 40)])
        m = dict(zip(t, cuts))
        return g["top"].map(m)
    d["line"] = d.groupby("page", group_keys=False).apply(_lines).astype(int)
    d["line_uid"] = d["page"].astype(str) + "_" + d["line"].astype(str)
    rows.append(d[["story", "pos", "page", "line", "line_uid", "top"]])
L = pd.concat(rows, ignore_index=True)
print(f"layout: {len(L)} words, {L.line_uid.nunique()} distinct lines across stories")

T = pd.read_parquet(ART / "words_traversal.parquet")
T = T.merge(L[["story", "pos", "line", "line_uid"]], on=["story", "pos"], how="left")
T["line_key"] = T["story"].astype(str) + "_" + T["line_uid"].astype(str)

# ---- R1: do large gaps span line boundaries? ----
f = pd.read_parquet(COUP / "reading_fixations.parquet").sort_values(["subject", "run", "tStart"])
f = f.merge(L[["story", "pos", "line_uid"]], on=["story", "pos"], how="left")
steps = []
for (s, r), g in f.groupby(["subject", "run"], sort=False):
    pos = g["pos"].to_numpy(); lu = g["line_uid"].to_numpy(); pg = g["page"].to_numpy()
    mw = g["is_mw"].to_numpy().astype(int)
    fwd = pos[1:] > pos[:-1]
    steps.append(pd.DataFrame({"gap": pos[1:] - pos[:-1] - 1, "forward": fwd,
                               "same_line": lu[1:] == lu[:-1], "same_page": pg[1:] == pg[:-1],
                               "mw": mw[:-1], "agree": mw[1:] == mw[:-1]}))
S = pd.concat(steps, ignore_index=True)
S = S[S.forward & S.agree]
b = pd.cut(S.gap, [-0.5, 0.5, 1.5, 4.5, 1e9], labels=["0", "1", "2-4", ">4"])
tab = S.groupby([b, "mw"], observed=True).agg(
    n=("gap", "size"),
    frac_line_change=("same_line", lambda x: float(1 - x.mean())),
    frac_page_change=("same_page", lambda x: float(1 - x.mean()))).round(3)
print("\nR1  Are large forward steps line transitions?")
print(tab.to_string())
rep["R1_gap_vs_line"] = json.loads(tab.reset_index().to_json(orient="records"))

# ---- R2: composition of excluded vs retained words, by state ----
T["retained"] = (T.skipped == 0) | (T.gap <= 4)
T = T[T.state_agree]
print("\nR2  Word-property composition of retained and excluded words")
comp = T.groupby(["retained", "is_mw"])[PROPS].mean().round(3)
cnt = T.groupby(["retained", "is_mw"]).size().rename("n")
print(pd.concat([comp, cnt], axis=1).to_string())
rep["R2_composition"] = json.loads(pd.concat([comp, cnt], axis=1).reset_index().to_json(orient="records"))
excl = T[~T.retained]
for p in PROPS:
    d = excl.groupby(["subject", "is_mw"])[p].mean().unstack().dropna()
    r = boot_ci((d[1] - d[0]).to_numpy())
    rep[f"R2_excluded_diff_{p}"] = r
    print(fmt(f"excluded words, MW-on {p}", r))

# ---- helper ----
def per_subject_D(df, minn=150):
    out = []
    for s, g in df.groupby("subject"):
        rec = {"subject": s}; ok = True
        for st, tag in [(0, "on"), (1, "mw")]:
            gs = g[g.is_mw == st]
            if len(gs) < minn or gs.skipped.nunique() < 2:
                ok = False; break
            for p in PROPS:
                rec[f"{p}_{tag}"] = somers_d(gs[p].to_numpy(), gs.skipped.to_numpy())
        if ok:
            out.append(rec)
    return pd.DataFrame(out)


def report(D, label):
    out = {}
    print(f"\n{label} (n={len(D)})")
    for p in PROPS:
        r = boot_ci((D[f"{p}_mw"] - D[f"{p}_on"]).to_numpy())
        r["D_on"] = float(D[f"{p}_on"].mean()); r["D_mw"] = float(D[f"{p}_mw"].mean())
        r["retention_pct"] = float(r["D_mw"] / r["D_on"] * 100)
        out[p] = r
        print(f"  {p:10s} D_on={r['D_on']:+.4f} D_mw={r['D_mw']:+.4f} ret={r['retention_pct']:5.1f}% "
              f"Δ={r['mean']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] p={r['p']:.2g} {r['n_pos']}/{r['n']}")
    return out


# ---- R3: layout-based criterion instead of gap size ----
# A word is analysable if the reader fixated at least one word on the SAME LINE both before
# and after it (or it was itself fixated). This makes no reference to gap size.
sub = []
fixed_by = {k: set(v["pos"]) for k, v in f.groupby(["subject", "run"])}
for (s, r), g in T.groupby(["subject", "run"], sort=False):
    fp = fixed_by.get((s, r), set())
    g = g.sort_values("pos")
    pos = g["pos"].to_numpy()
    lk = g["line_key"].to_numpy()
    isfix = np.isin(pos, list(fp))
    ok = np.zeros(len(pos), bool)
    for key in pd.unique(lk):
        m = lk == key
        idx = np.where(m)[0]
        fx = idx[isfix[idx]]
        if len(fx) == 0:
            continue
        lo, hi = pos[fx].min(), pos[fx].max()
        ok[idx] = (pos[idx] >= lo) & (pos[idx] <= hi)
    gg = g.copy(); gg["analysable"] = ok
    sub.append(gg)
T2 = pd.concat(sub, ignore_index=True)
print(f"\nR3  Layout-based criterion: analysable fraction overall {T2.analysable.mean():.3f}, "
      f"on-task {T2.loc[T2.is_mw==0,'analysable'].mean():.3f}, MW {T2.loc[T2.is_mw==1,'analysable'].mean():.3f}")
A = T2[T2.analysable]
per = A.groupby(["subject", "is_mw"]).skipped.mean().unstack().dropna()
rep["R3_skiprate_diff"] = boot_ci((per[1] - per[0]).to_numpy())
print(f"  skip rate on-task {A.loc[A.is_mw==0,'skipped'].mean():.3f}  MW {A.loc[A.is_mw==1,'skipped'].mean():.3f}")
print(fmt("  skip-rate diff", rep["R3_skiprate_diff"]))
rep["R3_selectivity"] = report(per_subject_D(A), "R3  Selectivity, line-interior words only")

# ---- R4: match the exclusion rate across states ----
# Within reader, drop the largest-gap retained MW words until the retained fraction matches
# the on-task retained fraction (and vice versa where MW retains more).
keep = []
for s, g in T.groupby("subject"):
    fr = g.groupby("is_mw").retained.mean()
    if 0 not in fr or 1 not in fr:
        continue
    target = min(fr[0], fr[1])
    for st in [0, 1]:
        gs = g[g.is_mw == st]
        ret = gs[gs.retained]
        n_target = int(round(target * len(gs)))
        if len(ret) > n_target:
            ret = ret.sort_values("gap").iloc[:n_target]
        keep.append(ret)
M = pd.concat(keep, ignore_index=True)
print(f"\nR4  Exclusion-matched set: {len(M)} words; retained fraction "
      f"on-task {M[M.is_mw==0].shape[0]}, MW {M[M.is_mw==1].shape[0]}")
per = M.groupby(["subject", "is_mw"]).skipped.mean().unstack().dropna()
rep["R4_skiprate_diff"] = boot_ci((per[1] - per[0]).to_numpy())
print(fmt("  skip-rate diff", rep["R4_skiprate_diff"]))
rep["R4_selectivity"] = report(per_subject_D(M), "R4  Selectivity, exclusion-rate matched")

# ---- R5: label validation ----
fx = pd.read_parquet(COUP / "reading_fixations.parquet")
lg = fx.assign(l=np.log(fx.fix_dur.clip(lower=1))).groupby(["subject", "is_mw"]).l.mean().unstack().dropna()
rep["R5_fixdur_pct"] = boot_ci((np.exp(lg[1] - lg[0]) - 1).to_numpy() * 100)
print("\nR5  Label validation")
print(fmt("  fixation duration change (%)", rep["R5_fixdur_pct"]))
rate = fx.groupby(["subject", "run"]).is_mw.mean()
rep["R5_mw_rate"] = {"fixation_level": float(fx.is_mw.mean()),
                     "per_reader_min": float(fx.groupby("subject").is_mw.mean().min()),
                     "per_reader_max": float(fx.groupby("subject").is_mw.mean().max()),
                     "readers_with_any": int((fx.groupby("subject").is_mw.sum() > 0).sum())}
print(f"  MW rate {rep['R5_mw_rate']['fixation_level']:.3f}; per-reader range "
      f"{rep['R5_mw_rate']['per_reader_min']:.3f}-{rep['R5_mw_rate']['per_reader_max']:.3f}; "
      f"{rep['R5_mw_rate']['readers_with_any']}/44 readers reported at least one span")

json.dump(rep, open(RES / "reviewer_checks.json", "w"), indent=2, default=float)
print(f"\nwrote {RES/'reviewer_checks.json'}")
