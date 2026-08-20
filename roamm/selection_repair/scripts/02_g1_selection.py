#!/usr/bin/env python3
"""G1/G2/G3 — lexical vs visual control of word skipping during mind-wandering.

Primary measure is Somers' D (rank-based, base-rate free) of a word property predicting
whether the word was skipped, computed per subject within each state.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy import stats
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ART, RES, boot_ci, fmt, somers_d, holm, RNG

G_PRIMARY = 4
PROPS = ["zipf", "length", "surprisal"]

T = pd.read_parquet(ART / "words_traversal.parquet")


def analysis_set(df, G=G_PRIMARY, agree=True):
    s = df[(df.skipped == 0) | (df.gap <= G)]
    if agree:
        s = s[s.state_agree]
    return s


def per_subject_D(df, props=PROPS, minn=150):
    """Somers' D per subject per state."""
    out = []
    for s, g in df.groupby("subject"):
        rec = {"subject": s}
        ok = True
        for st, tag in [(0, "on"), (1, "mw")]:
            gs = g[g.is_mw == st]
            if len(gs) < minn or gs.skipped.nunique() < 2:
                ok = False
                break
            rec[f"n_{tag}"] = len(gs)
            rec[f"skiprate_{tag}"] = float(gs.skipped.mean())
            for p in props:
                rec[f"{p}_{tag}"] = somers_d(gs[p].to_numpy(), gs.skipped.to_numpy())
        if ok:
            out.append(rec)
    return pd.DataFrame(out)


def report(D, props=PROPS, label=""):
    res, ps = {}, []
    for p in props:
        d = (D[f"{p}_mw"] - D[f"{p}_on"]).to_numpy()
        r = boot_ci(d)
        r["D_on"] = float(D[f"{p}_on"].mean())
        r["D_mw"] = float(D[f"{p}_mw"].mean())
        r["retention_pct"] = float(D[f"{p}_mw"].mean() / D[f"{p}_on"].mean() * 100)
        res[p] = r
        ps.append(r["p"])
    adj = holm(ps)
    for p, a in zip(props, adj):
        res[p]["p_holm"] = float(a)
    if label:
        print(f"\n--- {label} (n={len(D)} subjects) ---")
        for p in props:
            r = res[p]
            print(f"  {p:10s} D_on={r['D_on']:+.4f} D_mw={r['D_mw']:+.4f} "
                  f"ret={r['retention_pct']:5.1f}%  Δ={r['mean']:+.4f} "
                  f"[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] p={r['p']:.2g} "
                  f"p_holm={r['p_holm']:.2g} {r['n_pos']}/{r['n']}")
    return res


rep = {}
S = analysis_set(T)
print(f"analysis set: {len(S)} words, {S.subject.nunique()} subjects, "
      f"skip rate on-task {S.loc[S.is_mw==0,'skipped'].mean():.3f} MW {S.loc[S.is_mw==1,'skipped'].mean():.3f}")

# ---------------- G1 PRIMARY ----------------
D = per_subject_D(S)
rep["G1_primary"] = report(D, label="G1 PRIMARY: Somers' D of property -> skip, MW vs on-task")
D.to_csv(ART / "somersD_primary.csv", index=False)

# ---------------- G2 lexical vs visual ----------------
lex = ((D.zipf_mw - D.zipf_on).abs() - 0).to_numpy()
# signed magnitude change: zipf D is positive (frequent->skipped), length D negative
mag = {}
for p in PROPS:
    sgn = np.sign(D[f"{p}_on"].mean())
    mag[p] = (sgn * (D[f"{p}_mw"] - D[f"{p}_on"])).to_numpy()   # >0 => stronger control in MW
rep["G2_magnitude_change"] = {p: boot_ci(v) for p, v in mag.items()}
print("\n--- G2: change in |control strength| (positive = property controls skipping MORE during MW) ---")
for p in PROPS:
    print(fmt(p, rep["G2_magnitude_change"][p]))
diff_lv = mag["zipf"] - mag["length"]
rep["G2_lexical_minus_visual"] = boot_ci(diff_lv)
print(fmt("zipf-change MINUS length-change", rep["G2_lexical_minus_visual"]))
t, p = stats.ttest_rel(mag["zipf"], mag["length"])
rep["G2_paired_t"] = {"t": float(t), "p": float(p)}
print(f"  paired t={t:+.2f} p={p:.3g}")

# ---------------- G3 controls ----------------
print("\n=== G3 CONTROLS ===")

# (a) gap-threshold sensitivity
rep["G3_gap_sensitivity"] = {}
print("\n(a) gap-threshold sensitivity (Δ Somers' D, MW - on-task):")
for G in [1, 2, 3, 4, 6, 10]:
    Dg = per_subject_D(analysis_set(T, G=G))
    r = {p: boot_ci((Dg[f"{p}_mw"] - Dg[f"{p}_on"]).to_numpy()) for p in PROPS}
    rep["G3_gap_sensitivity"][str(G)] = r
    print(f"  G={G:<3d} " + "  ".join(f"{p}: {r[p]['mean']:+.4f} (p={r[p]['p']:.2g})" for p in PROPS))

# (b) deep MW
deep = S[(S.is_mw == 0) | (S.mw_frac >= 0.999)]
rep["G3_deep_mw"] = report(per_subject_D(deep), label="(b) deep MW only (mw_frac = 1)")

# (c) page-position tertiles
S2 = S.copy()
S2["ppos"] = S2.groupby(["subject", "run"])["pos"].transform(lambda x: pd.qcut(x, 3, labels=[0, 1, 2], duplicates="drop"))
rep["G3_position_tertile"] = {}
print("\n(c) within run-position tertiles:")
for t_ in [0, 1, 2]:
    Dt = per_subject_D(S2[S2.ppos == t_], minn=80)
    r = {p: boot_ci((Dt[f"{p}_mw"] - Dt[f"{p}_on"]).to_numpy()) for p in PROPS}
    rep["G3_position_tertile"][str(t_)] = r
    print(f"  tertile {t_} (n={len(Dt)}) " + "  ".join(f"{p}: {r[p]['mean']:+.4f} (p={r[p]['p']:.2g})" for p in PROPS))

# (d) leave-one-story-out
rep["G3_loso_story"] = {}
print("\n(d) leave-one-story-out:")
for st in sorted(S.story.dropna().unique()):
    Ds = per_subject_D(S[S.story != st], minn=100)
    r = {p: boot_ci((Ds[f"{p}_mw"] - Ds[f"{p}_on"]).to_numpy()) for p in PROPS}
    rep["G3_loso_story"][str(st)] = r
    print(f"  drop {str(st):24s} " + "  ".join(f"{p}: {r[p]['mean']:+.4f} (p={r[p]['p']:.2g})" for p in PROPS))

# (e) pseudo-MW: relocate each MW span to a random on-task location of equal length
print("\n(e) pseudo-MW control (200 relocations, matched span count and length):")
obs = {p: rep["G1_primary"][p]["mean"] for p in PROPS}
spans = []
for (s, r_), g in S.sort_values(["subject", "run", "pos"]).groupby(["subject", "run"], sort=False):
    mw = g.is_mw.to_numpy()
    i = 0
    while i < len(mw):
        if mw[i] == 1:
            j = i
            while j + 1 < len(mw) and mw[j + 1] == 1:
                j += 1
            spans.append((s, r_, j - i + 1))
            i = j + 1
        else:
            i += 1
Son = S[S.is_mw == 0].sort_values(["subject", "run", "pos"]).reset_index(drop=True)
grp_idx = {k: v.to_numpy() for k, v in Son.groupby(["subject", "run"]).groups.items()}
null = {p: [] for p in PROPS}
for it in range(200):
    lab = np.zeros(len(Son), bool)
    for (s, r_, L) in spans:
        idx = grp_idx.get((s, r_))
        if idx is None or len(idx) <= L:
            continue
        st = RNG.integers(0, len(idx) - L)
        lab[idx[st:st + L]] = True
    fake = Son.copy()
    fake["is_mw"] = lab.astype(int)
    Df = per_subject_D(fake, minn=150)
    if len(Df) < 20:
        continue
    for p in PROPS:
        null[p].append(float((Df[f"{p}_mw"] - Df[f"{p}_on"]).mean()))
rep["G3_pseudo_mw"] = {}
for p in PROPS:
    v = np.array(null[p])
    pv = float((np.abs(v) >= abs(obs[p])).mean())
    rep["G3_pseudo_mw"][p] = {"observed": obs[p], "null_mean": float(v.mean()),
                              "null_sd": float(v.std()), "p_perm": pv, "n_iter": int(len(v))}
    print(f"  {p:10s} observed {obs[p]:+.4f} vs pseudo null {v.mean():+.4f} +/- {v.std():.4f} "
          f"-> p_perm = {pv:.4f}")

# (f) logistic secondary, and base-rate-matched
print("\n(f) secondary: per-subject logistic slopes (latent scale, NOT base-rate free):")
from sklearn.linear_model import LogisticRegression
rows = []
for s, g in S.groupby("subject"):
    rec = {"subject": s}
    ok = True
    for st, tag in [(0, "on"), (1, "mw")]:
        gs = g[g.is_mw == st].dropna(subset=PROPS)
        if len(gs) < 150 or gs.skipped.nunique() < 2:
            ok = False
            break
        X = np.column_stack([(gs[p] - gs[p].mean()) / (gs[p].std() + 1e-9) for p in PROPS])
        clf = LogisticRegression(C=1e6, max_iter=3000).fit(X, gs.skipped.to_numpy())
        for p, b in zip(PROPS, clf.coef_[0]):
            rec[f"{p}_{tag}"] = float(b)
    if ok:
        rows.append(rec)
L = pd.DataFrame(rows)
rep["G3_logistic"] = {p: boot_ci((L[f"{p}_mw"] - L[f"{p}_on"]).to_numpy()) for p in PROPS}
for p in PROPS:
    r = rep["G3_logistic"][p]
    print(f"  {p:10s} b_on={L[f'{p}_on'].mean():+.3f} b_mw={L[f'{p}_mw'].mean():+.3f} " + fmt("Δ", r, width=3))

json.dump(rep, open(RES / "g1_selection.json", "w"), indent=2, default=float)
print(f"\nwrote {RES/'g1_selection.json'}")
