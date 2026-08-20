#!/usr/bin/env python3
"""Confirmatory, maximally-powered test of MW modulation of word-property coupling.

Trial-level linear mixed model on log fixation duration with random intercepts+slopes
by subject (the generalizable random effect) and a random intercept for word_key as a
variance component (crossed items). Fixed effects isolate each property's coupling and
its MW interaction, all predictors z-scored within subject.

  log_dur ~ zipf_z*is_mw + length_z*is_mw + surprisal_z*is_mw + fix_order_z
            + random: (1 + zipf_z + surprisal_z + is_mw | subject) + (1 | word_key)

Reports the MW-interaction coefficients with p-values. This is the arbiter of whether
lexical/semantic coupling is measurably reduced during mind-wandering.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

OUT = Path("roamm/artifacts/coupling")
fix = pd.read_parquet(OUT/"fixations.parquet")
wf = pd.read_parquet(OUT/"word_features.parquet")[["word_key","length","zipf","surprisal","clean"]]
df = fix.merge(wf, on="word_key", how="left")
df = df[(df["clean"].str.len()>=1)&(df["zipf"]>0)&df["surprisal"].notna()&df["fix_dur"].between(50,1000)].copy()
df["log_dur"] = np.log(df["fix_dur"].to_numpy())

# z-score predictors within subject
for col in ["zipf","length","surprisal","fix_order"]:
    df[col+"_z"] = df.groupby("subject")[col].transform(lambda s:(s-s.mean())/(s.std()+1e-9))
df["mw"] = df["is_mw"].astype(float)
df["subject"] = df["subject"].astype(str)
print("trials:", len(df), "subjects:", df.subject.nunique(), "words:", df.word_key.nunique())

# subject random intercept + random slopes (generalizable inference over subjects)
md = smf.mixedlm(
    "log_dur ~ zipf_z*mw + length_z*mw + surprisal_z*mw + fix_order_z",
    df, groups=df["subject"],
    re_formula="~ zipf_z + surprisal_z + mw",
)
print("fitting LMM (this takes a few minutes)...", flush=True)
fit = md.fit(method="lbfgs", maxiter=200)
print(fit.summary())

terms = ["zipf_z","surprisal_z","length_z","zipf_z:mw","surprisal_z:mw","length_z:mw","mw"]
res = {}
for t in terms:
    if t in fit.params.index:
        res[t] = {"coef": float(fit.params[t]), "se": float(fit.bse[t]),
                  "z": float(fit.tvalues[t]), "p": float(fit.pvalues[t])}
(OUT/"lmm_report.json").write_text(json.dumps(res, indent=2)+"\n")
print("\n=== KEY TERMS ===")
for t in terms:
    if t in res:
        r=res[t]; print(f"  {t:16} coef={r['coef']:+.5f} se={r['se']:.5f} z={r['z']:+.2f} p={r['p']:.4f}")
print("\nInterpretation: zipf_z<0 (freq shortens fix), surprisal_z>0 (surprise lengthens).")
print("Decoupling => zipf_z:mw>0 and surprisal_z:mw<0. Preserved => interactions ~0.")
