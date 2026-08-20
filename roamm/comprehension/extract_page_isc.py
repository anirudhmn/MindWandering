#!/usr/bin/env python3
"""Per-page inter-subject alignment (ISC) and per-page coupling slopes.

isc_analysis.py computed one ISC per subject per condition. To ask whether alignment
tracks *comprehension* we need it at the granularity of the outcome: one value per page,
because one multiple-choice question was asked per page.

For each word we build a leave-one-subject-out template from the ON-TASK responses of
the other readers, then correlate a reader's page-worth of words against that template.

Two flavours of alignment, deliberately separated:
  isc_*_raw    : alignment of the raw per-word response
  isc_*_resid  : alignment after each reader's OWN lexical model (zipf, length,
                 surprisal) has been regressed out of their per-word response.
                 This is shared response that word properties cannot explain, i.e. it is
                 not a restatement of the lexical-coupling channel.

In the same pass we compute the per-page coupling slopes themselves -- the channels the
paper reports as preserved during mind-wandering -- so that alignment and coupling can be
raced against the same outcome:
  b_zipf_gaze      : fixation duration vs word frequency
  b_surp_gaze      : fixation duration vs GPT-2 in-context surprisal
  b_surp_n400      : centroparietal 300-450 ms amplitude vs surprisal
plus page-level reading covariates (coverage, fixation count, mean fixation duration).

Output: artifacts/comprehension/page_isc.parquet, one row per (sub_id, story, page).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
COUP = ROOT / "roamm" / "artifacts" / "coupling"
OUT = ROOT / "roamm" / "artifacts" / "comprehension"

MIN_WORDS = 25          # words per page needed for a page-level estimate
FIX_RANGE = (50, 1000)  # ms
P2P_MAX_UV = 150.0      # FRP artifact rejection, as in the coupling pipeline

rep: dict = {}

# --------------------------------------------------------------- word-level responses
fx = pd.read_parquet(COUP / "reading_fixations.parquet")
fx = fx[fx["fix_dur"].between(*FIX_RANGE)].copy()
fx["logdur"] = np.log(fx["fix_dur"].to_numpy())

gaze = (
    fx.groupby(["subject", "story", "page", "word_key"], observed=True)
    .agg(
        val=("logdur", "mean"),
        is_mw=("is_mw", "max"),
        zipf=("zipf", "first"),
        surprisal=("surprisal", "first"),
        length=("length", "first"),
        n_fix=("logdur", "size"),
        firstpass=("is_firstpass", "max"),
    )
    .reset_index()
)

frp = pd.read_parquet(
    COUP / "fixations_frp.parquet",
    columns=["subject", "story", "page", "word_key", "is_mw", "fix_dur", "frp_cp_N400", "frp_occ_P2", "frp_p2p", "frp_valid"],
)
frp = frp[frp["frp_valid"] & frp["fix_dur"].between(*FIX_RANGE) & (frp["frp_p2p"] * 1e6 <= P2P_MAX_UV)].copy()
for c in ["frp_cp_N400", "frp_occ_P2"]:
    frp[c] = frp[c] * 1e6
neural = (
    frp.groupby(["subject", "story", "page", "word_key"], observed=True)
    .agg(n400=("frp_cp_N400", "mean"), occp2=("frp_occ_P2", "mean"))
    .reset_index()
)
W = gaze.merge(neural, on=["subject", "story", "page", "word_key"], how="left")
rep["n_word_obs"] = int(len(W))
rep["n_word_obs_with_frp"] = int(W["n400"].notna().sum())

# lexical design used both for residualising and for the per-page slopes
W["surprisal_f"] = W["surprisal"].fillna(W["surprisal"].mean())
W["zipf_f"] = W["zipf"].replace(0.0, np.nan)
W["zipf_f"] = W["zipf_f"].fillna(W["zipf_f"].mean())
W["length_f"] = W["length"].fillna(W["length"].mean())


def residualise_within_subject(df, col, feats=("zipf_f", "length_f", "surprisal_f")):
    """Remove each reader's own linear lexical model from their per-word response."""
    out = np.full(len(df), np.nan)
    for _, idx in df.groupby("subject", observed=True).indices.items():
        y = df[col].to_numpy()[idx]
        ok = np.isfinite(y)
        if ok.sum() < 50:
            continue
        X = np.column_stack([np.ones(ok.sum())] + [df[f].to_numpy()[idx][ok] for f in feats])
        beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
        r = np.full(len(idx), np.nan)
        r[ok] = y[ok] - X @ beta
        out[idx] = r
    return out


W["val_resid"] = residualise_within_subject(W, "val")
W["n400_resid"] = residualise_within_subject(W, "n400")
W["occp2_resid"] = residualise_within_subject(W, "occp2")


def loo_template(df, col):
    """leave-one-subject-out mean of `col` over ON-TASK rows, per word_key."""
    v = df[col].to_numpy(float)
    ontask = ((df["is_mw"].to_numpy() == 0) & np.isfinite(v)).astype(float)
    contrib = np.where(ontask > 0, v, 0.0)
    g = pd.DataFrame({"w": df["word_key"].to_numpy(), "s": contrib, "c": ontask}).groupby("w").sum()
    S = g["s"].reindex(df["word_key"].to_numpy()).to_numpy()
    C = g["c"].reindex(df["word_key"].to_numpy()).to_numpy()
    num = S - contrib
    den = C - ontask
    return np.where(den >= 3, num / np.maximum(den, 1), np.nan)


for col in ["val", "val_resid", "n400", "n400_resid", "occp2", "occp2_resid"]:
    W["tmpl_" + col] = loo_template(W, col)


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < MIN_WORDS:
        return np.nan, int(ok.sum())
    x, y = a[ok], b[ok]
    if x.std() < 1e-9 or y.std() < 1e-9:
        return np.nan, int(ok.sum())
    return float(np.corrcoef(x, y)[0, 1]), int(ok.sum())


def slope(y, X):
    """OLS slope on the first column of X (after an intercept), NaN if underpowered."""
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    if ok.sum() < MIN_WORDS:
        return np.nan
    A = np.column_stack([np.ones(ok.sum()), X[ok]])
    if np.linalg.matrix_rank(A) < A.shape[1]:
        return np.nan
    beta, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    return float(beta[1])


rows = []
for (subj, story, page), g in W.groupby(["subject", "story", "page"], observed=True):
    rec = {"subject": int(subj), "story_phys": story, "page": int(page)}
    for name, col in [
        ("gaze_raw", "val"), ("gaze_resid", "val_resid"),
        ("n400_raw", "n400"), ("n400_resid", "n400_resid"),
        ("occp2_raw", "occp2"), ("occp2_resid", "occp2_resid"),
    ]:
        r, n = corr(g[col].to_numpy(float), g["tmpl_" + col].to_numpy(float))
        rec["isc_" + name] = r
        rec["n_" + name] = n
    Xg = np.column_stack([g["zipf_f"], g["length_f"], g["surprisal_f"]])
    Xs = np.column_stack([g["surprisal_f"], g["zipf_f"], g["length_f"]])
    rec["b_zipf_gaze"] = slope(g["val"].to_numpy(float), Xg)
    rec["b_surp_gaze"] = slope(g["val"].to_numpy(float), Xs)
    rec["b_surp_n400"] = slope(g["n400"].to_numpy(float), Xs)
    rec["b_zipf_n400"] = slope(g["n400"].to_numpy(float), Xg)
    rec["n_words_fixated"] = int(len(g))
    rec["n_fixations"] = int(g["n_fix"].sum())
    rec["mean_logdur"] = float(g["val"].mean())
    rec["frac_firstpass"] = float(g["firstpass"].mean())
    rec["frac_words_mw"] = float((g["is_mw"] == 1).mean())
    rows.append(rec)

P = pd.DataFrame(rows)

# page coverage = fraction of that page's words this reader ever fixated
npage = W.groupby(["story", "page"], observed=True)["word_key"].nunique().rename("n_words_page").reset_index()
P = P.merge(npage, left_on=["story_phys", "page"], right_on=["story", "page"], how="left").drop(columns=["story"])
P["coverage"] = P["n_words_fixated"] / P["n_words_page"]

# Fisher-z the correlations
for c in [c for c in P.columns if c.startswith("isc_")]:
    P[c + "_z"] = np.arctanh(P[c].clip(-0.999, 0.999))

subs = sorted(pd.read_parquet(OUT / "pages.parquet")["sub_id"].unique())
P["sub_id"] = P["subject"].map({i: s for i, s in enumerate(subs)})

P.to_parquet(OUT / "page_isc.parquet", index=False)

rep["n_pages_with_isc"] = int(len(P))
rep["coverage"] = {k: float(v) for k, v in P["coverage"].describe().items()}
rep["isc_means"] = {c: float(P[c].mean()) for c in P.columns if c.startswith("isc_") and not c.endswith("_z")}
rep["isc_available"] = {c: int(P[c].notna().sum()) for c in P.columns if c.startswith("isc_") and not c.endswith("_z")}
rep["slope_means"] = {c: float(P[c].mean()) for c in ["b_zipf_gaze", "b_surp_gaze", "b_surp_n400", "b_zipf_n400"]}
rep["slope_available"] = {c: int(P[c].notna().sum()) for c in ["b_zipf_gaze", "b_surp_gaze", "b_surp_n400", "b_zipf_n400"]}
(OUT / "page_isc_report.json").write_text(json.dumps(rep, indent=2) + "\n")
print(json.dumps(rep, indent=2))
print(f"wrote {OUT/'page_isc.parquet'} {P.shape}")
