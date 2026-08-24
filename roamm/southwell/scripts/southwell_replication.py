#!/usr/bin/env python3
"""Southwell et al. (2020) global-gaze comprehension model, rebuilt on ROAMM.

Southwell, Gregg, Bixler and D'Mello (Cognitive Science 44:e12905) predict post-reading
comprehension of long connected texts from seven *global* page-level gaze features, with
participant-level cross-validation, and obtain observed-vs-predicted correlations of .384,
.362 and .372 across three datasets.  In their Dataset 2 they then regress observed
comprehension on the model's prediction and on self-caught mind-wandering together, and find
the gaze prediction significant while mind-wandering is not.

That is the opposite ordering to the one this manuscript reports, so it needs answering rather
than citing.  The two results are at different levels: theirs asks who comprehends more, from
rate-like features averaged over a page; ours asks which content a reader lost, from where on
the page the lapse fell.  This script tests the reconciliation directly by rebuilding their
model, their metric and their head-to-head comparison on our data.

What replicates is the model (held-out r = 0.338 against their 0.384, 0.362 and 0.372).  Their
head-to-head does not resolve on 44 readers -- neither term is reliable -- so it is reported
here as uninformative rather than as a replication or a contradiction.  The separation between
the two records appears one level further down, within the page, and that comparison lives in
`roamm/localisation/` rather than here.

Outputs results/southwell_replication.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[3]
ITER = ROOT / "roamm" / "southwell"
RES = ITER / "results"
COUP = ROOT / "roamm" / "artifacts" / "coupling"
COMP = ROOT / "roamm" / "artifacts" / "comprehension"

SEED = 6803
N_FOLDS = 4          # participant-level, as in Southwell et al.
N_REPS = 100         # they repeat the whole procedure 100 times and take the median
MAD_TRUNC = 2.5      # their outlier rule
MIN_READ_S = 1.0     # their page-exclusion rule
MIN_FIX = 2
HORIZ_DEG = 30.0     # their conservative horizontal-saccade threshold

FEATURES = [
    "mean_fix_dur",
    "n_fixations",
    "regression_fix_prop",
    "mean_saccade_len",
    "horiz_saccade_prop",
    "fix_dispersion",
    "reading_time",
]


def page_features() -> pd.DataFrame:
    """The seven global page-level features, computed per reader and page."""
    fx = pd.read_parquet(
        COUP / "all_fixations.parquet",
        columns=["subject", "story", "page", "tStart", "fix_dur", "x", "y", "fix_order_all"],
    ).dropna(subset=["page", "x", "y"])
    fx = fx.sort_values(["subject", "story", "page", "fix_order_all"])

    g = fx.groupby(["subject", "story", "page"], observed=True)
    dx = fx["x"].diff()
    dy = fx["y"].diff()
    same = (
        fx["subject"].eq(fx["subject"].shift())
        & fx["story"].eq(fx["story"].shift())
        & fx["page"].eq(fx["page"].shift())
    )
    fx["sacc_len"] = np.where(same, np.hypot(dx, dy), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ang = np.degrees(np.arctan2(np.abs(dy), np.abs(dx)))
    fx["is_horiz"] = np.where(same, (ang <= HORIZ_DEG).astype(float), np.nan)

    # Regressions are interword: a fixation landing left of the previous one on the same line,
    # or on an earlier line.  Southwell index fixations to words; screen coordinates are the
    # tracking-robust equivalent and are what their feature is designed to survive on.
    back = np.where(same, ((dy < -5) | ((np.abs(dy) <= 5) & (dx < 0))).astype(float), np.nan)
    fx["is_regression"] = back

    out = g.agg(
        mean_fix_dur=("fix_dur", "mean"),
        n_fixations=("fix_dur", "size"),
        regression_fix_prop=("is_regression", "mean"),
        mean_saccade_len=("sacc_len", "mean"),
        horiz_saccade_prop=("is_horiz", "mean"),
        t0=("tStart", "min"),
        t1=("tStart", "max"),
        last_dur=("fix_dur", "last"),
    ).reset_index()
    out["reading_time"] = (out["t1"] - out["t0"]) + out["last_dur"] / 1000.0

    # Fixation dispersion: RMS distance of each fixation from that page's mean fixation.
    cen = g[["x", "y"]].transform("mean")
    fx["d2"] = (fx["x"] - cen["x"]) ** 2 + (fx["y"] - cen["y"]) ** 2
    disp = g["d2"].mean().pow(0.5).rename("fix_dispersion").reset_index()
    return out.merge(disp, on=["subject", "story", "page"], how="left")


def load() -> pd.DataFrame:
    pages = pd.read_parquet(COMP / "pages_full.parquet")
    subs = sorted(pages["sub_id"].unique())
    feat = page_features()
    feat["sub_id"] = feat["subject"].map({i: s for i, s in enumerate(subs)})
    norm = lambda v: v.astype(str).str.lower().str.replace(r"[^a-z]+", "", regex=True)
    feat["reading_key"] = norm(feat["story"])
    pages["reading_key"] = norm(pages["reading"])

    d = pages.merge(
        feat.drop(columns=["subject", "story"]),
        on=["sub_id", "reading_key", "page"],
        how="inner",
        suffixes=("", "_f"),
    )
    d = d.dropna(subset=FEATURES + ["correct"])
    d = d[(d["reading_time"] >= MIN_READ_S) & (d["n_fixations"] > MIN_FIX)]
    return d.reset_index(drop=True)


def mad_truncate(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        v = d[c].to_numpy(dtype=float)
        med = np.median(v)
        mad = np.median(np.abs(v - med)) or 1e-12
        z = (v - med) / mad
        hi = v[np.abs(z) <= MAD_TRUNC]
        lo_v, hi_v = (hi.min(), hi.max()) if len(hi) else (v.min(), v.max())
        d[c] = np.clip(v, lo_v, hi_v)
    return d


def cv_predict(d: pd.DataFrame, cols: list[str], rng: np.random.Generator,
               shuffle: bool = False) -> np.ndarray:
    """Participant-level 4-fold CV; returns held-out predicted probabilities."""
    subs = d["sub_id"].unique()
    fold = dict(zip(rng.permutation(subs), np.arange(len(subs)) % N_FOLDS))
    f = d["sub_id"].map(fold).to_numpy()
    y = d["correct"].to_numpy(dtype=int)
    if shuffle:
        y = rng.permutation(y)
    x = d[cols].to_numpy(dtype=float)
    pred = np.zeros(len(d))
    for k in range(N_FOLDS):
        te, tr = f == k, f != k
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-12
        m = LogisticRegression(max_iter=2000, C=1.0)
        m.fit((x[tr] - mu) / sd, y[tr])
        pred[te] = m.predict_proba((x[te] - mu) / sd)[:, 1]
    return pred


def subject_level_r(d: pd.DataFrame, pred: np.ndarray) -> tuple[float, float]:
    t = pd.DataFrame({"sub_id": d["sub_id"], "obs": d["correct"], "pred": pred})
    a = t.groupby("sub_id").mean(numeric_only=True)
    r, p = stats.pearsonr(a["obs"], a["pred"])
    return float(r), float(p)


def ci_of(v: list[float]) -> list[float]:
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    d = load()
    d = mad_truncate(d, FEATURES)
    rep: dict = {
        "n_pages": int(len(d)),
        "n_readers": int(d["sub_id"].nunique()),
        "observed_accuracy": float(d["correct"].mean()),
        "southwell_reference_r": [0.384, 0.362, 0.372],
        "features": FEATURES,
        "n_folds": N_FOLDS,
        "n_reps": N_REPS,
    }

    models = {
        "eye_movements_plus_reading_time": FEATURES,
        "reading_time_only": ["reading_time"],
    }
    preds_keep: dict[str, np.ndarray] = {}
    for name, cols in models.items():
        rs, ps = [], []
        best = None
        for i in range(N_REPS):
            rng = np.random.default_rng(SEED + i)
            pr = cv_predict(d, cols, rng)
            r, p = subject_level_r(d, pr)
            rs.append(r); ps.append(p)
            if best is None or abs(r - np.median(rs)) < best[0]:
                best = (abs(r - np.median(rs)), pr)
        preds_keep[name] = best[1]
        rep[name] = {
            "median_r": float(np.median(rs)),
            "range": [float(np.min(rs)), float(np.max(rs))],
            "ci95_across_reps": ci_of(rs),
            "median_p": float(np.median(ps)),
        }

    rs = []
    for i in range(N_REPS):
        rng = np.random.default_rng(SEED + 10000 + i)
        rs.append(subject_level_r(d, cv_predict(d, FEATURES, rng, shuffle=True))[0])
    rep["shuffled_control"] = {
        "median_r": float(np.median(rs)), "range": [float(np.min(rs)), float(np.max(rs))]
    }

    # Southwell's discriminability test, on their terms: does the gaze model's prediction
    # survive alongside self-caught mind-wandering, between readers?
    pr = preds_keep["eye_movements_plus_reading_time"]
    t = pd.DataFrame({
        "sub_id": d["sub_id"], "obs": d["correct"], "pred": pr,
        "mw": d["is_MWreported"].astype(float), "mw_frac": d["mw_frac_page"].astype(float),
    }).groupby("sub_id").mean(numeric_only=True)
    z = (t - t.mean()) / t.std(ddof=1)
    x = np.column_stack([np.ones(len(z)), z["pred"], z["mw"]])
    b, *_ = np.linalg.lstsq(x, z["obs"].to_numpy(), rcond=None)
    resid = z["obs"].to_numpy() - x @ b
    dof = len(z) - x.shape[1]
    se = np.sqrt(np.diag(np.linalg.pinv(x.T @ x)) * (resid @ resid) / dof)
    tv = b / se
    pv = 2 * stats.t.sf(np.abs(tv), dof)
    crit = stats.t.ppf(0.975, dof)
    rep["between_reader_head_to_head"] = {
        "note": "standardised; predictors are the gaze model's prediction and the reader's "
                "self-caught mind-wandering rate",
        "n_readers": int(len(z)),
        "gaze_prediction": {"beta": float(b[1]), "ci95": [float(b[1] - crit * se[1]),
                                                          float(b[1] + crit * se[1])],
                            "t": float(tv[1]), "p": float(pv[1])},
        "mind_wandering_rate": {"beta": float(b[2]), "ci95": [float(b[2] - crit * se[2]),
                                                              float(b[2] + crit * se[2])],
                                "t": float(tv[2]), "p": float(pv[2])},
        "r2": float(1 - (resid @ resid) / ((z["obs"] - z["obs"].mean()) ** 2).sum()),
        "southwell_dataset2": {"gaze_beta": 0.40, "gaze_ci": [0.23, 0.57],
                               "mw_beta": 0.13, "mw_ci": [-0.04, 0.30], "mw_p": 0.13},
    }

    # The same two predictors within reader, which is the level our localisation result lives
    # at: does either say which *page* was comprehended, once the reader's own mean is removed?
    w = pd.DataFrame({"sub_id": d["sub_id"], "obs": d["correct"].astype(float),
                      "pred": pr, "mw": d["is_MWreported"].astype(float)})
    for c in ("obs", "pred", "mw"):
        w[c] = w[c] - w.groupby("sub_id")[c].transform("mean")
    within = {}
    for c in ("pred", "mw"):
        r, p = stats.pearsonr(w[c], w["obs"])
        within[c] = {"partial_r_within_reader": float(r), "p": float(p)}
    rep["within_reader_page_level"] = within

    (RES / "southwell_replication.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
