#!/usr/bin/env python3
"""Per (reader, item) reading and neural measures on the EVIDENCE region vs a matched CONTROL
region of the same page.

This is the unit the whole item-anchored idea rests on. Each comprehension question is
answered by a specific ~26-word stretch of the page (12% of it). For every reader x item we
measure what happened while the eyes were on that stretch, and -- critically -- the same
measures on an equal-size stretch of the SAME page that no answer option maps to. The
control region is what turns "did you read the answer" into a test rather than a restatement
of "did you read the page".

Gate questions this file exists to answer:
  * is evidence-region coverage variable, or does everyone read it? (no variance -> no test)
  * are there enough fixation-locked EEG epochs on evidence regions for a subsequent-memory
    contrast, once artifact rejection is applied?

Output: artifacts/comprehension/evidence_trials.parquet, one row per (sub_id, item).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
COUP = ROOT / "roamm" / "artifacts" / "coupling"
OUT = ROOT / "roamm" / "artifacts" / "comprehension"

FIX_RANGE = (50, 1000)
P2P_MAX_UV = 150.0

E = pd.read_parquet(OUT / "item_evidence.parquet")
pages = pd.read_parquet(OUT / "pages_full.parquet")

fx = pd.read_parquet(COUP / "reading_fixations.parquet")
fx = fx[fx["fix_dur"].between(*FIX_RANGE)].copy()
frp = pd.read_parquet(
    COUP / "fixations_frp.parquet",
    columns=["onset_abs_idx", "frp_cp_N400", "frp_occ_P2", "frp_occ_N1", "frp_front_late", "frp_p2p", "frp_valid"],
)
fx = fx.merge(frp, on="onset_abs_idx", how="left")
fx["frp_ok"] = fx["frp_valid"].fillna(False) & (fx["frp_p2p"].fillna(1) * 1e6 <= P2P_MAX_UV)
for c in ["frp_cp_N400", "frp_occ_P2", "frp_occ_N1", "frp_front_late"]:
    fx[c] = np.where(fx["frp_ok"], fx[c] * 1e6, np.nan)

subs = sorted(pages["sub_id"].unique())
fx["sub_id"] = fx["subject"].map({i: s for i, s in enumerate(subs)})

# word_key -> (item, region) membership
memb = []
for _, r in E.iterrows():
    memb += [{"word_key": k, "item": r["item"], "region": "evidence"} for k in r["evidence_word_keys"]]
    memb += [{"word_key": k, "item": r["item"], "region": "control"} for k in r["control_word_keys"]]
memb = pd.DataFrame(memb).drop_duplicates(["word_key", "item", "region"])

F = fx.merge(memb, on="word_key", how="inner")


def agg(g):
    first = g[g["fix_order"] == 0] if "fix_order" in g else g
    return pd.Series({
        "n_words": g["word_key"].nunique(),
        "n_fix": len(g),
        "dwell_ms": g["fix_dur"].sum(),
        "mean_fix_ms": g["fix_dur"].mean(),
        "n_refix": int((g["fix_order"] > 0).sum()),
        "n_reread": int((~g["is_firstpass"]).sum()),
        "mw_frac": float((g["is_mw"] == 1).mean()),
        "n_frp": int(g["frp_ok"].sum()),
        "n400": g.loc[g["frp_ok"], "frp_cp_N400"].mean(),
        "occ_n1": g.loc[g["frp_ok"], "frp_occ_N1"].mean(),
        "occ_p2": g.loc[g["frp_ok"], "frp_occ_P2"].mean(),
        "front_late": g.loc[g["frp_ok"], "frp_front_late"].mean(),
    })


A = F.groupby(["sub_id", "item", "region"], observed=True).apply(agg, include_groups=False).reset_index()
W = A.pivot(index=["sub_id", "item"], columns="region")
W.columns = [f"{b}_{a}" for a, b in W.columns]
W = W.reset_index()

# every reader x item cell should exist even when the region was never fixated
grid = pd.MultiIndex.from_product([subs, E["item"].tolist()], names=["sub_id", "item"]).to_frame(index=False)
W = grid.merge(W, on=["sub_id", "item"], how="left")
for c in [c for c in W.columns if c.startswith(("evidence_", "control_"))]:
    if c.split("_", 1)[1] in ("n_words", "n_fix", "dwell_ms", "n_refix", "n_reread", "n_frp"):
        W[c] = W[c].fillna(0.0)

D = W.merge(E.drop(columns=["evidence_word_keys", "control_word_keys"]), on="item", how="left")
D = D.merge(
    pages[["sub_id", "item", "correct", "skipped", "correct_answered", "mw", "page_dur",
           "n_gaze_raw", "coverage", "n_fixations", "understand", "prior_knowledge", "run", "page"]],
    on=["sub_id", "item"], how="inner",
)
D["evidence_cov"] = D["evidence_n_words"] / D["n_evidence_words"]
D["control_cov"] = D["control_n_words"] / D["n_control_words"].replace(0, np.nan)
D["evidence_read"] = (D["evidence_n_fix"] > 0).astype(int)
D["evidence_dwell_per_word"] = D["evidence_dwell_ms"] / D["n_evidence_words"]
D["control_dwell_per_word"] = D["control_dwell_ms"] / D["n_control_words"].replace(0, np.nan)

D.to_parquet(OUT / "evidence_trials.parquet", index=False)

rep = {
    "n_trials": int(len(D)),
    "n_readers": int(D["sub_id"].nunique()),
    "n_items": int(D["item"].nunique()),
    "evidence_coverage": {k: float(v) for k, v in D["evidence_cov"].describe().items()},
    "control_coverage": {k: float(v) for k, v in D["control_cov"].describe().items()},
    "frac_trials_evidence_never_fixated": float((D["evidence_n_fix"] == 0).mean()),
    "frac_trials_evidence_cov_below_half": float((D["evidence_cov"] < 0.5).mean()),
    "frp_epochs_on_evidence_total": int(D["evidence_n_frp"].sum()),
    "frp_epochs_per_trial_median": float(D["evidence_n_frp"].median()),
    "trials_with_ge5_evidence_frp": int((D["evidence_n_frp"] >= 5).sum()),
    "by_item_type": D.groupby("item_type").agg(
        n_trials=("correct", "size"), acc=("correct", "mean"),
        ev_cov=("evidence_cov", "mean"), ev_frp=("evidence_n_frp", "mean")).round(3).to_dict("index"),
}
(OUT / "evidence_gate.json").write_text(json.dumps(rep, indent=2, default=str) + "\n")
print(json.dumps(rep, indent=2, default=str))
