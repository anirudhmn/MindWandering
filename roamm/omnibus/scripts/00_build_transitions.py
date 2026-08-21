#!/usr/bin/env python3
"""Transition table and candidate sets for the omnibus coupling test.

The unit is one fixation-to-fixation transition with both words mapped inside a page's genuine
reading interval, taken from the word-mapped saccade table. Word position is reading order
within the page, so return sweeps are forward moves and cannot be scored as regressions.

For each transition the candidate set is the words at page positions pos-20 .. pos+20. That
window contains the true target on 99.1% of transitions; the remainder are dropped and counted.
Each candidate also carries a reader-state variable, the number of times this reader has already
fixated that word on this page, which belongs with geometry rather than with the text.

Needs the raw dataset (see README). Writes the tables the later scripts read.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ART, ROOT, W, NC

SAC = ROOT / "roamm/artifacts/coupling/saccades.parquet"
LAYOUT = ROOT / "roamm/artifacts/coupling/words_layout.parquet"
MULTI = ROOT / "roamm/artifacts/coupling/word_multiscale.parquet"

lay = pd.read_parquet(LAYOUT)
ms = pd.read_parquet(MULTI)[["word_key", "gpt2_s_sent", "gpt2_gain_long", "gpt2_gain_shuf"]]
lay = lay.merge(ms, on="word_key", how="left").rename(
    columns={"gpt2_s_sent": "s_local", "gpt2_gain_long": "gain_long", "gpt2_gain_shuf": "gain_shuf"})
lay = lay.sort_values(["story", "page", "pos"]).reset_index(drop=True)
lay["widx"] = np.arange(len(lay))
lay = lay.merge(lay.groupby(["story", "page"])["pos"].max().rename("page_max_pos"),
                on=["story", "page"], how="left")
pstart = {k: int(g["widx"].min()) for k, g in lay.groupby(["story", "page"])}
pmax = {k: int(g["pos"].max()) for k, g in lay.groupby(["story", "page"])}

sac = pd.read_parquet(SAC)
sac = sac[sac["fix_dur"].between(50, 1000) & sac["n_pos"].notna()].copy()
sac["pos"] = sac["pos"].astype(int)
sac["n_pos"] = sac["n_pos"].astype(int)
sac["delta_i"] = sac["n_pos"] - sac["pos"]
n_all = len(sac)
sac = sac[sac["delta_i"].abs() <= W].copy()
print(f"transitions {n_all} -> {len(sac)} inside +-{W} words ({100*len(sac)/n_all:.2f}%)", flush=True)

sac = sac.sort_values(["subject", "story", "page", "fix_order_all"]).reset_index(drop=True)
sac["log_fix_dur"] = np.log(sac["fix_dur"].to_numpy())
sac["target_idx"] = sac["delta_i"] + W
sac = sac.merge(lay[["story", "page", "page_max_pos"]].drop_duplicates(), on=["story", "page"], how="left")
sac = sac.merge(lay[["word_key", "line_len"]], on="word_key", how="left")
sac["page_prog"] = sac["pos"] / sac["page_max_pos"].clip(lower=1)
sac["log_in_amp"] = np.log1p(sac["in_amp_px"].fillna(sac["in_amp_px"].median()))
sac["log_prev_dur"] = np.log(sac["prev_fix_dur"].fillna(sac["fix_dur"].median()).clip(lower=50))
sac["mw"] = sac["is_mw"].astype(float)

# contiguous same-state runs within a reader's run, for the episode-preserving permutation
chg = ((sac["mw"].to_numpy() != np.r_[np.nan, sac["mw"].to_numpy()[:-1]]) |
       (sac["run_uid"].to_numpy() != np.r_[-1, sac["run_uid"].to_numpy()[:-1]]))
sac["block_id"] = np.cumsum(chg)

n = len(sac)
cand = np.full((n, NC), -1, np.int32)
prior = np.zeros((n, NC), np.int16)
offs = np.arange(-W, W + 1)
for (subj, story, page), g in sac.groupby(["subject", "story", "page"], sort=False):
    mx = pmax[(story, page)]
    p = g["pos"].to_numpy()
    cpos = p[:, None] + offs[None, :]
    ok = (cpos >= 0) & (cpos <= mx)
    idx = g.index.to_numpy()
    cand[idx] = np.where(ok, pstart[(story, page)] + cpos, -1).astype(np.int32)
    # times this reader already fixated each candidate on this page, strictly before now
    T = len(g)
    oh = np.zeros((T, mx + 1), np.int16)
    oh[np.arange(T), p] = 1
    cum = np.cumsum(oh, 0) - oh
    prior[idx] = np.where(ok, cum[np.arange(T)[:, None], np.clip(cpos, 0, mx)], 0).astype(np.int16)

true_widx = lay.set_index("word_key")["widx"].reindex(sac["n_word_key"]).to_numpy()
agree = float(np.mean(cand[np.arange(n), sac["target_idx"].to_numpy()] == true_widx))
print("target present at its candidate index:", agree, flush=True)
assert agree == 1.0, "candidate indexing does not recover the true target"

keep = ["subject", "story", "page", "run_uid", "fix_order_all", "pos", "line", "line_pos",
        "line_len", "x", "y", "center_x", "center_y", "page_max_pos", "page_prog", "log_in_amp",
        "log_prev_dur", "first_pass", "is_mw", "mw", "block_id", "log_fix_dur", "fix_dur",
        "target_idx", "delta_i", "kind", "word_key", "n_word_key"]
sac[keep].to_parquet(ART / "policy_trans.parquet", index=False)
lay.to_parquet(ART / "policy_words.parquet", index=False)
np.save(ART / "policy_cand_widx.npy", cand)
np.save(ART / "policy_cand_prior.npy", prior)

meta = dict(n_transitions_all=int(n_all), n_transitions=int(n), window=W, n_candidates=NC,
            coverage=float(len(sac) / n_all), target_agreement=agree,
            mw_rate=float(sac["mw"].mean()), n_readers=int(sac.subject.nunique()),
            n_words=int(len(lay)))
(ART / "policy_build.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
