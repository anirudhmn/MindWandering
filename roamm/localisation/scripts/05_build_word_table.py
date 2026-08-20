#!/usr/bin/env python3
"""Word-level table: every stimulus word with its sentence's importance and the covariates
needed to argue that an importance effect is not something else.

Joins on `word_key` (a unique token *instance*), so a sentence that straddles a page boundary
contributes different word_keys to each page and carries that page's own importance rating.

Covariates built here, each because it is a live alternative explanation for an importance
effect:
  zipf / length / surprisal        important sentences contain rarer, longer, less predictable words
  word_pos_in_sentence            sentence-initial and sentence-final words are read differently
  sent_pos_on_page                topic sentences come early (Hyona & Lorch); pure position effect
  n_words_sentence                long sentences are both more important and more effortful
  line_idx / is_line_first/last   line beginnings/ends attract fixations for oculomotor reasons
  is_page_boundary_sentence       sentence 0 of each page was already read on the previous page

Line assignment uses the selection-analysis rule: cluster bbox `top` within page with a 40 px
tolerance (~90 px line spacing). rank(top) is wrong -- glyph heights jitter the tops.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path(__file__).resolve().parents[1] / "artifacts"
ROOT = Path(__file__).resolve().parents[3]
STIM = ROOT / "data" / "derivatives" / "stimuli" / "wiki_stories"
COUP = ROOT / "roamm" / "artifacts" / "coupling"
STEMS = ["pluto", "the_voynich_manuscript", "history_of_film", "serena_williams", "prisoners_dilemma"]


def line_index(tops: np.ndarray, tol: float = 40.0) -> np.ndarray:
    order = np.argsort(tops, kind="stable")
    lab = np.empty(len(tops), int)
    cur, anchor = 0, tops[order[0]]
    for j in order:
        if tops[j] - anchor > tol:
            cur += 1
            anchor = tops[j]
        lab[j] = cur
    return lab


imp = pd.read_parquet(ART / "importance_llm.parquet")
cen = pd.read_parquet(ART / "centrality_lm.parquet")[["story_phys", "sentence_id", "centrality_lm"]]
qw_p = ART / "importance_qwen.parquet"
qw = pd.read_parquet(qw_p) if qw_p.exists() else None

rows = []
for stem in STEMS:
    c = pd.read_csv(STIM / f"{stem}_coordinates.csv")
    for pg, g in c.groupby("page", sort=True):
        g = g.reset_index(drop=True)
        sids = list(dict.fromkeys(g["sentence_id"]))
        sidx = {s: i for i, s in enumerate(sids)}
        li = line_index(g["top"].to_numpy(float))
        for i, r in g.iterrows():
            rows.append({"word_key": r["word_key"], "story_phys": stem, "page": int(pg),
                         "sentence_id": r["sentence_id"], "sent_idx": sidx[r["sentence_id"]],
                         "n_sentences_page": len(sids), "line_idx": int(li[i]),
                         "word_pos_on_page": int(i), "n_words_page": int(len(g)),
                         "top": float(r["top"]), "left": float(r["left"])})
W = pd.DataFrame(rows)

# word position within the sentence *as laid out on this page*
W["word_pos_in_sentence"] = W.groupby(["story_phys", "page", "sentence_id"]).cumcount()
W["n_words_sentence"] = W.groupby(["story_phys", "page", "sentence_id"])["word_key"].transform("size")
W["rel_pos_in_sentence"] = W["word_pos_in_sentence"] / W["n_words_sentence"].clip(lower=1)
W["sent_pos_on_page"] = W["sent_idx"] / (W["n_sentences_page"] - 1).clip(lower=1)
W["is_page_boundary_sentence"] = (W["sent_idx"] == 0).astype(int)
W["n_lines_page"] = W.groupby(["story_phys", "page"])["line_idx"].transform("max") + 1
W["line_pos"] = W["line_idx"] / (W["n_lines_page"] - 1).clip(lower=1)
W["word_pos_in_line"] = W.groupby(["story_phys", "page", "line_idx"])["left"].rank(method="first").astype(int) - 1
W["n_words_line"] = W.groupby(["story_phys", "page", "line_idx"])["word_key"].transform("size")
W["is_line_first"] = (W["word_pos_in_line"] == 0).astype(int)
W["is_line_last"] = (W["word_pos_in_line"] == W["n_words_line"] - 1).astype(int)

W = W.merge(imp[["story_phys", "page", "sent_idx", "importance_llm", "in_summary_llm"]],
            on=["story_phys", "page", "sent_idx"], how="left", validate="m:1")
W = W.merge(cen, on=["story_phys", "sentence_id"], how="left", validate="m:1")
if qw is not None:
    W = W.merge(qw[["story_phys", "page", "sent_idx", "importance_qwen", "importance_qwen_exp"]],
                on=["story_phys", "page", "sent_idx"], how="left", validate="m:1")

wf = pd.read_parquet(COUP / "word_features.parquet")[["word_key", "clean", "length", "zipf", "surprisal"]]
W = W.merge(wf, on="word_key", how="left", validate="1:1")
W["lemma"] = W["clean"].str.lower()
W["is_content"] = (W["zipf"] < 5.0).astype(int)   # crude, only used descriptively

assert W["importance_llm"].notna().all(), "unmatched words"
assert W["word_key"].is_unique
W.to_parquet(ART / "word_importance.parquet", index=False)

print(f"{len(W)} words, {W.word_key.nunique()} unique keys, {W.groupby(['story_phys','page']).ngroups} pages")
print(f"importance coverage {W.importance_llm.notna().mean():.3f}; centrality {W.centrality_lm.notna().mean():.3f}"
      + (f"; qwen {W.importance_qwen.notna().mean():.3f}" if qw is not None else "  [qwen pending]"))
print("\nword-level correlations of importance with the alternative explanations:")
cols = ["zipf", "length", "surprisal", "rel_pos_in_sentence", "sent_pos_on_page",
        "n_words_sentence", "line_pos", "is_line_first"]
print(W[["importance_llm"] + cols].corr(method="spearman")["importance_llm"].round(3).to_string())

# how much importance variation survives holding the word type fixed?
g = W.groupby("lemma")["importance_llm"]
W["imp_within_lemma"] = W["importance_llm"] - g.transform("mean")
multi = W[g.transform("size") >= 2]
var_ok = multi[multi.groupby("lemma")["importance_llm"].transform("std").fillna(0) > 0]
print(f"\nwithin-word-type: {len(multi)} tokens whose lemma occurs >=2x; "
      f"{len(var_ok)} with importance variation across occurrences "
      f"({len(var_ok)/len(W):.1%} of all tokens); "
      f"SD of within-lemma demeaned importance = {var_ok['imp_within_lemma'].std():.3f}")
