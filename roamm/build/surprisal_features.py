#!/usr/bin/env python3
"""Compute GPT-2 word surprisal for every stimulus word_key.

For each unique sentence, run GPT-2, get per-BPE-token surprisal (-log2 p(token|left
context)), then aggregate tokens to the display words (in reading order) via a running
character cursor over the sentence. Surprisal of a word = sum of its BPE tokens'
surprisals. Words are matched to word_key by (sentence_id, sent_pos).
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

OUT = Path("roamm/artifacts/coupling")
wf = pd.read_parquet(OUT / "word_features.parquet")

tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(dev)
print("device:", dev)

def word_surprisals(sentence: str, words: list[str]) -> list[float]:
    """Return surprisal per display word (reading order) for one sentence."""
    enc = tok(sentence, return_offsets_mapping=True, return_tensors="pt")
    ids = enc["input_ids"].to(dev)
    offsets = enc["offset_mapping"][0].tolist()
    with torch.no_grad():
        logits = model(ids).logits[0]                      # [T, V]
    logp = torch.log_softmax(logits.float(), dim=-1)
    # surprisal of token t (t>=1) under context < t
    tok_surp = np.full(ids.shape[1], np.nan)
    tgt = ids[0]
    for t in range(1, ids.shape[1]):
        tok_surp[t] = -(logp[t - 1, tgt[t]].item()) / np.log(2.0)
    # map each display word to a char span via cursor, then sum tokens overlapping it
    surp = []
    cur = 0
    low = sentence.lower()
    for w in words:
        core = re.sub(r"[^a-z0-9]", "", str(w).lower())
        if not core:
            surp.append(np.nan); continue
        # find next occurrence of the word's first alnum run at/after cursor
        m = re.search(re.escape(core[:min(len(core), 12)]), low[cur:])
        if m is None:
            surp.append(np.nan); continue
        w_start = cur + m.start()
        w_end = w_start + len(core)  # approximate span over core chars
        # advance cursor to end of the matched display token in the raw sentence
        cur = w_start + m.end() - m.start()
        s = 0.0; hit = False
        for t in range(1, len(offsets)):
            a, b = offsets[t]
            if b <= w_start or a >= w_end:
                continue
            if not np.isnan(tok_surp[t]):
                s += tok_surp[t]; hit = True
        surp.append(s if hit else np.nan)
    return surp

rows = []
for sid, g in wf.sort_values("sent_pos").groupby("sentence_id"):
    sent = str(g["sentence"].iloc[0])
    words = g["words"].tolist()
    sp = word_surprisals(sent, words)
    for wk, s in zip(g["word_key"].tolist(), sp):
        rows.append((wk, s))
surp = pd.DataFrame(rows, columns=["word_key", "surprisal"])
out = wf.merge(surp, on="word_key", how="left")
out.to_parquet(OUT / "word_features.parquet", index=False)
print("surprisal coverage:", float(out["surprisal"].notna().mean()))
print(out["surprisal"].describe().to_string())
# sanity: surprisal should correlate negatively with zipf (frequent words less surprising)
m = out[["zipf", "surprisal", "length"]].dropna()
print("\ncorr(surprisal, zipf) =", round(float(np.corrcoef(m.surprisal, m.zipf)[0,1]), 3),
      " corr(surprisal, length) =", round(float(np.corrcoef(m.surprisal, m.length)[0,1]), 3))
