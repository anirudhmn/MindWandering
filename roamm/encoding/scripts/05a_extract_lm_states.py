#!/usr/bin/env python3
"""GPT-2 layerwise hidden states for every corpus word.

Uses the same boundary-correct scheme as the surprisal predictors: each sentence is run in one
pass as [BOS] + the whole preceding passage + the sentence, and per-token states are aggregated
to words by character offsets. A word's state is taken at its final subword token, the position
that has attended to the whole word.

Writes a [n_words, n_layers, hidden] array, about 400 MB, which is not redistributed.
"""
from __future__ import annotations
import json, os, re
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ART, COUP

os.environ.setdefault("HF_HUB_OFFLINE", "1")
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

wf = pd.read_parquet(COUP / "word_features.parquet")
wf["words"] = wf["words"].astype("string").fillna(wf["clean"].astype("string")).fillna("the").astype(str)
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModel.from_pretrained("gpt2", output_hidden_states=True).eval().to(DEV)
BOS = tok.bos_token_id
print(f"{model.config.n_layer + 1} layers, {model.config.n_embd} dimensions, device {DEV}", flush=True)


def sentence_order(s):
    m = re.search(r"_(\d+)$", str(s))
    return int(m.group(1)) if m else 0


keys, vecs = [], []
for story, sg in wf.groupby("story_file"):
    sents = sg.drop_duplicates("sentence_id").copy()
    sents["ord"] = sents["sentence_id"].map(sentence_order)
    prior = []
    for sid in sents.sort_values("ord")["sentence_id"]:
        g = sg[sg.sentence_id == sid].sort_values("sent_pos")
        words = g["words"].astype(str).tolist()
        text = " " + " ".join(words)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        sent_ids, offs = enc["input_ids"], enc["offset_mapping"]
        spans, cur = [], 1
        for w in words:
            spans.append((cur, cur + len(w)))
            cur += len(w) + 1
        keep = 1023 - len(sent_ids)
        seq = [BOS] + (list(prior[-keep:]) if keep > 0 else []) + list(sent_ids)
        with torch.no_grad():
            hs = model(torch.tensor([seq], device=DEV)).hidden_states
        H = torch.stack([h[0].float().cpu() for h in hs], 0).numpy()
        start = len(seq) - len(sent_ids)
        for wi, (a, b) in enumerate(spans):
            hit = [ti for ti, (o0, o1) in enumerate(offs) if o0 < b and o1 > a] or [min(wi, len(offs) - 1)]
            keys.append(g["word_key"].tolist()[wi])
            vecs.append(H[:, start + hit[-1], :])
        prior = prior + list(sent_ids)
    print("  ", story, len(keys), "words", flush=True)

V = np.stack(vecs, 0).astype(np.float32)
np.save(ART / "lm_layer_states.npy", V)
pd.DataFrame({"word_key": keys}).to_parquet(ART / "lm_layer_keys.parquet", index=False)
(ART / "lm_layer_states.json").write_text(json.dumps(
    dict(shape=list(V.shape), coverage=float(len(set(keys)) / wf.word_key.nunique())), indent=2))
print("wrote", V.shape, "coverage", len(set(keys)) / wf.word_key.nunique())
