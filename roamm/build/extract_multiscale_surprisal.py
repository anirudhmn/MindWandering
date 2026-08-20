#!/usr/bin/env python3
"""BOUNDARY-CORRECT multi-scale surprisal (v3) — fixes the iter-51.5 v2 sentence-initial
token drop (482/482 sentences). Same output schema as word_multiscale_v2.parquet.

Method: each sentence is scored in ONE forward pass as [BOS] + ctx_ids + sent_ids, where
  sent_ids = tok(" " + " ".join(sentence_words))   # fixed block, identical across scopes
  ctx_ids  = tok(" ".join(context_words))          # scope-specific, prior text
Per-token surprisals over the sent_ids positions are aggregated to words by char offsets into
the (leading-space) sentence string. This guarantees (a) 100% word coverage incl. the sentence-
initial word, and (b) byte-identical target token ids across s_sent/s_prev1/s_prev3/s_long/s_shuf.

gain_X = s_sent - s_X (>0 => context X eased the word). Matches v2 column names exactly.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = Path("roamm/artifacts/coupling")
ARTV3 = OUT
ARTV3.mkdir(parents=True, exist_ok=True)
LN2 = np.log(2.0)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
RNG = np.random.default_rng(2024)
wf = pd.read_parquet(OUT / "word_features.parquet")
# 2 corpus words have a NaN surface form (pandas-3 StringDtype leaks <NA> into joins); fill them.
wf["words"] = wf["words"].astype("string").fillna(wf["clean"].astype("string")).fillna("the").astype(str)


def sord(s):
    m = re.search(r"_(\d+)$", str(s)); return int(m.group(1)) if m else 0


class LM:
    def __init__(self, name):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name).eval().to(DEV)
        self.bos = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id

    def ids(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"] if text else []

    def sent_block(self, words):
        """token ids + per-token char offsets for ' '+join(words) (leading space)."""
        text = " " + " ".join(str(w) for w in words)
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return enc["input_ids"], enc["offset_mapping"], text

    @torch.no_grad()
    def tok_surprisal(self, ctx_ids, sent_ids):
        keep = 1023 - len(sent_ids)
        ctx = list(ctx_ids[-keep:]) if keep > 0 else []
        seq = [self.bos] + ctx + list(sent_ids)
        logp = torch.log_softmax(self.model(torch.tensor([seq], device=DEV)).logits[0].float(), -1)
        start = len(seq) - len(sent_ids)
        idx = torch.arange(start, len(seq))
        return -(logp[idx - 1, torch.tensor(seq)[idx].to(DEV)]).cpu().numpy() / LN2


def word_char_spans_in_block(words, block_text):
    """char spans of each word in ' '+join(words): word i starts after leading space + joins."""
    spans, cur = [], 1  # skip leading space
    for i, w in enumerate(words):
        w = str(w)
        start = cur
        end = start + len(w)
        spans.append((start, end))
        cur = end + 1  # + the join space
    return spans


def process(lm, scopes):
    res = {}
    for story, sg in wf.groupby("story_file"):
        sents = sg.drop_duplicates("sentence_id").copy()
        sents["ord"] = sents["sentence_id"].map(sord)
        sids = sents.sort_values("ord")["sentence_id"].tolist()
        swords = {}
        for sid in sids:
            g = sg[sg.sentence_id == sid].sort_values("sent_pos")
            swords[sid] = (g["word_key"].tolist(), g["words"].astype(str).tolist())
        prior_words = []           # running list of all prior surface words in story
        prior_ids = []             # cached token ids of the full prior passage
        for pos, sid in enumerate(sids):
            wkeys, words = swords[sid]
            sent_ids, offs, block_text = lm.sent_block(words)
            spans = word_char_spans_in_block(words, block_text)
            surp = {}
            if "s_sent" in scopes:
                surp["s_sent"] = lm.tok_surprisal([], sent_ids)
            if "s_prev1" in scopes:
                ctx = [w for j in range(max(0, pos - 1), pos) for w in swords[sids[j]][1]]
                surp["s_prev1"] = lm.tok_surprisal(lm.ids(" ".join(ctx)), sent_ids)
            if "s_prev3" in scopes:
                ctx = [w for j in range(max(0, pos - 3), pos) for w in swords[sids[j]][1]]
                surp["s_prev3"] = lm.tok_surprisal(lm.ids(" ".join(ctx)), sent_ids)
            if "s_long" in scopes:
                surp["s_long"] = lm.tok_surprisal(prior_ids, sent_ids)
            if "s_shuf" in scopes:
                budget = min(len(prior_ids), 512)
                others = [j for j in range(len(sids)) if j != pos]
                RNG.shuffle(others)
                acc, ntok = [], 0
                for j in others:
                    w = swords[sids[j]][1]
                    acc += w
                    ntok += len(lm.ids(" ".join(w)))
                    if ntok >= budget:
                        break
                cids = lm.ids(" ".join(acc))[:budget] if acc else []
                surp["s_shuf"] = lm.tok_surprisal(cids, sent_ids)
            # aggregate tokens -> words by char-span overlap in block_text
            for wk, (o0, o1) in zip(wkeys, spans):
                d = res.setdefault(wk, {})
                for sc in scopes:
                    tot, hit = 0.0, False
                    for k, (a, b) in enumerate(offs):
                        if b <= o0 or a >= o1:
                            continue
                        tot += surp[sc][k]; hit = True
                    if hit:
                        d[sc] = tot
            prior_words += words
            prior_ids = lm.ids(" ".join(prior_words))
        print(f"    {story}: {len(sids)} sents", flush=True)
    return res


def main():
    print("device:", DEV)
    print("[GPT-2] full ladder...")
    g2 = process(LM("gpt2"), ["s_sent", "s_prev1", "s_prev3", "s_long", "s_shuf"])
    print("[Pythia-160m] cross-model...")
    py = process(LM("EleutherAI/pythia-160m"), ["s_sent", "s_long", "s_shuf"])

    rows = []
    for wk in wf["word_key"]:
        r = {"word_key": wk}
        for sc, v in g2.get(wk, {}).items():
            r["gpt2_" + sc] = v
        for sc, v in py.get(wk, {}).items():
            r["pythia_" + sc] = v
        rows.append(r)
    ms = pd.DataFrame(rows)
    for pre in ["gpt2", "pythia"]:
        for sc in ["s_prev1", "s_prev3", "s_long", "s_shuf"]:
            c = f"{pre}_{sc}"
            if c in ms and f"{pre}_s_sent" in ms:
                ms[f"{pre}_gain_{sc.replace('s_', '')}"] = ms[f"{pre}_s_sent"] - ms[c]
    ms.to_parquet(ARTV3 / "word_multiscale.parquet", index=False)

    # ---- diagnostics incl. the coverage/bug check ----
    si = wf.merge(ms, on="word_key")
    si["is_sent_initial"] = si.groupby("sentence_id")["sent_pos"].transform("min").eq(si["sent_pos"])
    cov = si["gpt2_s_sent"].notna().mean()
    print(f"\ncoverage gpt2_s_sent: {cov:.4f}  (v2 was 0.955 — missing 482 sentence-initial words)")
    print(f"sentence-initial coverage: {si[si.is_sent_initial]['gpt2_s_sent'].notna().mean():.4f} "
          f"(v2 ~0 for non-story-initial)")
    m = si.dropna(subset=["gpt2_s_sent", "gpt2_s_long"])
    print("mean gains  prev1 %.3f  prev3 %.3f  long %.3f  shuf %.3f" %
          tuple(m[f"gpt2_gain_{k}"].mean() for k in ["prev1", "prev3", "long", "shuf"]))
    print("sent-initial gain_long %.3f vs rest %.3f" %
          (si[si.is_sent_initial]["gpt2_gain_long"].mean(),
           si[~si.is_sent_initial]["gpt2_gain_long"].mean()))
    mm = si.dropna(subset=["gpt2_s_long", "pythia_s_long"])
    print("cross-model corr(gpt2_s_long,pythia_s_long) = %.3f" %
          np.corrcoef(mm["gpt2_s_long"], mm["pythia_s_long"])[0, 1])
    print("wrote word_multiscale.parquet", ms.shape)


if __name__ == "__main__":
    main()
