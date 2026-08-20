#!/usr/bin/env python3
"""Unsupervised LM information-centrality: how much does each sentence help predict what follows?

The third importance measure, and the only one involving no judgement at all. For every
sentence s_i we compare the language model's negative log-likelihood of the NEXT 150 words of
the story under two contexts that differ only by whether s_i is present:

    centrality_i = [ NLL(future | context WITHOUT s_i) - NLL(future | context WITH s_i) ] / n_tok

A sentence that carries the page's load makes the rest of the text much easier to predict;
a decorative aside changes nothing. This is the same construction as the earlier lesion
LM (D = S_lesioned - S_full) turned around: there the target was one downstream word, here it
is the whole downstream passage, which is why it survives where the lesion analysis failed
(that gate died because ROAMM has few tight local dependencies -- a diffuse whole-passage
readout does not need them).

Boundary handling follows the unit-tested rule from earlier work on this dataset: tokenisation is done on the
fully assembled string per condition, never by concatenating separately tokenised pieces, and
the scored span is located by character offsets (tok_end > span_start & tok_start < span_end).
Both conditions score the byte-identical future string, so the difference cannot come from
tokenisation drift.

Position confound: sentences late on a page have less page left, so the future window is drawn
from the continuing STORY text, not just the page, giving every sentence a comparable amount of
future. Residual position effects are handled downstream by including sentence position as a
covariate in every model.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ART = Path(__file__).resolve().parents[1] / "artifacts"
MODEL = "gpt2"
FUTURE_WORDS = 150
CTX_WORDS = 300          # preceding context cap, in words
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL).to(DEV).eval()


def nll_of_span(text: str, span_start: int, span_end: int) -> tuple[float, int]:
    """Mean NLL over the tokens overlapping [span_start, span_end) of `text`."""
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=1024)
    off = enc["offset_mapping"][0].numpy()
    ids = enc["input_ids"].to(DEV)
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    logp = torch.log_softmax(logits[:-1], dim=-1)
    tgt = ids[0, 1:]
    tok_nll = -logp[torch.arange(len(tgt)), tgt].cpu().numpy()      # nll of token j+1
    sel = [j for j in range(1, len(off)) if off[j][1] > span_start and off[j][0] < span_end]
    if not sel:
        return np.nan, 0
    v = tok_nll[[j - 1 for j in sel]]
    return float(v.mean()), len(v)


def main():
    inv = json.loads((ART / "sentence_inventory.json").read_text())
    # per story, the page-ordered sentence list; consecutive pages share a boundary sentence,
    # so build each story's text from unique sentence_ids in page order
    stories: dict[str, list[dict]] = {}
    for key, sents in inv.items():
        story, page = key.split("|")
        stories.setdefault(story, [])
        for s in sents:
            if not stories[story] or stories[story][-1]["sentence_id"] != s["sentence_id"]:
                stories[story].append({"sentence_id": s["sentence_id"], "sentence": s["sentence"].strip(),
                                       "page": int(page), "sent_idx": s["sent_idx"]})

    rows = []
    for story, sents in stories.items():
        texts = [s["sentence"] for s in sents]
        for i, s in enumerate(sents):
            pre = texts[max(0, i - 40):i]
            # trim preceding context to CTX_WORDS words, keeping the most recent
            while sum(len(p.split()) for p in pre) > CTX_WORDS and pre:
                pre = pre[1:]
            fut_words: list[str] = []
            for t in texts[i + 1:]:
                fut_words += t.split()
                if len(fut_words) >= FUTURE_WORDS:
                    break
            if len(fut_words) < 40:
                rows.append({**s, "story_phys": story, "centrality_lm": np.nan, "n_future_tok": 0})
                continue
            future = " ".join(fut_words[:FUTURE_WORDS])
            ctx_with = (" ".join(pre + [texts[i]])).strip()
            ctx_without = (" ".join(pre)).strip()
            t_with = (ctx_with + " " + future) if ctx_with else future
            t_without = (ctx_without + " " + future) if ctx_without else future
            nll_w, n1 = nll_of_span(t_with, len(t_with) - len(future), len(t_with))
            nll_o, n2 = nll_of_span(t_without, len(t_without) - len(future), len(t_without))
            rows.append({**s, "story_phys": story,
                         "centrality_lm": float(nll_o - nll_w), "n_future_tok": int(min(n1, n2))})
        print(f"  {story}: {len(sents)} unique sentences", flush=True)

    C = pd.DataFrame(rows)
    C.to_parquet(ART / "centrality_lm.parquet", index=False)
    ok = C["centrality_lm"].notna()
    print(f"\nscored {ok.sum()}/{len(C)} unique sentences (model={MODEL}, future={FUTURE_WORDS}w)")
    print(C.loc[ok, "centrality_lm"].describe().round(4).to_string())
    print(f"\nfraction with positive centrality: {(C.loc[ok,'centrality_lm'] > 0).mean():.3f}")
    top = C.loc[ok].nlargest(5, "centrality_lm")[["story_phys", "page", "sent_idx", "centrality_lm", "sentence"]]
    bot = C.loc[ok].nsmallest(5, "centrality_lm")[["story_phys", "page", "sent_idx", "centrality_lm", "sentence"]]
    print("\nMOST informative:")
    for _, r in top.iterrows():
        print(f"  {r.centrality_lm:+.4f} {r.story_phys} p{r.page} s{r.sent_idx}: {r.sentence[:95]}")
    print("LEAST informative:")
    for _, r in bot.iterrows():
        print(f"  {r.centrality_lm:+.4f} {r.story_phys} p{r.page} s{r.sent_idx}: {r.sentence[:95]}")


if __name__ == "__main__":
    main()
