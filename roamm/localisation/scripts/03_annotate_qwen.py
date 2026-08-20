#!/usr/bin/env python3
"""Second, independent, fully reproducible LLM annotator for text-based importance.

The LLM-annotation literature's central warning is that a single model with a single prompt
gives results that move when either changes ("LLM hacking"), so an annotation used as a
measurement instrument needs at least a second rater and a reported agreement. This is that
second rater: Qwen2.5-1.5B-Instruct, open weights, run locally.

Deterministic by construction -- nothing is sampled. The model sees the page and one target
sentence, and the rating is read straight off the next-token distribution restricted to the
digits 1-5:

    importance_qwen      = argmax over {1..5}
    importance_qwen_exp  = sum_k k * P(k) / sum_k P(k)     (graded, less quantisation noise)

The rubric text is character-for-character the rubric in 01_annotate_importance.py, and the
model never sees the comprehension questions. One call per sentence, target named explicitly,
so there is no output-order or list-position bias.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ART = Path(__file__).resolve().parents[1] / "artifacts"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

RUBRIC = """5 essential   states the page's core proposition/definition/mechanism/outcome; a reader who missed it could not say what the page was about
4 important   substantive supporting fact a summary of the page would include
3 moderate    relevant elaboration that fills out an important point
2 minor       peripheral specifics, secondary examples, incidental names/dates/numbers
1 negligible  aside, parenthetical trivia, decorative detail"""

TEMPLATE = """You are annotating text-based importance for a reading-comprehension study.

Below is one page of an encyclopedia article, with its sentences numbered.

--- PAGE ---
{page}
--- END PAGE ---

Rate how IMPORTANT sentence [{idx}] is, relative to the other sentences on this page, using this scale:

{rubric}

Sentence [{idx}]: {target}

Answer with a single digit from 1 to 5 and nothing else."""


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(DEV).eval()
    digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, 6)]

    inv = json.loads((ART / "sentence_inventory.json").read_text())
    rows = []
    for n, (key, sents) in enumerate(inv.items(), 1):
        story, page = key.split("|")
        page_txt = "\n".join(f"[{s['sent_idx']}] {s['sentence'].strip()}" for s in sents)
        for s in sents:
            msg = [{"role": "user", "content": TEMPLATE.format(
                page=page_txt, idx=s["sent_idx"], rubric=RUBRIC, target=s["sentence"].strip())}]
            ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt").to(DEV)
            with torch.no_grad():
                logits = model(ids).logits[0, -1].float()
            p = torch.softmax(logits[digit_ids], dim=-1).cpu().numpy()
            rows.append({"story_phys": story, "page": int(page), "sent_idx": s["sent_idx"],
                         "sentence_id": s["sentence_id"],
                         "importance_qwen": int(np.argmax(p)) + 1,
                         "importance_qwen_exp": float((p * np.arange(1, 6)).sum()),
                         "p_top": float(p.max())})
        print(f"  [{n}/50] {key}", flush=True)

    Q = pd.DataFrame(rows)
    Q.to_parquet(ART / "importance_qwen.parquet", index=False)
    print(f"\n{len(Q)} ratings, model={MODEL}, deterministic (argmax over digit logits)")
    print(Q["importance_qwen"].value_counts().sort_index().to_string())
    print(f"mean graded {Q.importance_qwen_exp.mean():.3f}  sd {Q.importance_qwen_exp.std():.3f}  "
          f"mean top-prob {Q.p_top.mean():.3f}")


if __name__ == "__main__":
    main()
