#!/usr/bin/env python3
"""Debiased per-item "answerable without reading" score.

v1 scored each item once with options in their original order. Letter-position bias is a
known artifact of logit-based multiple-choice scoring, and with 50 items a couple of biased
calls move the headline. Here each item is scored under 8 random option orderings and the
probability assigned to the GOLD option is averaged, giving a continuous per-item score that
does not depend on where the correct option happened to sit.

The output is the key covariate for everything downstream: `p_gold_nopassage` is how much of
an item is answerable from plausibility and world knowledge alone, so
`1 - p_gold_nopassage` is the reading-dependent headroom the physiology could possibly
explain.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
RD = ROOT / "reading_data"
STIM = ROOT / "data" / "derivatives" / "stimuli" / "wiki_stories"
OUT = ROOT / "roamm" / "artifacts" / "comprehension"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N_ORDER = 8
RNG = np.random.default_rng(5)

XLSX = {
    "Pluto": "pluto", "The Voynich Manuscript": "the_voynich_manuscript",
    "History of Film": "history_of_film", "Serena Williams": "serena_williams",
    "Prisoners Dilemma": "prisoners_dilemma",
}
LETTERS = ["A", "B", "C", "D"]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
model.eval()
letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]


@torch.no_grad()
def p_gold(question, options, gold, passage=None, n_order=N_ORDER):
    """mean probability mass on the correct option across random option orderings."""
    ps, hits = [], []
    for _ in range(n_order):
        perm = RNG.permutation(len(options))
        opts = "\n".join(f"{LETTERS[i]}. {options[perm[i]]}" for i in range(len(options)))
        body = "" if passage is None else f"Passage:\n{passage}\n\n"
        user = (f"{body}Question: {question}\n{opts}\n\n"
                "Answer with the single letter of the best option.")
        text = tok.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
        )
        logits = model(**tok(text, return_tensors="pt")).logits[0, -1]
        p = torch.softmax(logits.float()[letter_ids], -1).numpy()
        slot = int(np.flatnonzero(perm == gold)[0])
        ps.append(float(p[slot]))
        hits.append(int(p.argmax() == slot))
    return float(np.mean(ps)), float(np.mean(hits))


E = pd.read_parquet(OUT / "item_evidence.parquet").set_index("item")
rows = []
for story, stem in XLSX.items():
    coords = pd.read_csv(STIM / f"{stem}_coordinates.csv")
    qs = pd.read_excel(RD / f"{stem}_questions.xlsx")
    qs = qs[qs["Answer"].notna()]
    for _, q in qs.iterrows():
        page = int(q["question_index"])
        item = f"{story}_p{page}"
        options = [str(q[f"option{k}"]).strip() for k in range(1, 5)]
        gold = int(q["Answer"]) - 1
        pg = coords[coords["page"] == page]
        page_text = " ".join(dict.fromkeys(pg["sentence"]))
        sids = list(dict.fromkeys(pg["sentence_id"]))
        ev = E.loc[item, "evidence_sentence_idx"]
        ev_text = " ".join(pg.loc[pg["sentence_id"] == sids[i], "sentence"].iloc[0]
                           for i in ev if i < len(sids))
        rec = {"item": item, "reading": story, "page": page,
               "item_type": E.loc[item, "item_type"], "negated": bool(E.loc[item, "negated"])}
        for cond, passage in [("nopassage", None), ("evidence", ev_text), ("fullpage", page_text)]:
            pg_, acc = p_gold(str(q["question_text"]).strip(), options, gold, passage)
            rec[f"p_gold_{cond}"] = pg_
            rec[f"acc_{cond}"] = acc
        rows.append(rec)
        print(f"{item:32s} nopassage p={rec['p_gold_nopassage']:.3f} "
              f"evidence p={rec['p_gold_evidence']:.3f} fullpage p={rec['p_gold_fullpage']:.3f}", flush=True)

L = pd.DataFrame(rows)
pages = pd.read_parquet(OUT / "pages_full.parquet")
L = L.merge(pages.groupby("item")["correct"].mean().rename("human_acc"), on="item")
L["reading_headroom"] = 1 - L["p_gold_nopassage"]
L["llm_reading_gain"] = L["p_gold_fullpage"] - L["p_gold_nopassage"]
L.to_parquet(OUT / "llm_answerability_v2.parquet", index=False)

rep = {
    "model": MODEL, "n_orderings_per_item": N_ORDER, "n_items": int(len(L)),
    "mean_p_gold": {c: float(L[f"p_gold_{c}"].mean()) for c in ["nopassage", "evidence", "fullpage"]},
    "mean_accuracy": {c: float(L[f"acc_{c}"].mean()) for c in ["nopassage", "evidence", "fullpage"]},
    "human_accuracy": float(L["human_acc"].mean()), "chance": 0.25,
    "by_item_type": L.groupby("item_type")[
        ["p_gold_nopassage", "p_gold_evidence", "p_gold_fullpage", "human_acc"]].mean().round(3).to_dict("index"),
    "corr_p_gold_nopassage_with_human_acc": float(np.corrcoef(L["p_gold_nopassage"], L["human_acc"])[0, 1]),
    "corr_p_gold_fullpage_with_human_acc": float(np.corrcoef(L["p_gold_fullpage"], L["human_acc"])[0, 1]),
    "evidence_vs_fullpage_p_gold_corr": float(np.corrcoef(L["p_gold_evidence"], L["p_gold_fullpage"])[0, 1]),
}
(OUT / "llm_answerability_v2_report.json").write_text(json.dumps(rep, indent=2, default=str) + "\n")
print()
print(json.dumps(rep, indent=2, default=str))
