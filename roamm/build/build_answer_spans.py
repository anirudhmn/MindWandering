#!/usr/bin/env python3
"""Locate, for every comprehension item, the sentence on the page that actually answers it.

The page-average analyses were blunt: a question is answered by one or two sentences out of
~217 words, so averaging any physiological measure over the whole page buries the signal
that matters. This builds the item-anchored ground truth instead.

For each of the 50 items we score every sentence on that page against every answer option
(embedding cosine + IDF-weighted content-word overlap) and record:

  target_sentence      the source of the CORRECT option        (positive-polarity items)
  evidence_sentences   union of the sources of the TRUE options (NOT/EXCEPT items, where the
                       correct response is the statement the page does NOT support, so the
                       reader must have encoded the other three)
  distractor_sentence  per wrong option, its best-matching sentence -- the raw material for
                       asking whether a reader's specific error follows from the specific
                       text they actually fixated

Item polarity matters and is detected explicitly: "Which is NOT true", "...EXCEPT", etc.
invert what counts as evidence, and silently treating them like positive items would point
the analysis at the one sentence the page never contained.

Outputs artifacts/comprehension/answer_spans.parquet (one row per item x option) and a
human-readable dump for verification.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
RD = ROOT / "reading_data"
STIM = ROOT / "data" / "derivatives" / "stimuli" / "wiki_stories"
OUT = ROOT / "roamm" / "artifacts" / "comprehension"

XLSX = {
    "Pluto": "pluto",
    "The Voynich Manuscript": "the_voynich_manuscript",
    "History of Film": "history_of_film",
    "Serena Williams": "serena_williams",
    "Prisoners Dilemma": "prisoners_dilemma",
}
NEG = re.compile(r"\bNOT\b|\bEXCEPT\b|\bnot true\b|\bleast\b", re.I)
STOP = set("""a an the of in on at to for and or but is are was were be been being as by with from
that this these those it its他 he she they them his her their which who whom what when where why how
do does did done can could would should may might will shall have has had not no than then so such
about into over under again further once here there all any both each few more most other some only
own same too very s t don now i you your we our us me my""".split())
TOKEN = re.compile(r"[a-z0-9']+")


def content(text):
    return [w for w in TOKEN.findall(str(text).lower()) if w not in STOP and len(w) > 2]


model = SentenceTransformer("all-MiniLM-L6-v2")
rows, dump = [], []

for story, stem in XLSX.items():
    coords = pd.read_csv(STIM / f"{stem}_coordinates.csv")
    qs = pd.read_excel(RD / f"{stem}_questions.xlsx")
    qs = qs[qs["Answer"].notna()].copy()

    # sentence inventory per page, in reading order, with the word_keys each sentence owns
    sents = (
        coords.groupby(["page", "sentence_id"], sort=False)
        .agg(sentence=("sentence", "first"), word_keys=("word_key", list), n_words=("word_key", "size"))
        .reset_index()
    )
    # IDF over the sentences of this story
    docs = [set(content(s)) for s in sents["sentence"]]
    df_counts: dict[str, int] = {}
    for dset in docs:
        for w in dset:
            df_counts[w] = df_counts.get(w, 0) + 1
    N = len(docs)
    idf = {w: np.log((N + 1) / (c + 0.5)) for w, c in df_counts.items()}

    sent_emb = model.encode(sents["sentence"].tolist(), normalize_embeddings=True, show_progress_bar=False)

    for _, q in qs.iterrows():
        page = int(q["question_index"])
        cand = sents[sents["page"] == page]
        if cand.empty:
            continue
        ci = cand.index.to_numpy()
        E = sent_emb[ci]
        qtext = str(q["question_text"]).strip()
        negated = bool(NEG.search(qtext))
        correct = int(q["Answer"])

        opt_rows = []
        for k in range(1, 5):
            opt = q.get(f"option{k}")
            if pd.isna(opt):
                continue
            probe = f"{qtext} {opt}"
            pe = model.encode([probe], normalize_embeddings=True, show_progress_bar=False)[0]
            cos = E @ pe
            # IDF overlap between the option's own content words and each sentence
            ow = set(content(opt))
            lex = np.array([
                sum(idf.get(w, 0.0) for w in ow & docs[j]) / max(sum(idf.get(w, 0.0) for w in ow), 1e-9)
                for j in ci
            ])
            score = 0.5 * cos + 0.5 * lex
            order = np.argsort(-score)
            best, second = order[0], order[1] if len(order) > 1 else order[0]
            r = cand.iloc[best]
            opt_rows.append({
                "reading": story, "story_phys": stem, "page": page,
                "item": f"{story}_p{page}", "question_text": qtext, "negated_item": negated,
                "option_index": k, "option_text": str(opt).strip(),
                "is_correct_option": k == correct,
                "sentence_id": r["sentence_id"], "sentence": r["sentence"],
                "sent_rank_in_page": int(best), "n_words_sentence": int(r["n_words"]),
                "word_keys": list(r["word_keys"]),
                "score": float(score[best]), "margin": float(score[best] - score[second]),
                "cos": float(cos[best]), "lex": float(lex[best]),
            })
        rows.extend(opt_rows)

        # readable dump for hand verification
        chosen = [o for o in opt_rows if o["is_correct_option"]]
        dump.append({
            "item": f"{story}_p{page}", "negated": negated, "question": qtext,
            "correct_option": chosen[0]["option_text"] if chosen else None,
            "matched_sentence": chosen[0]["sentence"] if chosen else None,
            "margin": round(chosen[0]["margin"], 3) if chosen else None,
            "n_sentences_on_page": int(len(cand)),
        })

S = pd.DataFrame(rows)
S.to_parquet(OUT / "answer_spans.parquet", index=False)
(OUT / "answer_spans_dump.json").write_text(json.dumps(dump, indent=2) + "\n")

print(f"items: {S['item'].nunique()}   option-rows: {len(S)}")
print(f"negated (NOT/EXCEPT) items: {S.groupby('item')['negated_item'].first().sum()} / {S['item'].nunique()}")
print(f"median sentence length (words): {S['n_words_sentence'].median():.0f}")
print(f"correct-option match margin: median {S[S.is_correct_option]['margin'].median():.3f}, "
      f"min {S[S.is_correct_option]['margin'].min():.3f}")
dupe = S.groupby("item")["sentence_id"].nunique()
print(f"items whose 4 options map to 4 distinct sentences: {(dupe == 4).sum()} / {len(dupe)}")
print(f"wrote {OUT/'answer_spans.parquet'}")
