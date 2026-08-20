#!/usr/bin/env python3
"""LLM (Claude Opus 5, in-context) annotation of task-oriented RELEVANCE: the evidence span
of every comprehension item.

The complement of 01_annotate_importance.py. Where importance is what the text needs, this is
what the QUESTION needs -- McCrudden & Schraw's second construct. Annotated after the
importance ratings were frozen, so the questions cannot have leaked into them.

Every one of the 50 items was read against its page and localised by hand to the page-local
sentence indices that carry the answer. Recorded per item:

  evidence      sentence indices a reader must have encoded to answer correctly
  item_type     single_fact  one/two adjacent sentences state the answer
                inferential  the answer follows from a few sentences, none states it
                integrative  the answer requires combining several sentences
                negated      "which is NOT true / EXCEPT / is wrong" -- the keyed option is
                             the one the page does NOT support, so evidence is the union of
                             the sentences supporting the three TRUE options
  key_sentence  for negated items, the sentence that contradicts the keyed option (None when
                the keyed option is simply absent from the page)
  mis_keyed     the shipped answer key disagrees with the page

Why the negated items matter: they are a built-in control on the entire localisation approach.
For them the prediction inverts -- performance should track broad page coverage rather than
dwell on one span, because the reader has to verify three separate statements. If span
localisation were noise, single_fact and negated items could not dissociate.

mis_keyed: Prisoners Dilemma p8 asks which statement is TRUE about optimal strategies and keys
option 2 ("independent of the number of rounds"), but the page states the opposite (the optimal
strategy depends on the percentage of defectors and on the length of the game) and separately
states option 4 verbatim. The item is unanswerable as keyed; it is flagged and dropped in the
predeclared sensitivity analysis rather than silently kept.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path(__file__).resolve().parents[1] / "artifacts"
STIM = Path(__file__).resolve().parents[3] / "data" / "derivatives" / "stimuli" / "wiki_stories"
RNG = np.random.default_rng(60)

# story_phys|page -> (evidence sentence idx, item_type, key_sentence, mis_keyed)
EV = {
 "pluto|0": ([2, 4, 6], "negated", 4, False),
 "pluto|1": ([4], "single_fact", None, False),
 "pluto|2": ([6, 7], "single_fact", None, False),
 "pluto|3": ([0, 1], "single_fact", None, False),
 "pluto|4": ([6], "single_fact", None, False),
 "pluto|5": ([5, 7, 8], "inferential", None, False),
 "pluto|6": ([4, 5], "single_fact", None, False),
 "pluto|7": ([4, 5, 6], "negated", None, False),
 "pluto|8": ([1, 2, 4], "negated", None, False),
 "pluto|9": ([1, 2, 3], "negated", None, False),
 "the_voynich_manuscript|0": ([2], "single_fact", None, False),
 "the_voynich_manuscript|1": ([1, 3, 4], "negated", 5, False),
 "the_voynich_manuscript|2": ([7], "single_fact", None, False),
 "the_voynich_manuscript|3": ([6], "single_fact", None, False),
 "the_voynich_manuscript|4": ([4], "single_fact", None, False),
 "the_voynich_manuscript|5": ([7], "single_fact", None, False),
 "the_voynich_manuscript|6": ([5], "single_fact", None, False),
 "the_voynich_manuscript|7": ([0, 1, 2, 3], "inferential", None, False),
 "the_voynich_manuscript|8": ([6, 7], "single_fact", None, False),
 "the_voynich_manuscript|9": ([8], "single_fact", None, False),
 "history_of_film|0": ([6], "single_fact", None, False),
 "history_of_film|1": ([8, 10, 13], "negated", 9, False),
 "history_of_film|2": ([5], "single_fact", None, False),
 "history_of_film|3": ([9], "single_fact", None, False),
 "history_of_film|4": ([0, 1, 2, 3], "integrative", None, False),
 "history_of_film|5": ([1], "single_fact", None, False),
 "history_of_film|6": ([3, 4], "single_fact", None, False),
 "history_of_film|7": ([1], "single_fact", None, False),
 "history_of_film|8": ([3, 4], "integrative", None, False),
 "history_of_film|9": ([3], "single_fact", None, False),
 "serena_williams|0": ([7], "single_fact", None, False),
 "serena_williams|1": ([4], "single_fact", None, False),
 "serena_williams|2": ([3, 4], "single_fact", None, False),
 "serena_williams|3": ([1, 2, 4], "negated", 5, False),
 "serena_williams|4": ([1], "single_fact", None, False),
 "serena_williams|5": ([2, 3], "single_fact", None, False),
 "serena_williams|6": ([3], "single_fact", None, False),
 "serena_williams|7": ([8], "single_fact", None, False),
 "serena_williams|8": ([8, 9], "single_fact", None, False),
 "serena_williams|9": ([5, 6], "negated", None, False),
 "prisoners_dilemma|0": ([0, 3, 7], "negated", 10, False),
 "prisoners_dilemma|1": ([0, 1, 4, 6], "negated", 5, False),
 "prisoners_dilemma|2": ([1], "single_fact", None, False),
 "prisoners_dilemma|3": ([4, 5], "negated", None, False),
 "prisoners_dilemma|4": ([0, 3, 4], "single_fact", None, False),
 "prisoners_dilemma|5": ([6, 7, 8], "single_fact", None, False),
 "prisoners_dilemma|6": ([8], "single_fact", None, False),
 "prisoners_dilemma|7": ([9], "single_fact", None, False),
 "prisoners_dilemma|8": ([6, 7, 11], "single_fact", None, True),
 "prisoners_dilemma|9": ([6, 7, 8], "negated", 8, False),
}

READING = {"pluto": "Pluto", "the_voynich_manuscript": "The Voynich Manuscript",
           "history_of_film": "History of Film", "serena_williams": "Serena Williams",
           "prisoners_dilemma": "Prisoners Dilemma"}
EXPOSED = {"pluto|0", "pluto|1", "pluto|2", "pluto|3", "pluto|4", "pluto|6",
           "the_voynich_manuscript|7", "history_of_film|4",
           "prisoners_dilemma|4", "prisoners_dilemma|5", "prisoners_dilemma|8"}


def main():
    inv = json.loads((ART / "sentence_inventory.json").read_text())
    assert set(EV) == set(inv), set(EV) ^ set(inv)
    rows = []
    for key, (ev, itype, keysent, mis) in EV.items():
        stem, page = key.split("|")
        page = int(page)
        sents = inv[key]
        n_sent = len(sents)
        assert all(0 <= i < n_sent for i in ev), f"{key}: evidence index out of range"
        assert keysent is None or 0 <= keysent < n_sent, f"{key}: key_sentence out of range"
        coords = pd.read_csv(STIM / f"{stem}_coordinates.csv")
        pg = coords[coords["page"] == page]
        sids = [s["sentence_id"] for s in sents]
        keys_by_sent = {s: pg.loc[pg["sentence_id"] == s, "word_key"].tolist() for s in sids}

        ev_keys = [k for i in ev for k in keys_by_sent[sids[i]]]
        # matched control region: equal word count, same page, sentences that are neither
        # evidence nor the contradicting sentence, sampled in a fixed order (seed 60)
        excl = set(ev) | ({keysent} if keysent is not None else set())
        free = [i for i in range(n_sent) if i not in excl]
        RNG.shuffle(free)
        ctrl: list[str] = []
        for i in free:
            if len(ctrl) >= len(ev_keys):
                break
            ctrl += keys_by_sent[sids[i]]
        ctrl = ctrl[: len(ev_keys)]

        rows.append({
            "item": f"{READING[stem]}_p{page}", "reading": READING[stem], "story_phys": stem,
            "page": page, "item_type": itype, "negated": itype == "negated",
            "mis_keyed": mis, "exposed_to_annotator": key in EXPOSED,
            "evidence_sent_idx": ev, "n_evidence_sentences": len(ev),
            "key_sentence_idx": keysent,
            "evidence_word_keys": ev_keys, "n_evidence_words": len(ev_keys),
            "control_word_keys": ctrl, "n_control_words": len(ctrl),
            "n_sentences_page": n_sent, "n_words_page": int(len(pg)),
            "evidence_frac_of_page": len(ev_keys) / len(pg),
        })

    E = pd.DataFrame(rows)
    E.to_parquet(ART / "item_evidence_llm.parquet", index=False)

    print(E.groupby("item_type").agg(n=("item", "size"), med_ev_sent=("n_evidence_sentences", "median"),
                                     med_ev_words=("n_evidence_words", "median"),
                                     med_frac=("evidence_frac_of_page", "median")).round(3).to_string())
    print(f"\nevidence region = {E.evidence_frac_of_page.median():.1%} of the page (median) -> "
          f"{1/E.evidence_frac_of_page.median():.1f}x targeting gain over page averaging")
    print(f"equal-size control obtained: {(E.n_control_words == E.n_evidence_words).sum()}/{len(E)}")
    print(f"mis-keyed items: {E.mis_keyed.sum()}   annotator-exposed items: {E.exposed_to_annotator.sum()}")

    # agreement with the earlier automatic (embedding+IDF) localisation, where it exists
    old_p = ART.parents[1] / "artifacts" / "comprehension" / "item_evidence.parquet"
    if old_p.exists():
        O = pd.read_parquet(old_p)[["item", "item_type", "evidence_sentence_idx", "evidence_word_keys"]]
        O.columns = ["item", "item_type_auto", "ev_auto", "ev_keys_auto"]
        M = E.merge(O, on="item", how="inner")
        jac, hit = [], []
        for _, r in M.iterrows():
            a, b = set(r["evidence_sent_idx"]), set(r["ev_auto"])
            jac.append(len(a & b) / max(len(a | b), 1))
            hit.append(len(a & b) > 0)
        M["jaccard"] = jac
        print(f"\nagreement with the automatic embedding+IDF spans (n={len(M)}): "
              f"mean Jaccard {np.mean(jac):.3f}, any-overlap {np.mean(hit):.1%}, "
              f"exact match {np.mean([j == 1 for j in jac]):.1%}")
        print("items where the two disagree entirely:")
        for _, r in M[M.jaccard == 0].iterrows():
            print(f"  {r['item']:32s} llm={r['evidence_sent_idx']} auto={list(r['ev_auto'])} ({r['item_type']})")
        M[["item", "item_type", "item_type_auto", "jaccard"]].to_csv(ART / "evidence_agreement.csv", index=False)


if __name__ == "__main__":
    main()
