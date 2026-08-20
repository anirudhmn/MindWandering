#!/usr/bin/env python3
"""LLM (Claude Opus 5, in-context) annotation of TEXT-BASED IMPORTANCE, question-blind.

=== Construct ===
McCrudden & Schraw (2007) separate two things that both get called "relevance":
  * text-based IMPORTANCE  -- how much a segment is needed to understand the text itself
  * task-oriented RELEVANCE -- how much a segment is needed for a specific task/question
ROAMM asks its comprehension question only AFTER the page is gone, so a reader cannot know
task relevance while reading. Importance is therefore the only relevance signal available
online, which is why it is annotated here question-blind and scored separately from the
answer spans (02_annotate_evidence.py).

=== Annotation protocol (v1, frozen) ===
Unit: sentence, as segmented in the stimulus coordinates files, indexed by position in
reading order within a page (`story|page|sent_idx`).
Annotator: Claude Opus 5 reading the full page in context (no question text visible).
Rubric, applied relative to the OTHER SENTENCES ON THE SAME PAGE:
  5 essential   states the page's core proposition/definition/mechanism/outcome; a reader
                who missed it could not say what the page was about
  4 important   substantive supporting fact a summary of the page would include
  3 moderate    relevant elaboration that fills out an important point
  2 minor       peripheral specifics, secondary examples, incidental names/dates/numbers
  1 negligible  aside, parenthetical trivia, decorative detail
Second field `in_summary`: would this sentence appear in a 2-3 sentence summary of the page?
This is the standard NLP salience operationalisation (summarisation = content selection)
and is recorded separately so it can serve as a convergent measure rather than a restatement.

=== Disclosure of partial unblinding ===
Before annotating, this session had incidentally seen 5 Pluto question stems (from an
inspection of pluto_questions.xlsx) and one-line answer descriptions for 7 items encoded in
build_answer_spans_v2.py's OVERRIDE dict (Pluto p2, Pluto p6, Voynich p7, History of Film p4,
Prisoners Dilemma p4, p5, p8). Ratings for every page were nonetheless assigned from the page
text alone. Because it cannot be undone, the exposed items are flagged (EXPOSED_ITEMS) and the
outcome analyses re-run with them dropped. Importance->gaze and importance x MW never involve
the questions at all, so exposure cannot act on them.

Note pages overlap: sentence 0 of page N is the tail of the last sentence of page N-1, so the
same sentence is rated twice (once per page) and its word_keys differ between pages. Ratings
are per (page, sentence) by design -- importance is judged relative to the page in view.
"""
from pathlib import Path
import json

OUT = Path(__file__).resolve().parents[1] / "artifacts"

EXPOSED_ITEMS = ["Pluto_p0", "Pluto_p1", "Pluto_p2", "Pluto_p3", "Pluto_p4",
                 "Pluto_p6", "The Voynich Manuscript_p7", "History of Film_p4",
                 "Prisoners Dilemma_p4", "Prisoners Dilemma_p5", "Prisoners Dilemma_p8"]

# story|page -> {sent_idx: [importance 1-5, in_summary 0/1]}
ANN = {
"pluto|0": {0:[5,1],1:[3,0],2:[4,0],3:[3,0],4:[4,0],5:[4,0],6:[4,0],7:[1,0],8:[2,0],9:[5,1],10:[4,1],11:[4,0]},
"pluto|1": {0:[5,1],1:[5,1],2:[3,0],3:[4,0],4:[4,1],5:[3,0],6:[4,0],7:[2,0],8:[3,0],9:[3,0]},
"pluto|2": {0:[3,0],1:[5,1],2:[2,0],3:[4,0],4:[3,0],5:[2,0],6:[3,0],7:[4,1],8:[5,1],9:[4,0]},
"pluto|3": {0:[4,0],1:[5,1],2:[5,1],3:[2,0],4:[3,0],5:[2,0],6:[4,0],7:[2,0],8:[2,0],9:[2,0],10:[5,1],11:[3,0]},
"pluto|4": {0:[3,0],1:[4,1],2:[3,0],3:[4,1],4:[2,0],5:[2,0],6:[4,1],7:[2,0],8:[1,0],9:[3,0],10:[3,0],11:[3,0]},
"pluto|5": {0:[3,0],1:[2,0],2:[2,0],3:[1,0],4:[1,0],5:[5,1],6:[4,1],7:[4,0],8:[5,1]},
"pluto|6": {0:[4,0],1:[1,0],2:[5,1],3:[3,0],4:[5,1],5:[5,1],6:[4,0],7:[3,0],8:[4,0],9:[3,0]},
"pluto|7": {0:[4,1],1:[2,0],2:[3,0],3:[5,1],4:[5,1],5:[3,0],6:[4,0],7:[5,1]},
"pluto|8": {0:[4,0],1:[5,1],2:[5,1],3:[3,0],4:[5,1],5:[5,1],6:[5,0],7:[3,0],8:[1,0],9:[3,0],10:[4,0]},
"pluto|9": {0:[4,1],1:[5,1],2:[4,0],3:[4,0],4:[3,0],5:[3,0],6:[4,1],7:[1,0],8:[2,0],9:[4,0],10:[3,0],11:[2,0]},

"the_voynich_manuscript|0": {0:[5,1],1:[5,1],2:[5,1],3:[4,0],4:[5,0],5:[3,0],6:[4,0],7:[3,0],8:[2,0],9:[3,0]},
"the_voynich_manuscript|1": {0:[3,0],1:[2,0],2:[4,1],3:[4,1],4:[3,0],5:[5,1],6:[4,0],7:[3,0],8:[4,0]},
"the_voynich_manuscript|2": {0:[4,0],1:[3,0],2:[5,1],3:[3,0],4:[2,0],5:[2,0],6:[4,0],7:[5,1],8:[4,1],9:[5,0],10:[4,0],11:[4,0]},
"the_voynich_manuscript|3": {0:[4,0],1:[4,1],2:[3,0],3:[2,0],4:[3,0],5:[3,0],6:[5,1],7:[3,0],8:[4,0],9:[4,0],10:[5,1],11:[4,0]},
"the_voynich_manuscript|4": {0:[4,1],1:[3,0],2:[5,1],3:[5,1],4:[2,0],5:[4,0],6:[5,1],7:[4,0],8:[3,0],9:[3,0],10:[2,0],11:[1,0],12:[4,0]},
"the_voynich_manuscript|5": {0:[4,1],1:[5,1],2:[4,0],3:[3,0],4:[4,1],5:[2,0],6:[4,1],7:[5,1],8:[4,0],9:[5,0]},
"the_voynich_manuscript|6": {0:[5,1],1:[4,0],2:[3,0],3:[5,1],4:[4,0],5:[4,0],6:[3,0],7:[4,1]},
"the_voynich_manuscript|7": {0:[4,1],1:[5,1],2:[5,1],3:[4,0],4:[3,0],5:[5,1],6:[3,0],7:[4,0],8:[3,0]},
"the_voynich_manuscript|8": {0:[3,0],1:[5,1],2:[5,1],3:[4,0],4:[5,1],5:[4,0],6:[4,1],7:[3,0],8:[4,0],9:[4,0]},
"the_voynich_manuscript|9": {0:[4,1],1:[4,0],2:[5,1],3:[5,1],4:[3,0],5:[4,0],6:[3,0],7:[3,0],8:[5,1],9:[4,0]},

"history_of_film|0": {0:[5,1],1:[5,1],2:[5,1],3:[4,0],4:[3,0],5:[3,0],6:[4,0],7:[4,0],8:[2,0],9:[3,0],10:[3,0],11:[4,0]},
"history_of_film|1": {0:[4,0],1:[4,1],2:[4,1],3:[5,1],4:[3,0],5:[5,1],6:[3,0],7:[4,0],8:[4,0],9:[4,0],10:[2,0],11:[4,0],12:[4,1],13:[4,0]},
"history_of_film|2": {0:[4,0],1:[3,0],2:[4,1],3:[4,0],4:[3,0],5:[4,1],6:[3,0],7:[5,1],8:[5,1],9:[4,0]},
"history_of_film|3": {0:[4,0],1:[3,0],2:[3,0],3:[4,0],4:[4,0],5:[5,1],6:[5,1],7:[3,0],8:[3,0],9:[4,0],10:[3,0],11:[5,1],12:[4,0]},
"history_of_film|4": {0:[4,1],1:[5,1],2:[5,1],3:[4,0],4:[3,0],5:[5,1],6:[4,0],7:[4,0],8:[3,0],9:[4,0],10:[4,0]},
"history_of_film|5": {0:[4,0],1:[5,1],2:[5,1],3:[4,0],4:[3,0],5:[4,1],6:[4,0],7:[5,1],8:[4,0]},
"history_of_film|6": {0:[4,0],1:[4,0],2:[5,1],3:[5,1],4:[5,1],5:[5,1],6:[4,0],7:[4,0],8:[4,0]},
"history_of_film|7": {0:[5,1],1:[4,1],2:[4,0],3:[5,1],4:[3,0],5:[4,0],6:[4,1]},
"history_of_film|8": {0:[4,0],1:[4,1],2:[5,1],3:[4,0],4:[4,0],5:[3,0],6:[5,1],7:[4,0],8:[5,1],9:[4,0]},
"history_of_film|9": {0:[4,0],1:[4,1],2:[5,1],3:[5,1],4:[4,0],5:[5,1],6:[4,0],7:[5,1],8:[4,0]},

"serena_williams|0": {0:[5,1],1:[5,1],2:[4,0],3:[3,0],4:[5,1],5:[4,0],6:[5,1],7:[4,0],8:[3,0],9:[4,0]},
"serena_williams|1": {0:[4,1],1:[4,0],2:[4,1],3:[3,0],4:[5,1],5:[4,0],6:[5,1],7:[5,1],8:[4,0],9:[3,0]},
"serena_williams|2": {0:[3,0],1:[5,1],2:[4,0],3:[3,0],4:[5,1],5:[5,1],6:[4,0],7:[5,1],8:[3,0],9:[3,0]},
"serena_williams|3": {0:[5,1],1:[4,0],2:[3,0],3:[5,1],4:[4,0],5:[4,0],6:[4,0],7:[3,0],8:[4,1]},
"serena_williams|4": {0:[4,0],1:[4,1],2:[5,1],3:[5,1],4:[4,0],5:[4,0],6:[4,1],7:[4,0],8:[4,0],9:[3,0]},
"serena_williams|5": {0:[3,0],1:[4,0],2:[5,1],3:[4,0],4:[3,0],5:[4,1],6:[5,1],7:[3,0],8:[3,0],9:[2,0],10:[4,0],11:[5,1]},
"serena_williams|6": {0:[4,0],1:[3,0],2:[3,0],3:[5,1],4:[5,1],5:[3,0],6:[3,0],7:[5,1],8:[4,0],9:[4,0]},
"serena_williams|7": {0:[4,0],1:[5,1],2:[5,1],3:[3,0],4:[2,0],5:[4,1],6:[3,0],7:[3,0],8:[3,0],9:[3,0],10:[3,0],11:[3,0]},
"serena_williams|8": {0:[3,0],1:[3,0],2:[5,1],3:[3,0],4:[3,0],5:[3,0],6:[5,1],7:[4,0],8:[5,1],9:[5,1],10:[4,0],11:[3,0],12:[2,0],13:[2,0],14:[3,0]},
"serena_williams|9": {0:[3,0],1:[5,1],2:[3,0],3:[5,1],4:[4,0],5:[5,1],6:[5,1],7:[4,0],8:[5,1]},

"prisoners_dilemma|0": {0:[5,1],1:[4,0],2:[5,1],3:[5,1],4:[4,0],5:[4,0],6:[4,0],7:[5,1],8:[5,1],9:[5,1],10:[5,1],11:[4,0]},
"prisoners_dilemma|1": {0:[4,0],1:[5,1],2:[5,1],3:[5,1],4:[5,1],5:[5,1],6:[4,0],7:[4,0]},
"prisoners_dilemma|2": {0:[4,1],1:[3,0],2:[3,0],3:[5,1],4:[5,1],5:[4,0],6:[5,1],7:[5,1],8:[4,0],9:[4,0]},
"prisoners_dilemma|3": {0:[4,0],1:[5,1],2:[5,1],3:[4,0],4:[4,0],5:[4,0],6:[5,1],7:[4,0],8:[5,1],9:[4,0],10:[5,1],11:[4,0]},
"prisoners_dilemma|4": {0:[5,1],1:[4,0],2:[4,0],3:[5,1],4:[5,1],5:[5,1],6:[5,1],7:[4,0]},
"prisoners_dilemma|5": {0:[4,0],1:[3,0],2:[3,0],3:[3,0],4:[5,1],5:[5,1],6:[5,1],7:[4,0],8:[5,1],9:[4,0],10:[5,1],11:[4,0],12:[4,0]},
"prisoners_dilemma|6": {0:[4,0],1:[5,1],2:[5,1],3:[4,0],4:[5,1],5:[4,0],6:[4,0],7:[3,0],8:[5,1],9:[5,1]},
"prisoners_dilemma|7": {0:[5,1],1:[5,1],2:[4,0],3:[5,1],4:[4,1],5:[4,0],6:[4,0],7:[2,0],8:[4,0],9:[5,1],10:[4,0],11:[4,0],12:[4,0],13:[3,0]},
"prisoners_dilemma|8": {0:[3,0],1:[4,1],2:[5,1],3:[4,0],4:[4,0],5:[5,1],6:[5,1],7:[5,1],8:[4,0],9:[4,0],10:[4,0],11:[4,0]},
"prisoners_dilemma|9": {0:[5,1],1:[5,1],2:[4,0],3:[5,1],4:[4,0],5:[4,0],6:[5,1],7:[5,1],8:[4,0],9:[4,0]},
}

if __name__ == "__main__":
    inv = json.loads((OUT / "sentence_inventory.json").read_text())
    assert set(inv) == set(ANN), (set(inv) ^ set(ANN))
    rows = []
    for key, sents in inv.items():
        a = ANN[key]
        assert set(a) == {s["sent_idx"] for s in sents}, f"{key}: index mismatch"
        story, page = key.split("|")
        for s in sents:
            imp, summ = a[s["sent_idx"]]
            assert imp in (1, 2, 3, 4, 5) and summ in (0, 1)
            rows.append({"story_phys": story, "page": int(page), "sent_idx": s["sent_idx"],
                         "sentence_id": s["sentence_id"], "n_words_on_page": s["n_words"],
                         "importance_llm": imp, "in_summary_llm": summ,
                         "sentence": s["sentence"]})
    import pandas as pd
    D = pd.DataFrame(rows)
    D.to_parquet(OUT / "importance_llm.parquet", index=False)
    print(f"{len(D)} sentence ratings, {D.story_phys.nunique()} stories, {D.groupby(['story_phys','page']).ngroups} pages")
    print(D["importance_llm"].value_counts().sort_index().to_string())
    print("\nmean importance by story:")
    print(D.groupby("story_phys")["importance_llm"].agg(["mean", "std"]).round(3).to_string())
    wp = D.groupby(["story_phys", "page"])["importance_llm"].std()
    print(f"\nwithin-page SD of importance: median {wp.median():.3f}, min {wp.min():.3f} "
          f"(pages with SD<0.5: {(wp < 0.5).sum()}/{len(wp)})")
    print(f"in_summary rate {D.in_summary_llm.mean():.3f}; "
          f"corr(importance,in_summary) r={D.importance_llm.corr(D.in_summary_llm):.3f}")
