#!/usr/bin/env python3
"""Build a validated page-level comprehension table.

The ROAMM release ships one 4-alternative multiple-choice question per page
(option 5 = "I am not sure" = an explicit skip), ~10 pages per story x 5 stories,
plus per-article understandability / prior-knowledge Likerts. Those live in
reading_data/ (trial_level_data.csv + the *_questions.xlsx item banks) and were
never joined to the physiology.

This script:
  1. parses per-trial response / answer-key vectors into per-page outcomes,
  2. cross-validates the derived outcomes against the shipped accuracy,
     filtered_accuracy and num_skipped columns,
  3. cross-validates the shipped answer keys against the item-bank xlsx files,
  4. joins to the page-level reading table (page timing + MW report),
  5. writes artifacts/comprehension/pages.parquet.

Output row = one (subject, story, page) = one comprehension question.
"""
from __future__ import annotations
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RD = ROOT / "reading_data"
OUT = ROOT / "roamm" / "artifacts" / "comprehension"
OUT.mkdir(parents=True, exist_ok=True)

SKIP_CODE = 5.0  # "I am not sure"

# story label (trial/page tables) -> item-bank xlsx stem
XLSX = {
    "Pluto": "pluto",
    "The Voynich Manuscript": "the_voynich_manuscript",
    "History of Film": "history_of_film",
    "Serena Williams": "serena_williams",
    "Prisoners Dilemma": "prisoners_dilemma",
}
# the physiology artifacts (reading_fixations.parquet) use the same stems as the item banks
STORY_PHYS = dict(XLSX)

report: dict = {}

# ---------------------------------------------------------------- item bank
items = []
for story, stem in XLSX.items():
    x = pd.read_excel(RD / f"{stem}_questions.xlsx")
    q = x[x["Answer"].notna()].copy()
    q["question_index"] = q["question_index"].astype(int)
    for _, r in q.iterrows():
        items.append(
            {
                "reading": story,
                "page": int(r["question_index"]),
                "item_answer": float(r["Answer"]),
                "question_text": str(r["question_text"]).strip(),
                "n_options": int(sum(pd.notna(r[f"option{k}"]) for k in range(1, 6))),
            }
        )
    # the two trailing rows are the Likert self-reports
    likert = x[x["Answer"].isna()]["question_text"].tolist()
    report.setdefault("likert_prompts", {})[story] = [str(s).strip() for s in likert]
items = pd.DataFrame(items)
report["n_items"] = int(len(items))
report["items_per_story"] = items.groupby("reading").size().to_dict()
report["n_options_unique"] = sorted(items["n_options"].unique().tolist())

# ---------------------------------------------------------------- trial level
tr = pd.read_csv(RD / "trial_level_data.csv")
rows = []
for _, t in tr.iterrows():
    keys = ast.literal_eval(t["answer_keys"])
    resp = ast.literal_eval(t["responses"])
    assert len(keys) == len(resp) == 10, (t["sub_id"], t["reading"], len(keys), len(resp))
    for page, (k, r) in enumerate(zip(keys, resp)):
        rows.append(
            {
                "sub_id": t["sub_id"],
                "run": int(t["run"]),
                "reading": t["reading"],
                "page": page,
                "key": float(k),
                "response": float(r),
                "understand": float(t["understand"]),
                "prior_knowledge": float(t["prior_knowledge"]),
                "trial_accuracy": float(t["accuracy"]),
                "trial_filtered_accuracy": float(t["filtered_accuracy"]),
                "trial_num_skipped": int(t["num_skipped"]),
            }
        )
q = pd.DataFrame(rows)
q["skipped"] = (q["response"] == SKIP_CODE).astype(int)
q["correct"] = ((q["response"] == q["key"]) & (q["skipped"] == 0)).astype(int)
# correctness conditional on having committed to an answer (NaN where skipped)
q["correct_answered"] = np.where(q["skipped"] == 1, np.nan, q["correct"])

# ------------------------------------------------------- validation A: shipped scores
chk = q.groupby(["sub_id", "reading"]).agg(
    acc=("correct", "mean"),
    nskip=("skipped", "sum"),
    facc=("correct_answered", "mean"),
    ship_acc=("trial_accuracy", "first"),
    ship_nskip=("trial_num_skipped", "first"),
    ship_facc=("trial_filtered_accuracy", "first"),
)
report["validation"] = {
    "accuracy_max_abs_err": float(np.nanmax(np.abs(chk["acc"] - chk["ship_acc"]))),
    "filtered_accuracy_max_abs_err": float(np.nanmax(np.abs(chk["facc"] - chk["ship_facc"]))),
    "num_skipped_mismatches": int((chk["nskip"] != chk["ship_nskip"]).sum()),
    "n_trials_checked": int(len(chk)),
}

# ------------------------------------------------------- validation B: answer keys
m = q.merge(items, on=["reading", "page"], how="left", validate="many_to_one")
report["validation"]["answer_key_mismatches_vs_xlsx"] = int((m["key"] != m["item_answer"]).sum())
report["validation"]["responses_out_of_range"] = int(
    ((m["response"] < 1) | (m["response"] > 5)).sum()
)

# ------------------------------------------------------------------ page level
pg = pd.read_csv(RD / "reading_time_data.csv").drop(columns=["Unnamed: 0"])
d = m.merge(
    pg,
    on=["sub_id", "run", "reading", "page"],
    how="inner",
    validate="one_to_one",
)
assert len(d) == len(q), (len(d), len(q))

d["mw_frac_page"] = (d["mw_dur"] / d["page_dur"]).fillna(0.0).clip(0, 1)
d["log_page_dur"] = np.log(d["page_dur"].clip(lower=0.5))
d["story_phys"] = d["reading"].map(STORY_PHYS)
d["item"] = d["reading"] + "_p" + d["page"].astype(str)
# reading-order covariates
d["run_z"] = (d["run"] - d["run"].mean()) / d["run"].std()
d["page_z"] = (d["page"] - d["page"].mean()) / d["page"].std()

report["n_pages"] = int(len(d))
report["n_subjects"] = int(d["sub_id"].nunique())
report["overall"] = {
    "p_correct": float(d["correct"].mean()),
    "p_skipped": float(d["skipped"].mean()),
    "p_correct_given_answered": float(d["correct_answered"].mean()),
    "p_mw_page": float(d["is_MWreported"].mean()),
    "chance_level": 0.25,
}
report["per_story"] = (
    d.groupby("reading")
    .agg(p_correct=("correct", "mean"), p_skip=("skipped", "mean"), p_mw=("is_MWreported", "mean"))
    .round(4)
    .to_dict("index")
)
item_acc = d.groupby("item")["correct"].mean()
report["item_difficulty"] = {
    "min": float(item_acc.min()),
    "max": float(item_acc.max()),
    "mean": float(item_acc.mean()),
    "sd": float(item_acc.std()),
    "n_items_at_or_below_chance": int((item_acc <= 0.25).sum()),
}
subj_acc = d.groupby("sub_id")["correct"].mean()
report["subject_accuracy"] = {
    "min": float(subj_acc.min()),
    "max": float(subj_acc.max()),
    "mean": float(subj_acc.mean()),
    "sd": float(subj_acc.std()),
}

d.to_parquet(OUT / "pages.parquet", index=False)
(OUT / "build_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")

print(json.dumps(report, indent=2, default=str))
print(f"\nwrote {OUT/'pages.parquet'}  {d.shape}")
