#!/usr/bin/env python3
"""Export all fixations and page/report intervals from the 47 GB synchronized frame.

Unlike the established fixation cache, this deliberately retains non-first-pass
fixations so post-report annotation and resumed reading can be reconstructed.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "roamm" / "artifacts" / "coupling"


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data" / "derivatives" / "features_df.pkl"
    print(f"loading {source} ...", flush=True)
    raw = pd.read_pickle(source)
    n = len(raw)
    print("loaded", raw.shape, flush=True)

    time = pd.to_numeric(raw["time"], errors="coerce").to_numpy()
    boundaries = np.empty(n, bool)
    boundaries[0] = False
    boundaries[1:] = time[1:] < time[:-1]
    run_uid = np.cumsum(boundaries, dtype=np.int32)
    subject = (run_uid // 5).astype(np.int16)
    run = pd.to_numeric(raw["run_num"], errors="coerce").to_numpy()

    # Page intervals are contiguous first-pass blocks. Metadata at the first sample
    # contains the actual onset/report-or-advance time for the whole block.
    fp = raw["first_pass_reading"].fillna(False).to_numpy(bool)
    page = pd.to_numeric(raw["page_num"], errors="coerce").to_numpy()
    prev_fp = np.r_[False, fp[:-1]]
    prev_page = np.r_[np.nan, page[:-1]]
    prev_uid = np.r_[-1, run_uid[:-1]]
    page_start_mask = fp & (~prev_fp | (page != prev_page) | (run_uid != prev_uid))
    pstarts = np.flatnonzero(page_start_mask)
    page_end_meta = pd.to_numeric(raw["page_end"], errors="coerce").to_numpy()
    page_start_meta = pd.to_numeric(raw["page_start"], errors="coerce").to_numpy()
    is_mw = raw["is_mw"].fillna(False).to_numpy(bool)
    mw_onset_arr = pd.to_numeric(raw["mw_onset"], errors="coerce").to_numpy()
    mw_offset_arr = pd.to_numeric(raw["mw_offset"], errors="coerce").to_numpy()
    story = raw["story_name"].astype("string").to_numpy()

    pages = []
    for i, s in enumerate(pstarts):
        e = s + 1
        while e < n and fp[e] and run_uid[e] == run_uid[s] and page[e] == page[s]:
            e += 1
        mw_values = np.flatnonzero(is_mw[s:e])
        mo = mw_onset_arr[s:e]
        mf = mw_offset_arr[s:e]
        pages.append({
            "subject": int(subject[s]),
            "run": int(run[s]),
            "run_uid": int(run_uid[s]),
            "story": str(story[s]),
            "page": int(page[s]),
            "page_start": float(page_start_meta[s]),
            "page_end": float(page_end_meta[s]),
            "has_mw": bool(len(mw_values)),
            "mw_onset": float(np.nanmin(mo)) if np.isfinite(mo).any() else np.nan,
            "mw_offset": float(np.nanmax(mf)) if np.isfinite(mf).any() else np.nan,
            "start_abs_idx": int(s),
            "end_abs_idx": int(e - 1),
        })
        if (i + 1) % 500 == 0:
            print("  pages", i + 1, "/", len(pstarts), flush=True)
    pages = pd.DataFrame(pages).sort_values(["subject", "run", "page"]).reset_index(drop=True)
    pages["next_page_start"] = pages.groupby(["subject", "run"])["page_start"].shift(-1)
    pages["post_interval_s"] = pages["next_page_start"] - pages["page_end"]
    pages.to_parquet(ART / "page_intervals.parquet", index=False)

    # One row per left-eye fixation, including annotation and resumed reading.
    fx_start = pd.to_numeric(raw["fix_L_tStart"], errors="coerce").to_numpy()
    valid = np.isfinite(fx_start)
    prev_fx = np.r_[np.nan, fx_start[:-1]]
    changed = (fx_start != prev_fx) | (run_uid != prev_uid)
    new_fix = valid & (changed | ~np.r_[False, valid[:-1]])
    ix = np.flatnonzero(new_fix)
    print("all left-eye fixations", len(ix), flush=True)

    def num(name: str) -> np.ndarray:
        return pd.to_numeric(raw[name], errors="coerce").to_numpy()[ix]

    fix = pd.DataFrame({
        "onset_abs_idx": ix.astype(np.int64),
        "subject": subject[ix],
        "run": run[ix].astype(np.int16),
        "run_uid": run_uid[ix],
        "story": story[ix],
        "tStart": fx_start[ix],
        "tSample_onset": num("tSample"),
        "fix_dur": num("fix_L_duration"),
        "x": num("fix_L_xAvg"),
        "y": num("fix_L_yAvg"),
        "pupil": num("fix_L_pupilAvg"),
        "first_pass": fp[ix],
        "page": page[ix],
        "is_mw": is_mw[ix],
        "word_key": raw["fix_L_fixed_word_key"].astype("string").to_numpy()[ix],
    })
    fix = fix.sort_values(["subject", "run", "tStart"]).reset_index(drop=True)
    fix["fix_order_all"] = fix.groupby(["subject", "run"]).cumcount().astype(np.int32)
    fix.to_parquet(ART / "all_fixations.parquet", index=False)

    report = {
        "n_samples": int(n),
        "n_subjects": int(pages.subject.nunique()),
        "n_runs": int(pages[["subject", "run"]].drop_duplicates().shape[0]),
        "n_pages": int(len(pages)),
        "n_mw_pages": int(pages.has_mw.sum()),
        "n_all_fixations": int(len(fix)),
        "n_firstpass_fixations": int(fix.first_pass.sum()),
        "n_nonfirstpass_fixations": int((~fix.first_pass).sum()),
        "median_post_interval_mw_s": float(pages.loc[pages.has_mw, "post_interval_s"].median()),
        "median_post_interval_nonmw_s": float(pages.loc[~pages.has_mw, "post_interval_s"].median()),
    }
    (ART / "extraction_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    del raw
    gc.collect()


if __name__ == "__main__":
    main()

