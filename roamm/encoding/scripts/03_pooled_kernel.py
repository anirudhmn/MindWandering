#!/usr/bin/env python3
"""Pooled text kernel, held out on both the reader and the article.

The text kernel

    zipf, length, within-sentence surprisal, and the two extended-context terms

is fitted across readers on ON-TASK fixations only, on the nuisance residual from 02, and
evaluated on a reader-by-article cell that contributed nothing to the fit. Because the normal
equations are additive over cells, each of the 220 leave-one-reader-out by leave-one-article-out
folds comes from three precomputed sums by inclusion and exclusion, so all of them cost one pass
over the residual cache.

Read-out per held-out fixation and channel:

    D = mean over 0 to 500 ms of  resid^2 - (resid - text prediction)^2      [uV^2]

--shuffle SEED permutes which word occupies which page position, preserving layout, event timing
and the marginal distribution of every predictor: the negative control.
--text limits the block to a subset, which is how the length-only and ocular checks are run.

lam is swept. The primary value maximises on-task D, a choice that acts on the denominator of
the retention ratio and so cannot manufacture a retention value; the whole sweep is reported.
"""
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import (ART, RES, EEG_CH, TEXT_BASE, SF, build_XtX, fit_ridge, predict_run)

LAGS = np.arange(0, 129)           # 0 to 500 ms, matching what 02 stores
LAGS_MS = LAGS / SF * 1000
NL = len(LAGS)
RESID = ART / "resid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lams", default="0.01,0.1,1,10,100,1000")
    ap.add_argument("--tag", default="real")
    ap.add_argument("--shuffle", type=int, default=0)
    ap.add_argument("--text", default="", help="comma list restricting the text block")
    a = ap.parse_args()
    lams = [float(x) for x in a.lams.split(",")]
    text = a.text.split(",") if a.text else list(TEXT_BASE)
    cols = [TEXT_BASE.index(c) for c in text]
    NT = len(text)
    if a.text:
        print("text block restricted to", text, flush=True)

    ev = pd.read_parquet(ART / "events.parquet")
    mu = ev[TEXT_BASE].mean().to_numpy()[cols]
    sd = ev[TEXT_BASE].std().to_numpy()[cols]

    perm_map = None
    if a.shuffle:
        rng = np.random.default_rng(a.shuffle)
        w = ev.drop_duplicates("word_key")[["word_key", "story", "page"] + TEXT_BASE]
        parts = []
        for _, g in w.groupby(["story", "page"]):
            g = g.copy()
            g[TEXT_BASE] = g[TEXT_BASE].to_numpy()[rng.permutation(len(g))]
            parts.append(g[["word_key"] + TEXT_BASE])
        perm_map = pd.concat(parts).set_index("word_key")[TEXT_BASE].to_dict("index")
        oai2wk = dict(zip(ev.onset_abs_idx.to_numpy(), ev.word_key.to_numpy()))
        print(f"negative control: word features permuted within page (seed {a.shuffle})", flush=True)

    cells = []
    for f in sorted(RESID.glob("s*_r*.npz")):
        d = np.load(f, allow_pickle=True)
        if perm_map is None:
            X = d["text"].astype(np.float64)[:, cols]
        else:
            X = np.array([[perm_map[oai2wk[int(o)]][c] for c in TEXT_BASE]
                          for o in d["onset_abs_idx"]], float)[:, cols]
        X = (X - mu) / sd
        mw = d["mw"].astype(np.float64)
        Xm = X * (1.0 - mw)[:, None]                       # the fit sees on-task fixations only
        rel = d["onset_rel"]
        o = np.argsort(rel)
        R = d["resid"].astype(np.float64)
        cells.append(dict(subj=int(f.stem.split("_")[0][1:]), story=str(d["story"]),
                          XtX=build_XtX(rel[o], Xm[o], LAGS),
                          XtY=np.einsum("ep,elc->plc", Xm, R, optimize=True).reshape(NT * NL, 64),
                          X=X, mw=mw, rel=rel, runlen=int(d["runlen"]), resid=d["resid"],
                          oai=d["onset_abs_idx"]))
    subs = sorted({c["subj"] for c in cells})
    stories = sorted({c["story"] for c in cells})
    print(f"{len(cells)} reader-by-run cells, {len(subs)} readers", flush=True)

    TOTx = sum(c["XtX"] for c in cells)
    TOTy = sum(c["XtY"] for c in cells)
    BSx = {s: sum(c["XtX"] for c in cells if c["subj"] == s) for s in subs}
    BSy = {s: sum(c["XtY"] for c in cells if c["subj"] == s) for s in subs}
    BTx = {t: sum(c["XtX"] for c in cells if c["story"] == t) for t in stories}
    BTy = {t: sum(c["XtY"] for c in cells if c["story"] == t) for t in stories}

    sweep = {}
    for lam_scale in lams:
        D, sj, mw, oai = [], [], [], []
        for c in cells:
            XtX = TOTx - BSx[c["subj"]] - BTx[c["story"]] + c["XtX"]
            XtY = TOTy - BSy[c["subj"]] - BTy[c["story"]] + c["XtY"]
            beta, _ = fit_ridge(XtX, XtY, NT, NL, lam_scale=lam_scale)
            pred = predict_run(c["runlen"], c["rel"], c["X"], beta, LAGS)
            rows = c["rel"][:, None] + LAGS[None, :]
            R = c["resid"].astype(np.float64)
            P = pred[rows]
            D.append(((R ** 2 - (R - P) ** 2).mean(axis=1)).astype(np.float32))
            sj.append(np.full(len(c["rel"]), c["subj"]))
            mw.append(c["mw"])
            oai.append(c["oai"])
        D, sj, mw = np.concatenate(D), np.concatenate(sj), np.concatenate(mw)
        Dm = D.mean(1)
        don = np.array([Dm[(sj == s) & (mw == 0)].mean() for s in subs])
        dmw = np.array([Dm[(sj == s) & (mw == 1)].mean() if ((sj == s) & (mw == 1)).sum() >= 30
                        else np.nan for s in subs])
        sweep[str(lam_scale)] = dict(D_on=float(np.nanmean(don)), D_mw=float(np.nanmean(dmw)),
                                     readers_positive=int((don > 0).sum()))
        print(f"lam={lam_scale:>8}: D_on={np.nanmean(don):+.6f}  D_mw={np.nanmean(dmw):+.6f}  "
              f"({int((don>0).sum())}/{len(subs)} readers)", flush=True)
        np.savez_compressed(ART / f"pooled_{a.tag}_lam{lam_scale}.npz", D=D, subject=sj, mw=mw,
                            onset_abs_idx=np.concatenate(oai), lags_ms=LAGS_MS,
                            channels=np.array(EEG_CH))
    (RES / f"lambda_sweep_{a.tag}.json").write_text(json.dumps(sweep, indent=2))
    print("wrote", RES / f"lambda_sweep_{a.tag}.json")


if __name__ == "__main__":
    main()
