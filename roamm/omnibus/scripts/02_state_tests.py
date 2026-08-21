#!/usr/bin/env python3
"""State tests on the omnibus read-out: gate, retention, ladder, placebo, permutation.

  gate        does the text block buy anything on held-out on-task reading, and does the
              within-page word permutation remove it
  retention   D(mind-wandering) / D(on-task), with a one-sided bound and the smallest
              change detectable at 80% power
  ladder      the state coefficient with reader, launch-line, word and reader-by-page
              fixed effects absorbed in turn, reader-clustered SE
  placebo     the known additive lengthening of fixations through the identical ladder
  permutation each reader's mind-wandering labels replaced by another reader's, transferred
              by article, page and word position, which preserves episode length, contiguity
              and where in the text episodes fall

Reads policy_D_*.npz from 01_fit_policy.py; writes one JSON per tag.
"""
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
from scipy import stats
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ART, RES, boot_ci, boot_ratio, per_reader, group_means, fe_ols

READOUTS = [("target", "Dtgt", "bits"), ("duration", "Ddur", "log-ms^2"),
            ("duration_nonlinear", "Ddur_mlp", "log-ms^2")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="real")
    ap.add_argument("--nperm", type=int, default=2000)
    a = ap.parse_args()

    z = np.load(ART / f"policy_D_{a.tag}.npz", allow_pickle=True)
    sj, mw = z["subject"], z["mw"]
    subs = np.unique(sj)
    scode = pd.factorize(sj)[0]
    wcode = pd.factorize(z["word_key"])[0]
    story = pd.Series(z["story"]).astype(str)
    page = pd.Series(z["page"]).astype(str)
    # launch location is the LINE. (story, page, line, line_pos) is a bijection with word_key,
    # so using it here would absorb exactly what the word fixed effect absorbs.
    line_fe = pd.factorize(story + "|" + page + "|" + pd.Series(z["line"]).astype(str))[0]
    reader_page = pd.factorize(pd.Series(sj).astype(str) + "|" + story + "|" + page)[0]
    kcode = pd.factorize(story + "|" + page + "|" + pd.Series(z["pos"]).astype(str))[0]

    res = {"tag": a.tag, "n_transitions": int(len(sj)), "n_mw": int(mw.sum()),
           "n_readers": int(len(subs))}

    for name, key, unit in READOUTS:
        v = np.asarray(z[key], float)
        fin = np.isfinite(v)
        don = per_reader(v, sj, (mw == 0) & fin, subs)
        dmw = per_reader(v, sj, (mw == 1) & fin, subs)
        ok = np.isfinite(don) & np.isfinite(dmw)
        gate = boot_ci(don)
        ret = boot_ratio(dmw[ok], don[ok])
        ret["mde80_pct"] = ret["mde80_pct"]
        entry = {"unit": unit, "gate": gate, "retention": ret,
                 "D_on": float(don[ok].mean()), "D_mw": float(dmw[ok].mean())}
        print(f"{name:18s} gate  {gate['mean']:+.6f} [{gate['ci'][0]:+.6f},{gate['ci'][1]:+.6f}] "
              f"t={gate['t']:.2f} p={gate['p']:.3g}  {gate['n_pos']}/{gate['n']} readers", flush=True)
        print(f"{name:18s} reten {ret['retention']:+.3f} [{ret['ci'][0]:+.3f},{ret['ci'][1]:+.3f}]  "
              f"one-sided lower {ret['one_sided_lower_95']:+.3f}  MDE80 {ret['mde80_pct']:.0f}%", flush=True)

        ladder = {}
        for lname, fes in [("L0 reader", [scode[fin]]),
                           ("L1 +launch line", [scode[fin], line_fe[fin]]),
                           ("L2 +word", [scode[fin], wcode[fin]]),
                           ("L3 +reader x page", [reader_page[fin], wcode[fin]])]:
            b, se, p = fe_ols(v[fin], mw[fin][:, None], fes, scode[fin])
            ladder[lname] = dict(beta=b, se=se, p=p)
            print(f"    {lname:20s} beta_mw={b:+.6f} se={se:.6f} p={p:.4g}", flush=True)
        entry["ladder"] = ladder
        res[name] = entry

    # placebo: a state effect that is known to be real, through the identical ladder
    y = np.asarray(z["log_fix_dur"], float)
    placebo = {}
    for lname, fes in [("L0 reader", [scode]), ("L1 +launch line", [scode, line_fe]),
                       ("L2 +word", [scode, wcode]), ("L3 +reader x page", [reader_page, wcode])]:
        b, se, p = fe_ols(y, mw[:, None], fes, scode)
        placebo[lname] = dict(beta_log=b, pct=float(100 * (np.exp(b) - 1)), se=se, p=p)
        print(f"placebo fixation duration {lname:20s} {100*(np.exp(b)-1):+.2f}%  p={p:.3g}", flush=True)
    res["placebo_fixation_duration"] = placebo

    # cross-reader label swap, preserving episode structure and where episodes fall
    nk = kcode.max() + 1
    sidx = {s: i for i, s in enumerate(subs)}
    rows = [np.flatnonzero(sj == s) for s in subs]   # precomputed: the permutation loop is hot
    donor = np.zeros((len(subs), nk))
    for i, s in enumerate(subs):
        donor[i, kcode[rows[i]]] = mw[rows[i]]
    perm = {}
    for name, key, _ in READOUTS:
        v = np.asarray(z[key], float)
        fin = np.isfinite(v)
        vs = np.where(fin, v, 0.0)
        don = per_reader(v, sj, (mw == 0) & fin, subs)
        dmw = per_reader(v, sj, (mw == 1) & fin, subs)
        ok = np.isfinite(don) & np.isfinite(dmw)
        obs = dmw[ok].mean() - don[ok].mean()
        rng = np.random.default_rng(66)
        null = np.empty(a.nperm)
        for b in range(a.nperm):
            order = rng.permutation(len(subs))
            fake = np.zeros_like(mw)
            for i, d in enumerate(order):
                r = rows[i]
                fake[r] = donor[d, kcode[r]]
            fo = group_means(vs, scode, (fake == 0) & fin, len(subs))
            fm = group_means(vs, scode, (fake == 1) & fin, len(subs))
            o2 = np.isfinite(fo) & np.isfinite(fm)
            null[b] = fm[o2].mean() - fo[o2].mean()
        perm[name] = dict(observed=float(obs), null_mean=float(null.mean()),
                          null_sd=float(null.std()),
                          z=float((obs - null.mean()) / (null.std() + 1e-15)),
                          p=float((np.abs(null - null.mean()) >= abs(obs - null.mean())).mean()),
                          n_draws=int(a.nperm))
        print(f"permutation {name:18s} obs={obs:+.6f} null={null.mean():+.6f}+-{null.std():.6f} "
              f"z={perm[name]['z']:+.2f} p={perm[name]['p']:.4f}", flush=True)
    res["permutation_cross_reader_swap"] = perm

    # is the extra dwell during mind-wandering predictable from the text at all (SI S2)
    TD = np.asarray(z["TD"], float)
    e = y - np.asarray(z["pred_text"], float)
    undirected = {}
    for lab, sel in [("mind_wandering", mw == 1), ("on_task", mw == 0)]:
        idx = np.flatnonzero(sel & np.isfinite(e))
        folds = np.array([hash(int(g)) % 5 for g in sj[idx]])
        pred = np.full(len(idx), np.nan)
        for f in range(5):
            tr_i, te_i = idx[folds != f], idx[folds == f]
            X = np.column_stack([np.ones(len(tr_i)), TD[tr_i]])
            XtX = X.T @ X
            w = np.linalg.solve(XtX + 1e-3 * np.trace(XtX) / X.shape[1] * np.eye(X.shape[1]),
                                X.T @ e[tr_i])
            pred[folds == f] = np.column_stack([np.ones(len(te_i)), TD[te_i]]) @ w
        r2 = 1 - np.sum((e[idx] - pred) ** 2) / np.sum((e[idx] - e[idx].mean()) ** 2)
        undirected[lab] = dict(held_out_r2=float(r2), n=int(len(idx)))
        print(f"undirected effort, held-out R2 ({lab}): {r2:+.5f}  n={len(idx)}", flush=True)
    res["undirected_effort"] = undirected

    (RES / f"state_tests_{a.tag}.json").write_text(json.dumps(res, indent=2))
    print("wrote", RES / f"state_tests_{a.tag}.json")


if __name__ == "__main__":
    main()
