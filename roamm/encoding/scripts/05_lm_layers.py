#!/usr/bin/env python3
"""Do language-model representations predict the reading brain better than word properties?

The text block of 03 is replaced by the leading principal components of GPT-2 hidden states at
layer l, and the same leave-one-reader-out by leave-one-article-out machinery is re-run per
layer. Layer 0 is the static input embedding, layer 12 is maximally contextual.

Two choices, both deliberate:

  * The components are fitted on all corpus words. This is an unsupervised transform of the
    stimulus alone, with no recording and no reader involved, so it cannot leak the outcome;
    fitting it per fold would break the inclusion-exclusion that makes the folds affordable.
  * hidden_states[0] is the token embedding plus the positional embedding, and deeper layers
    carry position too. Position in the passage is a geometry confound, so every layer's states
    are residualised on within-article token position, linear and quadratic, before the
    decomposition. The same correction is applied at every layer, so the comparison is fair.

A layer whose on-task gain does not clear the word-permutation control of 03 is not interpreted.

Needs the residual cache from 02 and the layer states from 05a_extract_lm_states.py.
"""
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
from scipy import stats
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ART, COUP, RES, build_XtX, fit_ridge, predict_run, boot_ci, boot_ratio

LAGS = np.arange(0, 129)
NL = len(LAGS)
RESID = ART / "resid"


def components(k, layers):
    V = np.load(ART / "lm_layer_states.npy")
    keys = pd.read_parquet(ART / "lm_layer_keys.parquet")["word_key"].to_numpy()
    # article label comes from the corpus table, not from the fixation table: a word that no
    # reader happened to fixate still belongs to its article and must be residualised with it
    story = pd.read_parquet(COUP / "word_features.parquet").set_index(
        "word_key")["story_file"].reindex(keys).to_numpy()
    out = {}
    for l in layers:
        H = V[:, l, :].astype(np.float64)
        for st in pd.unique(story):
            m = story == st
            t = np.arange(m.sum(), dtype=float)
            t = (t - t.mean()) / (t.std() + 1e-9)
            Z = np.column_stack([np.ones(m.sum()), t, t ** 2])
            H[m] -= Z @ np.linalg.lstsq(Z, H[m], rcond=None)[0]
        H -= H.mean(0)
        U, S, _ = np.linalg.svd(H, full_matrices=False)
        P = U[:, :k] * S[:k]
        out[l] = dict(P=((P - P.mean(0)) / (P.std(0) + 1e-9)),
                      evr=float((S[:k] ** 2).sum() / (S ** 2).sum()))
        print(f"layer {l:2d}: {k} components carry {out[l]['evr']*100:.1f}% of the "
              f"position-residualised state variance", flush=True)
    return keys, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--layers", default="0,2,4,6,8,10,12")
    ap.add_argument("--lam", type=float, default=None)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    lam = a.lam
    if lam is None:
        sw = json.loads((RES / "lambda_sweep_real.json").read_text())
        lam = float(max(sw, key=lambda kk: sw[kk]["D_on"]))
    ref = json.loads((RES / "lambda_sweep_real.json").read_text())[str(lam)]["D_on"]

    keys, feats = components(a.k, layers)
    kpos = {kk: i for i, kk in enumerate(keys)}
    ev = pd.read_parquet(ART / "events.parquet")[["onset_abs_idx", "word_key"]]
    oai2wk = dict(zip(ev.onset_abs_idx.to_numpy(), ev.word_key.to_numpy()))

    cells = []
    for f in sorted(RESID.glob("s*_r*.npz")):
        d = np.load(f, allow_pickle=True)
        cells.append(dict(subj=int(f.stem.split("_")[0][1:]), story=str(d["story"]),
                          widx=np.array([kpos[oai2wk[int(o)]] for o in d["onset_abs_idx"]]),
                          mw=d["mw"].astype(np.float64), rel=d["onset_rel"],
                          runlen=int(d["runlen"]), resid=d["resid"]))
    subs = sorted({c["subj"] for c in cells})
    stories = sorted({c["story"] for c in cells})
    NT = a.k
    out = {"word_property_reference_D_on": ref, "k": a.k}

    for l in layers:
        P = feats[l]["P"]
        for c in cells:
            X = P[c["widx"]]
            Xm = X * (1.0 - c["mw"])[:, None]
            o = np.argsort(c["rel"])
            c["_XtX"] = build_XtX(c["rel"][o], Xm[o], LAGS)
            c["_XtY"] = np.einsum("ep,elc->plc", Xm, c["resid"].astype(np.float64),
                                  optimize=True).reshape(NT * NL, 64)
            c["_X"] = X
        TOTx = sum(c["_XtX"] for c in cells)
        TOTy = sum(c["_XtY"] for c in cells)
        BSx = {s: sum(c["_XtX"] for c in cells if c["subj"] == s) for s in subs}
        BSy = {s: sum(c["_XtY"] for c in cells if c["subj"] == s) for s in subs}
        BTx = {t: sum(c["_XtX"] for c in cells if c["story"] == t) for t in stories}
        BTy = {t: sum(c["_XtY"] for c in cells if c["story"] == t) for t in stories}
        Ds, sjs, mws = [], [], []
        for c in cells:
            beta, _ = fit_ridge(TOTx - BSx[c["subj"]] - BTx[c["story"]] + c["_XtX"],
                                TOTy - BSy[c["subj"]] - BTy[c["story"]] + c["_XtY"],
                                NT, NL, lam_scale=lam)
            pred = predict_run(c["runlen"], c["rel"], c["_X"], beta, LAGS)
            rows = c["rel"][:, None] + LAGS[None, :]
            R = c["resid"].astype(np.float64)
            Ds.append((R ** 2 - (R - pred[rows]) ** 2).mean(axis=1).mean(axis=1))
            sjs.append(np.full(len(c["rel"]), c["subj"]))
            mws.append(c["mw"])
        D, sj, mw = np.concatenate(Ds), np.concatenate(sjs), np.concatenate(mws)
        don = np.array([D[(sj == s) & (mw == 0)].mean() for s in subs])
        dmw = np.array([D[(sj == s) & (mw == 1)].mean() if ((sj == s) & (mw == 1)).sum() >= 30
                        else np.nan for s in subs])
        ok = np.isfinite(don) & np.isfinite(dmw)
        gate = boot_ci(don)
        ret = boot_ratio(dmw[ok], don[ok])
        out[str(l)] = dict(evr=feats[l]["evr"], D_on=float(don.mean()),
                           D_mw=float(np.nanmean(dmw)), t=gate["t"], p=gate["p"],
                           readers_positive=int((don > 0).sum()),
                           retention=ret["retention"], retention_ci=ret["ci"],
                           one_sided_lower_95=ret["one_sided_lower_95"],
                           mde80_pct=ret["mde80_pct"],
                           beats_word_properties=bool(don.mean() > ref))
        print(f"layer {l:2d}: D_on={don.mean():+.6f} (t={gate['t']:.1f}, "
              f"{int((don>0).sum())}/{len(subs)})  retention={out[str(l)]['retention']:+.3f}  "
              f"MDE80={out[str(l)]['mde80_pct']:.0f}%", flush=True)
        for c in cells:
            for kk in ("_XtX", "_XtY", "_X"):
                c.pop(kk, None)

    dons = np.array([out[str(l)]["D_on"] for l in layers])
    rets = np.array([out[str(l)]["retention"] for l in layers])
    passing = [l for l in layers if out[str(l)]["D_on"] > 0 and out[str(l)]["p"] < 0.05]
    rho_all, p_all = stats.spearmanr(layers, rets)
    rho_ok, p_ok = stats.spearmanr(passing, [out[str(l)]["retention"] for l in passing])
    out["depth_profile"] = dict(
        layers=layers, D_on=dons.tolist(), retention=rets.tolist(),
        best_layer=int(layers[int(np.argmax(dons))]), gate_passing_layers=passing,
        spearman_all=dict(rho=float(rho_all), p=float(p_all)),
        spearman_gate_passing=dict(rho=float(rho_ok), p=float(p_ok)))
    (RES / "lm_layers.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["depth_profile"], indent=2))


if __name__ == "__main__":
    main()
