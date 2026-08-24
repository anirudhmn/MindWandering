#!/usr/bin/env python3
"""Bridge between embedding-to-EEG alignment loss and the global rescaling of the
fixation-locked response.

Sun & Jangraw (2026) report that ridge encoding models mapping language-model embeddings
onto fixation-aligned EEG predict held-out activity less well during mind-wandering, with
the reduction much larger for spectral power than for fixation-related potentials.  This
script asks whether the FRP part of that reduction is what a content-independent
proportional rescaling of the fixation-locked response already predicts.

Encoding accuracy is a Pearson correlation, so it factorises exactly:

    r = beta * sigma_pred / sigma_obs

where beta is the slope of observed on predicted.  Hence

  r_mw / r_on = (beta_mw/beta_on) * (sigma_pred_mw/sigma_pred_on) * (sigma_obs_on/sigma_obs_mw)
                 ^ response gain     ^ stimulus term (~1 by design)  ^ noise term

The response-gain term is the quantity the topographic analysis estimates independently as
0.742 (a 25.8% attenuation) from a leave-one-reader-out proportional fit that leaves no
reliable residual.  If the gain term recovered here matches it, and the encoding weights
point the same way in both states, then the FRP alignment loss is a change in how large the
response is, not in what it encodes.

The encoder is pooled across readers, as in the target study: per-reader fits are at the
signal-to-noise floor (SI S11) and cannot resolve the contrast either way.

Outputs results/bridge_alignment.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
ITER = ROOT / "roamm" / "bridge"
RES = ITER / "results"
COUP = ROOT / "roamm" / "artifacts" / "coupling"
GPT2 = ROOT / "roamm" / "encoding" / "artifacts"

SEED = 6703
WIN_MS = (200.0, 300.0)          # Sun & Jangraw's FRP alignment peak
BASE_MS = (-100.0, 0.0)
N_FOLDS = 5                      # page-level, as in their pipeline
LAMBDAS = np.array([1e1, 1e2, 1e3, 1e4, 1e5])
N_BOOT = 2000
LAYERS = (0, 2, 4, 6, 8, 10, 12)
PC_GRID = (16, 50)

# Independent estimate from roamm/topography (leave-one-reader-out proportional fit).
GAIN_TOPO = 0.7422362234104763
GAIN_TOPO_CI = (1 - 0.39678726345300674, 1 - 0.13831990323960786)


def load_targets() -> tuple[np.ndarray, pd.DataFrame]:
    """Baseline-corrected mean amplitude in the 200-300 ms window, per channel."""
    t = np.load(COUP / "frp_epochs_time.npy") * 1000.0
    win = (t >= WIN_MS[0]) & (t <= WIN_MS[1])
    base = (t >= BASE_MS[0]) & (t <= BASE_MS[1])
    ep = np.load(COUP / "frp_epochs.npy", mmap_mode="r")

    n = ep.shape[0]
    y = np.empty((n, ep.shape[1]), dtype=np.float64)
    for i in range(0, n, 20000):
        blk = np.asarray(ep[i : i + 20000], dtype=np.float64)
        y[i : i + 20000] = blk[:, :, win].mean(axis=2) - blk[:, :, base].mean(axis=2)
    return y * 1e6, pd.read_parquet(COUP / "fixations_frp.parquet")


def build_sample(y: np.ndarray, meta: pd.DataFrame):
    """Per-reader balanced MW/on-task sample, scaled by each reader's on-task SD.

    Scaling uses on-task statistics only, so the mind-wandering/on-task amplitude ratio --
    the quantity under test -- passes through unchanged.
    """
    is_mw = meta["is_mw"].to_numpy().astype(bool)
    subj = meta["subject"].to_numpy()
    keep, labels, readers = [], [], []
    for s in np.unique(subj):
        sel = subj == s
        mw_ix = np.flatnonzero(sel & is_mw)
        on_ix = np.flatnonzero(sel & ~is_mw)
        if len(mw_ix) < 50:
            continue
        rng = np.random.default_rng(SEED + int(s))
        on_ix = rng.choice(on_ix, size=len(mw_ix), replace=False)
        sd = y[np.flatnonzero(sel & ~is_mw)].std(axis=0) + 1e-12
        y[mw_ix] /= sd
        y[on_ix] /= sd
        keep.append(np.concatenate([mw_ix, on_ix]))
        labels.append(np.concatenate([np.ones(len(mw_ix), bool), np.zeros(len(mw_ix), bool)]))
        readers.append(np.full(2 * len(mw_ix), s))
    return np.concatenate(keep), np.concatenate(labels), np.concatenate(readers)


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    g = x.T @ x + lam * np.eye(x.shape[1])
    return np.linalg.solve(g, x.T @ y)


def cv_encode(emb_w: np.ndarray, y: np.ndarray, lab: np.ndarray, fold: np.ndarray, n_pc: int):
    """Page-level cross-validated prediction, plus state-specific encoders."""
    pred = np.zeros_like(y)
    w_mw, w_on = [], []
    halves: dict[str, list] = {k: [] for k in ("onA", "onB", "mwA", "mwB")}
    for f in range(N_FOLDS):
        te, tr = fold == f, fold != f
        mu = emb_w[tr].mean(axis=0)
        _, _, vt = np.linalg.svd(emb_w[tr] - mu, full_matrices=False)
        comp = vt[:n_pc]
        xtr, xte = (emb_w[tr] - mu) @ comp.T, (emb_w[te] - mu) @ comp.T
        m, sd = xtr.mean(axis=0), xtr.std(axis=0) + 1e-12
        xtr, xte = (xtr - m) / sd, (xte - m) / sd

        ytr = y[tr]
        ymu = ytr.mean(axis=0)
        inner = np.random.default_rng(SEED + f).random(tr.sum()) < 0.8
        best, best_lam = -np.inf, LAMBDAS[0]
        for lam in LAMBDAS:
            b = ridge_fit(xtr[inner], ytr[inner] - ytr[inner].mean(axis=0), lam)
            p, o = xtr[~inner] @ b, ytr[~inner] - ytr[inner].mean(axis=0)
            sc = float(np.mean([np.corrcoef(p[:, c], o[:, c])[0, 1] for c in range(p.shape[1])]))
            if sc > best:
                best, best_lam = sc, lam
        pred[te] = xte @ ridge_fit(xtr, ytr - ymu, best_lam) + ymu

        for store, mask in ((w_mw, lab[tr]), (w_on, ~lab[tr])):
            yy = ytr[mask]
            store.append(ridge_fit(xtr[mask], yy - yy.mean(axis=0), best_lam))

        # Matched-size half-splits, so that a between-state weight similarity can be read
        # against the within-state reliability its own noise permits.
        rh = np.random.default_rng(SEED + 100 + f)
        on_ix, mw_ix = np.flatnonzero(~lab[tr]), np.flatnonzero(lab[tr])
        h = min(len(on_ix), len(mw_ix)) // 2
        on_ix, mw_ix = rh.permutation(on_ix)[: 2 * h], rh.permutation(mw_ix)[: 2 * h]
        for name, part in (
            ("onA", on_ix[:h]), ("onB", on_ix[h:]), ("mwA", mw_ix[:h]), ("mwB", mw_ix[h:])
        ):
            yy = ytr[part]
            halves[name].append(ridge_fit(xtr[part], yy - yy.mean(axis=0), best_lam))
    return (
        pred,
        np.mean(w_mw, axis=0),
        np.mean(w_on, axis=0),
        {k: np.mean(v, axis=0) for k, v in halves.items()},
    )


def suff_stats(pred: np.ndarray, obs: np.ndarray, chans: np.ndarray) -> np.ndarray:
    """Per-channel sufficient statistics [n, Sp, So, Spp, Soo, Spo] for one group of trials.

    Correlations, slopes and SDs over any union of groups are exact functions of these sums,
    which is what makes a 2000-draw reader bootstrap affordable.
    """
    p, o = pred[:, chans], obs[:, chans]
    n = np.full(len(chans), float(len(p)))
    return np.stack([n, p.sum(0), o.sum(0), (p * p).sum(0), (o * o).sum(0), (p * o).sum(0)])


def from_suff(s: np.ndarray) -> dict:
    n, sp, so, spp, soo, spo = s
    cpp = spp - sp * sp / n
    coo = soo - so * so / n
    cpo = spo - sp * so / n
    return {
        "r": float(np.mean(cpo / np.sqrt(cpp * coo))),
        "beta": float(np.mean(cpo / cpp)),
        "sigma_pred": float(np.mean(np.sqrt(cpp / n))),
        "sigma_obs": float(np.mean(np.sqrt(coo / n))),
        "per_channel_r": (cpo / np.sqrt(cpp * coo)).tolist(),
    }


def run(layer: int, n_pc: int, emb: np.ndarray, word_ix, y, lab, readers, fold) -> dict:
    emb_w = emb[word_ix]
    pred, w_mw, w_on, halves = cv_encode(emb_w, y, lab, fold, n_pc)

    # Channels selected on on-task performance alone, which is blind to the state contrast.
    on = ~lab
    r_on_all = np.array(
        [np.corrcoef(pred[on, c], y[on, c])[0, 1] for c in range(y.shape[1])]
    )
    chans = np.flatnonzero(r_on_all >= np.quantile(r_on_all, 0.75))

    uniq = np.unique(readers)
    ss = {
        st: np.stack([
            suff_stats(pred[(readers == s) & m], y[(readers == s) & m], chans) for s in uniq
        ])
        for st, m in (("on", on), ("mw", lab))
    }
    s_on = from_suff(ss["on"].sum(0))
    s_mw = from_suff(ss["mw"].sum(0))

    out = {
        "layer": layer, "n_pc": n_pc, "n_trials": int(len(y)),
        "n_readers": int(len(np.unique(readers))),
        "n_channels_used": int(len(chans)),
        "on_task": {k: v for k, v in s_on.items() if k != "per_channel_r"},
        "mind_wandering": {k: v for k, v in s_mw.items() if k != "per_channel_r"},
    }
    out["alignment"] = {
        "r_on": s_on["r"], "r_mw": s_mw["r"],
        "diff": s_on["r"] - s_mw["r"],
        "ratio": s_mw["r"] / s_on["r"] if s_on["r"] != 0 else float("nan"),
    }
    out["decomposition"] = {
        "gain_term": s_mw["beta"] / s_on["beta"],
        "stimulus_term": s_mw["sigma_pred"] / s_on["sigma_pred"],
        "noise_term": s_on["sigma_obs"] / s_mw["sigma_obs"],
    }
    out["decomposition"]["product"] = float(
        np.prod([out["decomposition"][k] for k in ("gain_term", "stimulus_term", "noise_term")])
    )

    # Bootstrap over readers, resampling the per-reader sufficient statistics.
    rng = np.random.default_rng(SEED)
    boots: dict[str, list[float]] = {k: [] for k in ("ratio", "diff", "gain_term", "noise_term")}
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        a = from_suff(ss["on"][pick].sum(0))
        b = from_suff(ss["mw"][pick].sum(0))
        boots["ratio"].append(b["r"] / a["r"] if a["r"] != 0 else np.nan)
        boots["diff"].append(a["r"] - b["r"])
        boots["gain_term"].append(b["beta"] / a["beta"])
        boots["noise_term"].append(a["sigma_obs"] / b["sigma_obs"])
    out["bootstrap_ci95"] = {
        k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
        for k, v in boots.items()
    }
    g = np.asarray(boots["gain_term"], dtype=float)
    out["gain_vs_topography"] = {
        "topographic_gain": GAIN_TOPO,
        "topographic_gain_ci95": list(GAIN_TOPO_CI),
        "encoding_gain": out["decomposition"]["gain_term"],
        "encoding_gain_ci95": out["bootstrap_ci95"]["gain_term"],
        "boot_p_two_sided_vs_topo": float(
            2 * min(np.nanmean(g < GAIN_TOPO), np.nanmean(g > GAIN_TOPO))
        ),
    }

    # Representational test: are the two states' encoders the same map at different size?
    a, b = w_on[:, chans].ravel(), w_mw[:, chans].ravel()
    hv = {k: v[:, chans].ravel() for k, v in halves.items()}
    cor = lambda u, v: float(np.corrcoef(u, v)[0, 1])
    within_on, within_mw = cor(hv["onA"], hv["onB"]), cor(hv["mwA"], hv["mwB"])
    between = float(np.mean([cor(hv[i], hv[j]) for i in ("onA", "onB") for j in ("mwA", "mwB")]))
    ceiling = np.sqrt(max(within_on * within_mw, 0.0))
    out["representation"] = {
        "weight_similarity_r": cor(a, b),
        "projection_scale": float((b @ a) / (a @ a)),
        "norm_ratio": float(np.linalg.norm(b) / np.linalg.norm(a)),
        "half_split_within_on": within_on,
        "half_split_within_mw": within_mw,
        "half_split_between_states": between,
        "noise_ceiling": float(ceiling),
        "similarity_over_ceiling": float(between / ceiling) if ceiling > 0 else float("nan"),
    }
    return out


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    y_all, meta = load_targets()
    keep, lab, readers = build_sample(y_all, meta)
    y = y_all[keep]

    keys = pd.read_parquet(GPT2 / "gpt2_layer_keys.parquet")["word_key"].to_numpy()
    key_ix = {k: i for i, k in enumerate(keys)}
    word_ix = meta["word_key"].map(key_ix).to_numpy().astype(int)[keep]

    page_id = (meta["story"].astype(str) + "|" + meta["page"].astype(str)).to_numpy()[keep]
    pages = np.unique(page_id)
    fold_of = {p: i % N_FOLDS for i, p in enumerate(np.random.default_rng(SEED).permutation(pages))}
    fold = np.array([fold_of[p] for p in page_id])

    states = np.load(GPT2 / "gpt2_layer_states.npy", mmap_mode="r")
    report = {
        "window_ms": list(WIN_MS), "baseline_ms": list(BASE_MS),
        "n_folds": N_FOLDS, "n_boot": N_BOOT, "seed": SEED,
        "n_trials": int(len(y)), "n_readers": int(len(np.unique(readers))),
        "n_mw_fixations": int(lab.sum()),
        "runs": [],
    }
    for layer in LAYERS:
        emb = np.asarray(states[:, layer, :], dtype=np.float64)
        for n_pc in PC_GRID:
            print(f"layer {layer} pc {n_pc} ...", flush=True)
            report["runs"].append(run(layer, n_pc, emb, word_ix, y, lab, readers, fold))

    best = max(report["runs"], key=lambda d: d["alignment"]["r_on"])
    report["primary"] = {"layer": best["layer"], "n_pc": best["n_pc"]}
    (RES / "bridge_alignment.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(best, indent=1))


if __name__ == "__main__":
    main()
