#!/usr/bin/env python3
"""Is the selection null an average over a diluted episode?

The relocation control in S3 moves each reader's mind-wandering spans to on-task
positions of matched length, holding the number of mind-wandering words fixed. That
tests displacement. It cannot see dilution: a brief interval in which selection is
sharply decoupled, followed by a long tail in which it is not, averages to the null we
report.

Here the marked span is cut into position bins and the selection contrast is recomputed
inside each. Same statistic as G1 (Somers' D of a word property predicting whether the
word was skipped, per reader, per state), same analysis set, same bootstrap. Three
binnings, because "brief" can mean three things:

  A  relative thirds of the span      dilution anywhere along it
  B  absolute words from span onset   a short decoupled interval at the start
  C  absolute words before span end   a short decoupled interval before noticing

Episodes are maximal runs of consecutive mind-wandering words in text order within a
reader and run; `pos` is complete, so a run break is a real state change. Binned
analyses use episodes of at least MIN_EP words, which is where a three-way cut means
anything; those episodes hold most of the mind-wandering words.

Each bin's D is compared against that reader's single on-task D, so the read-out is the
same retention percentage the paper reports. The paired early-versus-late test uses only
readers with a defined D in both bins and is the direct test of a position gradient.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy import stats
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ART, RES, boot_ci, somers_d, holm  # noqa: E402

PROPS = ["zipf", "length", "surprisal"]
G_PRIMARY = 4
MIN_EP = 15
MINN = 60


def episodes(T):
    """Label maximal runs of consecutive mind-wandering words and index within them."""
    T = T.sort_values(["subject", "run", "pos"]).reset_index(drop=True)
    brk = ((T.subject != T.subject.shift()) | (T.run != T.run.shift())
           | (T.is_mw.shift() != 1) | (T.pos != T.pos.shift() + 1))
    start = (T.is_mw == 1) & brk
    T["ep_id"] = np.where(T.is_mw == 1, start.cumsum(), -1)
    mw = T[T.is_mw == 1]
    ln = mw.groupby("ep_id").size()
    T["ep_len"] = T.ep_id.map(ln).fillna(0).astype(int)
    T["idx_on"] = np.where(T.is_mw == 1, T.groupby("ep_id").cumcount(), -1)
    T["idx_off"] = np.where(T.is_mw == 1, T.ep_len - 1 - T.idx_on, -1)
    T["rel"] = np.where(T.ep_len > 1, T.idx_on / (T.ep_len - 1).clip(lower=1), np.nan)
    return T


def analysis_set(T, G=G_PRIMARY):
    s = T[(T.skipped == 0) | (T.gap <= G)]
    return s[s.state_agree]


def d_on_task(S, minn=150):
    """One on-task Somers' D per reader, the baseline every bin is read against."""
    out = {}
    for s, g in S[S.is_mw == 0].groupby("subject"):
        if len(g) < minn or g.skipped.nunique() < 2:
            continue
        out[s] = {p: somers_d(g[p].to_numpy(), g.skipped.to_numpy()) for p in PROPS}
    return pd.DataFrame(out).T


def d_by_bin(MW, bincol, minn=MINN):
    """Somers' D per reader per bin, for mind-wandering words only."""
    rows = []
    for (s, b), g in MW.groupby(["subject", bincol], observed=True):
        if len(g) < minn or g.skipped.nunique() < 2:
            continue
        rec = {"subject": s, "bin": b, "n": len(g), "skiprate": float(g.skipped.mean())}
        for p in PROPS:
            rec[p] = somers_d(g[p].to_numpy(), g.skipped.to_numpy())
        rows.append(rec)
    return pd.DataFrame(rows)


def compare(D_bin, D_on, label, labels):
    """Retention and one-sided attenuation bound per bin, against each reader's own baseline."""
    res = {}
    print(f"\n=== {label} ===")
    for b in labels:
        sub = D_bin[D_bin["bin"] == b].set_index("subject")
        common = sub.index.intersection(D_on.index)
        r = {"n_readers": int(len(common)),
             "n_words": int(sub.loc[common, "n"].sum()) if len(common) else 0}
        ps = []
        for p in PROPS:
            v = (sub.loc[common, p] - D_on.loc[common, p]).to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v) < 3:
                r[p] = {"n": int(len(v))}
                ps.append(np.nan)
                continue
            ci = boot_ci(v)
            on = float(D_on.loc[common, p].mean())
            mw = float(sub.loc[common, p].mean())
            sgn = np.sign(on)
            # one-sided 95% bound on attenuation, as a percentage of the on-task effect
            lo = np.percentile(_boot(v), 5) if sgn > 0 else -np.percentile(_boot(v), 95)
            ci.update(D_on=on, D_mw=mw,
                      retention_pct=float(mw / on * 100) if on else np.nan,
                      attenuation_excluded_pct=float(-lo / abs(on) * 100) if on else np.nan)
            r[p] = ci
            ps.append(ci["p"])
        adj = holm([x for x in ps if np.isfinite(x)])
        k = 0
        for p in PROPS:
            if isinstance(r[p], dict) and "p" in r[p]:
                r[p]["p_holm"] = float(adj[k])
                k += 1
        res[str(b)] = r
        line = f"  {str(b):<22} readers={r['n_readers']:>2} words={r['n_words']:>6}  "
        for p in PROPS:
            if "retention_pct" in r[p]:
                line += (f"{p}: {r[p]['retention_pct']:6.1f}% "
                         f"(Δ={r[p]['mean']:+.4f} [{r[p]['ci'][0]:+.4f},{r[p]['ci'][1]:+.4f}]) ")
        print(line)
    return res


def _boot(v, n=10000, seed=59):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    return v[idx].mean(axis=1)


def paired(D_bin, first, last, label):
    """Direct within-reader test of a position gradient."""
    a = D_bin[D_bin["bin"] == first].set_index("subject")
    b = D_bin[D_bin["bin"] == last].set_index("subject")
    common = a.index.intersection(b.index)
    out = {"n_readers": int(len(common))}
    print(f"\n--- {label}: {first} vs {last}, paired within reader (n={len(common)}) ---")
    for p in PROPS:
        v = (b.loc[common, p] - a.loc[common, p]).to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v) < 3:
            out[p] = {"n": int(len(v))}
            continue
        r = boot_ci(v)
        _, pv = stats.wilcoxon(v) if len(v) >= 6 else (np.nan, np.nan)
        r.update(wilcoxon_p=float(pv) if np.isfinite(pv) else None,
                 mean_first=float(a.loc[common, p].mean()),
                 mean_last=float(b.loc[common, p].mean()))
        out[p] = r
        print(f"  {p:10s} {r['mean_first']:+.4f} -> {r['mean_last']:+.4f}  "
              f"Δ={r['mean']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] "
              f"p={r['p']:.3g} wilcoxon p={pv:.3g}")
    return out


def pooled_interaction(MW):
    """Property x position interaction pooled over readers, with reader fixed effects.

    Linear probability model of skipping on the standardized property, its interaction
    with relative position in the span, and a reader dummy set; standard errors are
    clustered by reader. Higher power than the per-reader D, and it tests the gradient
    directly rather than bin by bin.
    """
    out = {}
    subs = np.sort(MW.subject.unique())
    Dsub = np.column_stack([(MW.subject == s).to_numpy(float) for s in subs])
    for p in PROPS:
        d = MW[["subject", "skipped", "rel", p]].dropna()
        if len(d) < 500:
            continue
        keep = MW.index.isin(d.index)
        x = ((d[p] - d[p].mean()) / d[p].std()).to_numpy()
        rel = (d.rel - 0.5).to_numpy()
        X = np.column_stack([x, rel, x * rel, Dsub[keep]])
        y = d.skipped.to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        XtX_inv = np.linalg.pinv(X.T @ X)
        meat = np.zeros_like(XtX_inv)
        for s in np.unique(d.subject):
            m = (d.subject == s).to_numpy()
            u = X[m].T @ resid[m]
            meat += np.outer(u, u)
        V = XtX_inv @ meat @ XtX_inv
        se = np.sqrt(np.diag(V))
        z = beta[2] / se[2]
        # start-to-end change in the property's influence, as a fraction of that
        # influence; delta method on beta_interaction / beta_property
        bp, bi = beta[0], beta[2]
        var = (V[2, 2] / bp ** 2 + bi ** 2 * V[0, 0] / bp ** 4
               - 2 * bi * V[0, 2] / bp ** 3)
        se_r = float(np.sqrt(max(var, 0)))
        ratio = float(bi / bp)
        out[p] = {"beta_prop": float(bp), "beta_interaction": float(bi),
                  "se_interaction": float(se[2]), "z": float(z),
                  "p": float(2 * stats.norm.sf(abs(z))),
                  "start_to_end_change_pct": 100 * ratio,
                  "start_to_end_ci_pct": [100 * (ratio - 1.96 * se_r),
                                          100 * (ratio + 1.96 * se_r)],
                  "n": int(len(d)), "n_readers": int(d.subject.nunique())}
        print(f"  {p:10s} property beta={bp:+.5f}  property x position={bi:+.5f} "
              f"(SE {se[2]:.5f}, z={z:+.2f}, p={out[p]['p']:.3g})  "
              f"start-to-end change {100 * ratio:+.1f}% "
              f"[{100 * (ratio - 1.96 * se_r):+.1f}, {100 * (ratio + 1.96 * se_r):+.1f}]")
    return out


def main():
    T = episodes(pd.read_parquet(ART / "words_traversal.parquet"))
    ep = T[T.is_mw == 1].groupby("ep_id").agg(n=("pos", "size"), subject=("subject", "first"))
    rep = {"episodes": {"n": int(len(ep)), "n_readers": int(ep.subject.nunique()),
                        "words": int(ep.n.sum()),
                        "length_words": {q: float(ep.n.quantile(q / 100))
                                         for q in [10, 25, 50, 75, 90]},
                        "mean_length": float(ep.n.mean()),
                        "n_long": int((ep.n >= MIN_EP).sum()),
                        "words_in_long": int(ep.n[ep.n >= MIN_EP].sum())},
           "params": {"min_episode_words": MIN_EP, "min_words_per_cell": MINN,
                      "gap_threshold": G_PRIMARY}}
    print(f"{len(ep)} episodes over {ep.subject.nunique()} readers, "
          f"median {ep.n.median():.0f} words, mean {ep.n.mean():.1f}; "
          f"{(ep.n >= MIN_EP).sum()} of at least {MIN_EP} words, holding "
          f"{ep.n[ep.n >= MIN_EP].sum()} of {ep.n.sum()} mind-wandering words")

    S = analysis_set(T)
    D_on = d_on_task(S)
    rep["baseline"] = {"n_readers": int(len(D_on)),
                       **{p: float(D_on[p].mean()) for p in PROPS}}
    print(f"\non-task baseline: {len(D_on)} readers, " +
          ", ".join(f"D_{p}={D_on[p].mean():+.4f}" for p in PROPS))

    MW = S[(S.is_mw == 1) & (S.ep_len >= MIN_EP)].copy()
    rep["n_mw_words_binned"] = int(len(MW))

    # A: relative thirds
    MW["binA"] = pd.cut(MW.rel, [-0.001, 1 / 3, 2 / 3, 1.001],
                        labels=["first third", "middle third", "last third"])
    DA = d_by_bin(MW, "binA")
    rep["A_relative_thirds"] = compare(DA, D_on, "A. Relative position in the span",
                                       ["first third", "middle third", "last third"])
    rep["A_paired"] = paired(DA, "first third", "last third", "A")

    # B: absolute words from onset
    MW["binB"] = pd.cut(MW.idx_on, [-1, 9, 29, 10 ** 6],
                        labels=["words 1-10", "words 11-30", "words 31+"])
    DB = d_by_bin(MW, "binB")
    rep["B_from_onset"] = compare(DB, D_on, "B. Words from the start of the span",
                                  ["words 1-10", "words 11-30", "words 31+"])
    rep["B_paired"] = paired(DB, "words 1-10", "words 31+", "B")

    # C: absolute words before the end
    MW["binC"] = pd.cut(MW.idx_off, [-1, 9, 29, 10 ** 6],
                        labels=["last 10", "10-30 before end", "earlier"])
    DC = d_by_bin(MW, "binC")
    rep["C_before_end"] = compare(DC, D_on, "C. Words before the end of the span",
                                  ["last 10", "10-30 before end", "earlier"])
    rep["C_paired"] = paired(DC, "last 10", "earlier", "C")

    print("\n=== Pooled property x position interaction (reader fixed effects, "
          "reader-clustered SE) ===")
    rep["pooled_interaction"] = pooled_interaction(MW)

    (RES / "position_within_episode.json").write_text(json.dumps(rep, indent=2))
    DA.to_csv(ART / "somersD_by_position.csv", index=False)
    print(f"\nwrote {RES / 'position_within_episode.json'}")


if __name__ == "__main__":
    main()
