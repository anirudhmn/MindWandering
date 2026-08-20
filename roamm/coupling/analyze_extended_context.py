#!/usr/bin/env python3
"""Corrected hierarchical coupling: simple slopes, equivalence tests, context ladder,
ORDERED-vs-SHUFFLED dissection, cross-model check, and a brain x behaviour modality test.
All predictors from the BOS-corrected word_multiscale_v2.parquet.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path("roamm/artifacts/coupling")
RNG = np.random.default_rng(23)
fix = pd.read_parquet(OUT / "fixations_frp.parquet")
wf = pd.read_parquet(OUT / "word_features.parquet")[["word_key", "length", "zipf", "clean"]]
ms = pd.read_parquet(OUT / "word_multiscale.parquet")
df = fix.merge(wf, on="word_key", how="left").merge(ms, on="word_key", how="left")
df["frp_cp_N400"] = df["frp_cp_N400"] * 1e6
df["p2p_uV"] = df["frp_p2p"] * 1e6
need = ["gpt2_s_sent", "gpt2_gain_long"]
df = df[(df["clean"].str.len() >= 1) & (df["zipf"] > 0) & df["fix_dur"].between(50, 1000)
        & df[need].notna().all(1)]
df["log_dur"] = np.log(df["fix_dur"].to_numpy())
print("rows:", len(df), "subjects:", df.subject.nunique())

def center(x): x = np.asarray(x, float); return x - x.mean()
def resid(y, X): b, *_ = np.linalg.lstsq(X, y, rcond=None); return y - X @ b

def fit(resp, disc_col, extra_cols=(), artifact=True, min_disc="gpt2_gain_long"):
    """Per-subject OLS; returns DataFrame of coefs incl on-task & MW simple slope for 'disc'."""
    recs = []
    for s, g in df.groupby("subject"):
        g = g[g[disc_col].notna()]
        if artifact and resp != "log_dur": g = g[g["p2p_uV"] <= 150]
        if len(g) < 300: continue
        mw = g["is_mw"].to_numpy().astype(float)
        if mw.sum() < 40 or (1 - mw).sum() < 150: continue
        zc, lc, slc = center(g["zipf"]), center(g["length"]), center(g["gpt2_s_sent"])
        cov = [np.ones(len(g)), zc, lc, slc]
        covn = ["int", "zipf", "length", "s_local"]
        extra = []
        for c in extra_cols:
            ec = center(g[c]); extra.append(ec); cov.append(ec); covn.append(c)
        dc = center(g[disc_col])
        dpure = resid(dc, np.column_stack(cov))     # discourse ⊥ covariates
        o = g["fix_order"].to_numpy().astype(float); oz = (o - o.mean()) / (o.std() + 1e-9)
        main = cov + [dpure, oz]; mnames = covn + ["disc", "order"]
        if resp != "log_dur":
            ld = np.log(g["fix_dur"].to_numpy()); main.append(ld - ld.mean()); mnames.append("logdur")
        M = np.column_stack(main)
        X = np.column_stack([M, mw, M[:, 1:] * mw[:, None]])
        names = mnames + ["is_mw"] + [n + "_x" for n in mnames[1:]]
        y = g[resp].to_numpy().astype(float); ok = np.isfinite(y)
        if ok.sum() < 300: continue
        b, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        d = dict(zip(names, b))
        d["disc_mw_slope"] = d["disc"] + d["disc_x"]     # simple slope during MW
        d["subject"] = s
        recs.append(d)
    return pd.DataFrame(recs)

def grp(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    t, p = stats.ttest_1samp(v, 0)
    boot = np.array([RNG.choice(v, len(v)).mean() for _ in range(10000)])
    return {"mean": float(v.mean()), "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "t": float(t), "p": float(p), "frac_pos": float((v > 0).mean()), "n": int(len(v))}

def tost(v, bound):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    se = v.std(ddof=1) / np.sqrt(len(v)); m = v.mean()
    p_lo = 1 - stats.t.cdf((m + bound) / se, len(v) - 1)
    p_hi = stats.t.cdf((m - bound) / se, len(v) - 1)
    return {"bound": float(bound), "mean": float(m), "p_tost": float(max(p_lo, p_hi)),
            "equivalent": bool(max(p_lo, p_hi) < 0.05)}

report = {}
# ---------- PRIMARY (corrected) ----------
for tag, resp in [("behavioral", "log_dur"), ("neural_N400", "frp_cp_N400")]:
    R = fit(resp, "gpt2_gain_long")
    on = grp(R["disc"]); ix = grp(R["disc_x"]); mws = grp(R["disc_mw_slope"])
    loc = grp(R["s_local"]); locx = grp(R["s_local_x"])
    bound = 0.5 * abs(on["mean"])       # SESOI = 50% of on-task discourse slope
    eq = tost(R["disc_mw_slope"] if resp != "log_dur" else R["disc_x"], bound)
    report[tag] = {"disc_on": on, "disc_x": ix, "disc_mw_slope": mws,
                   "local_on": loc, "local_x": locx,
                   "equiv_SESOI": bound,
                   "equiv_test": eq}
    print(f"\n[{tag}]  (corrected, n={on['n']})")
    print(f"  DISCOURSE on-task ={on['mean']:+.4f} p={on['p']:.1e} | "
          f"xMW ={ix['mean']:+.4f} CI[{ix['ci'][0]:+.4f},{ix['ci'][1]:+.4f}] p={ix['p']:.3f}")
    print(f"  DISCOURSE MW simple slope ={mws['mean']:+.4f} CI[{mws['ci'][0]:+.4f},{mws['ci'][1]:+.4f}] p={mws['p']:.3f}")
    print(f"  LOCAL     on-task ={loc['mean']:+.4f} p={loc['p']:.2f} | xMW ={locx['mean']:+.4f} p={locx['p']:.2f}")
    kind = "MW slope≡0" if resp != "log_dur" else "interaction≡0 (preserved)"
    print(f"  EQUIVALENCE {kind} within ±{bound:.4f}: {eq['equivalent']} (p_tost={eq['p_tost']:.3f})")

# ---------- CONTEXT LADDER + SHUFFLED CONTROL ----------
print("\n[context ladder — neural N400 disc(on) & disc:MW]")
ladder = {}
for col in ["gpt2_gain_prev1", "gpt2_gain_prev3", "gpt2_gain_long", "gpt2_gain_shuf"]:
    R = fit("frp_cp_N400", col)
    on = grp(R["disc"]); ix = grp(R["disc_x"])
    ladder[col] = {"disc_on": on, "disc_x": ix}
    print(f"  {col:20s} on={on['mean']:+.4f} p={on['p']:.1e} | xMW={ix['mean']:+.4f} p={ix['p']:.3f}")
report["ladder"] = ladder

# ---------- ORDERED vs SHUFFLED dissection (both in the model) ----------
print("\n[ordered-vs-shuffled dissection — neural N400]")
Rd = fit("frp_cp_N400", "gpt2_gain_long", extra_cols=["gpt2_gain_shuf"])
# here 'disc' = gain_long residualised on covariates incl gain_shuf -> ORDERED long-range component
ordered_on = grp(Rd["disc"]); ordered_x = grp(Rd["disc_x"])
shuf_on = grp(Rd["gpt2_gain_shuf"]); shuf_x = grp(Rd["gpt2_gain_shuf_x"])
report["dissection"] = {"ordered_disc_on": ordered_on, "ordered_disc_x": ordered_x,
                        "shuffled_on": shuf_on, "shuffled_x": shuf_x}
print(f"  ORDERED long-range: on={ordered_on['mean']:+.4f} p={ordered_on['p']:.1e} | xMW={ordered_x['mean']:+.4f} p={ordered_x['p']:.3f}")
print(f"  SHUFFLED (topic)  : on={shuf_on['mean']:+.4f} p={shuf_on['p']:.2f} | xMW={shuf_x['mean']:+.4f} p={shuf_x['p']:.2f}")

# ---------- CROSS-MODEL (Pythia) ----------
Rp = fit("frp_cp_N400", "pythia_gain_long")
on = grp(Rp["disc"]); ix = grp(Rp["disc_x"])
report["cross_model_pythia"] = {"disc_on": on, "disc_x": ix}
print(f"\n[cross-model Pythia N400] on={on['mean']:+.4f} p={on['p']:.1e} | xMW={ix['mean']:+.4f} p={ix['p']:.3f}")

# ---------- MODALITY x DISCOURSE x MW (standardised outcomes) ----------
# per subject: does disc:MW attenuation differ between brain and behaviour?
beh = fit("log_dur", "gpt2_gain_long"); neu = fit("frp_cp_N400", "gpt2_gain_long")
mrg = beh[["subject", "disc", "disc_x"]].merge(neu[["subject", "disc", "disc_x"]], on="subject", suffixes=("_beh", "_neu"))
# standardise each modality's interaction by its own on-task |slope| (relative attenuation)
rel_beh = mrg["disc_x_beh"] / mrg["disc_beh"].abs()
rel_neu = mrg["disc_x_neu"] / mrg["disc_neu"].abs()
tt, pp = stats.ttest_rel(rel_neu.replace([np.inf, -np.inf], np.nan).dropna(),
                         rel_beh.replace([np.inf, -np.inf], np.nan).dropna()) \
    if False else stats.ttest_1samp((rel_neu - rel_beh).replace([np.inf, -np.inf], np.nan).dropna(), 0)
report["modality_interaction"] = {"rel_atten_neural": float(rel_neu.replace([np.inf,-np.inf],np.nan).mean()),
                                  "rel_atten_behav": float(rel_beh.replace([np.inf,-np.inf],np.nan).mean()),
                                  "paired_t": float(tt), "p": float(pp)}
print(f"\n[modality x disc x MW] relative attenuation neural={report['modality_interaction']['rel_atten_neural']:+.2f} "
      f"vs behav={report['modality_interaction']['rel_atten_behav']:+.2f}  paired p={pp:.3f}")

(OUT / "extended_context_report.json").write_text(json.dumps(report, indent=2) + "\n")
print("\nwrote extended_context_report.json")
