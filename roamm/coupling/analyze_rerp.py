#!/usr/bin/env python3
"""Gauntlet on the overlap-corrected rERP kernels.

For each subject we have deconvolved kernels beta_p(tau, channel). Tests:
 1. Sanity: on-task surprisal kernel = N400 (centroparietal negativity ~300-450 ms),
    frequency kernel = occipitotemporal effect ~150-290 ms. Group one-sample tests.
 2. LANDMARK FORK: surprisal:mw interaction kernel over the N400 ROI/window — does deep
    semantic coupling selectively decouple during MW once overlap is removed?
    Reported with (a) a-priori ROI/window t-test, (b) cluster-based permutation across
    time on the ROI (sign-flip, 5000 perms), (c) fraction of on-task slope (equivalence).
 3. Same for zipf:mw (should stay null) and the mw additive kernel.
 4. Deconvolved vs naive: does removing overlap change the interaction estimate?
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats

OUT = Path("roamm/artifacts/coupling")
betas = np.load(OUT/"rerp_betas.npy")             # [S, P, L, 64]
meta = json.loads((OUT/"rerp_meta.json").read_text())
t = np.array(meta["lags_ms"]); PRED = meta["pred_names"]; CH = meta["channels"]
chi = {c:i for i,c in enumerate(CH)}
pidx = {p:i for i,p in enumerate(PRED)}
S = betas.shape[0]
RNG = np.random.default_rng(2026)

OCC = ['PO7','PO8','PO3','PO4','O1','O2','Oz','POz','P7','P8','P9','P10','Iz']
CP  = ['Cz','CPz','Pz','CP1','CP2','C1','C2','P1','P2']
def roi(names): return [chi[c] for c in names if c in chi]

def kernel(pred, chans):
    """[S, L] average over channels of the predictor kernel."""
    return betas[:, pidx[pred]][:, :, roi(chans)].mean(axis=2)

def win(tlo, thi):
    return (t >= tlo) & (t <= thi)

def group_scalar(vals):
    vals = vals[np.isfinite(vals)]
    tt, p = stats.ttest_1samp(vals, 0.0)
    boot = np.array([RNG.choice(vals, len(vals)).mean() for _ in range(10000)])
    ci = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(vals.mean()), "ci": [float(ci[0]), float(ci[1])],
            "t": float(tt), "p": float(p), "n": int(len(vals)), "frac_neg": float((vals<0).mean())}

def cluster_perm(kern, tmask, n_perm=5000):
    """1-D temporal cluster-based permutation (sign-flip) on [S, L] within tmask."""
    K = kern[:, tmask]; tt = t[tmask]
    def clusters(x):
        tvals = x.mean(0) / (x.std(0, ddof=1)/np.sqrt(x.shape[0]) + 1e-12)
        thr = stats.t.ppf(0.975, x.shape[0]-1)
        mass = []; cur = 0.0; span = None; out = []
        sig = np.abs(tvals) > thr
        i = 0
        while i < len(sig):
            if sig[i]:
                j = i; s = 0.0
                while j < len(sig) and sig[j] and np.sign(tvals[j])==np.sign(tvals[i]):
                    s += tvals[j]; j += 1
                out.append((i, j, s)); i = j
            else: i += 1
        return out, tvals
    obs, tvals = clusters(K)
    if not obs:
        return {"n_clusters": 0, "min_p": 1.0, "sig_windows": []}
    obs_max = max(abs(c[2]) for c in obs)
    null = np.empty(n_perm)
    for b in range(n_perm):
        signs = RNG.choice([-1, 1], K.shape[0])[:, None]
        cl, _ = clusters(K*signs)
        null[b] = max((abs(c[2]) for c in cl), default=0.0)
    res = []
    for (i, j, mass) in obs:
        p = float((null >= abs(mass)).mean())
        res.append({"t_start": float(tt[i]), "t_end": float(tt[j-1]),
                    "mass": float(mass), "p": p})
    return {"n_clusters": len(obs), "min_p": min(r["p"] for r in res),
            "sig_windows": [r for r in res if r["p"] < 0.05], "all": res}

report = {"n_subjects": S}

# 1. sanity: on-task kernels
surp_cp = kernel("surprisal", CP)
freq_occ = kernel("zipf", OCC)
report["sanity_surprisal_N400_300_450"] = group_scalar(surp_cp[:, win(300,450)].mean(1))
report["sanity_freq_occ_150_290"] = group_scalar(freq_occ[:, win(150,290)].mean(1))

# 2. surprisal decoupling (the fork)
surpmw_cp = kernel("surprisal:mw", CP)
report["surprisal_x_mw_N400_roi"] = group_scalar(surpmw_cp[:, win(300,450)].mean(1))
report["surprisal_x_mw_cluster"] = cluster_perm(surpmw_cp, win(0,498))
# equivalence: interaction as fraction of on-task surprisal N400 slope (per subject)
ot = surp_cp[:, win(300,450)].mean(1); inter = surpmw_cp[:, win(300,450)].mean(1)
frac = []
for _ in range(10000):
    idx = RNG.integers(0, S, S)
    frac.append(-inter[idx].mean()/ (ot[idx].mean() + 1e-12))
report["surprisal_attenuation_frac"] = {
    "median": float(np.median(frac)), "ci": [float(np.percentile(frac,2.5)), float(np.percentile(frac,97.5))]}

# 3. zipf decoupling (control, expect null) + additive mw
freqmw_occ = kernel("zipf:mw", OCC)
report["zipf_x_mw_occ_150_290"] = group_scalar(freqmw_occ[:, win(150,290)].mean(1))
report["zipf_x_mw_cluster"] = cluster_perm(freqmw_occ, win(0,498))
mw_occ = kernel("mw", OCC); mw_cp = kernel("mw", CP)
report["mw_additive_occ_150_290"] = group_scalar(mw_occ[:, win(150,290)].mean(1))
report["mw_additive_cp_300_450"] = group_scalar(mw_cp[:, win(300,450)].mean(1))

# save kernels for plotting
np.savez(OUT/"rerp_kernels.npz", t=t,
         surp_cp=surp_cp, freq_occ=freq_occ, surpmw_cp=surpmw_cp,
         freqmw_occ=freqmw_occ, mw_cp=mw_cp, mw_occ=mw_occ,
         intercept_occ=kernel("intercept", OCC), intercept_cp=kernel("intercept", CP))
(OUT/"rerp_report.json").write_text(json.dumps(report, indent=2)+"\n")

def show(k):
    r = report[k]
    if "mean" in r:
        print(f"{k}: {r['mean']:+.4f} uV CI[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] t={r['t']:+.2f} p={r['p']:.4f}")
    else:
        print(f"{k}: {json.dumps(r)}")
print("\n=== SANITY (measures valid) ===")
show("sanity_surprisal_N400_300_450"); show("sanity_freq_occ_150_290")
print("\n=== SURPRISAL DECOUPLING FORK ===")
show("surprisal_x_mw_N400_roi")
print("cluster:", report["surprisal_x_mw_cluster"]["sig_windows"], "min_p=", report["surprisal_x_mw_cluster"]["min_p"])
print("attenuation frac: %.0f%% CI[%.0f%%,%.0f%%]" % (
    report["surprisal_attenuation_frac"]["median"]*100,
    report["surprisal_attenuation_frac"]["ci"][0]*100, report["surprisal_attenuation_frac"]["ci"][1]*100))
print("\n=== FREQUENCY DECOUPLING (control) ===")
show("zipf_x_mw_occ_150_290")
print("cluster:", report["zipf_x_mw_cluster"]["sig_windows"], "min_p=", report["zipf_x_mw_cluster"]["min_p"])
print("\n=== MW ADDITIVE KERNEL ===")
show("mw_additive_occ_150_290"); show("mw_additive_cp_300_450")
print("\nwrote rerp_report.json + rerp_kernels.npz")
