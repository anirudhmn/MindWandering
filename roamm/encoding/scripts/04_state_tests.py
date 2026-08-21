#!/usr/bin/env python3
"""State tests and controls on the pooled encoding read-out (supplementary section S11).

  gate       does the pooled text kernel predict held-out data, and does the within-page word
             permutation remove it
  positive   does the nuisance model's mind-wandering kernel reproduce the additive
             occipitotemporal offset the main text reports; a null from a machine that cannot
             find a known effect would not be interpretable
  state      the paired within-reader difference in D between states, with the smallest change
             detectable at 80% power, compared against the single-window contrast
  topography where the gain sits, which is how the ocular caution in S11 is established

The retention ratio D(mind-wandering)/D(on-task) is reported but is not the primary statistic
here: several readers have a non-positive on-task value, which makes a ratio of means unstable.
The paired difference is used instead.

Reads the pooled read-outs from 03_pooled_kernel.py.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from scipy import stats
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ART, RES, OCC, CP, EEG_CH, lag_grid, boot_ci

# what the same channels give in the single-window contrast of the main text
FROZEN = {"occ": dict(mde80_pct=43, attenuation_excluded_pct=8),
          "cp": dict(mde80_pct=144, attenuation_excluded_pct=97)}


def views(D, channels):
    ch = list(channels)
    return {"all64": D.mean(1),
            "occ": D[:, [ch.index(c) for c in OCC]].mean(1),
            "cp": D[:, [ch.index(c) for c in CP]].mean(1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", default=None, help="default: the value maximising on-task D")
    a = ap.parse_args()
    sweep = json.loads((RES / "lambda_sweep_real.json").read_text())
    lam = a.lam or max(sweep, key=lambda k: sweep[k]["D_on"])
    z = np.load(ART / f"pooled_real_lam{lam}.npz", allow_pickle=True)
    D, sj, mw = z["D"], z["subject"], z["mw"]
    subs = np.unique(sj)
    res = {"lam_scale": lam, "n_fixations": int(len(sj)), "n_readers": int(len(subs)),
           "n_mw": int(mw.sum())}

    for name, v in views(D, z["channels"]).items():
        don = np.array([v[(sj == s) & (mw == 0)].mean() for s in subs])
        dmw = np.array([v[(sj == s) & (mw == 1)].mean() if ((sj == s) & (mw == 1)).sum() >= 30
                        else np.nan for s in subs])
        ok = np.isfinite(don) & np.isfinite(dmw)
        gate = boot_ci(don)
        diff = boot_ci(dmw[ok] - don[ok])
        base = don[ok].mean()
        entry = dict(
            gate=gate,
            paired_difference=dict(
                delta=diff["mean"], ci=diff["ci"], t=diff["t"], p=diff["p"],
                pct_of_on_task=float(100 * diff["mean"] / base),
                pct_ci=[float(100 * diff["ci"][0] / base), float(100 * diff["ci"][1] / base)],
                mde80_pct=float(100 * 2.802 * diff["sd"] / abs(base))),
            readers_with_D_on_le_0=int((ok & (don <= 0)).sum()),
            D_on=float(base), D_mw=float(dmw[ok].mean()))
        if name in FROZEN:
            entry["single_window_contrast"] = FROZEN[name]
        res[name] = entry
        print(f"{name:6s} gate {gate['mean']:+.6f} t={gate['t']:.2f} p={gate['p']:.3g} "
              f"{gate['n_pos']}/{gate['n']}   delta {diff['mean']:+.6f} "
              f"({100*diff['mean']/base:+.0f}% of on-task) p={diff['p']:.3f}  "
              f"MDE80 {100*2.802*diff['sd']/abs(base):.0f}%", flush=True)

    # negative controls
    shuf = {}
    for f in sorted(ART.glob(f"pooled_shuf*_lam{lam}.npz")):
        zz = np.load(f, allow_pickle=True)
        dd, ss, mm = zz["D"].mean(1), zz["subject"], zz["mw"]
        don = np.array([dd[(ss == s) & (mm == 0)].mean() for s in np.unique(ss)])
        shuf[f.stem] = dict(D_on=float(don.mean()), readers_positive=int((don > 0).sum()))
        print(f"negative control {f.stem}: D_on={don.mean():+.6f}  "
              f"{int((don>0).sum())}/{len(don)} readers", flush=True)
    res["negative_controls"] = shuf

    # restricted text blocks: where each part of the block predicts
    restricted = {}
    lat_front = ["F8", "FT8", "AF8", "F7", "FT7", "AF7"]
    for f in sorted(ART.glob(f"pooled_only_*_lam{lam}.npz")) + [ART / f"pooled_real_lam{lam}.npz"]:
        zz = np.load(f, allow_pickle=True)
        ch = list(zz["channels"])
        t = zz["D"][zz["mw"] == 0].mean(0)
        restricted[f.stem] = dict(
            all64=float(t.mean()),
            occ=float(t[[ch.index(c) for c in OCC]].mean()),
            cp=float(t[[ch.index(c) for c in CP]].mean()),
            lateral_frontal=float(t[[ch.index(c) for c in lat_front]].mean()),
            top_channels=[ch[i] for i in np.argsort(t)[::-1][:6]])
        r = restricted[f.stem]
        print(f"{f.stem:34s} all={r['all64']:+.5f} occ={r['occ']:+.5f} cp={r['cp']:+.5f} "
              f"lat-front={r['lateral_frontal']:+.5f}  top {', '.join(r['top_channels'][:4])}",
              flush=True)
    res["text_block_restrictions"] = restricted

    # positive control: the additive occipitotemporal offset carried by the nuisance kernel
    NUIS = ["intercept", "logdur_z", "order_z", "log_in_amp_z", "log_out_amp_z", "page_prog_z",
            "mw", "mw:logdur_z"]
    _, lags_ms = lag_grid(-100, 500)
    B = np.stack([np.load(ART / "resid" / f"nuisbeta_s{int(s):02d}.npy") for s in subs])
    w = (lags_ms >= 150) & (lags_ms <= 290)
    off = B[:, NUIS.index("mw")][:, w][:, :, [EEG_CH.index(c) for c in OCC]].mean(axis=(1, 2))
    t, p = stats.ttest_1samp(off, 0)
    res["positive_control_mw_occipital_offset_uV"] = dict(
        mean=float(off.mean()), t=float(t), p=float(p), ci=boot_ci(off)["ci"],
        main_text_value=0.087)
    print(f"positive control, mind-wandering occipitotemporal offset 150-290 ms: "
          f"{off.mean():+.4f} uV (t={t:.2f}, p={p:.3g}); main text reports +0.087", flush=True)

    res["topography_D_on"] = {c: float(D[mw == 0][:, i].mean()) for i, c in enumerate(z["channels"])}
    (RES / "state_tests.json").write_text(json.dumps(res, indent=2))
    print("wrote", RES / "state_tests.json")


if __name__ == "__main__":
    main()
