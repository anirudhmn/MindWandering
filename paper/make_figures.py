#!/usr/bin/env python3
"""Main figures for the mindless-reading manuscript. All panels are drawn from stored
analysis artifacts; nothing is simulated."""
from __future__ import annotations
import json, glob, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
COUP = ROOT / "roamm/artifacts/coupling"
SEL  = ROOT / "roamm/selection_repair"
LOC  = ROOT / "roamm/localisation"
TOP  = ROOT / "roamm/topography"
ATT  = ROOT / "roamm/attention_index"
ZA   = ROOT / "zuco/artifacts"
OUT  = ROOT / "paper/figs"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.titlesize": 8.5, "axes.labelsize": 8,
})
ON, MW, GOAL, GREY = "#3B6EA5", "#C0392B", "#7D3C98", "#7F8C8D"
CM = 1 / 2.54


def panel_label(ax, s, dx=-0.16, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="left")


def sem(x, axis=0):
    return np.nanstd(x, axis=axis, ddof=1) / np.sqrt(np.sum(np.isfinite(x), axis=axis))


def _auc(y, score):
    """Area under the ROC curve, from ranks (no sklearn dependency)."""
    y = np.asarray(y).astype(bool); score = np.asarray(score, float)
    r = stats.rankdata(score)
    n1, n0 = int(y.sum()), int((~y).sum())
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


g0 = json.load(open(SEL / "results/g0_skip_audit.json"))
g0b = json.load(open(SEL / "results/g0b_traversal.json"))
g1 = json.load(open(SEL / "results/g1_selection.json"))
g4 = json.load(open(SEL / "results/g4_repair.json"))
g56 = json.load(open(SEL / "results/g5_g6_duration.json"))
g7 = json.load(open(SEL / "results/g7_zuco_session.json"))
neq = json.load(open(SEL / "results/neural_equivalence.json"))
isc = json.load(open(COUP / "isc_report.json"))
iscv = json.load(open(COUP / "isc_verify_report.json"))
rerp = json.load(open(COUP / "rerp_report.json"))
lmm = json.load(open(COUP / "lmm_report.json"))
land = json.load(open(COUP / "landmark_summary.json"))
Dsom = pd.read_csv(SEL / "artifacts/somersD_primary.csv")
rev = json.load(open(SEL / "results/reviewer_checks.json"))
npow = json.load(open(SEL / "results/neural_power.json"))
mech = json.load(open(SEL / "results/mechanism_tests.json"))
mctl = json.load(open(SEL / "results/mechanism_controls.json"))

# ----------------------------------------------- artifacts for the 2026-08 sections
topo = json.load(open(TOP / "results/topographic_rerp_report.json"))
mwloc = json.load(open(LOC / "results/mw_localisation_deepen.json"))
mwctl = json.load(open(LOC / "results/mw_localisation_control.json"))
outc = json.load(open(LOC / "results/outcome.json"))
GRAD = pd.read_csv(LOC / "results/mw_overlap_gradient.csv")

OCC_ROI = ["PO7", "PO8", "PO3", "PO4", "O1", "O2", "Oz", "POz", "P7", "P8", "P9", "P10", "Iz"]
BOOT = 20000


def pct_of_base(delta, base, seed=63):
    """Per-reader change as a percentage of the group engaged-state effect, with a
    percentile bootstrap over readers. This is the estimator used throughout the paper."""
    d = np.asarray(delta, float)
    d = d[np.isfinite(d)] / base * 100      # signed: base carries the sign of the effect
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(BOOT, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(d)


def roamm_occ_frequency():
    """Per-reader occipitotemporal frequency response, on-task and mind-wandering."""
    B = np.load(COUP / "rerp_betas.npy")
    meta = json.load(open(COUP / "rerp_meta.json"))
    T = np.asarray(meta["lags_ms"]); ip = {x: i for i, x in enumerate(meta["pred_names"])}
    occ = [meta["channels"].index(c) for c in OCC_ROI]
    tm = (T >= 150) & (T <= 290)
    sc = 1e6 if np.nanmax(np.abs(B[:, ip["zipf"]])) < 1e-3 else 1
    on = B[:, ip["zipf"]][:, tm][:, :, occ].mean(axis=(1, 2)) * sc
    inter = B[:, ip["zipf:mw"]][:, tm][:, :, occ].mean(axis=(1, 2)) * sc
    return on, on + inter, inter


_ZB = {}


def zuco_behaviour():
    """Per-reader frequency slope on log gaze duration in ZuCo, normal reading vs
    instructed relation search. Recomputed exactly as in the selection and repair analysis, script 05."""
    if _ZB:
        return _ZB["NR"], _ZB["TSR"]
    recs = []
    for task in ["NR", "TSR"]:
        ling = pd.read_parquet(ZA / f"linguistic_{task}.parquet")
        for mp in sorted(glob.glob(str(ZA / f"frp/meta_*_{task}.parquet"))):
            m = pd.read_parquet(mp).copy()
            m["subject"] = os.path.basename(mp).split("_")[1]
            recs.append(m.merge(ling, on=["task", "sent_idx", "word_idx"], how="left"))
    Z = pd.concat(recs, ignore_index=True).dropna(subset=["zipf"])
    Z["logGD"] = np.log((Z.GD / 500 * 1000).clip(lower=1))
    out = {}
    for task, g0_ in Z.groupby("task"):
        d = {}
        for sbj, g in g0_.groupby("subject"):
            g = g.dropna(subset=["logGD", "zipf"])
            if len(g) < 80:
                continue
            z = (g.zipf - g.zipf.mean()) / g.zipf.std()
            d[sbj] = float(np.polyfit(z, g.logGD, 1)[0])
        out[task] = d
    common = sorted(set(out["NR"]) & set(out["TSR"]))
    _ZB["NR"] = np.array([out["NR"][k] for k in common])
    _ZB["TSR"] = np.array([out["TSR"][k] for k in common])
    return _ZB["NR"], _ZB["TSR"]


def zuco_occ_frequency():
    """Per-reader occipital frequency rERP in ZuCo, 150-290 ms, NR vs TSR."""
    chan = json.load(open(ZA / "chanlocs_105.json"))
    occ = [i for i, r in enumerate(chan) if r["X"] < -5 and r["Z"] < 4]

    def one(subj, task):
        d = np.load(ZA / f"rerp/rerp_{subj}_{task}.npz", allow_pickle=True)
        preds = [str(x) for x in d["preds"]]
        t = d["lags"] / 500 * 1000
        w = (t >= 150) & (t <= 290)
        return float(np.nanmean(d["beta"][preds.index("zipf")][w][:, occ]))

    subs = sorted({os.path.basename(f).split("_")[1]
                   for f in glob.glob(str(ZA / "rerp/rerp_*_NR.npz"))})
    subs = [s for s in subs if os.path.exists(ZA / f"rerp/rerp_{s}_TSR.npz")]
    return (np.array([one(s, "NR") for s in subs]),
            np.array([one(s, "TSR") for s in subs]))


def attention_index():
    """The text-driven and level-based indices, one row per fixation."""
    f = ATT / "results/attention_index.parquet"
    if not f.exists():
        raise SystemExit(f"missing {f}: run roamm/attention_index/"
                         "scripts/01_text_attention.py first")
    return pd.read_parquet(f)


def paired_panel(ax, a, b, labels, colors, ylabel, title, note=None, fmt="{:.3f}"):
    """Reader-by-reader change between two states, in the style of Fig. 3a."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    for u, v in zip(a, b):
        ax.plot([0, 1], [u, v], color="0.78", lw=0.5, zorder=1)
    ax.plot(np.zeros_like(a), a, "o", color=colors[0], ms=3.0, alpha=0.85, zorder=2)
    ax.plot(np.ones_like(b), b, "o", color=colors[1], ms=3.0, alpha=0.85, zorder=2)
    ax.plot([0, 1], [a.mean(), b.mean()], "-", color="k", lw=1.9, zorder=3)
    ax.plot([0, 1], [a.mean(), b.mean()], "o", color="k", ms=4.5, zorder=4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_xlim(-0.35, 1.35); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    if note:
        ax.text(0.5, 0.965, note, transform=ax.transAxes, fontsize=6.6, ha="center", va="top")


# =============================================================== FIGURE 1
def figure1():
    fig = plt.figure(figsize=(18 * CM, 7.0 * CM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.80], width_ratios=[1, 1.35],
                          hspace=0.75, wspace=0.24, left=0.125, right=0.985, top=0.90, bottom=0.11)

    # (a) competing predictions
    ax = fig.add_subplot(gs[0, 0]); panel_label(ax, "a", dx=-0.30)
    x = np.linspace(0, 1, 100)
    ax.plot(x, np.ones_like(x), color=ON, lw=1.6, label="engaged")
    ax.plot(x, 1 - 0.55 * x, color=MW, lw=1.6, ls="--", label="decoupling account")
    ax.plot(x, 1 - 0.02 * x, color=MW, lw=1.6, ls=":", label="state-shift account")
    ax.set_xlabel("degree of disengagement")
    ax.set_ylabel("word to eye coupling")
    ax.set_xticks([]); ax.set_yticks([0, 1]); ax.set_yticklabels(["0", "intact"])
    ax.set_ylim(-0.05, 1.32)
    ax.legend(frameon=False, loc="lower left", handlelength=1.8, fontsize=6.5)
    ax.set_title("Competing predictions", loc="left")

    # (b) three decisions, evenly spaced word slots
    ax = fig.add_subplot(gs[0, 1]); panel_label(ax, "b", dx=-0.10)
    words = ["The", "aurorae", "on", "Pluto", "were", "first", "seen", "in", "2015"]
    xs = np.arange(len(words), dtype=float)
    for w, xx in zip(words, xs):
        ax.text(xx, 0.0, w, fontsize=6.6, family="monospace", ha="center", va="center")
    skipped = {2, 5, 7}
    for i, xx in enumerate(xs):
        if i in skipped:
            ax.plot(xx, 0.55, "x", ms=5, color=GREY, mew=1.3)
        else:
            ax.plot(xx, 0.55, "o", ms=4.5, color=ON, zorder=3)
    ax.annotate("", xy=(3.0, 0.95), xytext=(1.0, 0.95),
                arrowprops=dict(arrowstyle="->", color=ON, lw=1.1))
    ax.text(2.0, 1.06, "selection: skip word 3", fontsize=6.5, color=ON, ha="center")
    ax.annotate("", xy=(1.0, 1.62), xytext=(4.0, 1.62),
                arrowprops=dict(arrowstyle="->", color="#B9770E", lw=1.1, ls="--"))
    ax.text(2.5, 1.73, "repair: regress to an earlier word", fontsize=6.5, color="#B9770E", ha="center")
    ax.annotate("", xy=(6.0, 0.50), xytext=(6.0, 1.15),
                arrowprops=dict(arrowstyle="->", color="k", lw=0.9))
    ax.text(6.0, 1.26, "duration:\nhow long to dwell", fontsize=6.5, ha="center")
    ax.set_xlim(-0.9, len(words) - 0.1); ax.set_ylim(-0.45, 2.05); ax.axis("off")
    ax.set_title("Three decisions the eyes make", loc="left")

    # (c) ROAMM
    ax = fig.add_subplot(gs[1, 0]); panel_label(ax, "c", dx=-0.30)
    ax.add_patch(Rectangle((0, 0.52), 10, 0.30, fc="#EDF1F5", ec="k", lw=0.7))
    for st, en in [(2.2, 3.5), (6.4, 7.4)]:
        ax.add_patch(Rectangle((st, 0.52), en - st, 0.30, fc=MW, alpha=0.40, ec="none"))
    ax.text(2.85, 0.36, "reported", fontsize=6.2, color=MW, ha="center")
    ax.text(0, 0.94, "ROAMM", fontsize=8, fontweight="bold")
    ax.text(2.3, 0.94, "44 readers, 5 articles", fontsize=6.8)
    ax.text(0, 0.14, "continuous self-paced reading, retrospective spans", fontsize=6.4, color=GREY)
    ax.text(0, -0.02, "64-channel EEG, 404,557 first-pass fixations", fontsize=6.4, color=GREY)
    ax.set_xlim(0, 10.2); ax.set_ylim(-0.12, 1.10); ax.axis("off")
    ax.set_title("Spontaneous disengagement", loc="left")

    # (d) ZuCo
    ax = fig.add_subplot(gs[1, 1]); panel_label(ax, "d", dx=-0.10)
    labs = [("NR", "deep", ON, 1), ("SR h1", "deep", ON, 1),
            ("TSR", "shallow", GOAL, 2), ("SR h2", "deep", ON, 2)]
    for i, (t, d, c, ss) in enumerate(labs):
        ax.add_patch(Rectangle((i * 2.5, 0.46), 2.1, 0.34, fc=c, alpha=0.30, ec="k", lw=0.7))
        ax.text(i * 2.5 + 1.05, 0.68, t, ha="center", va="center", fontsize=7, fontweight="bold")
        ax.text(i * 2.5 + 1.05, 0.55, d, ha="center", va="center", fontsize=6.3)
        ax.text(i * 2.5 + 1.05, 0.33, f"session {ss}", ha="center", fontsize=6.2, color=GREY)
    ax.text(0, 0.94, "ZuCo", fontsize=8, fontweight="bold")
    ax.text(1.6, 0.94, "12 readers, isolated sentences, 105-channel EEG", fontsize=6.8)
    ax.text(0, 0.10, "task order fixed for every reader; sessions on separate days",
            fontsize=6.4, color=GREY)
    ax.set_xlim(-0.2, 10.2); ax.set_ylim(0.02, 1.10); ax.axis("off")
    ax.set_title("Instructed disengagement", loc="left")

    fig.savefig(OUT / "fig1_design.pdf"); fig.savefig(OUT / "fig1_design.png", dpi=300)
    plt.close(fig); print("fig1 done")


# =============================================================== FIGURE 2
def figure2():
    K = np.load(COUP / "rerp_kernels.npz")
    t = K["t"] * 1000 if np.max(np.abs(K["t"])) < 10 else K["t"]
    fig = plt.figure(figsize=(13.5 * CM, 11.0 * CM))
    gs = fig.add_gridspec(2, 2, wspace=0.46, hspace=0.62, left=0.155, right=0.955, top=0.91, bottom=0.085)

    for j, (key, roi, ttl, col, note) in enumerate([
            ("freq_occ", "occipitotemporal", "Frequency", ON,
             f"+{rerp['sanity_freq_occ_150_290']['mean']:.3f} µV/SD, $p$ = 3.9×10$^{{-10}}$"),
            ("surp_cp", "centroparietal", "Surprisal", MW,
             f"{rerp['sanity_surprisal_N400_300_450']['mean']:.3f} µV/SD, $p$ = 3.6×10$^{{-6}}$")]):
        ax = fig.add_subplot(gs[0, j]); panel_label(ax, "abcd"[j], dx=-0.26)
        Y = K[key] * 1e6 if np.nanmax(np.abs(K[key])) < 1e-3 else K[key]
        m, s = np.nanmean(Y, 0), sem(Y)
        ax.fill_between(t, m - s, m + s, color=col, alpha=0.25, lw=0)
        ax.plot(t, m, color=col, lw=1.5)
        w = (150, 290) if j == 0 else (300, 450)
        ax.axvspan(*w, color="k", alpha=0.06, lw=0)
        ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6, ls=":")
        ax.set_xlabel("time from fixation onset (ms)")
        ax.set_ylabel(f"{roi}\nregression ERP (µV/SD)" if j == 0 else "regression ERP (µV/SD)",
                      fontsize=7.4 if j == 0 else 8)
        ax.set_title(ttl, loc="left")
        ax.text(0.045, 0.06, note, transform=ax.transAxes, fontsize=6.6,
                bbox=dict(fc="w", ec="none", pad=1.0))
        ax.set_xlim(-100, 500)

    # (c) ZuCo replication of the frequency kernel
    ax = fig.add_subplot(gs[1, 0]); panel_label(ax, "c", dx=-0.30)
    chan = json.load(open(ZA / "chanlocs_105.json"))
    occ = [i for i, r in enumerate(chan) if r["X"] < -5 and r["Z"] < 4]
    curves = []
    for f in sorted(glob.glob(str(ZA / "rerp/rerp_*_NR.npz"))):
        d = np.load(f, allow_pickle=True)
        preds = [str(p) for p in d["preds"]]
        bi = preds.index("zipf") if "zipf" in preds else 1
        curves.append(np.nanmean(d["beta"][bi][:, occ], axis=1))
        lags = d["lags"]
    C = np.array(curves)
    tz = lags / 500 * 1000 if np.max(np.abs(lags)) > 10 else lags * 1000
    tz = np.linspace(-100, 500, C.shape[1]) if np.ptp(tz) > 2000 else tz
    m, s = np.nanmean(C, 0), sem(C)
    ax.fill_between(tz, m - s, m + s, color="#1E8449", alpha=0.25, lw=0)
    ax.plot(tz, m, color="#1E8449", lw=1.5)
    ax.axvspan(150, 290, color="k", alpha=0.06, lw=0)
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("time from fixation onset (ms)"); ax.set_ylabel("regression ERP (µV/SD)")
    ax.set_title("Frequency, replication", loc="left")
    ax.text(0.045, 0.06, "+0.139 µV/SD, cluster $p$ = 0.040\nZuCo, $n$ = 12, 105-channel EGI",
            transform=ax.transAxes, fontsize=6.6, bbox=dict(fc="w", ec="none", pad=1.0))
    ax.set_xlim(-100, 500)

    # (d) behaviour at the three levels
    ax = fig.add_subplot(gs[1, 1]); panel_label(ax, "d", dx=-0.26)
    vals = [abs(Dsom.zipf_on.mean()), abs(Dsom.length_on.mean()), abs(Dsom.surprisal_on.mean())]
    rg = [abs(g4["regression_to_difficulty"][p]["D_on"]) for p in ["zipf", "length", "surprisal"]]
    cr = [abs(g4["corrective_selectivity"][p]["D_on"]) for p in ["zipf", "length", "surprisal"]]
    x = np.arange(3); w = 0.27
    ax.bar(x - w, vals, w, color=ON, label="selection (skip)")
    ax.bar(x, cr, w, color="#B9770E", label="repair (corrective return)")
    ax.bar(x + w, rg, w, color=GREY, label="repair (regression)")
    ax.set_xticks(x); ax.set_xticklabels(["freq.", "length", "surp."])
    ax.set_ylabel("|Somers' $D$|, on-task")
    ax.set_title("Eye decisions track words", loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=6.2)
    ax.set_ylim(0, 0.80)

    fig.savefig(OUT / "fig2_instrument.pdf"); fig.savefig(OUT / "fig2_instrument.png", dpi=300)
    plt.close(fig); print("fig2 done")


# =============================================================== FIGURE 3
def figure3():
    fig = plt.figure(figsize=(18 * CM, 10.4 * CM))
    gs = fig.add_gridspec(2, 3, wspace=0.62, hspace=0.66, left=0.105, right=0.985,
                          top=0.90, bottom=0.135)

    # (a) selection
    ax = fig.add_subplot(gs[0, 0]); panel_label(ax, "a", dx=-0.31, dy=1.17)
    props = ["zipf", "length", "surprisal"]
    for i, p in enumerate(props):
        on, mw = Dsom[f"{p}_on"].abs(), Dsom[f"{p}_mw"].abs()
        for a, b in zip(on, mw):
            ax.plot([i - .17, i + .17], [a, b], color="0.75", lw=0.45, zorder=1)
        ax.plot([i - .17] * len(on), on, "o", color=ON, ms=2.6, alpha=.85, zorder=2)
        ax.plot([i + .17] * len(mw), mw, "o", color=MW, ms=2.6, alpha=.85, zorder=2)
        ax.plot([i - .17, i + .17], [on.mean(), mw.mean()], "-", color="k", lw=1.8, zorder=3)
        ax.text(i, 0.80, f"{g1['G1_primary'][p]['retention_pct']:.0f}%", ha="center",
                fontsize=7.5, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(["frequency", "length", "surprisal"])
    ax.set_ylabel("|Somers' $D$|, property→skip", fontsize=7.2); ax.set_ylim(0, 0.87)
    ax.set_title("Selection", loc="left")
    ax.plot([], [], "o", color=ON, ms=3, label="on-task"); ax.plot([], [], "o", color=MW, ms=3, label="MW")
    ax.legend(frameon=False, loc="lower left", ncol=2, columnspacing=0.8)

    # (b) repair
    ax = fig.add_subplot(gs[0, 1]); panel_label(ax, "b", dx=-0.22, dy=1.17)
    fams = [("corrective\nreturn", "corrective_selectivity"), ("refixation", "refixation_selectivity"),
            ("regression", "regression_to_difficulty")]
    x = np.arange(len(fams)); w = 0.26
    for j, p in enumerate(props):
        v = [g4[k][p]["D_mw"] / g4[k][p]["D_on"] * 100 for _, k in fams]
        ax.bar(x + (j - 1) * w, v, w, color=[ON, GREY, "#B9770E"][j], label=p)
    ax.axhline(100, color="k", lw=0.7, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([a for a, _ in fams], fontsize=6.2)
    ns = [g4[k]["zipf"]["n"] for _, k in fams]
    for xi, nn in zip(x, ns):
        ax.text(xi, 4, f"$n$={nn}", ha="center", fontsize=5.8, color="0.35")
    ax.set_ylabel("retention of selectivity (%)"); ax.set_ylim(0, 155)
    ax.set_title("Repair", loc="left")
    ax.legend(frameon=False, ncol=3, columnspacing=0.5, loc="upper center", fontsize=6.0,
              handlelength=1.0, handletextpad=0.4)

    # (c) duration, across-word vs within-token
    ax = fig.add_subplot(gs[0, 2]); panel_label(ax, "c", dx=-0.22, dy=1.17)
    aw = g56["G5_across_word_signed_retention"]; fe = g56["G5"]["logGD"]
    SL = pd.read_csv(SEL / "artifacts/duration_slopes_by_state.csv")
    rng = np.random.default_rng(59)
    labs, est, lo_, hi_, cols = [], [], [], [], []
    for p in ["zipf", "surprisal", "length"]:
        base = aw[p]["beta_on"]
        d = (SL[f"{p}_mw"] - SL[f"{p}_on"]).to_numpy()
        d = d[np.isfinite(d)]
        bm = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1) / base * 100
        labs.append(p); est.append(float(d.mean() / base * 100)); cols.append(MW)
        lo_.append(float(np.percentile(bm, 2.5))); hi_.append(float(np.percentile(bm, 97.5)))
        v = fe[f"mw_x_{p}"]
        labs.append(p); est.append(v["beta"] / base * 100); cols.append("k")
        h = abs(1.96 * v["se"] / base * 100)
        lo_.append(est[-1] - h); hi_.append(est[-1] + h)
    err = None
    y = np.arange(len(est))[::-1]
    ax.axvspan(-20, 20, color="k", alpha=0.06, lw=0)
    est = np.array(est); lo_ = np.array(lo_); hi_ = np.array(hi_)
    ax.errorbar(est, y, xerr=[est - lo_, hi_ - est], fmt="none", ecolor="0.5", lw=1.0)
    for e, yy, c in zip(est, y, cols):
        ax.plot(e, yy, "o", ms=4.5, color=c, zorder=3)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(y); ax.set_yticklabels([f"{l}, {'across words' if i%2==0 else 'within token'}"
                                          for i, l in enumerate(labs)], fontsize=5.9)
    ax.set_xlabel("change in coupling during MW\n(% of on-task)")
    ax.set_title("Duration", loc="left"); ax.set_xlim(-24, 26)

    # (d) neural kernels split by state
    K = np.load(COUP / "rerp_kernels.npz"); t = K["t"] * 1000 if np.max(np.abs(K["t"])) < 10 else K["t"]
    for j, (base, inter, roi, ttl, w) in enumerate([
            ("freq_occ", "freqmw_occ", "occipitotemporal", "Frequency kernel", (150, 290)),
            ("surp_cp", "surpmw_cp", "centroparietal", "Surprisal kernel", (300, 450))]):
        ax = fig.add_subplot(gs[1, j]); panel_label(ax, "de"[j], dx=-0.22, dy=1.17)
        B = K[base] * (1e6 if np.nanmax(np.abs(K[base])) < 1e-3 else 1)
        I = K[inter] * (1e6 if np.nanmax(np.abs(K[inter])) < 1e-3 else 1)
        for Y, c, lab in [(B, ON, "on-task"), (B + I, MW, "mind-wandering")]:
            m, s = np.nanmean(Y, 0), sem(Y)
            ax.fill_between(t, m - s, m + s, color=c, alpha=0.22, lw=0)
            ax.plot(t, m, color=c, lw=1.4, label=lab)
        ax.axvspan(*w, color="k", alpha=0.06, lw=0)
        ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6, ls=":")
        ax.set_xlabel("time from fixation onset (ms)"); ax.set_ylabel(f"{roi}\n(µV/SD)")
        ax.set_title(ttl, loc="left"); ax.set_xlim(-100, 500)
        ax.legend(frameon=False, loc="upper left", fontsize=6.2)
        pv = rerp["zipf_x_mw_occ_150_290"]["p"] if j == 0 else rerp["surprisal_x_mw_N400_roi"]["p"]
        ax.text(0.97, 0.05, f"interaction $p$ = {pv:.2f}\nno cluster in epoch",
                transform=ax.transAxes, fontsize=6.2, ha="right")

    # (f) equivalence summary
    ax = fig.add_subplot(gs[1, 2]); panel_label(ax, "f", dx=-0.22, dy=1.17)
    rows = [("selection, frequency", "skip_selectivity_zipf"),
            ("selection, length", "skip_selectivity_length"),
            ("selection, surprisal", "skip_selectivity_surprisal"),
            ("duration, surprisal", "duration_surprisal"),
            ("duration, frequency", "duration_zipf"),
            ("neural, frequency", "NP:frequency (occipitotemporal)"),
            ("neural, surprisal", "NP:surprisal (centroparietal N400)")]
    y = np.arange(len(rows))[::-1]
    for (lab, k), yy in zip(rows, y):
        if k.startswith("NP:"):
            e = npow[k[3:]]
            m, c0, c1 = e["change_pct"], e["ci95"][0], e["ci95"][1]
            col = "#1E8449" if e["MDE_pct_80power"] < 50 else "#B9770E"
        else:
            e = neq["equivalence"][k]
            sg = np.sign(e["on_task_effect"])
            m = e["pct"]["mean"] * sg
            c0, c1 = sorted([e["pct"]["ci"][0] * sg, e["pct"]["ci"][1] * sg])
            col = "#1E8449"
        ax.errorbar([m], [yy], xerr=[[m - c0], [c1 - m]], fmt="o", ms=4,
                    color=col, ecolor=col, lw=1.1, capsize=1.8)
    ax.axvspan(-20, 20, color="k", alpha=0.07, lw=0)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    ax.set_xlabel("change in coupling during MW\n(% of on-task effect)")
    ax.set_title("Precision of each contrast", loc="left"); ax.set_xlim(-125, 125)
    ax.set_xticks([-100, -50, 0, 50, 100])

    fig.savefig(OUT / "fig3_preserved.pdf"); fig.savefig(OUT / "fig3_preserved.png", dpi=300)
    plt.close(fig); print("fig3 done")


# =============================================================== FIGURE 4
def figure4():
    fig = plt.figure(figsize=(18 * CM, 10.6 * CM))
    gs = fig.add_gridspec(2, 3, wspace=0.58, hspace=0.62, left=0.105, right=0.985, top=0.91, bottom=0.085)

    ax = fig.add_subplot(gs[0, 0]); panel_label(ax, "a", dx=-0.22, dy=1.17)
    raw = [g0["skip_rate"]["on_task"], g0["skip_rate"]["mw"]]
    cor = [g0b["skip_rate_by_G"]["4"]["on_task"], g0b["skip_rate_by_G"]["4"]["mw"]]
    x = np.arange(2)
    ax.bar(x - 0.19, raw, 0.36, color=[ON, MW], alpha=0.40, edgecolor="k", lw=0.6)
    ax.bar(x + 0.19, cor, 0.36, color=[ON, MW], edgecolor="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(["on-task", "MW"])
    ax.set_ylabel("skipping rate"); ax.set_ylim(0, 0.93)
    ax.text(0.5, 0.855, "+0.168, $p$ = 7×10$^{-11}$", ha="center", fontsize=6.3, color=GREY)
    ax.text(0.5, 0.775, "−0.042, $p$ = 2×10$^{-6}$", ha="center", fontsize=6.3, fontweight="bold")
    ax.bar([np.nan], [np.nan], color="0.5", alpha=0.40, label="any unfixated word")
    ax.bar([np.nan], [np.nan], color="0.3", label="scan-path defined")
    ax.legend(frameon=False, loc="upper left", fontsize=6.0, handlelength=1.0,
              bbox_to_anchor=(0.0, 0.80))
    ax.set_title("Definition decides the sign", loc="left")

    ax = fig.add_subplot(gs[0, 1]); panel_label(ax, "b", dx=-0.22, dy=1.17)
    Gs = [1, 2, 3, 4, 6, 10]
    d = [g0b["skip_rate_by_G"][str(G)]["diff"]["mean"] for G in Gs]
    lo = [g0b["skip_rate_by_G"][str(G)]["diff"]["ci"][0] for G in Gs]
    hi = [g0b["skip_rate_by_G"][str(G)]["diff"]["ci"][1] for G in Gs]
    ax.errorbar(Gs, d, yerr=[np.array(d) - lo, np.array(hi) - np.array(d)], fmt="o-",
                color=MW, ms=4, lw=1.2, capsize=2)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("maximum stepped-over words"); ax.set_ylabel("skip rate, MW − on-task")
    ax.set_title("Robust to the threshold", loc="left")

    # (c) large steps are line transitions
    ax = fig.add_subplot(gs[0, 2]); panel_label(ax, "c", dx=-0.22, dy=1.17)
    R1 = pd.DataFrame(rev["R1_gap_vs_line"])
    order = ["0", "1", "2-4", ">4"]
    xg = np.arange(len(order)); w = 0.36
    for st, c, lab in [(0, ON, "on-task"), (1, MW, "MW")]:
        v = [float(R1[(R1.gap == g) & (R1.mw == st)].frac_line_change.iloc[0]) * 100 for g in order]
        ax.bar(xg + (st - 0.5) * w, v, w, color=c, edgecolor="k", lw=0.6, label=lab)
    ax.set_xticks(xg); ax.set_xticklabels(order)
    ax.set_xlabel("words stepped over"); ax.set_ylabel("steps crossing a text line (%)")
    ax.set_ylim(0, 112); ax.legend(frameon=False, loc="upper left", fontsize=6.4)
    ax.set_title("Large steps cross lines", loc="left")

    ax = fig.add_subplot(gs[1, 0]); panel_label(ax, "d", dx=-0.22, dy=1.17)
    b = json.load(open(SEL / "results/blackout_anatomy.json"))
    o = b["offword_fraction_same_page"]
    ax.bar([0, 1], [o["on_task"], o["mw"]], 0.5, color=[ON, MW], edgecolor="k", lw=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["on-task", "MW"])
    ax.set_ylabel("off-word fraction\nof within-page time")
    ax.set_ylim(0, 0.26)
    ax.text(0.5, 0.215, f"$p$ = {o['diff']['p']:.2f}", ha="center", fontsize=7)
    ax.set_title("Interval between fixations", loc="left")

    # (e) layout-based replication
    ax = fig.add_subplot(gs[1, 1]); panel_label(ax, "e", dx=-0.22, dy=1.17)
    props3 = ["zipf", "length", "surprisal"]
    xg = np.arange(3); w = 0.38
    for k, c, lab in [("G1", ON, "gap criterion"), ("R3_selectivity", "#1E8449", "line-interior only")]:
        src = g1["G1_primary"] if k == "G1" else rev["R3_selectivity"]
        v = [src[p]["retention_pct"] for p in props3]
        e = [[abs(src[p]["ci"][i] / src[p]["D_on"] * 100) for p in props3] for i in (0, 1)]
        ax.bar(xg + (0 if k == "G1" else 1) * w - w / 2, v, w, color=c, edgecolor="k", lw=0.6, label=lab)
    ax.axhline(100, color="k", lw=0.7, ls="--")
    ax.set_xticks(xg); ax.set_xticklabels(["freq.", "length", "surp."])
    ax.set_ylabel("retention of selectivity (%)"); ax.set_ylim(0, 152)
    ax.legend(frameon=False, loc="upper center", fontsize=6.2, ncol=2,
              columnspacing=0.8, handlelength=1.0, handletextpad=0.4)
    ax.set_title("Independent of the criterion", loc="left")

    ax = fig.add_subplot(gs[1, 2]); panel_label(ax, "f", dx=-0.22, dy=1.17)
    Tt = pd.read_parquet(SEL / "artifacts/words_traversal.parquet")
    Tt = Tt[((Tt.skipped == 0) | (Tt.gap <= 4)) & Tt.state_agree]
    f = pd.read_parquet(COUP / "reading_fixations.parquet").sort_values(["subject", "run", "tStart"])
    gg = f.groupby(["subject", "run"], sort=False)
    f["refix"] = (gg["pos"].shift(-1) == f["pos"]).astype(float)
    f.loc[gg["pos"].shift(-1).isna(), "refix"] = np.nan
    Wd = f[f.is_firstpass == 1].groupby(["subject", "word_key"]).agg(
        GD=("fix_dur", "sum"), is_mw=("is_mw", "first")).reset_index()
    C = pd.read_parquet(SEL / "artifacts/corrective_regressions.parquet")

    def pc(df, col, log=False):
        if log:
            lg = df.assign(_l=np.log(df[col].clip(lower=1))).groupby(["subject", "is_mw"])._l.mean().unstack().dropna()
            return float((np.exp(lg[1] - lg[0]) - 1).mean() * 100)
        p = df.groupby(["subject", "is_mw"])[col].mean().unstack().dropna()
        return float((p[1] / p[0] - 1).mean() * 100)

    items = [("gaze duration", pc(Wd, "GD", True), 1.7e-6),
             ("regressions", pc(f.dropna(subset=["regression_out"]), "regression_out"),
              g4["regression_rate"]["diff"]["p"]),
             ("refixations", pc(f.dropna(subset=["refix"]), "refix"), g4["refixation_rate"]["diff"]["p"]),
             ("corrective\nreturns", pc(C, "corrective"), g4["corrective_rate"]["diff"]["p"]),
             ("skipping", pc(Tt, "skipped"), g0b["skip_rate_by_G"]["4"]["diff"]["p"])]
    v = [i[1] for i in items]; yy = np.arange(len(items))[::-1]
    ax.barh(yy, v, 0.6, color=[MW if q > 0 else "#1E8449" for q in v], edgecolor="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.7)
    for q, y0, (_, _, p) in zip(v, yy, items):
        # negative bars are labelled inside the bar so the text cannot reach the tick labels
        ax.text(q + 1.4, y0, f"{q:+.1f}%", va="center", ha="left", fontsize=6.6,
                color="k" if q > 0 else "w")
    ax.set_yticks(yy); ax.set_yticklabels([i[0] for i in items], fontsize=6.8)
    ax.set_xlabel("change during MW (%)"); ax.set_xlim(-34, 34)
    ax.set_title("Reading becomes effortful", loc="left")

    fig.savefig(OUT / "fig4_measurement.pdf"); fig.savefig(OUT / "fig4_measurement.png", dpi=300)
    plt.close(fig); print("fig4 done")


# =============================================================== FIGURE 5
def figure5():
    """Skimming and mind-wandering are different states (Section: sec:goal)."""
    fig = plt.figure(figsize=(18 * CM, 11.4 * CM))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.42], wspace=0.68, hspace=0.95,
                          left=0.105, right=0.985, top=0.875, bottom=0.135)

    SL = pd.read_csv(SEL / "artifacts/duration_slopes_by_state.csv")
    r_on, r_mw = SL.zipf_on.to_numpy(), SL.zipf_mw.to_numpy()
    z_nr, z_tsr = zuco_behaviour()
    n_on, n_mw, n_int = roamm_occ_frequency()
    zn_nr, zn_tsr = zuco_occ_frequency()

    # (a, b) the eye-movement policy
    ax = fig.add_subplot(gs[0, 0]); panel_label(ax, "a", dx=-0.40, dy=1.30)
    paired_panel(ax, np.abs(r_on), np.abs(r_mw), ["on-task", "mind-\nwandering"], [ON, MW],
                 "|frequency \u2192 log\ngaze duration|", "Behaviour, spontaneous")
    ax.set_ylim(0, 0.42)
    ax.text(0.5, 0.955, f"{g56['G5_across_word_signed_retention']['zipf']['retention_pct']:.0f}% retained",
            transform=ax.transAxes, fontsize=6.8, ha="center", va="top")

    ax = fig.add_subplot(gs[0, 1]); panel_label(ax, "b", dx=-0.30, dy=1.30)
    paired_panel(ax, np.abs(z_nr), np.abs(z_tsr), ["normal\nreading", "instructed\nsearch"],
                 [ON, GOAL], "", "Behaviour, instructed")
    ax.set_ylim(0, 0.42)
    ax.text(0.5, 0.955, f"{abs(z_tsr.mean() / z_nr.mean()) * 100:.0f}% retained",
            transform=ax.transAxes, fontsize=6.8, ha="center", va="top")

    # (d, e) the response to a fixated word
    ax = fig.add_subplot(gs[1, 0]); panel_label(ax, "d", dx=-0.40, dy=1.30)
    paired_panel(ax, n_on, n_mw, ["on-task", "mind-\nwandering"], [ON, MW],
                 "occipital frequency\nresponse (\u00b5V/SD)", "Brain, spontaneous")
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_ylim(-0.30, 0.90)
    ax.text(0.5, 0.955, f"{n_mw.mean() / n_on.mean() * 100:.0f}% retained",
            transform=ax.transAxes, fontsize=6.8, ha="center", va="top")

    ax = fig.add_subplot(gs[1, 1]); panel_label(ax, "e", dx=-0.30, dy=1.30)
    paired_panel(ax, zn_nr, zn_tsr, ["normal\nreading", "instructed\nsearch"], [ON, GOAL],
                 "", "Brain, instructed")
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_ylim(-0.16, 0.48)
    ax.text(0.5, 0.955, f"{zn_tsr.mean() / zn_nr.mean() * 100:.0f}% retained",
            transform=ax.transAxes, fontsize=6.8, ha="center", va="top")

    # (c) the dissociation, one estimator for all four cells
    ax = fig.add_subplot(gs[:, 2]); panel_label(ax, "c", dx=-0.36, dy=1.10)
    rows = [
        ("mind-wandering", MW, pct_of_base(r_mw - r_on, r_on.mean())),
        ("instructed search", GOAL, pct_of_base(z_tsr - z_nr, z_nr.mean())),
        ("mind-wandering", MW, pct_of_base(n_int, n_on.mean())),
        ("instructed search", GOAL, pct_of_base(zn_tsr - zn_nr, zn_nr.mean())),
    ]
    y = np.array([3.35, 2.35, 1.0, 0.0])
    ax.axvspan(-20, 20, color="k", alpha=0.06, lw=0)
    ax.axvline(0, color="k", lw=0.8)
    ticks = []
    for (st, col, (m, lo, hi, n)), yy in zip(rows, y):
        ax.errorbar([m], [yy], xerr=[[m - lo], [hi - m]], fmt="o", ms=5.5, color=col,
                    ecolor=col, lw=1.4, capsize=2.4, zorder=3)
        ax.text(m, yy + 0.15, f"{m:+.0f}%", ha="center", fontsize=6.8, color=col)
        ticks.append(f"{st}\n($n$={n})")
    ax.set_yticks(y); ax.set_yticklabels(ticks, fontsize=6.8)
    ax.set_ylim(-0.62, 3.95)
    ax.axhline(1.72, color="0.85", lw=0.9)
    for yy, lab in [(0.985, "eye-movement policy"), (0.485, "response to a fixated word")]:
        ax.text(0.015, yy, lab.upper(), transform=ax.transAxes, fontsize=5.8,
                color="0.30", fontweight="bold", va="top")
    ax.set_xlabel("change in frequency coupling\n(% of the engaged state)")
    ax.set_xlim(-72, 108); ax.set_xticks([-50, -25, 0, 25, 50, 75, 100])
    ax.set_title("Only the goal decouples", loc="left")
    ax.text(0.015, 0.015, "shaded: $\\pm$20% equivalence region",
            transform=ax.transAxes, fontsize=5.9, color="0.40")

    fig.savefig(OUT / "fig5_states.pdf"); fig.savefig(OUT / "fig5_states.png", dpi=300)
    plt.close(fig); print("fig5 done")


# =============================================================== FIGURE 6
def figure6():
    """What changes in the brain is gain, not pattern (Section: sec:gain)."""
    import mne
    mne.set_log_level("error")

    B = np.load(COUP / "rerp_betas.npy")
    meta = json.load(open(COUP / "rerp_meta.json"))
    t = np.asarray(meta["lags_ms"])
    ip = {x: i for i, x in enumerate(meta["pred_names"])}
    w = (t >= 150) & (t <= 290)
    cen = lambda x: x - x.mean(-1, keepdims=True)
    gfp = lambda x: np.sqrt(np.mean(cen(x) ** 2, -1))

    on = B[:, ip["intercept"]][:, w].mean(1).mean(0)
    mwd = B[:, ip["mw"]][:, w].mean(1).mean(0)
    P = topo["mw_vs_fixation_crossfit_proportional_residual"]
    alpha = P["loo_scale_mean"]                       # negative: an attenuation
    pred = alpha * on
    resid = mwd - pred

    names = ["AFz" if c == "Afz" else c for c in meta["channels"]]
    info = mne.create_info(names, 256.0, "eeg")
    info.set_montage("biosemi64", on_missing="warn")

    fig = plt.figure(figsize=(18 * CM, 12.2 * CM))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 0.98], hspace=0.40,
                             left=0.095, right=0.945, top=0.93, bottom=0.095)
    top = outer[0].subgridspec(1, 7, width_ratios=[1, 0.075, 0.34, 1, 1, 1, 0.075],
                               wspace=0.30)
    bot = outer[1].subgridspec(1, 3, wspace=0.70)

    v1 = float(np.abs(cen(on)).max())
    v2 = float(max(np.abs(cen(mwd)).max(), np.abs(cen(pred)).max(), np.abs(cen(resid)).max()))
    panels = [
        (0, on, v1, "Ordinary fixation field", f"GFP {gfp(on):.3f} \u00b5V", "a"),
        (3, mwd, v2, "MW minus on-task",
         f"GFP {gfp(mwd):.3f} \u00b5V\n$p$ = 2\u00d710$^{{-4}}$", "b"),
        (4, pred, v2, f"Predicted: {alpha*100:+.0f}% \u00d7 a", "one free scalar", "c"),
        (5, resid, v2, "Residual", f"GFP {P['group_mean_gfp_uV']:.3f} \u00b5V\n"
                                   f"$p$ = {P['p_signflip']:.2f}", "d"),
    ]
    for col, m, vr, ttl, note, lab in panels:
        ax = fig.add_subplot(top[0, col]); panel_label(ax, lab, dx=-0.02, dy=1.30)
        im, _ = mne.viz.plot_topomap(cen(m), info, axes=ax, show=False, cmap="RdBu_r",
                                     vlim=(-vr, vr), contours=4, sensors=False)
        ax.set_title(ttl, fontsize=7.4, pad=6)
        ax.text(0.5, -0.13, note, transform=ax.transAxes, fontsize=6.4, ha="center",
                va="top")
        if col in (0, 5):
            cax = fig.add_subplot(top[0, col + 1])
            fig.colorbar(im, cax=cax)
            cax.tick_params(labelsize=6.0)
            cax.set_title("\u00b5V", fontsize=6.5, pad=4)

    # (e) every channel: the difference is the on-task field, scaled
    ax = fig.add_subplot(bot[0, 0]); panel_label(ax, "e", dx=-0.26)
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    xs = np.linspace(cen(on).min() * 1.08, cen(on).max() * 1.08, 50)
    ax.plot(xs, alpha * xs, color="k", lw=1.2, ls="--",
            label=f"$-${abs(alpha)*100:.0f}% of the\non-task field")
    ax.plot(cen(on), cen(mwd), "o", ms=3.4, color=MW, alpha=0.85)
    ci = P["spatial_r_boot_ci95"]
    ax.text(0.03, 0.05, f"$r$ = {-P['absolute_spatial_r']:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]"
                        f"\n$R^2$ = {P['group_map_variance_explained_r2']:.2f}, 64 sensors",
            transform=ax.transAxes, fontsize=6.4)
    ax.set_xlabel("on-task fixation field (\u00b5V)")
    ax.set_ylabel("MW \u2212 on-task (\u00b5V)")
    ax.legend(frameon=False, loc="upper right", fontsize=6.2, handlelength=1.6)
    ax.set_title("One scalar describes it", loc="left")

    # (f) time-resolved global field power of the difference field
    ax = fig.add_subplot(bot[0, 1]); panel_label(ax, "f", dx=-0.30)
    S = B[:, ip["mw"]]                                   # readers x lags x channels
    g = gfp(S.mean(0))
    rng = np.random.default_rng(63)
    signs = rng.choice([-1.0, 1.0], size=(1000, S.shape[0]))
    null = gfp(np.einsum("ps,slc->plc", signs, S) / S.shape[0])
    lo, hi = np.percentile(null, [2.5, 97.5], axis=0)
    cl = topo["time_resolved_field_clusters"]["mw"]["significant"][0]
    ax.axvspan(cl["start_ms"], cl["end_ms"], color=MW, alpha=0.13, lw=0)
    ax.fill_between(t, lo, hi, color=GREY, alpha=0.30, lw=0, label="sign-flip null, 95%")
    ax.plot(t, g, color=MW, lw=1.5, label="observed")
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlim(-100, 500); ax.set_ylim(0, 0.132)
    ax.set_xlabel("time from fixation onset (ms)")
    ax.set_ylabel("global field power of the\nMW difference (\u00b5V)")
    ax.text(0.975, 0.975, f"{cl['start_ms']:.0f}\u2013{cl['end_ms']:.0f} ms\n"
                          f"cluster $p$ = {cl['p_cluster_fwer_family']:.4f}",
            transform=ax.transAxes, fontsize=6.3, ha="right", va="top")
    ax.legend(frameon=False, loc="lower left", fontsize=6.2, ncol=2, columnspacing=1.0)
    ax.set_title("When it happens", loc="left")

    # (g) field tests before and after the rescaling
    ax = fig.add_subplot(bot[0, 2]); panel_label(ax, "g", dx=-0.30)
    W = topo["window_field_tests"]["mw_lexical"]
    items = [("MW\ndifference", W["group_mean_gfp_uV"], W["null_95"], W["p_signflip"]),
             ("residual after\nrescaling", P["group_mean_gfp_uV"], P["null_95"], P["p_signflip"])]
    for i, (lab, v, n95, pv) in enumerate(items):
        ax.add_patch(Rectangle((i - 0.32, n95[0]), 0.64, n95[1] - n95[0],
                               fc=GREY, alpha=0.32, ec="none", zorder=1))
        ax.bar(i, v, 0.44, color=MW if i == 0 else GREY, edgecolor="k", lw=0.6, zorder=2)
        ax.text(i, v + 0.004, f"$p$ = {pv:.4f}" if pv < 0.01 else f"$p$ = {pv:.2f}",
                ha="center", fontsize=6.6, fontweight="bold" if pv < 0.05 else "normal")
    ax.set_xticks([0, 1]); ax.set_xticklabels([a for a, *_ in items], fontsize=6.8)
    ax.set_ylabel("global field power,\n150\u2013290 ms (\u00b5V)")
    ax.set_ylim(0, 0.092); ax.set_xlim(-0.75, 1.75)
    ax.plot([], [], "s", color=GREY, alpha=0.45, ms=6, label="sign-flip null, 95%")
    ax.legend(frameon=False, loc="upper right", fontsize=6.2)
    ax.set_title("Nothing is left over", loc="left")

    fig.savefig(OUT / "fig6_gain.pdf"); fig.savefig(OUT / "fig6_gain.png", dpi=300)
    plt.close(fig); print("fig6 done")


# =============================================================== FIGURE 7
def _spans(m):
    """Start and end indices of each run of True in a boolean array."""
    e = np.diff(np.r_[0, np.asarray(m).astype(int), 0])
    return list(zip(np.where(e == 1)[0], np.where(e == -1)[0]))


def _pick_session(c, min_spans=5, min_fix=600):
    """The session whose two state contrasts sit closest to the group means, among
    sessions with at least `min_spans` reported spans. A fixed rule, not a hand pick."""
    tgt = {}
    for col in ["attention", "slow"]:
        g = c.groupby(["subject", "is_mw"])[col].mean().unstack().dropna()
        tgt[col] = float((g[True] - g[False]).mean())
    best, bd = None, np.inf
    for (sub, run), g in c.groupby(["subject", "run"]):
        if len(g) < min_fix or len(_spans(g.is_mw.values)) < min_spans:
            continue
        d = np.hypot(g[g.is_mw].attention.mean() - g[~g.is_mw].attention.mean() - tgt["attention"],
                     g[g.is_mw].slow.mean() - g[~g.is_mw].slow.mean() - tgt["slow"])
        if d < bd:
            best, bd = (sub, run), d
    return best


def _peri_onset(c, col, pre=120, post=240, min_ep=30):
    """Reader-average index trace around the onset of a reported span."""
    lags = np.arange(-pre, post + 1)
    per_reader = []
    for sub, gs_ in c.groupby("subject"):
        ep = []
        for (_, run), g in gs_.groupby(["subject", "run"]):
            v = g.sort_values("fix_order_all")[col].to_numpy()
            for a_, _b in _spans(g.sort_values("fix_order_all").is_mw.values):
                idx = a_ + lags
                ok = (idx >= 0) & (idx < len(v))
                seg = np.full(len(lags), np.nan)
                seg[ok] = v[idx[ok]]
                ep.append(seg)
        if len(ep) >= 3:
            per_reader.append(np.nanmean(np.array(ep), axis=0))
    R = np.array(per_reader)
    return lags, R


def figure7():
    """A continuous index of text-driven reading does not move (Section: sec:index)."""
    c = attention_index()
    fig = plt.figure(figsize=(18 * CM, 12.0 * CM))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.14], width_ratios=[1.28, 0.8, 0.8, 1.22],
                          hspace=0.66, wspace=0.60, left=0.075, right=0.985,
                          top=0.895, bottom=0.105)
    LEV, TXT = "#B9770E", ON

    # (a) one session, both indices, reported spans shaded
    sub, run = _pick_session(c)
    g = c[(c.subject == sub) & (c.run == run)].sort_values("fix_order_all").reset_index(drop=True)
    ax = fig.add_subplot(gs[0, :]); panel_label(ax, "a", dx=-0.045, dy=1.22)
    for a_, b_ in _spans(g.is_mw.values):
        ax.axvspan(a_, b_, color=MW, alpha=0.18, lw=0)
    WD = 101                                    # display smoothing only
    for col, colr, lab in [("slow", LEV, "level index (fixation duration)"),
                           ("attention", TXT, "text-driven index (word properties \u2192 duration)")]:
        y = pd.Series(g[col].values).rolling(WD, center=True, min_periods=30).mean()
        ax.plot(np.arange(len(g)), y, color=colr, lw=1.4, label=lab)
    ax.axhline(50, color="k", lw=0.6, ls=":")
    ax.set_xlim(0, len(g) - 1); ax.set_ylim(20, 97)
    ax.set_xlabel("fixation number within the session")
    ax.set_ylabel("index (percentile)")
    hh, ll = ax.get_legend_handles_labels()
    ax.legend(hh + [Rectangle((0, 0), 1, 1, fc=MW, alpha=0.18, ec="none")],
              ll + ["reported mind-wandering"], frameon=False, loc="upper left",
              ncol=3, fontsize=6.4, columnspacing=1.4, handlelength=1.6,
              borderaxespad=0.3)
    ax.set_title("Two continuous indices through a reading session", loc="left")
    ax.text(1.0, 1.05, f"reader {sub}, {g.story.iloc[0].replace('_', ' ')}, "
                       f"{len(g)} fixations; {WD}-fixation display smoothing",
            transform=ax.transAxes, fontsize=6.2, ha="right", color="0.35")

    # (b) the same two indices, aligned to the onset of a reported span
    ax = fig.add_subplot(gs[1, 0]); panel_label(ax, "b", dx=-0.30)
    rng0 = np.random.default_rng(8)
    smooth = lambda v: pd.Series(v).rolling(21, center=True, min_periods=8).mean().to_numpy()
    for col, colr, lab in [("slow", LEV, "level"), ("attention", TXT, "text-driven")]:
        lags, R = _peri_onset(c, col)
        R = np.array([smooth(r) for r in R])
        m_ = np.nanmean(R, axis=0)
        bs = np.nanmean(R[rng0.integers(0, len(R), size=(2000, len(R)))], axis=1)
        lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
        ax.fill_between(lags, lo, hi, color=colr, alpha=0.22, lw=0)
        ax.plot(lags, m_, color=colr, lw=1.4, label=lab)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.axhline(50, color="k", lw=0.6, ls=":")
    ax.set_xlabel("fixations from span onset")
    ax.set_ylabel("index (percentile)")
    ax.legend(frameon=False, loc="lower right", fontsize=6.2, ncol=2, columnspacing=1.0)
    ax.set_xlim(lags[0], lags[-1])
    ax.set_title("Aligned to span onset", loc="left")

    # (c, d) every reader
    for j, (col, ttl, colr) in enumerate([
            ("attention", "Text-driven index", TXT), ("slow", "Level index", LEV)]):
        ax = fig.add_subplot(gs[1, 1 + j]); panel_label(ax, "cd"[j], dx=-0.42)
        gg = c.groupby(["subject", "is_mw"])[col].mean().unstack().dropna()
        a_, b_ = gg[False].to_numpy(), gg[True].to_numpy()
        paired_panel(ax, a_, b_, ["on-task", "mind-\nwandering"], [ON, MW],
                     "index (percentile)" if j == 0 else "", ttl)
        ax.axhline(50, color="k", lw=0.6, ls=":")
        ax.set_ylim(37.5, 74)
        ax.text(0.5, 0.985, f"{int((b_ < a_).sum())}/{len(a_)} readers\nlower during MW",
                transform=ax.transAxes, fontsize=6.3, ha="center", va="top")

    # (e) the contrast and the classifier
    ax = fig.add_subplot(gs[1, 3]); panel_label(ax, "e", dx=-0.26)
    rng = np.random.default_rng(64)
    rows = []
    for col, colr, lab in [("attention", TXT, "text-driven\nindex"), ("slow", LEV, "level\nindex")]:
        gg = c.groupby(["subject", "is_mw"])[col].mean().unstack().dropna()
        d = (gg[True] - gg[False]).to_numpy()
        bs = d[rng.integers(0, len(d), size=(20000, len(d)))].mean(axis=1)
        pv = 2 * min((bs > 0).mean(), (bs < 0).mean())
        sc = -c[col] if col == "attention" else c[col]
        auc = float(_auc(c.is_mw.to_numpy(), sc.to_numpy()))
        rows.append((lab, colr, d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), pv, auc))
    y = np.arange(len(rows))[::-1]
    ax.axvline(0, color="k", lw=0.8)
    for (lab, colr, m_, lo, hi, pv, auc), yy in zip(rows, y):
        ax.errorbar([m_], [yy], xerr=[[m_ - lo], [hi - m_]], fmt="o", ms=5.5, color=colr,
                    ecolor=colr, lw=1.4, capsize=2.4, zorder=3)
        ptxt = "$p$ < 10$^{-4}$" if pv < 1e-4 else f"$p$ = {pv:.2f}"
        xt = float(np.clip(m_, -3.2, 3.2))
        ax.text(xt, yy + 0.22, f"{m_:+.1f} points\n{ptxt}", ha="center", fontsize=6.4, color=colr)
        ax.text(xt, yy - 0.42, f"AUC {auc:.3f}", ha="center", fontsize=6.4, color="0.35")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7.0)
    ax.set_ylim(-0.62, 1.75)
    ax.set_xlabel("MW \u2212 on-task\n(percentile points, 95% CI)")
    ax.set_xlim(-6.0, 6.0); ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_title("Group contrast", loc="left")

    fig.savefig(OUT / "fig7_index.pdf"); fig.savefig(OUT / "fig7_index.png", dpi=300)
    plt.close(fig); print("fig7 done")


# =============================================================== FIGURE 8
def figure8():
    fig = plt.figure(figsize=(18 * CM, 10.6 * CM))
    gs = fig.add_gridspec(2, 3, wspace=0.52, hspace=0.62, left=0.078, right=0.985, top=0.91, bottom=0.085)
    D = np.load(COUP / "dynamics_subjavg.npz")
    t = D["t"]

    ax = fig.add_subplot(gs[0, 0]); panel_label(ax, "a")
    for key, c, lab in [("on_pupil", MW, "MW onset"), ("pseudo_pupil", GREY, "matched control")]:
        Y = D[key]; m, s = np.nanmean(Y, 0), sem(Y)
        ax.fill_between(t, m - s, m + s, color=c, alpha=0.22, lw=0)
        ax.plot(t, m, color=c, lw=1.4, label=lab)
    ax.axvline(0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("time from onset (s)"); ax.set_ylabel("pupil (z)")
    ax.set_title("Pupil dilates at onset", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=6.2)
    ax.text(0.97, 0.05, "+0.20 z, $p$ = 1.5×10$^{-4}$", transform=ax.transAxes, fontsize=6.5,
            ha="right", bbox=dict(fc="w", ec="none", pad=1.0))

    ax = fig.add_subplot(gs[0, 1]); panel_label(ax, "b")
    for key, c, lab in [("on_beta", MW, "MW onset"), ("pseudo_beta", GREY, "matched control")]:
        Y = D[key]; m, s = np.nanmean(Y, 0), sem(Y)
        ax.fill_between(t, m - s, m + s, color=c, alpha=0.22, lw=0)
        ax.plot(t, m, color=c, lw=1.4, label=lab)
    ax.axvline(0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("time from onset (s)"); ax.set_ylabel("central beta power (z)")
    ax.set_title("Cortical desynchronisation", loc="left")
    ax.text(0.97, 0.05, "−0.074 z, $p$ = 2.4×10$^{-4}$", transform=ax.transAxes, fontsize=6.5,
            ha="right", bbox=dict(fc="w", ec="none", pad=1.0))

    # (c) regressions accumulate across the episode
    ax = fig.add_subplot(gs[0, 2]); panel_label(ax, "c")
    obs = mctl["C_observed"]["regression_out"]
    bl = ["0-2", "2-5", "5-10", ">10"]
    base_on = float(np.nanmean(pd.read_parquet(COUP / "reading_fixations.parquet")
                               .query("is_mw == 0").regression_out))
    v = [b - base_on for b in obs["bin_means"]]
    v0 = v[0]
    ax.plot(range(4), v, "o-", color=MW, ms=5, lw=1.6, label="mind-wandering")
    nm = mctl["C_control"]["regression_out"]["null_mean"]
    ns = mctl["C_control"]["regression_out"]["null_sd"]
    xs = np.arange(4) - 0.0
    lo = v0 + (nm - 1.96 * ns) * xs
    hi = v0 + (nm + 1.96 * ns) * xs
    ax.fill_between(range(4), lo, hi, color=GREY, alpha=0.30, lw=0,
                    label="matched control, 95%")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(range(4)); ax.set_xticklabels(bl)
    ax.set_xlabel("time since episode onset (s)")
    ax.set_ylabel("regression rate above\nthe reader's own baseline")
    ax.legend(frameon=False, loc="upper left", fontsize=6.2)
    ax.text(0.035, 0.60, f"trend $p$ = {obs['trend']['p']:.4f}\nvs control $p$ < 0.005",
            transform=ax.transAxes, fontsize=6.2, ha="left", va="top")
    ax.set_title("Regressions accumulate", loc="left")

    # (d) inter-subject alignment, decomposed
    ax = fig.add_subplot(gs[1, 0]); panel_label(ax, "d")
    keys = [("y_raw", "raw"), ("y_z", "effort\nremoved"), ("y_resid", "word properties\nremoved")]
    x = np.arange(3); w = 0.36
    for j, (k, lab) in enumerate(keys):
        e = mech["D_isc"][k]
        ax.bar(j - w / 2, e["isc_on"], w, color=ON, edgecolor="k", lw=0.6)
        ax.bar(j + w / 2, e["isc_mw"], w, color=MW, edgecolor="k", lw=0.6)
        star = "*" if e["diff"]["p"] < 0.05 else "n.s."
        ax.text(j, max(e["isc_on"], e["isc_mw"]) + 0.012, star, ha="center", fontsize=7.5,
                fontweight="bold" if star == "*" else "normal")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in keys], fontsize=5.9)
    ax.set_ylabel("inter-subject correlation"); ax.set_ylim(0, 0.46)
    ax.plot([], [], "s", color=ON, ms=4, label="on-task"); ax.plot([], [], "s", color=MW, ms=4, label="MW")
    ax.legend(frameon=False, loc="upper right", fontsize=6.2)
    ax.set_title("Alignment loss is in the residual", loc="left")

    # (e) the extra effort is not aimed at difficult words
    ax = fig.add_subplot(gs[1, 1]); panel_label(ax, "e")
    for meas, c, lab in [("logdur", MW, "fixation duration"),
                         ("regression_out", "#B9770E", "regressions"),
                         ("refix", GREY, "refixations")]:
        q = np.array(mech["B_targeting"][meas]["per_quartile_delta"])
        ax.plot(range(4), q / q.mean(), "o-", color=c, ms=4, lw=1.4, label=lab)
    ax.axhline(1, color="k", lw=0.7, ls="--")
    ax.set_xticks(range(4)); ax.set_xticklabels(["easy", "2", "3", "hard"])
    ax.set_xlabel("word difficulty quartile")
    ax.set_ylabel("MW increase, relative to\nits own average (1 = uniform)")
    ax.set_ylim(0.3, 1.9); ax.legend(frameon=False, loc="upper left", fontsize=6.2)
    ax.text(0.97, 0.06, "all trends $p$ > 0.25", transform=ax.transAxes, fontsize=6.2, ha="right")
    ax.set_title("Extra effort is undirected", loc="left")

    ax = fig.add_subplot(gs[9, 9]) if False else None
    g = isc["gaze_fixdur"]
    ax = fig.add_subplot(gs[1, 2]); panel_label(ax, "f")
    conds = ["NR", "SR1", "SR2", "TSR"]
    v = [abs(g7["slopes"]["zipf"][c]["mean"]) for c in conds]
    ax.bar(range(4), v, 0.6, color=[ON, ON, "#7FA8CD", GOAL], edgecolor="k", lw=0.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["NR\ns1", "SR h1\ns1", "SR h2\ns2", "TSR\ns2"], fontsize=6.6)
    ax.set_ylabel("|frequency → log gaze duration|"); ax.set_ylim(0, 0.175)
    c = g7["contrasts"]["zipf"]
    ax.annotate("", xy=(1, 0.150), xytext=(2, 0.150), arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.9))
    ax.text(1.5, 0.154, f"session {c['session_effect_SR1_to_SR2']['retention_pct']:.0f}%",
            ha="center", fontsize=6.3, color=GREY)
    ax.annotate("", xy=(2, 0.126), xytext=(3, 0.126), arrowprops=dict(arrowstyle="<->", color="k", lw=0.9))
    ax.text(2.5, 0.130, f"task {c['task_within_session2_SR2_to_TSR']['retention_pct']:.0f}%",
            ha="center", fontsize=6.3, fontweight="bold")
    ax.set_title("A goal does decouple", loc="left")

    fig.savefig(OUT / "fig8_changes.pdf"); fig.savefig(OUT / "fig8_changes.png", dpi=300)
    plt.close(fig); print("fig8 done")


# =============================================================== FIGURE 9
def figure9():
    """Comprehension fails where the mind was, not where the eyes were (sec:comp)."""
    fig = plt.figure(figsize=(18 * CM, 7.4 * CM))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.18, 0.98], wspace=0.60,
                          left=0.075, right=0.985, top=0.855, bottom=0.235)

    # (a) accuracy by where the lapse fell
    ax = fig.add_subplot(gs[0, 0]); panel_label(ax, "a", dx=-0.34)
    D = mwloc["R1_descriptive"]
    bars = [("no lapse\nanywhere", D["acc_no_mw_anywhere"], D["n_no_mw_anywhere"], ON),
            ("lapse on the page,\nnot on the answer", D["acc_mw_page_no_mw_on_evidence"],
             D["n_no_mw_on_evidence"], "#E59866"),
            ("lapse on the\nanswer span", D["acc_mw_on_evidence"], D["n_mw_on_evidence"], MW)]
    for i, (lab, v, n, col) in enumerate(bars):
        ax.bar(i, v, 0.62, color=col, edgecolor="k", lw=0.6)
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=6.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"none\n($n$={bars[0][2]})", f"elsewhere\n($n$={bars[1][2]})",
                        f"answer span\n($n$={bars[2][2]})"], fontsize=5.9)
    ax.set_xlabel("where the reader lapsed")
    ax.set_ylabel("comprehension accuracy"); ax.set_ylim(0, 0.78)
    ax.axhline(0.56, color="k", lw=0.8, ls="--")
    ax.text(2.45, 0.712, "no-passage\nceiling", fontsize=5.9, ha="right", va="center",
            color="0.30")
    ax.set_title("Where the lapse fell", loc="left")

    # (b) the overlap gradient across 1000 random regions
    ax = fig.add_subplot(gs[0, 1]); panel_label(ax, "b", dx=-0.24)
    R3 = mwloc["R3_overlap_gradient"]
    ax.plot(GRAD.overlap, GRAD.stat, "o", ms=2.6, color=GREY, alpha=0.42)
    xs = np.linspace(0, max(GRAD.overlap.max(), 1.0), 50)
    ax.plot(xs, R3["intercept"] + R3["slope"] * xs, color="k", lw=1.5)
    ax.plot([1.0], [R3["observed_true_span"]], "*", ms=13, color=MW, zorder=4)
    ax.plot([0], [R3["stat_at_zero_overlap"]], "o", ms=6, mfc="w", mec="k", mew=1.2, zorder=4)
    ax.annotate(f"{R3['stat_at_zero_overlap']:+.3f} at zero overlap",
                xy=(0, R3["stat_at_zero_overlap"]), xytext=(0.10, 0.016),
                fontsize=6.2, arrowprops=dict(arrowstyle="->", lw=0.8, color="k"))
    ax.text(0.46, -0.004, "1000 random regions,\nsize held constant", fontsize=6.2,
            color="0.35", va="top")
    ax.text(0.80, -0.033, f"slope {R3['slope']:.3f}\n$p$ = 1.5\u00d710$^{{-14}}$",
            fontsize=6.4, ha="center")
    ax.text(0.985, -0.0685, "the true\nanswer span", fontsize=6.4, ha="right", va="top",
            color=MW)
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("overlap of the region with the answer span")
    ax.set_ylabel("effect of a lapse in the region\non accuracy")
    ax.set_xlim(-0.05, 1.10)
    ax.set_title("Damage tracks distance from the answer", loc="left")

    # (c) the mind localises, the eyes do not
    ax = fig.add_subplot(gs[0, 2]); panel_label(ax, "c", dx=-0.26)
    T2 = outc["T2_random_region_permutation"]
    rows = [("lapse on the span\n(the mind)", MW, mwctl["observed_evidence_stat"],
             mwctl["null_q"]["2.5"], mwctl["null_q"]["97.5"], mwctl["null_q"]["50"],
             mwctl["percentile_of_observed"], mwctl["p_one_sided_more_negative"], "<"),
            ("reading the span\n(the eyes)", ON, T2["observed"],
             T2["null_mean"] - 1.96 * T2["null_sd"], T2["null_mean"] + 1.96 * T2["null_sd"],
             T2["null_mean"], T2["percentile"], T2["p_one_sided"], "=")]
    for i, (lab, col, obs, lo, hi, med, pctl, pv, rel) in enumerate(rows):
        yy = 1 - i
        ax.plot([lo, hi], [yy, yy], color=GREY, lw=6, alpha=0.45, solid_capstyle="butt",
                zorder=1)
        ax.plot([med], [yy], "|", ms=9, color="0.35", mew=1.2, zorder=2)
        ax.plot([obs], [yy], "o", ms=7, color=col, zorder=3)
        ax.text(float(np.clip(obs, -0.034, 0.008)), yy - 0.16,
                f"{pctl:.0f}th percentile of the null\n"
                f"$p$ {'<' if pv < 0.002 else '='} "
                f"{'0.001' if pv < 0.002 else f'{pv:.2f}'}",
                ha="center", va="top", fontsize=6.1, color=col)
    ax.axvline(0, color="k", lw=0.7, ls=":")
    ax.set_yticks([1, 0]); ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    ax.set_ylim(-0.92, 1.42)
    ax.set_xlabel("effect on accuracy, against\nits random-region null")
    ax.set_xlim(-0.082, 0.082); ax.set_xticks([-0.05, 0, 0.05])
    ax.plot([], [], "s", color=GREY, alpha=0.45, ms=6, label="random-region null, 95%")
    ax.legend(frameon=False, loc="lower left", fontsize=6.2, bbox_to_anchor=(-0.02, -0.02))
    ax.set_title("The mind, not the eyes", loc="left")

    fig.savefig(OUT / "fig9_comprehension.pdf")
    fig.savefig(OUT / "fig9_comprehension.png", dpi=300)
    plt.close(fig); print("fig9 done")


if __name__ == "__main__":
    figure1(); figure2(); figure3(); figure4()
    figure5(); figure6(); figure7(); figure8(); figure9()
    print("all figures written to", OUT)
