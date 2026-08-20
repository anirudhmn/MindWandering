#!/usr/bin/env python3
"""Synthesis figure for the localisation analysis."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common60 import RES, FIG

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8.5,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a9892"
GRID = dict(color="#e6e5e1", lw=0.7, zorder=0)

H1 = json.loads((RES / "gates_h1.json").read_text())
PERM = json.loads((RES / "h1_stimulus_permutation.json").read_text())
OUT = json.loads((RES / "outcome.json").read_text())
MWC = json.loads((RES / "mw_localisation_control.json").read_text())
DEEP = json.loads((RES / "mw_localisation_deepen.json").read_text())
GRAD = pd.read_csv(RES / "mw_overlap_gradient.csv")
NEU = json.loads((RES / "neural.json").read_text())

fig = plt.figure(figsize=(11.4, 9.6))
gs = GridSpec(3, 2, figure=fig, hspace=0.62, wspace=0.28,
              left=0.085, right=0.975, top=0.925, bottom=0.06)

# ---------------------------------------------------------------- A effects on reading
ax = fig.add_subplot(gs[0, 0])
rows = [("log gaze duration", "log_gaze"), ("first fixation dur.", "log_ffd"),
        ("P(refixation)", "refix"), ("P(regression out)", "regr_out")]
ys = np.arange(len(rows))[::-1]
ax.axvline(0, color=MUTED, lw=0.9, zorder=1)
ax.grid(axis="x", **GRID)
SURVIVES = {"refix"}
for y, (lab, k) in zip(ys, rows):
    b = H1["H1"][k]["subject_bootstrap_within_lemma"]
    col = BLUE if k in SURVIVES else (ORANGE if b["p"] < 0.05 else MUTED)
    ax.plot([b["ci"][0], b["ci"][1]], [y, y], color=col, lw=2.2, solid_capstyle="round", zorder=3)
    ax.plot([b["mean"]], [y], "o", ms=7.5, color=col, mec="white", mew=1.4, zorder=4)
    ax.text(b["ci"][1] + 0.0009, y, f"p={b['p']:.3f}", va="center", ha="left",
            fontsize=7.4, color=INK2)
ax.text(0.995, 0.03, "reader-level bootstrap; blue = also survives\nthe stimulus-level test in panel B",
        transform=ax.transAxes, fontsize=7.0, color=INK2, ha="right", va="bottom")
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=8.2)
ax.set_xlim(-0.0055, 0.0155)
ax.set_xlabel("effect per SD of sentence importance\n(within word type, page + lemma fixed effects)")
ax.set_title("A   Readers return to important words —\n     they do not linger on first pass", loc="left")

# ---------------------------------------------------------------- B stimulus permutation
ax = fig.add_subplot(gs[0, 1])
for k, lab, col, off in [("refix", "P(refixation)", BLUE, 0), ("log_gaze", "gaze duration", ORANGE, 1)]:
    p = PERM[k]
    lo, hi = p["null_mean"] - 1.96 * p["null_sd"], p["null_mean"] + 1.96 * p["null_sd"]
    y = 1 - off
    ax.add_patch(plt.Rectangle((lo, y - 0.17), hi - lo, 0.34, color="#eeedea", ec=MUTED, lw=0.7, zorder=2))
    ax.plot([p["null_mean"]], [y], "|", ms=13, color=MUTED, mew=1.4, zorder=3)
    ax.plot([p["observed_beta"]], [y], "D", ms=8, color=col, mec="white", mew=1.4, zorder=5)
    ax.text(p["observed_beta"], y + 0.27, f"observed\np={p['p_one_sided']:.3f}", ha="center",
            va="bottom", fontsize=7.4, color=col, fontweight="bold")
    ax.text(lo - 0.0004, y, lab, ha="right", va="center", fontsize=8.2, color=INK)
ax.axvline(0, color=MUTED, lw=0.9, zorder=1)
ax.grid(axis="x", **GRID)
ax.set_yticks([]); ax.set_ylim(-0.75, 1.85); ax.set_xlim(-0.0075, 0.0075)
ax.set_xlabel("pooled coefficient per SD of importance")
ax.set_title("B   Reshuffling the ratings among the same page's\n     sentences: only refixation survives", loc="left")

# ---------------------------------------------------------------- C reading vs MW localisation
ax = fig.add_subplot(gs[1, 0])
t2 = OUT["T2_random_region_permutation"]
panels = [
    ("where the EYES went\n(evidence coverage)", t2["null_mean"], t2["null_sd"], t2["observed"],
     t2["p_one_sided"], t2["percentile"], AQUA),
    ("where the MIND was\n(MW on evidence)", MWC["null_mean"], MWC["null_sd"],
     MWC["observed_evidence_stat"], MWC["p_one_sided_more_negative"], MWC["percentile_of_observed"], ORANGE),
]
for i, (lab, nm, nsd, obs, p, pct, col) in enumerate(panels):
    y = (1 - i) * 1.25
    lo, hi = nm - 1.96 * nsd, nm + 1.96 * nsd
    ax.add_patch(plt.Rectangle((lo, y - 0.15), hi - lo, 0.30, color="#eeedea", ec=MUTED, lw=0.7, zorder=2))
    ax.plot([nm], [y], "|", ms=13, color=MUTED, mew=1.4, zorder=3)
    ax.plot([obs], [y], "D", ms=8.5, color=col, mec="white", mew=1.4, zorder=5)
    ax.text(obs, y - 0.24, f"p={p:.3f} ({pct:.0f}th pct)", ha="center", va="top",
            fontsize=7.4, color=col, fontweight="bold")
    ax.text(-0.105, y + 0.24, lab.replace("\n", " "), ha="left", va="bottom", fontsize=8.3, color=INK)
ax.axvline(0, color=MUTED, lw=0.9, zorder=1)
ax.grid(axis="x", **GRID)
ax.set_yticks([]); ax.set_ylim(-0.75, 1.95); ax.set_xlim(-0.107, 0.068)
ax.set_xlabel("effect on answering the item correctly\n(grey band = 1000 random equal-size regions on the same page)")
ax.set_title("C   The answer span is unremarkable for reading\n     and decisive for attention", loc="left")

# ---------------------------------------------------------------- D accuracy gradient
ax = fig.add_subplot(gs[1, 1])
d = DEEP["R1_descriptive"]
bars = [("no MW\nanywhere on page", d["acc_no_mw_anywhere"], d["n_no_mw_anywhere"], MUTED),
        ("MW on page,\nnot on the answer", d["acc_mw_page_no_mw_on_evidence"], d["n_no_mw_on_evidence"], BLUE),
        ("MW while the eyes\nwere on the answer", d["acc_mw_on_evidence"], d["n_mw_on_evidence"], ORANGE)]
x = np.arange(3)
ax.grid(axis="y", **GRID)
for xi, (lab, v, n, col) in zip(x, bars):
    ax.add_patch(plt.Rectangle((xi - 0.33, 0), 0.66, v, color=col, zorder=3,
                               joinstyle="round", lw=0))
    ax.text(xi, v + 0.017, f"{v:.3f}", ha="center", fontsize=8.6, color=INK, fontweight="bold")
    ax.text(xi, 0.028, f"n={n}", ha="center", fontsize=7.2, color="white")
ax.axhline(0.25, color=VIOLET, lw=1.1, ls=(0, (4, 2)), zorder=4)
ax.text(2.62, 0.258, "chance", fontsize=7.2, color=VIOLET, ha="right", va="bottom")
ax.annotate("", xy=(2, bars[2][1] + 0.022), xytext=(1, bars[1][1] + 0.022),
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4, shrinkA=3, shrinkB=3))
ax.text(1.5, 0.615, "− 23.5 points", ha="center", fontsize=8.8, color=ORANGE, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars], fontsize=7.8)
ax.set_xlim(-0.7, 2.68)
ax.set_ylim(0, 0.84); ax.set_ylabel("P(item answered correctly)")
ax.set_title("D   Mind-wandering costs 4 points, unless it lands\n     on the answer — then 27", loc="left")

# ---------------------------------------------------------------- E overlap gradient
ax = fig.add_subplot(gs[2, 0])
g = DEEP["R3_overlap_gradient"]
ax.grid(**GRID)
ax.scatter(GRAD["overlap"], GRAD["stat"], s=7, color=BLUE, alpha=0.32, lw=0, zorder=3)
xx = np.linspace(GRAD["overlap"].min(), GRAD["overlap"].max(), 50)
ax.plot(xx, g["intercept"] + g["slope"] * xx, color=INK, lw=1.8, zorder=5)
ax.axhline(g["observed_true_span"], color=ORANGE, lw=1.4, ls=(0, (5, 2)), zorder=4)
ax.text(GRAD["overlap"].min(), g["observed_true_span"] + 0.0035, "observed, true answer span",
        color=ORANGE, fontsize=7.6, ha="left", va="bottom", fontweight="bold")
ax.text(0.02, 0.955, f"slope {g['slope']:+.3f}    r={g['r']:+.2f}    p={g['p']:.0e}",
        transform=ax.transAxes, fontsize=7.8, color=INK2, va="top")
ax.set_xlabel("overlap of the random region with the true answer span")
ax.set_ylabel("MW effect on\nanswering correctly")
ax.set_title("E   The cost scales with how much of the answer\n     the lapse covered", loc="left")

# ---------------------------------------------------------------- F neural bounded null
ax = fig.add_subplot(gs[2, 1])
rois = [("occipital N1", "frp_occ_N1"), ("occipital P2", "frp_occ_P2"),
        ("central N400", "frp_cp_N400"), ("frontal late", "frp_front_late")]
ys = np.arange(len(rois))[::-1]
ax.axvline(0, color=MUTED, lw=0.9, zorder=1)
ax.grid(axis="x", **GRID)
for y, (lab, k) in zip(ys, rois):
    v = NEU["importance_main"][k]["pooled_within_lemma"]
    ax.add_patch(plt.Rectangle((-v["mde_80_uv"], y - 0.3), 2 * v["mde_80_uv"], 0.6,
                               color="#f2f1ee", ec=MUTED, lw=0.6, zorder=2))
    ax.plot([v["ci"][0], v["ci"][1]], [y, y], color=VIOLET, lw=2.2, solid_capstyle="round", zorder=3)
    ax.plot([v["beta"]], [y], "o", ms=7, color=VIOLET, mec="white", mew=1.4, zorder=4)
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rois], fontsize=8.2)
ax.set_xlabel("µV per SD of importance   (shaded = detectable effect at 80% power)")
ax.set_xlim(-0.034, 0.048)
ax.set_ylim(-0.75, 3.6)
ax.text(0.995, 0.995, "no importance-evoked FRP effect above ±0.02 µV —\nsmaller than half the N400 surprisal effect\nmeasured in these same epochs",
        transform=ax.transAxes, fontsize=7.3, color=INK2, ha="right", va="top")
ax.set_title("F   Nothing in the evoked response, with the bound\n     that makes that informative", loc="left")

fig.suptitle("Semantic importance, mind-wandering and comprehension in ROAMM  —  "
             "44 readers, 50 items, 2,200 trials",
             fontsize=11.5, fontweight="bold", x=0.085, ha="left", y=0.978)
fig.savefig(FIG / "localisation_synthesis.png", dpi=220, bbox_inches="tight")
fig.savefig(FIG / "localisation_synthesis.pdf", bbox_inches="tight")
print("wrote", FIG / "localisation_synthesis.png")
