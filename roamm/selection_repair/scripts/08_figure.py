#!/usr/bin/env python3
"""Synthesis figure for the selection and repair analysis."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ART, RES, FIG

g0 = json.load(open(RES / "g0_skip_audit.json"))
g0b = json.load(open(RES / "g0b_traversal.json"))
g1 = json.load(open(RES / "g1_selection.json"))
g4 = json.load(open(RES / "g4_repair.json"))
g56 = json.load(open(RES / "g5_g6_duration.json"))
g7 = json.load(open(RES / "g7_zuco_session.json"))
D = pd.read_csv(ART / "somersD_primary.csv")

ON, MW, ACC = "#4C72B0", "#C44E52", "#55A868"

# ---- subject-level paired percentage changes for panel C ----
from common import COUP
_f = pd.read_parquet(COUP / "reading_fixations.parquet").sort_values(["subject", "run", "tStart"])
_g = _f.groupby(["subject", "run"], sort=False)
_f["refix"] = (_g["pos"].shift(-1) == _f["pos"]).astype(float)
_f.loc[_g["pos"].shift(-1).isna(), "refix"] = np.nan
_W = (_f[_f.is_firstpass == 1].groupby(["subject", "word_key"])
      .agg(GD=("fix_dur", "sum"), is_mw=("is_mw", "first")).reset_index())
_T = pd.read_parquet(ART / "words_traversal.parquet")
_T = _T[((_T.skipped == 0) | (_T.gap <= 4)) & _T.state_agree]
_C = pd.read_parquet(ART / "corrective_regressions.parquet")


def pct_change(df, col, log=False):
    """Subject-level paired % change, MW vs on-task."""
    p = df.groupby(["subject", "is_mw"])[col].mean().unstack().dropna()
    if log:
        lg = df.assign(_l=np.log(df[col].clip(lower=1))).groupby(["subject", "is_mw"])._l.mean().unstack().dropna()
        v = np.exp(lg[1] - lg[0]) - 1
    else:
        v = p[1] / p[0] - 1
    return float(v.mean() * 100)


PANEL_C = [("gaze duration", pct_change(_W, "GD", log=True), 1.7e-6),
           ("regressions", pct_change(_f.dropna(subset=["regression_out"]), "regression_out"),
            g4["regression_rate"]["diff"]["p"]),
           ("refixations", pct_change(_f.dropna(subset=["refix"]), "refix"),
            g4["refixation_rate"]["diff"]["p"]),
           ("corrective\nreturns", pct_change(_C, "corrective"), g4["corrective_rate"]["diff"]["p"]),
           ("skipping", pct_change(_T, "skipped"), g0b["skip_rate_by_G"]["4"]["diff"]["p"])]

fig = plt.figure(figsize=(15.5, 10.2))
gs = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.30,
                      left=0.06, right=0.985, top=0.855, bottom=0.075)

# ---- A: skip measurement ----
ax = fig.add_subplot(gs[0, 0])
raw = [g0["skip_rate"]["on_task"], g0["skip_rate"]["mw"]]
cor = [g0b["skip_rate_by_G"]["4"]["on_task"], g0b["skip_rate_by_G"]["4"]["mw"]]
x = np.arange(2)
ax.bar(x - 0.19, raw, 0.36, color=[ON, MW], alpha=.45, edgecolor="k", lw=.6, label="raw (any unfixated word)")
ax.bar(x + 0.19, cor, 0.36, color=[ON, MW], edgecolor="k", lw=.6, label="scan-path corrected")
for xi, (a, b) in enumerate(zip(raw, cor)):
    ax.text(xi - 0.19, a + .012, f"{a:.3f}", ha="center", fontsize=8.5)
    ax.text(xi + 0.19, b + .012, f"{b:.3f}", ha="center", fontsize=8.5, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(["on-task", "mind-wandering"])
ax.set_ylabel("word skipping rate"); ax.set_ylim(0, .78)
ax.set_title("A  The 'more skipping' marker is a\nmeasurement artefact", fontsize=10.5, loc="left")
ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.86))
ax.annotate("raw: +.168, p=7e-11", (0.5, .735), ha="center", fontsize=8, color="grey")
ax.annotate("corrected: −.042, p=2e-6", (0.5, .695), ha="center", fontsize=8,
            color="k", fontweight="bold")

# ---- B: G1 selectivity ----
ax = fig.add_subplot(gs[0, 1])
props = ["zipf", "length", "surprisal"]
for i, p in enumerate(props):
    on, mw = D[f"{p}_on"].abs(), D[f"{p}_mw"].abs()
    for a, b in zip(on, mw):
        ax.plot([i - .16, i + .16], [a, b], color="grey", lw=.5, alpha=.35)
    ax.plot([i - .16] * len(on), on, "o", color=ON, ms=3.5, alpha=.75)
    ax.plot([i + .16] * len(mw), mw, "o", color=MW, ms=3.5, alpha=.75)
    ax.plot([i - .16, i + .16], [on.mean(), mw.mean()], "-", color="k", lw=2.2)
    r = g1["G1_primary"][p]
    ax.text(i, 0.80, f"{r['retention_pct']:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.text(i, 0.755, f"p={r['p']:.2f}", ha="center", fontsize=7.5, color="grey")
ax.set_xticks(range(3)); ax.set_xticklabels(props)
ax.set_ylabel("|Somers' D|   property → skip"); ax.set_ylim(0, .87)
ax.set_title("B  Lexical control of *which* words are\nskipped is fully preserved (equiv. ±10%)",
             fontsize=10.5, loc="left")
ax.plot([], [], "o", color=ON, label="on-task"); ax.plot([], [], "o", color=MW, label="MW")
ax.legend(fontsize=7.5, frameon=False, loc="lower left")

# ---- C: repair channel ----
ax = fig.add_subplot(gs[0, 2])
names = [i[0] for i in PANEL_C]; vals = [i[1] for i in PANEL_C]
cols = [MW if v > 0 else ACC for v in vals]
ax.barh(range(len(PANEL_C)), vals, color=cols, edgecolor="k", lw=.6, height=.62)
ax.axvline(0, color="k", lw=.8)
for i, (nm, v, p) in enumerate(PANEL_C):
    ax.text(v + (1.2 if v > 0 else -1.2), i, f"{v:+.1f}%   p={p:.2g}", va="center",
            ha="left" if v > 0 else "right", fontsize=8)
ax.set_yticks(range(len(PANEL_C))); ax.set_yticklabels(names, fontsize=8.5)
ax.set_xlabel("change during MW (%, subject-level paired)"); ax.set_xlim(-46, 46)
ax.invert_yaxis()
ax.set_title("C  MW makes reading *more* effortful,\nnot more cursory", fontsize=10.5, loc="left")

# ---- D: duration identification ----
ax = fig.add_subplot(gs[1, 0])
aw = g56["G5_across_word_signed_retention"]; fe = g56["G5"]["logGD"]
labels, est, lo, hi = [], [], [], []
for p in ["zipf", "surprisal", "length"]:
    base = abs(aw[p]["beta_on"])
    labels.append(f"{p}\nacross words")
    est.append((aw[p]["retention_pct"] - 100))
    se = 100 * abs(aw[p]["beta_mw"] - aw[p]["beta_on"]) / base / max(abs(aw[p]["t"]), 1e-9)
    lo.append(est[-1] - 1.96 * se); hi.append(est[-1] + 1.96 * se)
    v = fe[f"mw_x_{p}"]
    sgn = -1 if aw[p]["beta_on"] < 0 else 1
    labels.append(f"{p}\nwithin token")
    est.append(sgn * v["beta"] / base * 100)
    s = sgn * v["se"] / base * 100
    lo.append(est[-1] - 1.96 * abs(s)); hi.append(est[-1] + 1.96 * abs(s))
y = np.arange(len(est))
cols = [MW if "across" in l else "#333333" for l in labels]
ax.errorbar(est, y, xerr=[np.array(est) - np.array(lo), np.array(hi) - np.array(est)],
            fmt="o", ms=5, color="k", ecolor="grey", lw=1.2, zorder=3)
for i, c in enumerate(cols):
    ax.plot(est[i], y[i], "o", ms=6.5, color=c, zorder=4)
ax.axvline(0, color="k", lw=.8)
ax.axvspan(-20, 20, color="grey", alpha=.12)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
ax.invert_yaxis(); ax.set_xlabel("change in coupling during MW (% of on-task)")
ax.set_title("D  The apparent 'enhancement' halves and\nloses significance within token instance",
             fontsize=10.5, loc="left")

# ---- E: scale audit ----
ax = fig.add_subplot(gs[1, 1])
keys = [("ROAMM_MW_zipf", "ROAMM MW\nzipf"), ("ROAMM_MW_surprisal", "ROAMM MW\nsurprisal"),
        ("ZuCo_TSR_zipf", "ZuCo skim\nzipf"), ("ZuCo_TSR_surprisal", "ZuCo skim\nsurprisal")]
x = np.arange(len(keys))
logv = [g56["G6"][k]["log_retention_pct"] for k, _ in keys]
rawv = [g56["G6"][k]["raw_retention_pct"] for k, _ in keys]
predv = [g56["G6"][k]["log_retention_predicted_by_additive_shift_pct"] for k, _ in keys]
ax.bar(x - .2, logv, .4, color="#8C8C8C", edgecolor="k", lw=.6, label="log scale (as reported)")
ax.bar(x + .2, rawv, .4, color="#2B4E72", edgecolor="k", lw=.6, label="raw ms scale")
ax.plot(x, predv, "v", color="crimson", ms=8, label="log value if coupling were unchanged")
ax.axhline(100, color="k", lw=.8, ls="--")
ax.set_xticks(x); ax.set_xticklabels([l for _, l in keys], fontsize=7.5)
ax.set_ylabel("coupling retention (%)")
ax.set_title("E  The log scale understates both effects", fontsize=10.5, loc="left")
ax.legend(fontsize=7, frameon=False, loc="upper right")

# ---- F: ZuCo session decomposition ----
ax = fig.add_subplot(gs[1, 2])
conds = ["NR", "SR1", "SR2", "TSR"]
lab = ["NR\n(deep, wiki)\nsession 1", "SR-h1\n(deep, movie)\nsession 1",
       "SR-h2\n(deep, movie)\nsession 2", "TSR\n(shallow, wiki)\nsession 2"]
v = [abs(g7["slopes"]["zipf"][c]["mean"]) for c in conds]
cc = [ON, ON, "#7BA7D4", MW]
ax.bar(range(4), v, color=cc, edgecolor="k", lw=.6, width=.66)
for i, val in enumerate(v):
    ax.text(i, val + .004, f"{val:.3f}", ha="center", fontsize=8.5)
ax.set_xticks(range(4)); ax.set_xticklabels(lab, fontsize=7)
ax.set_ylabel("|zipf → log gaze duration|"); ax.set_ylim(0, .175)
c = g7["contrasts"]["zipf"]
ax.annotate("", xy=(1, .152), xytext=(2, .152), arrowprops=dict(arrowstyle="<->", color="dimgrey"))
ax.text(1.5, .156, f"SESSION  {c['session_effect_SR1_to_SR2']['retention_pct']:.0f}%  p=.020",
        ha="center", fontsize=7.5, color="dimgrey")
ax.annotate("", xy=(2, .128), xytext=(3, .128), arrowprops=dict(arrowstyle="<->", color="k"))
ax.text(2.5, .132, f"TASK  {c['task_within_session2_SR2_to_TSR']['retention_pct']:.0f}%  p=.002",
        ha="center", fontsize=7.5, fontweight="bold")
ax.set_title("F  The ZuCo goal effect survives a\nwithin-session control (but is smaller)",
             fontsize=10.5, loc="left")

fig.suptitle("Mind-wandering spares every level of word-to-eye control; what changes is effort, not selectivity",
             fontsize=14, x=0.06, ha="left", y=0.975, fontweight="bold")
fig.text(0.06, 0.945, "ROAMM n=44 readers / 404,557 fixations  •  ZuCo n=12 readers  •  "
                      "subject-level inference, 10,000-sample bootstrap CIs",
         fontsize=8.5, color="dimgrey", ha="left")
out = FIG / "selection_repair_synthesis.png"
fig.savefig(out, dpi=190, facecolor="white")
print("wrote", out)
