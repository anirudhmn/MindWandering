#!/usr/bin/env python3
"""Reference-invariant full-scalp tests of the existing overlap-corrected rERP."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ITER = ROOT / "roamm" / "topography"
ART = ITER / "artifacts"
RES = ITER / "results"
FIG = ITER / "figures"
COUP = ROOT / "roamm" / "artifacts" / "coupling"
RNG = np.random.default_rng(6303)
N_PERM = 5000


def center_map(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=-1, keepdims=True)


def gfp(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(center_map(x) ** 2, axis=-1))


def normalize_maps(x: np.ndarray) -> np.ndarray:
    z = center_map(x)
    return z / (gfp(z)[..., None] + 1e-12)


def gmd(a: np.ndarray, b: np.ndarray) -> float:
    """Global map dissimilarity of two channel vectors."""
    aa = normalize_maps(np.asarray(a)[None])[0]
    bb = normalize_maps(np.asarray(b)[None])[0]
    # Polarity is retained: opposite maps are physiologically different fields.
    return float(np.sqrt(np.mean((aa - bb) ** 2)))


def field_test(maps: np.ndarray, signs: np.ndarray) -> dict:
    """Test GFP of the group-average field against subject sign flips."""
    obs = float(gfp(maps.mean(axis=0)))
    null = gfp(np.einsum("ps,sc->pc", signs, maps) / maps.shape[0])
    return {
        "group_mean_gfp_uV": obs,
        "p_signflip": float((1 + (null >= obs).sum()) / (len(null) + 1)),
        "null_mean": float(null.mean()),
        "null_95": [float(x) for x in np.percentile(null, [2.5, 97.5])],
    }


def map_compare(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired randomization TANOVA on window-mean subject maps."""
    an = normalize_maps(a)
    bn = normalize_maps(b)
    obs = gmd(an.mean(0), bn.mean(0))
    null = np.empty(N_PERM)
    for p in range(N_PERM):
        swap = RNG.random(len(a)) < .5
        ap = an.copy()
        bp = bn.copy()
        ap[swap], bp[swap] = bn[swap], an[swap]
        null[p] = gmd(ap.mean(0), bp.mean(0))
    spatial_r = float(np.corrcoef(center_map(a.mean(0)), center_map(b.mean(0)))[0, 1])
    boot_r = []
    for _ in range(5000):
        ix = RNG.integers(0, len(a), len(a))
        boot_r.append(np.corrcoef(
            center_map(a[ix].mean(0)), center_map(b[ix].mean(0))
        )[0, 1])
    return {
        "gmd": obs,
        "p_paired_tanova": float((1 + (null >= obs).sum()) / (N_PERM + 1)),
        "group_map_spatial_r": spatial_r,
        "spatial_r_boot_ci95": [float(x) for x in np.percentile(boot_r, [2.5, 97.5])],
    }


def crossfit_proportional_residual(
    effect: np.ndarray, base: np.ndarray, signs: np.ndarray
) -> dict:
    """Does the effect contain a field beyond a signed rescaling of the base map?

    The proportional coefficient for each held-out reader is estimated from the
    other readers' group maps, preventing the held-out map from fitting itself.
    """
    residual = np.empty_like(effect)
    slopes = np.empty(len(effect))
    for s in range(len(effect)):
        keep = np.arange(len(effect)) != s
        e = center_map(effect[keep].mean(axis=0))
        b = center_map(base[keep].mean(axis=0))
        slope = float(np.dot(e, b) / (np.dot(b, b) + 1e-12))
        slopes[s] = slope
        residual[s] = effect[s] - slope * base[s]
    boot_scale = []
    boot_r = []
    for _ in range(10000):
        ix = RNG.integers(0, len(effect), len(effect))
        e = center_map(effect[ix].mean(axis=0))
        b = center_map(base[ix].mean(axis=0))
        boot_scale.append(float(np.dot(e, b) / (np.dot(b, b) + 1e-12)))
        boot_r.append(float(np.corrcoef(e, b)[0, 1]))
    test = field_test(residual, signs)
    test.update({
        "loo_scale_mean": float(slopes.mean()),
        "loo_scale_range": [float(slopes.min()), float(slopes.max())],
        "group_map_variance_explained_r2": float(
            np.corrcoef(center_map(effect.mean(0)), center_map(base.mean(0)))[0, 1] ** 2
        ),
        "absolute_spatial_r": float(abs(
            np.corrcoef(center_map(effect.mean(0)), center_map(base.mean(0)))[0, 1]
        )),
        "group_scale_boot_ci95": [
            float(x) for x in np.percentile(boot_scale, [2.5, 97.5])
        ],
        "remaining_response_ratio_mw_over_on_task": float(1 + np.mean(slopes)),
        "attenuation_percent": float(-100 * np.mean(slopes)),
        "attenuation_percent_boot_ci95": [
            float(x) for x in -100 * np.percentile(boot_scale, [97.5, 2.5])
        ],
        "spatial_r_boot_ci95": [
            float(x) for x in np.percentile(boot_r, [2.5, 97.5])
        ],
    })
    return test


def clusters_above(x: np.ndarray, threshold: np.ndarray) -> list[tuple[int, int, float]]:
    above = x > threshold
    out = []
    i = 0
    while i < len(x):
        if not above[i]:
            i += 1
            continue
        j = i
        mass = 0.0
        while j < len(x) and above[j]:
            mass += x[j] - threshold[j]
            j += 1
        out.append((i, j, float(mass)))
        i = j
    return out


def family_field_clusters(kernels: dict[str, np.ndarray], time: np.ndarray) -> dict:
    """Max-cluster sign-flip test across kernels and the entire epoch."""
    names = list(kernels)
    nsub = next(iter(kernels.values())).shape[0]
    obs = np.stack([gfp(k.mean(axis=0)) for k in kernels.values()])
    null = np.empty((N_PERM, len(names), len(time)), np.float32)
    for p in range(N_PERM):
        sign = RNG.choice([-1.0, 1.0], nsub)
        for ki, k in enumerate(kernels.values()):
            null[p, ki] = gfp(np.einsum("s,stc->tc", sign, k) / nsub)
    threshold = np.percentile(null, 95, axis=0)
    null_max = np.zeros(N_PERM)
    for p in range(N_PERM):
        best = 0.0
        for ki in range(len(names)):
            cc = clusters_above(null[p, ki], threshold[ki])
            best = max(best, max((c[2] for c in cc), default=0.0))
        null_max[p] = best
    report = {}
    for ki, name in enumerate(names):
        rows = []
        for i, j, mass in clusters_above(obs[ki], threshold[ki]):
            rows.append({
                "start_ms": float(time[i]),
                "end_ms": float(time[j - 1]),
                "mass": mass,
                "p_cluster_fwer_family": float(
                    (1 + (null_max >= mass).sum()) / (N_PERM + 1)
                ),
            })
        report[name] = {
            "clusters": rows,
            "significant": [r for r in rows if r["p_cluster_fwer_family"] < .05],
        }
    return report


def holm_adjust(rows: list[dict], key: str = "p_paired_tanova") -> None:
    p = np.array([r[key] for r in rows])
    order = np.argsort(p)
    running = 0.0
    out = np.zeros(len(p))
    for rank, ix in enumerate(order):
        running = max(running, (len(p) - rank) * p[ix])
        out[ix] = min(running, 1.0)
    for row, adj in zip(rows, out):
        row["p_holm_family_B"] = float(adj)


def montage_info(channels: list[str]) -> mne.Info:
    names = ["AFz" if c == "Afz" else c for c in channels]
    info = mne.create_info(names, 256.0, "eeg")
    info.set_montage("biosemi64", on_missing="warn")
    return info


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    betas = np.load(COUP / "rerp_betas.npy")
    meta = json.loads((COUP / "rerp_meta.json").read_text())
    time = np.asarray(meta["lags_ms"], float)
    pidx = {p: i for i, p in enumerate(meta["pred_names"])}
    windows = {
        "lexical": (time >= 150) & (time <= 290),
        "semantic": (time >= 300) & (time <= 450),
    }
    maps = {
        "intercept_lexical": betas[:, pidx["intercept"]][:, windows["lexical"]].mean(1),
        "zipf_lexical": betas[:, pidx["zipf"]][:, windows["lexical"]].mean(1),
        "mw_lexical": betas[:, pidx["mw"]][:, windows["lexical"]].mean(1),
        "zipf_mw_lexical": betas[:, pidx["zipf:mw"]][:, windows["lexical"]].mean(1),
        "surprisal_semantic": betas[:, pidx["surprisal"]][:, windows["semantic"]].mean(1),
        "surprisal_mw_semantic": betas[:, pidx["surprisal:mw"]][:, windows["semantic"]].mean(1),
    }
    signs = RNG.choice([-1.0, 1.0], size=(N_PERM, betas.shape[0]))
    field = {
        name: field_test(value, signs)
        for name, value in maps.items()
        if name != "intercept_lexical"
    }
    field["intercept_lexical"] = field_test(maps["intercept_lexical"], signs)
    comparisons = {
        "frequency_vs_surprisal": map_compare(
            maps["zipf_lexical"], maps["surprisal_semantic"]
        ),
        "mw_vs_on_task_fixation": map_compare(
            maps["mw_lexical"], maps["intercept_lexical"]
        ),
        "frequency_interaction_vs_frequency": map_compare(
            maps["zipf_mw_lexical"], maps["zipf_lexical"]
        ),
        "surprisal_interaction_vs_surprisal": map_compare(
            maps["surprisal_mw_semantic"], maps["surprisal_semantic"]
        ),
    }
    holm_adjust(list(comparisons.values()))
    proportional = crossfit_proportional_residual(
        maps["mw_lexical"], maps["intercept_lexical"], signs
    )

    kernels = {
        "zipf": betas[:, pidx["zipf"]],
        "surprisal": betas[:, pidx["surprisal"]],
        "mw": betas[:, pidx["mw"]],
        "zipf_mw": betas[:, pidx["zipf:mw"]],
        "surprisal_mw": betas[:, pidx["surprisal:mw"]],
    }
    cluster = family_field_clusters(kernels, time)

    b2_field = field["mw_lexical"]
    b2_topo = comparisons["mw_vs_on_task_fixation"]
    if b2_field["p_signflip"] < .05 and proportional["p_signflip"] >= .05:
        verdict = (
            "nonzero MW field is explained by a signed rescaling of the ordinary "
            "fixation field: subtractive gain/attenuation, not a new configuration"
        )
    elif proportional["p_signflip"] < .05:
        verdict = (
            "MW contains a reliable residual field beyond proportional gain: mixed "
            "gain and configuration change"
        )
    else:
        verdict = "earlier ROI MW effect does not establish a reliable full-scalp field"

    report = {
        "n_subjects": int(betas.shape[0]),
        "n_permutations": N_PERM,
        "window_field_tests": field,
        "window_tanova": comparisons,
        "mw_vs_fixation_crossfit_proportional_residual": proportional,
        "time_resolved_field_clusters": cluster,
        "B2_verdict": verdict,
        "interaction_topology_gate": {
            "frequency_interaction_field_reliable": bool(
                field["zipf_mw_lexical"]["p_signflip"] < .05
            ),
            "surprisal_interaction_field_reliable": bool(
                field["surprisal_mw_semantic"]["p_signflip"] < .05
            ),
            "rule": (
                "Normalized topology of an interaction is not interpreted when the "
                "interaction field itself fails the sign-flip field test."
            ),
        },
        "interpretation_limit": (
            "Scalp-field configuration is reference-invariant but is not equivalent "
            "to anatomical source localization."
        ),
    }
    (RES / "topographic_rerp_report.json").write_text(json.dumps(report, indent=2) + "\n")

    info = montage_info(meta["channels"])
    plot_names = [
        "zipf_lexical", "surprisal_semantic", "mw_lexical",
        "zipf_mw_lexical", "surprisal_mw_semantic",
    ]
    titles = [
        "Frequency\n150–290 ms", "Surprisal\n300–450 ms",
        "MW additive\n150–290 ms", "Frequency × MW\n150–290 ms",
        "Surprisal × MW\n300–450 ms",
    ]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), constrained_layout=True)
    vmax = max(float(np.abs(maps[n].mean(0)).max()) for n in plot_names)
    for ax, name, title in zip(axes, plot_names, titles):
        im, _ = mne.viz.plot_topomap(
            maps[name].mean(0), info, axes=ax, show=False, cmap="RdBu_r",
            vlim=(-vmax, vmax), contours=4,
        )
        ax.set_title(title, fontsize=10)
    cbar = fig.colorbar(im, ax=axes, shrink=.72, pad=.02)
    cbar.set_label("rERP coefficient (µV)")
    fig.suptitle("Full-scalp overlap-corrected reading fields", fontsize=13)
    fig.savefig(FIG / "topographic_rerp_figure.png", dpi=220)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
