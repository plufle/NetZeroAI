"""Generates the chart images used in reports/Final_Report.md.

Not part of the netzero pipeline itself -- a reporting utility that reads the
pipeline's artifacts. Run from the project root after the pipeline:
    python reports/build_report_charts.py

Colors follow the dashboard's validated palette (app.py): a one-hue ordinal
ramp for tiers, max three categorical hues per chart, small multiples instead
of more hues, and no dual-axis charts.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import constants as C  # noqa: E402
from src.components.index_validation import global_median_imputation_composite  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets" / "final"
ASSETS.mkdir(parents=True, exist_ok=True)

TIER_COLORS = dict(zip(C.TIER_LABELS, ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]))
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, MUTED, GRID, CONTEXT = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#d6d5ce"
DIM_LABELS = {d: d.replace("_", " ").title() for d in C.DIMENSIONS}
DIM_COLS = [f"{d}_score" for d in C.DIMENSIONS]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": "#c3c2b7",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

df = pd.read_csv(C.FINAL_TABLE_PATH)
with open(C.MANIFEST_PATH) as f:
    manifest = json.load(f)


def save(fig, name):
    fig.savefig(ASSETS / name)
    plt.close(fig)


def tier_legend(ax, **kw):
    handles = [plt.Rectangle((0, 0), 1, 1, color=TIER_COLORS[t]) for t in C.TIER_LABELS]
    ax.legend(handles, C.TIER_LABELS, title="Tier", **kw)


# 1. Indicator distributions (why percentile normalization)
fig, axes = plt.subplots(1, 5, figsize=(17, 3.4))
for ax, col in zip(axes, C.INDICATOR_DIRECTIONS):
    ax.hist(df[col], bins=30, color=SERIES[0], edgecolor="white", linewidth=0.6)
    ax.axvline(df[col].median(), color=INK_2, linewidth=1, linestyle=":")
    ax.set_title(col, fontsize=10)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Countries")
fig.suptitle("Raw indicator distributions, 194 countries (dotted line = median)", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save(fig, "indicator_distributions.png")

# 2. Top/bottom 15
ranked = df.sort_values("netzero_index_score", ascending=False)
tb = pd.concat([ranked.head(15), ranked.tail(15)]).sort_values("netzero_index_score")
fig, ax = plt.subplots(figsize=(8, 9))
labels = [f"{c} *" if imp else c for c, imp in zip(tb["country"], tb[[f"{i}_imputed" for i in C.INDICATOR_DIRECTIONS]].any(axis=1))]
ax.barh(labels, tb["netzero_index_score"], color=[TIER_COLORS[t] for t in tb["tier"]], height=0.7)
ax.set_xlabel("Composite index score (0-100)")
ax.set_xlim(0, 100)
ax.grid(axis="y", visible=False)
ax.set_title("Top 15 and bottom 15 countries, 2022 (* = at least one imputed indicator)")
tier_legend(ax, loc="lower right")
save(fig, "top_bottom_15.png")

# 3. Sensitivity scenarios: two measures -> two panels on a shared scenario axis
sens = manifest["sensitivity"]
scen = pd.DataFrame(sens["weight_scenarios"] + sens["method_alternatives"]).iloc[::-1]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
a1.barh(scen["scenario"], scen["spearman_vs_baseline"], color=SERIES[0], height=0.6)
a1.set_xlim(0.7, 1.0)
a1.set_title("Rank correlation with baseline (Spearman)")
a2.barh(scen["scenario"], scen["tier_change_pct"], color=SERIES[0], height=0.6)
a2.set_title("Countries changing tier (%)")
for ax, col, fmt in [(a1, "spearman_vs_baseline", "{:.2f}"), (a2, "tier_change_pct", "{:.0f}%")]:
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(scen[col]):
        ax.text(v, y, " " + fmt.format(v), va="center", fontsize=8.5, color=INK_2)
fig.suptitle("Sensitivity of the index to weighting and method choices", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save(fig, "sensitivity_scenarios.png")

# 4. Weight sensitivity of the top 25 ranks
mc = sens["monte_carlo"]
spread = f"±{mc['weight_spread']:.0%}"
top = df.sort_values("rank").head(25).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 7.5))
ax.hlines(top["country"], top["weight_rank_p05"], top["weight_rank_p95"], color=CONTEXT, linewidth=4)
ax.scatter(top["rank"], top["country"], c=[TIER_COLORS[t] for t in top["tier"]], s=45, zorder=3, edgecolor="white", linewidth=1)
ax.set_xlabel("Rank (1 = best)")
ax.set_xlim(0, top["weight_rank_p95"].max() + 5)
ax.grid(axis="y", visible=False)
ax.set_title(f"Top 25: baseline rank (dot) and 5th-95th percentile rank (bar)\n"
             f"across {mc['n_draws']} random weightings, each dimension {spread} (not a confidence interval)")
save(fig, "rank_uncertainty_top25.png")

# 5. Tier stability by tier (grouped, shared bins)
bins = np.linspace(0, 1.0001, 11)
counts = {t: np.histogram(df.loc[df["tier"] == t, "weight_tier_stability"], bins=bins)[0] for t in C.TIER_LABELS}
fig, ax = plt.subplots(figsize=(9, 4))
width = 0.1 / 5
for i, t in enumerate(C.TIER_LABELS):
    ax.bar(bins[:-1] + width * (i + 0.5), counts[t], width=width * 0.9, color=TIER_COLORS[t], align="edge", label=t)
ax.set_xlim(0.3, 1.01)
ax.set_xlabel(f"Share of random weightings ({spread} per dimension) in which the country keeps its baseline tier")
ax.set_ylabel("Countries")
ax.grid(axis="x", visible=False)
ax.set_title("Tier stability: outer tiers are robust, middle tiers flip often")
ax.legend(title="Tier")
save(fig, "tier_stability.png")

# 5b. Imputation sensitivity: rank range when the imputed value spans its peers' observed range
imp = df[df["impute_rank_best"].notna()].sort_values("rank", ascending=False)
fig, ax = plt.subplots(figsize=(8.5, 8.5))
ax.hlines(imp["country"], imp["impute_rank_best"], imp["impute_rank_worst"], color=CONTEXT, linewidth=4)
ax.scatter(imp["rank"], imp["country"], c=[TIER_COLORS[t] for t in imp["tier"]], s=40, zorder=3, edgecolor="white", linewidth=1)
ax.set_xlabel("Rank (1 = best)")
ax.set_xlim(0, len(df) + 2)
ax.grid(axis="y", visible=False)
ax.set_title("Countries with an imputed value: baseline rank (dot) and best-worst rank (bar)\n"
             "when the imputed value spans its imputation peer group's observed range")
tier_legend(ax, loc="lower left")
save(fig, "imputation_sensitivity.png")

# 6. Tier x segment agreement + segment profiles
agree = manifest["tier_cluster_agreement"]
crosstab = pd.crosstab(df["tier"], df["cluster_label"]).reindex(C.TIER_LABELS)
profiles = pd.DataFrame(agree["cluster_profiles"]).set_index("cluster_label")[DIM_COLS]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1, 1.25]})
for ax, data, title, vmax, fmt in [
    (a1, crosstab, f"Tier x KMeans segment (ARI {agree['adjusted_rand_index']:.2f})", crosstab.values.max(), "{:d}"),
    (a2, profiles, "Segment profiles: mean dimension score", 100, "{:.0f}"),
]:
    im = ax.imshow(data.values, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(data.shape[1]))
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(data.index)
    ax.grid(False)
    for (r, c), v in np.ndenumerate(data.values):
        ax.text(c, r, fmt.format(int(v) if fmt == "{:d}" else v), ha="center", va="center",
                color="white" if v > vmax * 0.55 else INK, fontsize=9)
    ax.set_title(title)
a1.set_xticklabels(crosstab.columns, rotation=20)
a2.set_xticklabels([DIM_LABELS[d] for d in C.DIMENSIONS], rotation=20)
fig.tight_layout()
save(fig, "tier_cluster_agreement.png")

# 7. Segments as PCA small multiples
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
var = manifest["pca_explained_variance"]
for ax, (segment, members) in zip(axes.flat, df.groupby("cluster_label")):
    ax.scatter(df["pca_1"], df["pca_2"], s=14, color=CONTEXT)
    ax.scatter(members["pca_1"], members["pca_2"], s=24, color=SERIES[0], edgecolor="white", linewidth=0.6)
    p = profiles.loc[segment]
    strongest, weakest = DIM_LABELS[p.idxmax().replace("_score", "")], DIM_LABELS[p.idxmin().replace("_score", "")]
    ax.set_title(f"{segment} ({len(members)} countries)\nstrongest: {strongest}\nweakest: {weakest}", fontsize=9.5)
for ax in axes[1]:
    ax.set_xlabel(f"PC1 ({var[0]:.0%} of variance)")
for ax in axes[:, 0]:
    ax.set_ylabel(f"PC2 ({var[1]:.0%})")
fig.suptitle("KMeans segments (k=4) in PCA space - each panel highlights one segment", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save(fig, "segments_pca.png")

# 8. Choosing k: three measures, three panels (no dual axis)
k_df = pd.DataFrame(manifest["elbow_silhouette_table"]).merge(pd.DataFrame(manifest["cluster_stability"]["by_k"]), on="k")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
for ax, col, title in zip(axes, ["inertia", "silhouette", "mean_ari"], ["Inertia (elbow)", "Silhouette", "Bootstrap stability (mean ARI)"]):
    ax.plot(k_df["k"], k_df[col], color=SERIES[0], linewidth=2, marker="o", markersize=6)
    ax.axvline(C.N_CLUSTERS_DEFAULT, color=MUTED, linewidth=1, linestyle=":")
    ax.set_title(title)
    ax.set_xlabel("k")
fig.suptitle(f"Choosing k (dotted line: chosen k={C.N_CLUSTERS_DEFAULT})", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save(fig, "k_selection.png")

# 9. Case studies: dimension breakdown for three countries (3 categorical slots)
cases = ["Qatar", "Macao", "Kuwait"]
fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(C.DIMENSIONS) + 1)
for i, country in enumerate(cases):
    row = df[df["country"] == country].iloc[0]
    vals = [row[c] for c in DIM_COLS] + [row["netzero_index_score"]]
    ax.bar(x + (i - 1) * 0.26, vals, width=0.24, color=SERIES[i], label=f"{country} ({row['tier']}, rank {row['rank']})")
ax.axhline(50, color=MUTED, linewidth=1, linestyle=":")
ax.set_xticks(x)
ax.set_xticklabels([DIM_LABELS[d].replace(" ", "\n", 1) for d in C.DIMENSIONS] + ["Composite\nindex"])
ax.set_ylim(0, 100)
ax.set_ylabel("Score (0-100; dotted = cohort median)")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
ax.set_title("Case studies: strong dimensions offsetting weak ones under equal weighting")
save(fig, "case_studies.png")

# 10. Effect of regional vs global-median imputation on the imputed countries
alt = df.assign(global_median_score=global_median_imputation_composite(df))
imputed = alt[alt["gdp_per_capita_imputed"]].copy()
imputed["delta"] = imputed["netzero_index_score"] - imputed["global_median_score"]
imputed = imputed.sort_values("delta")
fig, ax = plt.subplots(figsize=(8.5, 8.5))
ax.hlines(imputed["country"], imputed["global_median_score"], imputed["netzero_index_score"], color=CONTEXT, linewidth=2)
ax.scatter(imputed["global_median_score"], imputed["country"], color=SERIES[1], s=30, zorder=3, label="Global-median imputation (DA1)")
ax.scatter(imputed["netzero_index_score"], imputed["country"], color=SERIES[0], s=30, zorder=3, label="Regional imputation (final)")
ax.set_xlabel("Composite index score")
ax.grid(axis="y", visible=False)
ax.legend(loc="upper left")
ax.set_title("Countries with imputed GDP: score under global vs regional imputation")
save(fig, "imputation_effect.png")

print("Saved charts to", ASSETS)
for p in sorted(ASSETS.glob("*.png")):
    print(" -", p.name)
