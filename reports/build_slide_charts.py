"""Generates the chart images used in the DA1 progress-review deck.

Not part of the netzero pipeline itself -- a one-off reporting utility.
Run from the project root: python reports/build_slide_charts.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import constants as C  # noqa: E402

sns.set_theme(style="whitegrid")
ASSETS = Path(__file__).resolve().parent / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(C.FINAL_TABLE_PATH)
with open(C.MANIFEST_PATH) as f:
    manifest = json.load(f)

# 1. Why the synthetic dataset was rejected
# global_warming_dataset.csv was removed from disk since the original audit; reuse the
# correlation heatmap generated from it in that earlier session instead of regenerating.
synthetic_src = Path(__file__).resolve().parent.parent / "global_warming_dataset.csv"
cached_heatmap = Path(__file__).resolve().parent.parent / "eda_output" / "correlation_heatmap.png"
if synthetic_src.exists():
    synthetic = pd.read_csv(synthetic_src)
    num_cols = synthetic.select_dtypes(include="number").columns.drop("Year")
    corr = synthetic[num_cols].corr()
    plt.figure(figsize=(9, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-0.05, vmax=0.05, square=True, cbar_kws={"label": "correlation"})
    plt.title("Attached dataset: all pairwise correlations ~0\n(confirms synthetic random noise, not real climate data)")
    plt.tight_layout()
    plt.savefig(ASSETS / "synthetic_correlation.png", dpi=150)
    plt.close()
elif cached_heatmap.exists():
    import shutil
    shutil.copy(cached_heatmap, ASSETS / "synthetic_correlation.png")
else:
    print("WARNING: no synthetic-dataset correlation image available (source csv and cache both missing)")

# 2. Heavy-tailed distribution justifying percentile normalization
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.histplot(df["co2_per_capita"], bins=30, ax=axes[0], color="#d1495b")
axes[0].set_title("CO2 per capita (t) — heavy right tail")
axes[0].axvline(df["co2_per_capita"].median(), color="black", linestyle="--", label="median")
axes[0].legend()
sns.histplot(df["gdp_per_capita"], bins=30, ax=axes[1], color="#2a9d8f")
axes[1].set_title("GDP per capita ($) — heavy right tail")
axes[1].axvline(df["gdp_per_capita"].median(), color="black", linestyle="--", label="median")
axes[1].legend()
plt.tight_layout()
plt.savefig(ASSETS / "indicator_distributions.png", dpi=150)
plt.close()

# 3. Top/bottom 15 by composite index
ranked = df.sort_values("netzero_index_score", ascending=False).reset_index(drop=True)
top_bottom = pd.concat([ranked.head(15), ranked.tail(15)]).sort_values("netzero_index_score")
plt.figure(figsize=(9, 9))
palette = {"Leader": "#2a9d8f", "Advancing": "#8ab17d", "Emerging": "#e9c46a", "Lagging": "#e76f51"}
sns.barplot(data=top_bottom, x="netzero_index_score", y="country", hue="tier", dodge=False, palette=palette)
plt.title("Top 15 / Bottom 15 Countries — Net Zero Readiness Index (2022, preliminary)")
plt.xlabel("Composite Index Score (0-100)")
plt.ylabel("")
plt.tight_layout()
plt.savefig(ASSETS / "top_bottom_15.png", dpi=150)
plt.close()

# 4. Tier distribution
plt.figure(figsize=(6, 4.5))
order = C.TIER_LABELS
sns.countplot(data=df, x="tier", order=order, palette=[palette[t] for t in order])
plt.title(f"Tier Distribution — {len(df)} Countries (2022 snapshot)")
plt.xlabel("")
plt.ylabel("Number of countries")
plt.tight_layout()
plt.savefig(ASSETS / "tier_distribution.png", dpi=150)
plt.close()

# 5. PCA + clusters
plt.figure(figsize=(7, 6))
sns.scatterplot(data=df, x="pca_1", y="pca_2", hue="cluster_label", palette="Set2", s=45)
plt.title("PCA Projection Colored by KMeans Segment (k=4)")
plt.tight_layout()
plt.savefig(ASSETS / "pca_clusters.png", dpi=150)
plt.close()

# 6. Elbow / silhouette
k_df = pd.DataFrame(manifest["elbow_silhouette_table"])
fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax2 = ax1.twinx()
ax1.plot(k_df["k"], k_df["inertia"], "o-", color="#264653", label="Inertia (elbow)")
ax2.plot(k_df["k"], k_df["silhouette"], "s--", color="#e76f51", label="Silhouette score")
ax1.set_xlabel("k (number of clusters)")
ax1.set_ylabel("Inertia", color="#264653")
ax2.set_ylabel("Silhouette score", color="#e76f51")
plt.title("Choosing k: Elbow + Silhouette (k=4 chosen)")
fig.tight_layout()
plt.savefig(ASSETS / "elbow_silhouette.png", dpi=150)
plt.close()

# 7. Qatar case study -- interpretability check
qatar = df[df["country"] == "Qatar"].iloc[0]
dims = list(C.DIMENSIONS)
vals = [qatar[f"{d}_score"] for d in dims] + [qatar["netzero_index_score"]]
names = [d.replace("_", "\n").title() for d in dims] + ["Composite\nIndex"]
plt.figure(figsize=(9, 4.8))
colors = ["#e76f51"] * len(dims) + ["#264653"]
bars = plt.bar(names, vals, color=colors, width=0.6)
plt.xticks(fontsize=9)
plt.axhline(50, color="gray", linestyle=":", label="cohort median")
for b, v in zip(bars, vals):
    plt.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center")
plt.ylim(0, 105)
plt.title("Case Study: Qatar — high GDP + improving trend offset\npoor emissions intensity & energy mix in an equal-weighted index")
plt.ylabel("Score (0-100)")
plt.legend()
plt.tight_layout()
plt.savefig(ASSETS / "qatar_case_study.png", dpi=150)
plt.close()

print("Saved charts to", ASSETS)
for p in sorted(ASSETS.glob("*.png")):
    print(" -", p.name)
