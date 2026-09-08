import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import constants as C
from src.pipeline.train_pipeline import TrainPipeline

st.set_page_config(page_title="NetZeroAI", layout="wide")


@st.cache_data
def load_data(_mtime):
    return pd.read_csv(C.FINAL_TABLE_PATH)


def load_manifest():
    with open(C.MANIFEST_PATH) as f:
        return json.load(f)


if not C.FINAL_TABLE_PATH.exists():
    with st.spinner("Building indicator table (first run)... this takes a few seconds."):
        TrainPipeline().run()

df = load_data(C.FINAL_TABLE_PATH.stat().st_mtime)
manifest = load_manifest()

DIMENSION_COLS = [f"{d}_score" for d in C.DIMENSIONS]
INDICATOR_LABELS = {
    "co2_per_capita": ("CO2 per capita", "t"),
    "ghg_per_capita": ("GHG per capita", "t"),
    "co2pc_trend_slope": ("CO2/capita trend (2013-22)", "t/yr"),
    "renewables_share_elec": ("Renewables share (electricity)", "%"),
    "gdp_per_capita": ("GDP per capita", "$"),
}

st.title("NetZeroAI — Net Zero Readiness Assessment")

page = st.sidebar.radio(
    "Section",
    ["Overview / Ranking", "Country Detail", "PCA & Clusters", "Methodology & Limitations"],
)

if page == "Overview / Ranking":
    st.subheader(f"Ranking — {len(df)} countries, reference year {C.REFERENCE_YEAR}")

    ranked = df.sort_values("netzero_index_score", ascending=False).reset_index(drop=True)
    ranked.index += 1
    st.dataframe(
        ranked[["country", "netzero_index_score", "tier"] + DIMENSION_COLS],
        use_container_width=True,
    )

    top_bottom = pd.concat([ranked.head(15), ranked.tail(15)]).sort_values("netzero_index_score")
    fig = px.bar(
        top_bottom,
        x="netzero_index_score",
        y="country",
        color="tier",
        orientation="h",
        title="Top 15 and Bottom 15 Countries by Net Zero Readiness Index",
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Country Detail":
    country = st.selectbox("Select a country", sorted(df["country"].unique()))
    row = df[df["country"] == country].iloc[0]

    st.metric("Net Zero Readiness Index", f"{row['netzero_index_score']:.1f} / 100", row["tier"])

    cols = st.columns(len(INDICATOR_LABELS))
    any_imputed = False
    for c, (indicator, (label, unit)) in zip(cols, INDICATOR_LABELS.items()):
        imputed = bool(row.get(f"{indicator}_imputed", False))
        any_imputed = any_imputed or imputed
        c.metric(label, f"{row[indicator]:.2f}{unit}" + (" *" if imputed else ""))
    if any_imputed:
        st.caption("* imputed value (missing in source data, filled with the cohort median)")

    categories = list(C.INDICATOR_DIRECTIONS)
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(r=[row[f"{c}_norm"] for c in categories], theta=categories, fill="toself", name=country)
    )
    fig.add_trace(
        go.Scatterpolar(r=[50] * len(categories), theta=categories, name="Global median (=50 by construction)", line=dict(dash="dot"))
    )
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

elif page == "PCA & Clusters":
    var_pct = sum(manifest["pca_explained_variance"]) * 100
    fig = px.scatter(
        df, x="pca_1", y="pca_2", color="cluster_label", hover_name="country",
        title=f"PCA projection (PC1+PC2 explain {var_pct:.1f}% of variance)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Choosing k (elbow / silhouette)")
    k_df = pd.DataFrame(manifest["elbow_silhouette_table"])
    st.line_chart(k_df.set_index("k")[["silhouette"]])
    st.caption(f"Silhouette score at k={C.N_CLUSTERS_DEFAULT}: {manifest['silhouette_k4']:.3f}")

    st.subheader("Tier vs. cluster agreement")
    st.caption("Tiers come from quantile cuts of the composite score; clusters are an independent KMeans view. This table shows how well they agree.")
    st.dataframe(pd.crosstab(df["tier"], df["cluster_label"]))

else:
    st.subheader("Methodology & Limitations")
    st.markdown(f"""
- **Reference year:** {C.REFERENCE_YEAR} (2023 has no GDP data in either source dataset)
- **Countries scored:** {manifest['final_country_count']}
- **Indicators (5, across 4 equally-weighted dimensions):** {', '.join(C.INDICATOR_DIRECTIONS)}
- **Normalization:** percentile rank, not min-max — robust to outliers like Qatar's CO2/capita or a high-GDP microstate
- **Tiers** come from quantile cuts of the composite score; KMeans clustering (k={C.N_CLUSTERS_DEFAULT}) is shown separately as an independent cross-check, not the tier source
- **Imputed value counts:** {manifest['imputed_counts']}
""")
    st.markdown("""
**Known limitations:**
- Cross-sectional 2022 snapshot — the trend dimension looks back to 2013-2022, but the index itself is not longitudinal
- `renewables_share_elec` doesn't count nuclear as renewable, so nuclear-heavy low-emission grids (e.g. France) look worse on energy transition than their real decarbonization suggests
- No continent/region field in the source data, so missing-value imputation uses the global cohort median rather than a region-adjusted one
- No policy/pledge-ambition dimension, no supervised classifier, no historical-responsibility/equity weighting — out of scope by design
- The originally attached `global_warming_dataset.csv` was audited and found to be synthetic random noise (max |correlation| ~0.01, both row-level and after grouping by country); it is not used anywhere in this framework
""")
