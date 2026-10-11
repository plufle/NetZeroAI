import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

from src import constants as C
from src.components.data_transformation import composite_score
from src.components.index_validation import rank_desc
from src.fingerprint import file_sha256, stale_reasons
from src.pipeline.predict_pipeline import CustomCountryData, InvalidIndicatorInput, PredictPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.utils import describe_bounds

st.set_page_config(page_title="NetZeroAI", page_icon="🌍", layout="wide")

# Colors are validated (CVD-safe, contrast) for the light theme set in .streamlit/config.toml.
# Tiers are ordered, so they use a one-hue ordinal ramp (light = Lagging, dark = Leader).
TIER_COLORS = dict(zip(C.TIER_LABELS, ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]))
TIER_ORDER = {"tier": C.TIER_LABELS}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]        # categorical, max 3 per chart
REFERENCE = "#898781"                              # reference lines (global median)
CONTEXT = "#d6d5ce"                                # de-emphasized background marks
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIMENSION_LABELS = {d: d.replace("_", " ").title() for d in C.DIMENSIONS}
DIMENSION_COLS = [f"{d}_score" for d in C.DIMENSIONS]
INDICATOR_LABELS = {
    "co2_per_capita": ("CO2 per capita", "t"),
    "ghg_per_capita": ("GHG per capita", "t"),
    "co2pc_trend_slope": ("CO2/capita trend (2013-22)", "t/yr"),
    "renewables_share_elec": ("Renewables share (electricity)", "%"),
    "gdp_per_capita": ("GDP per capita", "$"),
}
SHORT_INDICATOR_LABELS = {
    "co2_per_capita": "CO2/capita", "ghg_per_capita": "GHG/capita", "co2pc_trend_slope": "CO2 trend",
    "renewables_share_elec": "Renewables", "gdp_per_capita": "GDP/capita",
}
WEIGHT_SPREAD = f"±{C.WEIGHT_DRAW_SPREAD:.0%}"
WEIGHT_BAND_HELP = (
    f"Weight-sensitivity result, not a confidence interval: 5th-95th percentile rank across {C.N_WEIGHT_DRAWS} "
    f"random weightings, each dimension weight drawn uniformly within {WEIGHT_SPREAD} of its baseline. "
    "It does not include uncertainty from imputed values."
)
IMPUTATION_BAND_HELP = (
    "Imputed countries only: best and worst rank when each imputed value is set to the lowest and highest "
    "observed value in the peer group it was imputed from (other countries held fixed)."
)
SCORE_TABLE_CONFIG = {
    "rank": st.column_config.NumberColumn("Rank"),
    "country": "Country",
    "region": "Region",
    "tier": "Tier",
    "netzero_index_score": st.column_config.ProgressColumn("Index score", min_value=0, max_value=100, format="%.1f"),
    **{f"{d}_score": st.column_config.NumberColumn(DIMENSION_LABELS[d], format="%.1f") for d in C.DIMENSIONS},
}


@st.cache_data
def load_artifacts(manifest_sha256):
    # The manifest hash is part of the cache key on purpose - no leading underscore, which
    # Streamlit would exclude - so every rebuild (which rewrites the manifest) invalidates it.
    df = pd.read_csv(C.FINAL_TABLE_PATH)
    df["imputed_any"] = df[[f"{c}_imputed" for c in C.INDICATOR_DIRECTIONS]].any(axis=1)
    with open(C.MANIFEST_PATH) as f:
        manifest = json.load(f)
    return df, manifest


# Every artifact must exist, match the hash recorded at build time, and come from the
# current raw inputs + configuration (src/fingerprint.py); otherwise rebuild.
stale = stale_reasons()
if stale:
    with st.spinner(f"Rebuilding the index ({'; '.join(stale)}) - about 20 seconds..."):
        TrainPipeline().run()
    stale = stale_reasons()
    if stale:
        st.error("Artifacts are still inconsistent after a rebuild: " + "; ".join(stale))
        st.stop()

df, manifest = load_artifacts(file_sha256(C.MANIFEST_PATH))


def tier_bar(data, x, y, title, height=None):
    fig = px.bar(
        data, x=x, y=y, color="tier", orientation="h", title=title,
        color_discrete_map=TIER_COLORS, category_orders=TIER_ORDER, height=height,
    )
    fig.update_layout(xaxis_title="Index score (0-100)", yaxis_title="", legend_title="Tier", bargap=0.25)
    return fig


# ---------------------------------------------------------------- pages


def page_overview():
    st.subheader(f"Net Zero Readiness ranking - {len(df)} countries, {C.REFERENCE_YEAR}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Countries scored", len(df))
    k2.metric("Median index score", f"{df['netzero_index_score'].median():.1f}")
    mc = manifest["sensitivity"]["monte_carlo"]
    k3.metric("Rank robustness (mean Spearman)", f"{mc['mean_spearman']:.2f}", help=(
        f"Average rank correlation with the baseline ranking across {mc['n_draws']} random weightings "
        f"(each weight varied +/-{mc['weight_spread']:.0%})."
    ))
    k4.metric("Countries with an imputed value", int(df["imputed_any"].sum()))

    fig = px.choropleth(
        df, locations="iso_code", color="netzero_index_score", hover_name="country",
        hover_data={"iso_code": False, "tier": True, "rank": True, "netzero_index_score": ":.1f"},
        # Fit the scale to the observed range (composites never reach 0 or 100) for more contrast.
        color_continuous_scale=BLUES,
        range_color=(10 * (df["netzero_index_score"].min() // 10), 10 * -(-df["netzero_index_score"].max() // 10)),
        labels={"netzero_index_score": "Index score"},
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=430, geo=dict(showframe=False, projection_type="natural earth"))
    st.plotly_chart(fig, use_container_width=True)

    f1, f2 = st.columns(2)
    regions = f1.multiselect("Region", sorted(df["region"].dropna().unique()))
    tiers = f2.multiselect("Tier", C.TIER_LABELS)
    view = df
    if regions:
        view = view[view["region"].isin(regions)]
    if tiers:
        view = view[view["tier"].isin(tiers)]

    table = view.sort_values("rank").assign(
        weight_band=lambda t: t["weight_rank_p05"].astype(str) + "-" + t["weight_rank_p95"].astype(str),
        # Int64 first: the CSV round trip turns these nullable ranks into floats ("1.0").
        imputation_band=lambda t: (t["impute_rank_best"].astype("Int64").astype("string") + "-"
                                   + t["impute_rank_worst"].astype("Int64").astype("string")).fillna(""),
    )[["rank", "country", "region", "netzero_index_score", "tier", "weight_band", "weight_tier_stability",
       "imputation_band"] + DIMENSION_COLS]
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            **SCORE_TABLE_CONFIG,
            "weight_band": st.column_config.TextColumn(
                f"Rank, weights {WEIGHT_SPREAD}", help=WEIGHT_BAND_HELP,
            ),
            "weight_tier_stability": st.column_config.ProgressColumn(
                f"Tier kept, weights {WEIGHT_SPREAD}", min_value=0, max_value=1, format="%.2f",
                help=f"Share of random weightings (each dimension weight {WEIGHT_SPREAD}) that keep the baseline tier.",
            ),
            "imputation_band": st.column_config.TextColumn(
                "Rank, imputed value varied", help=IMPUTATION_BAND_HELP,
            ),
        },
    )
    st.caption(
        f"Rank ranges are sensitivity results, not confidence intervals. \"Weights {WEIGHT_SPREAD}\": 5th-95th percentile rank "
        "when each dimension weight is drawn uniformly within that range of its baseline. \"Imputed value varied\" "
        "(imputed countries only): best-worst rank when the imputed value spans its peer group's observed range."
    )
    st.download_button(
        "Download full scored table (CSV)", df.to_csv(index=False).encode(), file_name=C.FINAL_TABLE_PATH.name, mime="text/csv",
    )

    ranked = df.sort_values("netzero_index_score", ascending=False)
    top_bottom = pd.concat([ranked.head(15), ranked.tail(15)]).sort_values("netzero_index_score")
    st.plotly_chart(
        tier_bar(top_bottom, "netzero_index_score", "country", "Top 15 and bottom 15 countries", height=700),
        use_container_width=True,
    )


def page_country():
    country = st.selectbox("Select a country", sorted(df["country"]), index=sorted(df["country"]).index("Germany"))
    row = df[df["country"] == country].iloc[0]
    peers = df[df["sub_region"] == row["sub_region"]]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Net Zero Readiness Index", f"{row['netzero_index_score']:.1f} / 100")
    m2.metric("Rank", f"{row['rank']} of {len(df)}", help="Rank under the baseline equal weighting")
    m3.metric(f"Rank range, weights {WEIGHT_SPREAD}", f"{row['weight_rank_p05']} - {row['weight_rank_p95']}", help=WEIGHT_BAND_HELP)
    m4.metric("Tier", row["tier"], help="Quartile of the composite score")
    st.caption(
        f"{row['region']} / {row['sub_region']} - keeps its tier in {row['weight_tier_stability']:.0%} of random "
        f"weightings ({WEIGHT_SPREAD} per dimension) - ML segment: {row['cluster_label']}"
    )

    cols = st.columns(len(INDICATOR_LABELS))
    notes = []
    for c, (indicator, (label, unit)) in zip(cols, INDICATOR_LABELS.items()):
        imputed = bool(row[f"{indicator}_imputed"])
        value = f"{row[indicator]:,.0f}" if indicator == "gdp_per_capita" else f"{row[indicator]:.2f}"
        c.metric(label + (" *" if imputed else ""), f"{value} {unit}")
        if imputed:
            notes.append(
                f"{label}: missing in source data, filled with the {row[f'{indicator}_impute_source']} "
                f"(peers' observed range {row[f'{indicator}_impute_min']:,.2f} - {row[f'{indicator}_impute_max']:,.2f})"
            )
    if notes:
        st.caption("\\* imputed - " + "; ".join(notes))
        tiers = (f"{row['impute_tier_low']}" if row["impute_tier_low"] == row["impute_tier_high"]
                 else f"{row['impute_tier_low']} to {row['impute_tier_high']}")
        st.warning(
            f"Imputation sensitivity: if the imputed value were anywhere in its peers' observed range, this country "
            f"would rank **{int(row['impute_rank_best'])} - {int(row['impute_rank_worst'])}** (tier: {tiers}). "
            "This uncertainty is separate from, and not included in, the weight-sensitivity rank range above."
        )

    left, right = st.columns(2)
    categories = list(C.INDICATOR_DIRECTIONS)
    labels = [SHORT_INDICATOR_LABELS[c] for c in categories]
    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=[row[f"{c}_norm"] for c in categories], theta=labels, fill="toself", name=country, line=dict(color=SERIES[0], width=2),
    ))
    radar.add_trace(go.Scatterpolar(
        r=[peers[f"{c}_norm"].median() for c in categories], theta=labels, name=f"{row['sub_region']} median",
        line=dict(color=SERIES[1], width=2),
    ))
    radar.add_trace(go.Scatterpolar(
        r=[50] * len(categories), theta=labels, name="Global median (50 by construction)", line=dict(color=REFERENCE, dash="dot", width=1.5),
    ))
    radar.update_layout(
        title="Indicator percentiles (higher = better)",
        polar=dict(radialaxis=dict(range=[0, 100], angle=90, tickangle=90, tickfont=dict(size=9))),
        legend=dict(orientation="h", y=-0.15), margin=dict(l=70, r=70),
    )
    left.plotly_chart(radar, use_container_width=True)

    dims = pd.DataFrame({
        "dimension": [DIMENSION_LABELS[d] for d in C.DIMENSIONS] * 2,
        "score": [row[f"{d}_score"] for d in C.DIMENSIONS] + [peers[f"{d}_score"].median() for d in C.DIMENSIONS],
        "series": [country] * len(C.DIMENSIONS) + [f"{row['sub_region']} median"] * len(C.DIMENSIONS),
    })
    bars = px.bar(dims, x="score", y="dimension", color="series", barmode="group", orientation="h",
                  title="Dimension scores vs regional peers", color_discrete_sequence=SERIES)
    bars.update_layout(xaxis_range=[0, 100], xaxis_title="Score (0-100)", yaxis_title="", legend=dict(orientation="h", y=-0.2, title=""))
    right.plotly_chart(bars, use_container_width=True)

    st.markdown(f"**{row['sub_region']} peers**")
    st.dataframe(
        peers.sort_values("rank")[["rank", "country", "netzero_index_score", "tier"] + DIMENSION_COLS],
        hide_index=True, use_container_width=True, column_config=SCORE_TABLE_CONFIG,
    )


def page_scenarios():
    tab_weights, tab_whatif = st.tabs(["Reweight the dimensions", "Score a hypothetical country"])

    with tab_weights:
        st.markdown(
            "The baseline index weights all four dimensions equally. Move the sliders to see how the "
            "ranking changes under a different set of priorities (weights are renormalized to sum to 100%)."
        )
        sliders = st.columns(len(C.DIMENSIONS))
        raw = {
            d: col.slider(DIMENSION_LABELS[d], 0, 100, int(C.DIMENSION_WEIGHTS[d] * 100), step=5, key=f"w_{d}")
            for col, d in zip(sliders, C.DIMENSIONS)
        }
        if sum(raw.values()) == 0:
            st.warning("Give at least one dimension a non-zero weight.")
        else:
            render_reweighting(raw)

    with tab_whatif:
        render_whatif()


def render_reweighting(raw):
    total = sum(raw.values())
    st.caption("Effective weights: " + ", ".join(f"{DIMENSION_LABELS[d]} {w / total:.0%}" for d, w in raw.items()))

    custom = df[["country", "rank", "tier", "netzero_index_score"] + DIMENSION_COLS].copy()
    custom["custom_score"] = composite_score(custom, raw)
    custom["custom_rank"] = rank_desc(custom["custom_score"])
    custom["custom_tier"] = pd.qcut(custom["custom_score"], q=4, labels=C.TIER_LABELS).astype(str)
    custom["rank_change"] = custom["rank"] - custom["custom_rank"]

    s1, s2, s3 = st.columns(3)
    s1.metric("Rank correlation with baseline", f"{stats.spearmanr(custom['custom_score'], custom['netzero_index_score']).statistic:.3f}")
    s2.metric("Countries changing tier", int((custom["custom_tier"] != custom["tier"]).sum()))
    s3.metric("Mean absolute rank change", f"{custom['rank_change'].abs().mean():.1f}")

    cols = ["country", "rank", "custom_rank", "rank_change", "tier", "custom_tier"]
    rename = {"country": "Country", "rank": "Baseline rank", "custom_rank": "New rank", "rank_change": "Change",
              "tier": "Baseline tier", "custom_tier": "New tier"}
    climbers = custom[custom["rank_change"] > 0].sort_values("rank_change", ascending=False).head(10)
    fallers = custom[custom["rank_change"] < 0].sort_values("rank_change").head(10)
    if climbers.empty and fallers.empty:
        st.info("No rank changes - these are the baseline weights. Move a slider to compare.")
    else:
        up, down = st.columns(2)
        up.markdown("**Biggest climbers**")
        up.dataframe(climbers[cols].rename(columns=rename), hide_index=True, use_container_width=True)
        down.markdown("**Biggest fallers**")
        down.dataframe(fallers[cols].rename(columns=rename), hide_index=True, use_container_width=True)

    with st.expander("Full re-weighted ranking"):
        st.dataframe(
            custom.sort_values("custom_rank")[cols + ["custom_score"]].rename(columns=rename | {"custom_score": "New score"}).round(1),
            hide_index=True, use_container_width=True,
        )


def render_whatif():
    st.markdown(
        "Enter raw indicator values for a hypothetical (or modified real) country. It is scored "
        "through the same fitted normalizer, tier boundaries and KMeans model as the 194 real countries, "
        "without changing them."
    )
    template = st.selectbox("Start from a real country's values", sorted(df["country"]), index=sorted(df["country"]).index("India"))
    base = df[df["country"] == template].iloc[0]
    inputs = st.columns(len(INDICATOR_LABELS))
    values = {}
    for col, (indicator, (label, unit)) in zip(inputs, INDICATOR_LABELS.items()):
        step = 500.0 if indicator == "gdp_per_capita" else (0.01 if indicator == "co2pc_trend_slope" else 0.1)
        bounds = C.INDICATOR_BOUNDS[indicator]          # the same bounds PredictPipeline enforces
        values[indicator] = col.number_input(
            f"{label} ({unit})", value=float(base[indicator]), step=step, key=f"in_{indicator}_{template}",
            min_value=bounds["min"], max_value=bounds["max"], help=f"Allowed: {describe_bounds(indicator)}",
        )

    try:
        result = PredictPipeline().predict(CustomCountryData(**values, country=f"Modified {template}").to_dataframe()).iloc[0]
    except (InvalidIndicatorInput, FileNotFoundError) as e:
        st.error(f"Can't score this country: {e}")
        return
    would_rank = int((df["netzero_index_score"].round(9) > round(result["netzero_index_score"], 9)).sum()) + 1

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Index score", f"{result['netzero_index_score']:.1f}", f"{result['netzero_index_score'] - base['netzero_index_score']:+.1f} vs {template}")
    r2.metric("Would rank", f"{would_rank} of {len(df) + 1}")
    r3.metric("Tier", str(result["tier"]))
    r4.metric("ML segment", result["cluster_label"])

    dims = pd.DataFrame({
        "dimension": [DIMENSION_LABELS[d] for d in C.DIMENSIONS] * 2,
        "score": [result[f"{d}_score"] for d in C.DIMENSIONS] + [base[f"{d}_score"] for d in C.DIMENSIONS],
        "series": ["Hypothetical"] * len(C.DIMENSIONS) + [f"{template} (actual)"] * len(C.DIMENSIONS),
    })
    fig = px.bar(dims, x="score", y="dimension", color="series", barmode="group", orientation="h", color_discrete_sequence=SERIES)
    fig.update_layout(xaxis_range=[0, 100], yaxis_title="", legend=dict(orientation="h", y=-0.2, title=""))
    st.plotly_chart(fig, use_container_width=True)


def permutation_p_text(p_value, n_permutations):
    # (hits + 1) / (n + 1): the smallest attainable value means no permutation reached the observed one.
    floor = 1 / (n_permutations + 1)
    return f"p < {floor:.3f}" if p_value <= floor + 1e-12 else f"p = {p_value:.3f}"


def page_validation():
    sens = manifest["sensitivity"]
    mc = sens["monte_carlo"]
    agree = manifest["tier_cluster_agreement"]
    stab = manifest["cluster_stability"]

    st.subheader("1. Sensitivity to weighting and method choices")
    st.markdown(
        "Each alternative is compared with the baseline (equal weights, percentile normalization, arithmetic "
        "mean, absolute trend slope, regional imputation). Tiers are re-cut by quartile under each alternative."
    )
    scen = pd.DataFrame(sens["weight_scenarios"] + sens["method_alternatives"]).drop(columns="weights", errors="ignore")
    st.dataframe(
        scen,
        hide_index=True,
        use_container_width=True,
        column_config={
            "scenario": "Scenario",
            "spearman_vs_baseline": st.column_config.ProgressColumn("Spearman vs baseline", min_value=0, max_value=1, format="%.3f"),
            "mean_abs_rank_shift": st.column_config.NumberColumn("Mean |rank shift|", format="%.1f"),
            "max_abs_rank_shift": "Max |rank shift|",
            "max_shift_country": "Most-moved country",
            "tier_change_pct": st.column_config.NumberColumn("% changing tier", format="%.1f"),
            "top10_overlap": "Top-10 overlap",
        },
    )

    st.subheader(f"2. Weight sensitivity: random weights {WEIGHT_SPREAD}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Spearman vs baseline", f"{mc['mean_spearman']:.3f}", help=f"Lowest across all draws: {mc['min_spearman']:.3f}")
    c2.metric("Median width, 5th-95th pct rank", f"{mc['median_rank_band_width']:.0f} places")
    c3.metric("Countries keeping tier in >=80% of draws", f"{mc['pct_countries_tier_stable_80']:.0f}%")
    c4.metric("Countries keeping tier in >=95% of draws", f"{mc['pct_countries_tier_stable_95']:.0f}%")
    st.caption(
        f"{mc['n_draws']} draws; each dimension weight drawn uniformly within {WEIGHT_SPREAD} of its baseline value, then "
        "renormalized. These bands are a sensitivity analysis under that assumed weight variation - not a statistical "
        "confidence interval - and they exclude imputation uncertainty (section 3). Rankings are broadly robust, but "
        "fine-grained rank positions and tier boundaries are not: read the index by tier and rank band."
    )
    default = df.sort_values("rank").head(20)["country"].tolist()
    chosen = st.multiselect("Countries to show", sorted(df["country"]), default=default)
    band = df[df["country"].isin(chosen)].sort_values("rank", ascending=False)
    if not band.empty:
        fig = go.Figure(go.Scatter(
            x=band["rank"], y=band["country"], mode="markers",
            marker=dict(size=9, color=[TIER_COLORS[t] for t in band["tier"]]),
            # Clipped at 0: the baseline rank isn't guaranteed to sit inside the 5th-95th band.
            error_x=dict(type="data", symmetric=False, array=(band["weight_rank_p95"] - band["rank"]).clip(lower=0),
                         arrayminus=(band["rank"] - band["weight_rank_p05"]).clip(lower=0)),
            hovertemplate="%{y}: rank %{x}<extra></extra>",
        ))
        fig.update_layout(
            title=f"Baseline rank with 5th-95th percentile rank under random weights {WEIGHT_SPREAD} (1 = best)",
            xaxis_title="Rank", height=max(300, 26 * len(band)), xaxis=dict(range=[0, len(df) + 1]),
        )
        st.plotly_chart(fig, use_container_width=True)
    hist = px.histogram(df, x="weight_tier_stability", color="tier", color_discrete_map=TIER_COLORS, category_orders=TIER_ORDER,
                        labels={"weight_tier_stability": f"Share of random weightings ({WEIGHT_SPREAD}) that keep the baseline tier",
                                "tier": "Tier"},
                        title="Tier stability across random weightings")
    # Shared bins, tiers side by side: the outer tiers are visibly the most stable (only one neighbour to slip into).
    hist.update_traces(xbins=dict(start=0, end=1.0001, size=0.1), marker_line=dict(color="#fcfcfb", width=1))
    hist.update_layout(barmode="group", yaxis_title="Countries", bargap=0.1, xaxis_range=[0, 1.0001])
    st.plotly_chart(hist, use_container_width=True)

    imp = sens["imputation"]
    st.subheader("3. Imputation sensitivity")
    i1, i2, i3 = st.columns(3)
    i1.metric("Countries with an imputed value", imp["n_countries"])
    i2.metric("...whose tier could change", imp["n_tier_can_change"])
    i3.metric("Median rank range", f"{imp['median_rank_range_width']:.0f} places")
    global_gdp = next((v["peer_range"] for r in imp["countries"] for c, v in r["imputed"].items()
                       if c == "gdp_per_capita" and v["source"].startswith("global")), None)
    global_note = (f" (GDP per capita \\${global_gdp[0]:,.0f} to \\${global_gdp[1]:,.0f})" if global_gdp else "")
    st.caption(
        "Each imputed value is set to the lowest and the highest observed value in the peer group it was imputed from, "
        "with every other country held fixed. Countries without regional peers - the Pacific islands - fall back to the "
        f"full global range{global_note}, so their ranges are very wide: the data simply cannot place them. These "
        "ranges are not part of the weight-sensitivity bands above."
    )
    st.dataframe(
        pd.DataFrame([
            {
                "Country": r["country"],
                "Imputed": ", ".join(INDICATOR_LABELS[c][0] for c in r["imputed"]),
                "Source": "; ".join(v["source"] for v in r["imputed"].values()),
                "Peer range": "; ".join(f"{v['peer_range'][0]:,.1f} - {v['peer_range'][1]:,.1f}" for v in r["imputed"].values()),
                "Rank": r["baseline_rank"],
                "Rank range": f"{r['rank_range'][0]} - {r['rank_range'][1]}",
                "Tier": r["baseline_tier"],
                "Tier range": r["tier_range"][0] if r["tier_range"][0] == r["tier_range"][1] else " to ".join(r["tier_range"]),
            }
            for r in imp["countries"]
        ]),
        hide_index=True, use_container_width=True,
    )

    st.subheader("4. Do the KMeans segments agree with the index tiers?")
    null = agree["permutation_baseline"]
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Adjusted Rand Index", f"{agree['adjusted_rand_index']:.3f}", help="Label-independent; 0 = chance, 1 = identical")
    a2.metric("Normalized mutual info", f"{agree['normalized_mutual_info']:.3f}", help="Label-independent; 0 = independent")
    a3.metric("Exact agreement", f"{agree['exact_agreement_pct']:.0f}%", help=(
        "Tier position equals segment position, with segments numbered by mean index score. Chance level "
        f"{null['exact_agreement_mean_pct']:.1f}%: the mean over {null['n_permutations']} random segment assignments "
        "re-numbered the same way."
    ))
    a4.metric("Within one step", f"{agree['within_one_step_pct']:.0f}%", help=f"Chance level {null['within_one_step_mean_pct']:.1f}%")
    left, right = st.columns(2)
    crosstab = pd.crosstab(df["tier"], df["cluster_label"]).reindex(C.TIER_LABELS)
    left.plotly_chart(
        px.imshow(crosstab, text_auto=True, color_continuous_scale=BLUES, title="Tier x segment (counts)", aspect="auto",
                  labels={"x": "KMeans segment", "y": "Index tier", "color": "Countries"}),
        use_container_width=True,
    )
    profiles = pd.DataFrame(agree["cluster_profiles"]).set_index("cluster_label")
    heat = profiles[DIMENSION_COLS].rename(columns={f"{d}_score": DIMENSION_LABELS[d] for d in C.DIMENSIONS})
    right.plotly_chart(
        px.imshow(heat, text_auto=".0f", color_continuous_scale=BLUES, zmin=0, zmax=100,
                  labels={"x": "", "y": "", "color": "Score"},
                  title="Segment profiles (mean dimension score)", aspect="auto"),
        use_container_width=True,
    )
    st.markdown(
        f"The label-independent measures (ARI {agree['adjusted_rand_index']:.2f}, NMI {agree['normalized_mutual_info']:.2f}) show "
        f"real but moderate agreement. Exact agreement is {agree['exact_agreement_pct']:.0f}% against a permutation baseline of "
        f"{null['exact_agreement_mean_pct']:.1f}% ({permutation_p_text(null['exact_agreement_p_value'], null['n_permutations'])}). "
        "The segments separate countries by "
        "**development profile** (PC1 contrasts emissions intensity with GDP), while tiers rank overall readiness. "
        "Most strong mismatches are wealthy, fossil-heavy economies (e.g. Kuwait, Oman, Singapore, Trinidad and Tobago) "
        "that the index places in *Emerging* but KMeans groups with the high-income, fast-decarbonizing *Leader*-dominated segment."
    )
    with st.expander(f"Countries whose tier and segment differ by 2+ steps ({len(agree['strong_disagreements'])})"):
        st.dataframe(pd.DataFrame(agree["strong_disagreements"]), hide_index=True, use_container_width=True)

    st.subheader("5. Are the KMeans segments stable?")
    by_k = pd.DataFrame(stab["by_k"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=by_k["k"], y=by_k["mean_ari"], mode="lines+markers", name="Mean ARI", line=dict(color=SERIES[0], width=2)))
    fig.add_trace(go.Scatter(x=by_k["k"], y=by_k["p05_ari"], mode="lines+markers", name="5th percentile ARI", line=dict(color=SERIES[1], width=2)))
    fig.update_layout(title=f"Bootstrap stability ({stab['n_bootstrap']} refits on {stab['subsample_fraction']:.0%} subsamples)",
                      xaxis_title="k", yaxis_title="ARI vs full-data clustering", yaxis_range=[0, 1.02])
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"At the chosen k={C.N_CLUSTERS_DEFAULT}, refits reproduce the full-data segments with mean ARI "
        f"{stab['selected_k']['mean_ari']:.2f}. k=2 and k=3 are more stable still, so k=4 is chosen for comparability "
        "with the four tiers, not because the data strongly favors it."
    )


def segment_summary(profile):
    """'strongest: X - weakest: Y' from a segment's mean dimension scores."""
    scores = {DIMENSION_LABELS[d]: profile[f"{d}_score"] for d in C.DIMENSIONS}
    return f"strongest: {max(scores, key=scores.get)} - weakest: {min(scores, key=scores.get)}"


def k_line(k_df, column, title):
    fig = go.Figure(go.Scatter(x=k_df["k"], y=k_df[column], mode="lines+markers", line=dict(color=SERIES[0], width=2), marker=dict(size=8)))
    fig.add_vline(x=C.N_CLUSTERS_DEFAULT, line=dict(color=REFERENCE, width=1, dash="dot"))
    fig.update_layout(title=title, xaxis_title="k", height=320, margin=dict(t=50, b=40))
    return fig


def page_pca():
    var = manifest["pca_explained_variance"]
    axis_labels = {"pca_1": f"PC1 ({var[0]:.0%})", "pca_2": f"PC2 ({var[1]:.0%})"}
    fig = px.scatter(
        df, x="pca_1", y="pca_2", color="tier", hover_name="country",
        color_discrete_map=TIER_COLORS, category_orders=TIER_ORDER, labels=axis_labels | {"tier": "Tier"},
        title=f"PCA projection, colored by index tier (PC1 + PC2 explain {sum(var):.0%} of variance)", height=540,
    )
    fig.update_traces(marker=dict(size=9, line=dict(color="#fcfcfb", width=1)))
    loadings = pd.DataFrame(manifest["pca_loadings"])
    scale = 3
    for indicator, (pc1, pc2) in loadings.iterrows():
        fig.add_annotation(x=pc1 * scale, y=pc2 * scale, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor="#52514e", text="")
        fig.add_annotation(x=pc1 * scale * 1.15, y=pc2 * scale * 1.15, showarrow=False, text=SHORT_INDICATOR_LABELS[indicator],
                           font=dict(size=12, color="#0b0b0b"), bgcolor="rgba(252,252,251,0.85)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Arrows show each normalized indicator's PCA loading (x3 for visibility); normalized indicators point toward "
        "*better* values. PC1 contrasts low emissions with high income - the main axis the KMeans segments split along."
    )

    st.subheader(f"KMeans segments (k={C.N_CLUSTERS_DEFAULT})")
    profiles = {p["cluster_label"]: p for p in manifest["tier_cluster_agreement"]["cluster_profiles"]}
    segments = sorted(profiles)
    grid = make_subplots(
        rows=2, cols=2, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.12, horizontal_spacing=0.05,
        subplot_titles=[f"{s} - {profiles[s]['n_countries']} countries<br><sup>{segment_summary(profiles[s])}</sup>" for s in segments],
    )
    for i, segment in enumerate(segments):
        r, c = divmod(i, 2)
        members = df[df["cluster_label"] == segment]
        grid.add_trace(go.Scatter(x=df["pca_1"], y=df["pca_2"], mode="markers", marker=dict(color=CONTEXT, size=6),
                                  hoverinfo="skip", showlegend=False), r + 1, c + 1)
        grid.add_trace(go.Scatter(x=members["pca_1"], y=members["pca_2"], mode="markers", text=members["country"],
                                  marker=dict(color=SERIES[0], size=8, line=dict(color="#fcfcfb", width=1)),
                                  hovertemplate="%{text}<extra></extra>", showlegend=False), r + 1, c + 1)
    grid.update_layout(height=640, margin=dict(t=80))
    grid.update_xaxes(title_text=axis_labels["pca_1"], row=2)
    grid.update_yaxes(title_text=axis_labels["pca_2"], col=1)
    st.plotly_chart(grid, use_container_width=True)
    st.caption("Each panel highlights one segment against all countries (grey). Segments are numbered by ascending mean index score.")

    left, mid, right = st.columns(3)
    left.markdown("**PCA loadings**")
    left.dataframe(loadings.rename(index={k: v[0] for k, v in INDICATOR_LABELS.items()}).round(3), use_container_width=True)
    k_df = pd.DataFrame(manifest["elbow_silhouette_table"])
    mid.plotly_chart(k_line(k_df, "inertia", "Inertia (elbow)"), use_container_width=True)
    right.plotly_chart(k_line(k_df, "silhouette", "Silhouette"), use_container_width=True)
    st.caption(
        f"Silhouette at k={C.N_CLUSTERS_DEFAULT}: {manifest['silhouette_k4']:.3f}; no k gives strong separation "
        "(all <= 0.35) and there is no sharp elbow - countries form a continuum more than distinct groups. "
        "k=4 is used for comparability with the four tiers; see the validation page for its bootstrap stability."
    )


def page_methodology():
    imputed = manifest["imputed_counts"]
    sens = manifest["sensitivity"]
    st.subheader("Methodology")
    st.markdown(f"""
- **Data:** Our World in Data CO2 & GHG and Energy datasets, reference year **{C.REFERENCE_YEAR}** (2023 has no GDP data
  in either source). UN M49 regions (via `lukes/ISO-3166-Countries-with-Regional-Codes`) for regional imputation.
- **Countries:** {manifest['final_country_count']} - real ISO-3 countries/territories with at most {C.MAX_MISSING_INDICATORS}
  of the 5 indicators missing.
- **Indicators -> dimensions (equal 25% weights):**
""" + "\n".join(
        f"  - {DIMENSION_LABELS[d]}: " + ", ".join(INDICATOR_LABELS[i][0] for i in cols) for d, cols in C.DIMENSIONS.items()
    ) + f"""
- **Normalization:** percentile rank against the {manifest['final_country_count']}-country cohort (0-100, higher = better).
  Chosen over min-max because the indicators are heavy-tailed - see the min-max scenario on the validation page.
- **Composite:** weighted arithmetic mean of the four dimension scores. **Tiers** are quartiles of the composite.
- **Missing data:** filled with the median of the most specific UN M49 group (intermediate region -> sub-region ->
  region) with at least {C.MIN_IMPUTATION_GROUP_OBS} observed values, else the global median. Imputed counts:
  {", ".join(f"{INDICATOR_LABELS[k][0]} {v}" for k, v in imputed.items() if v)}.
- **ML cross-check:** StandardScaler + PCA (2 components, visualization) + KMeans (k={C.N_CLUSTERS_DEFAULT}) on the same
  normalized indicators, independent of the weights. Segments are an independent view, not the tier source.
""")
    with st.expander("Imputed values and where they came from"):
        st.dataframe(pd.DataFrame(manifest["imputation_details"]), hide_index=True, use_container_width=True)

    bias = sens["trend_level_bias"]
    st.subheader("Known limitations")
    st.markdown(f"""
- **Fine-grained ranks are weight-sensitive.** Rankings correlate strongly across random weightings ({WEIGHT_SPREAD} per
  dimension; mean Spearman {sens['monte_carlo']['mean_spearman']:.2f}), but only
  {sens['monte_carlo']['pct_countries_tier_stable_80']:.0f}% of countries keep their tier in >=80% of them. Use tiers and
  rank ranges, not exact positions. The rank ranges are sensitivity results under that assumed weight variation, not
  confidence intervals.
- **Territorial emissions accounting.** OWID emissions are production-based: import-dependent territories (e.g. Macao,
  which imports most of its electricity) look cleaner than their consumption footprint. Macao also has an imputed GDP.
- **Imputation load is concentrated in GDP** ({imputed['gdp_per_capita']} countries, mostly small island states).
  Pacific island states have no observed regional peers and fall back to the global median. A regional median can be
  far off in either direction - it understates Brunei and the Bahamas (both in the bottom 10) and likely overstates
  Sudan (Northern Africa median, driven by Libya, Algeria, Egypt). Varying each imputed value across its peers' observed
  range could change the tier of {sens['imputation']['n_tier_can_change']} of {sens['imputation']['n_countries']} imputed
  countries (validation page, section 3); this is not included in the weight-sensitivity rank ranges.
- **Renewables share of electricity excludes nuclear**, so nuclear-heavy low-carbon grids (e.g. France) score lower on
  energy transition than their decarbonization suggests.
- **Trend dimension:** the absolute 2013-2022 slope correlates mildly with emissions level (Spearman
  {bias['absolute_slope']:.2f}); a relative %/yr trend correlates even more ({bias['relative_trend']:.2f}) and barely changes
  rankings, so this reflects real convergence (rich countries cutting, developing countries growing), not a scaling artifact.
- **Equal weighting is a transparent default, not an expert-elicited scheme** - wealthy, carbon-intensive economies
  (e.g. Qatar) can score moderately because high GDP and an improving trend offset poor emissions. The geometric-mean
  scenario on the validation page shows the effect of penalizing such imbalance.
- **Clusters != readiness.** KMeans segments reflect development profile; they agree with tiers only moderately
  (ARI {manifest['tier_cluster_agreement']['adjusted_rand_index']:.2f}).
- **Out of scope by design:** no policy/pledge-ambition dimension, no supervised classifier, no historical-responsibility
  or equity weighting; a cross-sectional {C.REFERENCE_YEAR} snapshot.
- The originally attached `global_warming_dataset.csv` was audited and found to be synthetic random noise
  (max |correlation| ~0.01); it is not used.
""")
    audit, build = manifest["data_audit"], manifest["build"]
    with st.expander("Build provenance and data audit"):
        st.markdown(
            "Checks that passed on this build: " + "; ".join(audit["checks_passed"]) + ". A violation stops the build "
            "with an error naming the offending rows; a changed input file or setting triggers a rebuild."
        )
        st.dataframe(
            pd.DataFrame([{"Input": v["file"], "Rows": audit["raw_rows"][k], "Bytes": v["bytes"], "SHA-256": v["sha256"]}
                          for k, v in build["inputs"].items()]),
            hide_index=True, use_container_width=True,
        )
        merge = audit["real_countries_at_reference_year"]
        st.markdown(
            f"Real countries at {C.REFERENCE_YEAR}: {merge['owid_co2']} in the CO2 file, {merge['owid_energy']} in the energy "
            f"file, {merge['merged']} in both. Only in the CO2 file: {', '.join(audit['unmatched_in_merge']['owid_co2_only'])}. "
            f"Only in the energy file: {', '.join(audit['unmatched_in_merge']['owid_energy_only'])}."
        )
        st.markdown(f"**Dropped for missing more than {C.MAX_MISSING_INDICATORS} indicator(s):**")
        st.dataframe(
            pd.DataFrame([{"Country": d["country"], "ISO": d["iso_code"],
                           "Missing": ", ".join(INDICATOR_LABELS[c][0] for c in d["missing_indicators"])}
                          for d in audit["dropped_countries"]]),
            hide_index=True, use_container_width=True,
        )
    st.caption(
        f"Artifacts built {manifest['built_at'][:19].replace('T', ' ')} UTC - pipeline v{manifest['pipeline_version']} - "
        f"build fingerprint {build['fingerprint'][:12]}"
    )


PAGES = {
    "Overview / Ranking": page_overview,
    "Country Detail": page_country,
    "Scenario Lab": page_scenarios,
    "Robustness & Validation": page_validation,
    "PCA & Clusters": page_pca,
    "Methodology & Limitations": page_methodology,
}

st.title("NetZeroAI - Net Zero Readiness Assessment")
page = st.sidebar.radio("Section", list(PAGES))
st.sidebar.caption(
    f"{len(df)} countries - {C.REFERENCE_YEAR} data - Our World in Data. "
    "Scores are percentile-based: 50 = the cohort median."
)
PAGES[page]()
