import itertools
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src import constants as C
from src.components.data_transformation import PercentileNormalizer, add_dimension_scores, assign_tiers, composite_score
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

FEATURE_COLS = [f"{ind}_norm" for ind in C.INDICATOR_DIRECTIONS]
DIMENSION_SCORE_COLS = [f"{d}_score" for d in C.DIMENSIONS]


def dimension_label(dimension):
    return dimension.replace("_", " ").title()


def rank_desc(scores):
    """Rank scores so that 1 = best (highest); ties share the best rank. Works
    column-wise on 2-D input. Scores are rounded first so that mathematically
    tied countries aren't split by ~1e-14 floating-point noise."""
    return stats.rankdata(-np.round(np.asarray(scores, dtype=float), 9), method="min", axis=0).astype(int)


def quartile_tiers(scores):
    """Tier codes 0..3 (Lagging..Leader) from quartile cuts. Matches
    pd.qcut(q=4) bin semantics; 2-D input is cut column-wise."""
    scores = np.asarray(scores)
    edges = np.quantile(scores, [0.25, 0.5, 0.75], axis=0)
    if scores.ndim == 1:
        return np.searchsorted(edges, scores, side="left")
    return (scores[:, None, :] > edges[None, :, :]).sum(axis=1)


def weight_scenarios():
    """Named alternative weightings: drop each dimension, double each dimension."""
    base = C.DIMENSION_WEIGHTS
    scenarios = {}
    for d in C.DIMENSIONS:
        scenarios[f"Drop {dimension_label(d)}"] = {k: (0.0 if k == d else w) for k, w in base.items()}
    for d in C.DIMENSIONS:
        scenarios[f"Double {dimension_label(d)}"] = {k: (2 * w if k == d else w) for k, w in base.items()}
    return scenarios


def minmax_dimension_scores(df):
    """Dimension scores rebuilt with min-max instead of percentile normalization,
    to show what the documented outlier-compression problem looks like."""
    out = pd.DataFrame(index=df.index)
    norm = {}
    for col, direction in C.INDICATOR_DIRECTIONS.items():
        lo, hi = df[col].min(), df[col].max()
        scaled = (df[col] - lo) / (hi - lo) * 100
        norm[col] = 100 - scaled if direction == "lower_better" else scaled
    for name, cols in C.DIMENSIONS.items():
        out[f"{name}_score"] = pd.concat([norm[c] for c in cols], axis=1).mean(axis=1)
    return out


def relative_trend_composite(df):
    """Composite with the decarbonization dimension rebuilt from the relative
    (%/yr) CO2-per-capita trend instead of the absolute t/yr slope - tests
    whether the absolute slope favors high emitters just because they have
    more tonnes available to cut."""
    trend = df[["co2pc_trend_pct"]].fillna(df["co2pc_trend_pct"].median())
    normalizer = PercentileNormalizer({"co2pc_trend_pct": "lower_better"}).fit(trend)
    swapped = df.assign(decarbonization_trajectory_score=normalizer.transform(trend)["co2pc_trend_pct_norm"])
    return composite_score(swapped)


def global_median_imputation_composite(df):
    """Composite if imputed cells used the global cohort median (the pre-v2
    method) instead of the regional hierarchy - re-fits the normalizer since
    the imputed values change each indicator's reference distribution."""
    alt = df.copy()
    for col in C.INDICATOR_DIRECTIONS:
        imputed = alt[f"{col}_imputed"].astype(bool)
        alt.loc[imputed, col] = alt.loc[~imputed, col].median()
    normalizer = PercentileNormalizer(C.INDICATOR_DIRECTIONS).fit(alt)
    return composite_score(add_dimension_scores(normalizer.transform(alt)))


def geometric_composite(df, weights=None):
    """Weighted geometric mean of dimension scores (floored at 1). Unlike the
    arithmetic mean, a very weak dimension can't be fully offset by a strong one."""
    weights = weights or C.DIMENSION_WEIGHTS
    total = sum(weights.values())
    log_sum = sum(np.log(df[f"{d}_score"].clip(lower=1)) * (w / total) for d, w in weights.items())
    return np.exp(log_sum)


def compare_to_baseline(name, scores, base_scores, base_ranks, base_tiers, countries, weights=None):
    ranks = rank_desc(scores)
    shift = np.abs(ranks - base_ranks)
    worst = int(np.argmax(shift))
    top10 = set(np.argsort(-np.asarray(scores))[:10])
    base_top10 = set(np.argsort(-np.asarray(base_scores))[:10])
    row = {
        "scenario": name,
        "spearman_vs_baseline": float(stats.spearmanr(scores, base_scores).statistic),
        "mean_abs_rank_shift": float(shift.mean()),
        "max_abs_rank_shift": int(shift.max()),
        "max_shift_country": countries[worst],
        "tier_change_pct": float((quartile_tiers(scores) != base_tiers).mean() * 100),
        "top10_overlap": len(top10 & base_top10),
    }
    if weights is not None:
        total = sum(weights.values())
        row["weights"] = {d: round(w / total, 4) for d, w in weights.items()}
    return row


class IndexValidation:
    """Robustness checks on the composite index and the clustering cross-check.
    Adds per-country rank-uncertainty columns and returns a report dict for
    the pipeline manifest."""

    def initiate_index_validation(self, train_df):
        logging.info("Starting index validation (sensitivity, tier/cluster agreement, cluster stability)")
        try:
            df = train_df.copy()
            countries = df["country"].tolist()
            base_scores = df["netzero_index_score"].to_numpy()
            base_ranks = rank_desc(base_scores)
            base_tiers = pd.Categorical(df["tier"], categories=C.TIER_LABELS).codes
            df["rank"] = base_ranks

            sensitivity = self._sensitivity(df, countries, base_scores, base_ranks, base_tiers)
            monte_carlo, mc_columns = self._weight_monte_carlo(df, base_scores, base_tiers)
            for col, values in mc_columns.items():
                df[col] = values
            sensitivity["monte_carlo"] = monte_carlo

            imputation, imputation_columns = self._imputation_sensitivity(df, load_object(C.PREPROCESSOR_PATH), base_scores)
            for col, values in imputation_columns.items():
                df[col] = values
            sensitivity["imputation"] = imputation

            model_bundle = load_object(C.MODEL_PATH)
            report = {
                "sensitivity": sensitivity,
                "tier_cluster_agreement": self._tier_cluster_agreement(df, base_tiers),
                "cluster_stability": self._cluster_stability(df, model_bundle),
                "pca_loadings": {
                    f"PC{i + 1}": dict(zip(C.INDICATOR_DIRECTIONS, map(float, comp)))
                    for i, comp in enumerate(model_bundle["pca"].components_)
                },
            }
            logging.info(
                "Validation summary: MC mean Spearman %.3f, tier/cluster ARI %.3f, bootstrap ARI k=%d %.3f",
                monte_carlo["mean_spearman"],
                report["tier_cluster_agreement"]["adjusted_rand_index"],
                C.N_CLUSTERS_DEFAULT,
                report["cluster_stability"]["selected_k"]["mean_ari"],
            )
            return df, report

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _sensitivity(df, countries, base_scores, base_ranks, base_tiers):
        args = (base_scores, base_ranks, base_tiers, countries)
        weight_rows = [
            compare_to_baseline(name, composite_score(df, w).to_numpy(), *args, weights=w)
            for name, w in weight_scenarios().items()
        ]
        method_rows = [
            compare_to_baseline("Geometric aggregation", geometric_composite(df).to_numpy(), *args),
            compare_to_baseline(
                "Min-max normalization", composite_score(minmax_dimension_scores(df)).to_numpy(), *args
            ),
            compare_to_baseline("Relative (%/yr) CO2 trend", relative_trend_composite(df).to_numpy(), *args),
            compare_to_baseline(
                "Global-median imputation", global_median_imputation_composite(df).to_numpy(), *args
            ),
        ]
        level_bias = {
            "absolute_slope": float(stats.spearmanr(df["co2_per_capita"], df["decarbonization_trajectory_score"]).statistic),
        }
        trend = df["co2pc_trend_pct"].fillna(df["co2pc_trend_pct"].median())
        level_bias["relative_trend"] = float(stats.spearmanr(df["co2_per_capita"], -trend).statistic)
        return {
            "weight_scenarios": weight_rows,
            "method_alternatives": method_rows,
            # Spearman(co2_per_capita, decarbonization score): >0 means high
            # emitters get systematically better trend scores.
            "trend_level_bias": level_bias,
        }

    @staticmethod
    def _weight_monte_carlo(df, base_scores, base_tiers):
        rng = np.random.default_rng(C.RANDOM_STATE)
        base_w = np.array([C.DIMENSION_WEIGHTS[d] for d in C.DIMENSIONS])
        draws = base_w * rng.uniform(1 - C.WEIGHT_DRAW_SPREAD, 1 + C.WEIGHT_DRAW_SPREAD, (C.N_WEIGHT_DRAWS, len(base_w)))
        draws /= draws.sum(axis=1, keepdims=True)

        scores = df[DIMENSION_SCORE_COLS].to_numpy() @ draws.T        # countries x draws
        ranks = rank_desc(scores)
        tiers = quartile_tiers(scores)
        spearman = [stats.spearmanr(scores[:, j], base_scores).statistic for j in range(scores.shape[1])]

        # Named weight_* on purpose: these bands describe sensitivity to the assumed
        # +/-WEIGHT_DRAW_SPREAD weight variation only - not a confidence interval, and
        # they ignore imputation uncertainty (see _imputation_sensitivity).
        p05, p50, p95 = np.percentile(ranks, [5, 50, 95], axis=1)
        tier_stability = (tiers == base_tiers[:, None]).mean(axis=1)
        columns = {
            "weight_rank_p05": p05.round().astype(int),
            "weight_rank_median": p50.round().astype(int),
            "weight_rank_p95": p95.round().astype(int),
            "weight_tier_stability": tier_stability,
        }
        band = p95 - p05
        summary = {
            "n_draws": C.N_WEIGHT_DRAWS,
            "weight_spread": C.WEIGHT_DRAW_SPREAD,
            "mean_spearman": float(np.mean(spearman)),
            "min_spearman": float(np.min(spearman)),
            "median_rank_band_width": float(np.median(band)),
            "max_rank_band_width": int(band.max()),
            "pct_countries_tier_stable_80": float((tier_stability >= 0.8).mean() * 100),
            "pct_countries_tier_stable_95": float((tier_stability >= 0.95).mean() * 100),
        }
        return summary, columns

    @staticmethod
    def _imputation_sensitivity(df, normalizer, base_scores):
        """For each country with an imputed value, re-score it with that value set to
        the lowest and the highest observed value in the peer group it was imputed from
        (every low/high combination if several values were imputed). The composite is
        monotonic in each indicator, so these corners bound the country's score. Other
        countries' scores and the fitted reference distribution are held fixed
        (one-at-a-time approximation)."""
        indicators = list(C.INDICATOR_DIRECTIONS)
        rounded = np.round(base_scores, 9)
        columns = {name: pd.Series(pd.NA, index=df.index, dtype=dtype) for name, dtype in [
            ("impute_score_low", "Float64"), ("impute_score_high", "Float64"),
            ("impute_rank_best", "Int64"), ("impute_rank_worst", "Int64"),
            ("impute_tier_low", "object"), ("impute_tier_high", "object"),
        ]}
        records = []
        for pos, idx in enumerate(df.index):
            which = [c for c in indicators if bool(df.at[idx, f"{c}_imputed"])]
            if not which:
                continue
            ranges = [(df.at[idx, f"{c}_impute_min"], df.at[idx, f"{c}_impute_max"]) for c in which]
            base_row = df.loc[idx, indicators].to_dict()
            variants = pd.DataFrame([base_row | dict(zip(which, corner)) for corner in itertools.product(*ranges)])
            scores = composite_score(add_dimension_scores(normalizer.transform(variants))).to_numpy()
            low, high = float(scores.min()), float(scores.max())
            others = np.delete(rounded, pos)
            rank_best, rank_worst = int((others > round(high, 9)).sum() + 1), int((others > round(low, 9)).sum() + 1)
            tier_low, tier_high = assign_tiers(pd.Series([low, high]), normalizer.tier_bin_edges).astype(str)

            for name, value in [("impute_score_low", low), ("impute_score_high", high), ("impute_rank_best", rank_best),
                                ("impute_rank_worst", rank_worst), ("impute_tier_low", tier_low), ("impute_tier_high", tier_high)]:
                columns[name][idx] = value
            records.append({
                "country": df.at[idx, "country"],
                "imputed": {c: {"value": float(df.at[idx, c]), "source": df.at[idx, f"{c}_impute_source"],
                                "peer_range": [float(lo), float(hi)]} for c, (lo, hi) in zip(which, ranges)},
                "baseline_rank": int(df.at[idx, "rank"]),
                "rank_range": [rank_best, rank_worst],
                "score_range": [low, high],
                "baseline_tier": df.at[idx, "tier"],
                "tier_range": [tier_low, tier_high],
            })

        widths = [r["rank_range"][1] - r["rank_range"][0] for r in records]
        summary = {
            "method": "imputed value set to the min and max observed value of its imputation peer group",
            "n_countries": len(records),
            "n_tier_can_change": sum(r["tier_range"][0] != r["tier_range"][1] for r in records),
            "median_rank_range_width": float(np.median(widths)) if widths else 0.0,
            "countries": sorted(records, key=lambda r: r["baseline_rank"]),
        }
        return summary, columns

    @staticmethod
    def _permutation_baseline(base_tiers, clusters, scores):
        """Chance exact / within-one agreement when segment membership is random.
        Shuffles the labels (keeping segment sizes), then re-numbers the shuffled
        segments by ascending mean index score - the same ordering step model_trainer
        applies - so the baseline includes the agreement that ordering alone creates."""
        rng = np.random.default_rng(C.RANDOM_STATE)
        n_segments = clusters.max() + 1
        exact, within_one = np.empty(C.N_PERMUTATIONS), np.empty(C.N_PERMUTATIONS)
        for i in range(C.N_PERMUTATIONS):
            shuffled = rng.permutation(clusters)
            means = np.array([scores[shuffled == k].mean() for k in range(n_segments)])
            ordered = np.argsort(np.argsort(means))[shuffled]
            gap = np.abs(base_tiers - ordered)
            exact[i], within_one[i] = (gap == 0).mean(), (gap <= 1).mean()
        return exact * 100, within_one * 100

    @classmethod
    def _tier_cluster_agreement(cls, df, base_tiers):
        clusters = df["cluster_id"].to_numpy()
        null_exact, null_within = cls._permutation_baseline(base_tiers, clusters, df["netzero_index_score"].to_numpy())

        profiles = (
            df.groupby("cluster_label")[DIMENSION_SCORE_COLS + ["netzero_index_score"]]
            .mean()
            .round(1)
            .assign(n_countries=df["cluster_label"].value_counts())
        )
        profiles["dominant_tier"] = df.groupby("cluster_label")["tier"].agg(lambda s: s.value_counts().idxmax())

        gap = np.abs(base_tiers - clusters)
        disagreements = df.loc[gap >= 2, ["country", "tier", "cluster_label", "netzero_index_score"] + DIMENSION_SCORE_COLS]
        exact, within_one = (gap == 0).mean() * 100, (gap <= 1).mean() * 100

        return {
            # Label-independent measures first: they need no segment ordering.
            "adjusted_rand_index": float(adjusted_rand_score(base_tiers, clusters)),
            "normalized_mutual_info": float(normalized_mutual_info_score(base_tiers, clusters)),
            # Position-based measures depend on numbering segments by mean score, so
            # their chance baseline repeats that step (see _permutation_baseline).
            "exact_agreement_pct": float(exact),
            "within_one_step_pct": float(within_one),
            "permutation_baseline": {
                "n_permutations": C.N_PERMUTATIONS,
                "exact_agreement_mean_pct": float(null_exact.mean()),
                "exact_agreement_p95_pct": float(np.percentile(null_exact, 95)),
                "exact_agreement_p_value": float(((null_exact >= exact).sum() + 1) / (C.N_PERMUTATIONS + 1)),
                "within_one_step_mean_pct": float(null_within.mean()),
                "within_one_step_p_value": float(((null_within >= within_one).sum() + 1) / (C.N_PERMUTATIONS + 1)),
            },
            "spearman_tier_vs_cluster": float(stats.spearmanr(base_tiers, clusters).statistic),
            "cluster_profiles": profiles.reset_index().to_dict(orient="records"),
            "strong_disagreements": disagreements.round(1).to_dict(orient="records"),
        }

    @staticmethod
    def _cluster_stability(df, model_bundle):
        """Refit KMeans on random subsamples and measure (ARI) how well each
        refit reproduces the full-data clustering, for every candidate k."""
        X = model_bundle["scaler"].transform(df[FEATURE_COLS])
        n_sub = int(len(X) * C.BOOTSTRAP_FRACTION)
        rng = np.random.default_rng(C.RANDOM_STATE)
        subsamples = [rng.choice(len(X), size=n_sub, replace=False) for _ in range(C.N_BOOTSTRAP)]

        by_k = []
        for k in C.K_RANGE:
            if k == C.N_CLUSTERS_DEFAULT:
                full_labels = model_bundle["kmeans"].predict(X)
            else:
                full_labels = KMeans(n_clusters=k, n_init=10, random_state=C.RANDOM_STATE).fit_predict(X)
            aris = [
                adjusted_rand_score(
                    full_labels,
                    KMeans(n_clusters=k, n_init=10, random_state=C.RANDOM_STATE + i).fit(X[idx]).predict(X),
                )
                for i, idx in enumerate(subsamples)
            ]
            by_k.append(
                {"k": k, "mean_ari": float(np.mean(aris)), "p05_ari": float(np.percentile(aris, 5)), "min_ari": float(np.min(aris))}
            )

        selected = next(row for row in by_k if row["k"] == C.N_CLUSTERS_DEFAULT)
        return {
            "n_bootstrap": C.N_BOOTSTRAP,
            "subsample_fraction": C.BOOTSTRAP_FRACTION,
            "by_k": by_k,
            "selected_k": selected,
        }
