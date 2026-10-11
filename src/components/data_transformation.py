import sys

import numpy as np
import pandas as pd

from src import constants as C
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


class PercentileNormalizer:
    """Fits a reference distribution per indicator from a cohort, then
    percentile-ranks any row (including a brand-new country, via
    predict_pipeline) against that fixed reference. Chosen over min-max
    because the indicators are heavy-tailed (e.g. co2_per_capita spans
    0.06-37.9, gdp_per_capita spans ~$689-$131,991) and min-max would let a
    single outlier compress everyone else's score toward 0."""

    def __init__(self, directions):
        self.directions = directions
        self.reference_ = {}
        self.tier_bin_edges = None

    def fit(self, df):
        for col in self.directions:
            self.reference_[col] = np.sort(df[col].dropna().values)
        return self

    def transform(self, df):
        out = df.copy()
        for col, direction in self.directions.items():
            ref = self.reference_[col]
            pct = np.searchsorted(ref, out[col].values, side="right") / len(ref) * 100
            if direction == "lower_better":
                pct = 100 - pct
            out[f"{col}_norm"] = pct
        return out


def add_dimension_scores(df):
    """Average each dimension's normalized indicators into `<dimension>_score`."""
    df = df.copy()
    for name, cols in C.DIMENSIONS.items():
        df[f"{name}_score"] = df[[f"{c}_norm" for c in cols]].mean(axis=1)
    return df


def composite_score(df, weights=None):
    """Weighted sum of dimension scores. `weights` defaults to the baseline
    DIMENSION_WEIGHTS and is renormalized to sum to 1."""
    weights = weights or C.DIMENSION_WEIGHTS
    total = sum(weights.values())
    return sum(df[f"{d}_score"] * (w / total) for d, w in weights.items())


def assign_tiers(scores, bin_edges):
    """Tier labels for scores using the training quartile edges. The outer bins are
    opened to +/-inf so a new country scoring beyond the cohort's min/max still gets
    the bottom/top tier."""
    bins = np.array(bin_edges, dtype=float)
    bins[0], bins[-1] = -np.inf, np.inf
    return pd.cut(scores, bins=bins, labels=C.TIER_LABELS)


class DataTransformation:
    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Starting data transformation")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            normalizer = PercentileNormalizer(C.INDICATOR_DIRECTIONS)
            normalizer.fit(train_df)

            train_df = add_dimension_scores(normalizer.transform(train_df))
            test_df = add_dimension_scores(normalizer.transform(test_df))
            train_df["netzero_index_score"] = composite_score(train_df)
            test_df["netzero_index_score"] = composite_score(test_df)

            train_tiers, bin_edges = pd.qcut(
                train_df["netzero_index_score"], q=4, labels=C.TIER_LABELS, retbins=True
            )
            train_df["tier"] = train_tiers
            # The 12-country anchor set is too small to qcut on its own, so
            # its members are bucketed into the TRAIN set's quantile bins.
            test_df["tier"] = pd.cut(
                test_df["netzero_index_score"], bins=bin_edges, labels=C.TIER_LABELS, include_lowest=True
            )

            normalizer.tier_bin_edges = bin_edges
            save_object(C.PREPROCESSOR_PATH, normalizer)
            logging.info(f"Saved fitted normalizer (+ tier bin edges) to {C.PREPROCESSOR_PATH}")

            return train_df, test_df

        except Exception as e:
            raise CustomException(e, sys)
