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


class DataTransformation:
    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Starting data transformation")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            normalizer = PercentileNormalizer(C.INDICATOR_DIRECTIONS)
            normalizer.fit(train_df)

            train_df = normalizer.transform(train_df)
            test_df = normalizer.transform(test_df)

            for name, cols in C.DIMENSIONS.items():
                norm_cols = [f"{c}_norm" for c in cols]
                train_df[f"{name}_score"] = train_df[norm_cols].mean(axis=1)
                test_df[f"{name}_score"] = test_df[norm_cols].mean(axis=1)

            train_df["netzero_index_score"] = sum(
                train_df[f"{d}_score"] * w for d, w in C.DIMENSION_WEIGHTS.items()
            )
            test_df["netzero_index_score"] = sum(
                test_df[f"{d}_score"] * w for d, w in C.DIMENSION_WEIGHTS.items()
            )

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
