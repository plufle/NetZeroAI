import sys

import pandas as pd

from src import constants as C
from src.components.data_transformation import add_dimension_scores, assign_tiers, composite_score
from src.exception import CustomException
from src.logger import logging
from src.utils import indicator_bound_violations, load_object

FEATURE_COLS = [f"{ind}_norm" for ind in C.INDICATOR_DIRECTIONS]


class InvalidIndicatorInput(ValueError):
    """Raw indicator values that are missing, non-finite or physically impossible."""


class CustomCountryData:
    """Packages raw indicator values for a single country - existing or
    hypothetical/what-if - into the shape the trained pipeline expects."""

    def __init__(
        self,
        co2_per_capita,
        ghg_per_capita,
        co2pc_trend_slope,
        renewables_share_elec,
        gdp_per_capita,
        country="Custom Country",
    ):
        self.data = {
            "country": country,
            "co2_per_capita": co2_per_capita,
            "ghg_per_capita": ghg_per_capita,
            "co2pc_trend_slope": co2pc_trend_slope,
            "renewables_share_elec": renewables_share_elec,
            "gdp_per_capita": gdp_per_capita,
        }

    def to_dataframe(self):
        return pd.DataFrame([self.data])


class PredictPipeline:
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        # Validated outside the try block so callers get InvalidIndicatorInput /
        # FileNotFoundError with a readable message rather than a wrapped CustomException.
        problems = indicator_bound_violations(df)
        if problems:
            raise InvalidIndicatorInput("Invalid indicator values - " + "; ".join(problems))
        missing = [p.name for p in (C.PREPROCESSOR_PATH, C.MODEL_PATH) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing fitted artifacts {missing} in {C.ARTIFACTS_DIR}. "
                "Build them with: python -m src.pipeline.train_pipeline"
            )

        try:
            normalizer = load_object(C.PREPROCESSOR_PATH)
            model_bundle = load_object(C.MODEL_PATH)

            df = df.copy()
            df[list(C.INDICATOR_DIRECTIONS)] = df[list(C.INDICATOR_DIRECTIONS)].astype(float)
            df = add_dimension_scores(normalizer.transform(df))
            df["netzero_index_score"] = composite_score(df)
            df["tier"] = assign_tiers(df["netzero_index_score"], normalizer.tier_bin_edges)

            X = model_bundle["scaler"].transform(df[FEATURE_COLS])
            pca_coords = model_bundle["pca"].transform(X)
            df["pca_1"], df["pca_2"] = pca_coords[:, 0], pca_coords[:, 1]

            raw_cluster = model_bundle["kmeans"].predict(X)
            df["cluster_id"] = [model_bundle["relabel"][c] for c in raw_cluster]
            df["cluster_label"] = df["cluster_id"].apply(lambda i: f"ML Segment {i + 1}")

            return df

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    logging.info("Running predict_pipeline smoke test")
    sample = CustomCountryData(
        co2_per_capita=5.0,
        ghg_per_capita=6.0,
        co2pc_trend_slope=-0.1,
        renewables_share_elec=60.0,
        gdp_per_capita=25000.0,
        country="Hypothetical Country",
    ).to_dataframe()
    result = PredictPipeline().predict(sample)
    print(result[["country", "netzero_index_score", "tier", "cluster_label"]].to_string(index=False))
