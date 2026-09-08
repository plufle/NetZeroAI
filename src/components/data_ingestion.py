import sys

import pandas as pd
from scipy import stats

from src import constants as C
from src.exception import CustomException
from src.logger import logging


class DataIngestion:
    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion")
        try:
            co2 = pd.read_csv(C.RAW_CO2_PATH)
            energy = pd.read_csv(C.RAW_ENERGY_PATH)

            co2_real = self._filter_real_countries(co2)
            energy_real = self._filter_real_countries(energy)

            co2_year = co2_real[co2_real["year"] == C.REFERENCE_YEAR]
            energy_year = energy_real[energy_real["year"] == C.REFERENCE_YEAR]

            merged = co2_year[
                ["iso_code", "country", "co2_per_capita", "ghg_per_capita", "gdp", "population"]
            ].merge(
                energy_year[["iso_code", "renewables_share_elec"]], on="iso_code", how="inner"
            )
            merged["gdp_per_capita"] = merged["gdp"] / merged["population"]

            slopes = self._compute_trend_slopes(co2_real, set(merged["iso_code"]))
            merged = merged.merge(slopes, on="iso_code", how="left")

            C.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            merged.to_csv(C.DATA_PATH, index=False)
            logging.info(f"Saved raw merged indicator table ({len(merged)} countries) to {C.DATA_PATH}")

            indicator_cols = list(C.INDICATOR_DIRECTIONS)
            n_missing = merged[indicator_cols].isna().sum(axis=1)
            retained = merged[n_missing <= C.MAX_MISSING_INDICATORS].copy()
            dropped = merged.loc[n_missing > C.MAX_MISSING_INDICATORS, "country"].tolist()
            logging.info(f"Dropped {len(dropped)} countries missing >{C.MAX_MISSING_INDICATORS} indicators: {dropped}")

            for col in indicator_cols:
                is_na = retained[col].isna()
                retained[f"{col}_imputed"] = is_na
                if is_na.any():
                    retained.loc[is_na, col] = retained[col].median()

            retained.to_csv(C.TRAIN_PATH, index=False)
            logging.info(f"Saved final analysis-ready table ({len(retained)} countries) to {C.TRAIN_PATH}")

            test_df = retained[retained["country"].isin(C.ANCHOR_COUNTRIES)].copy()
            test_df.to_csv(C.TEST_PATH, index=False)
            logging.info(f"Saved {len(test_df)}-country anchor verification set to {C.TEST_PATH}")

            return C.TRAIN_PATH, C.TEST_PATH

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _filter_real_countries(df):
        real = df[df["iso_code"].astype(str).str.match(C.REAL_ISO_REGEX, na=False)]
        return real[~real["iso_code"].isin(C.EXCLUDE_ISO)]

    @staticmethod
    def _compute_trend_slopes(co2_real, iso_codes):
        window = co2_real[co2_real["year"].between(C.TREND_START_YEAR, C.TREND_END_YEAR)]
        rows = []
        for iso in iso_codes:
            sub = window.loc[window["iso_code"] == iso, ["year", "co2_per_capita"]].dropna()
            slope = None
            if len(sub) >= C.MIN_TREND_YEARS:
                slope = stats.linregress(sub["year"], sub["co2_per_capita"]).slope
            rows.append({"iso_code": iso, "co2pc_trend_slope": slope})
        return pd.DataFrame(rows)


if __name__ == "__main__":
    DataIngestion().initiate_data_ingestion()
