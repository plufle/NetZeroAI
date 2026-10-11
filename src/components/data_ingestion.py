import sys

import pandas as pd
from scipy import stats

from src import constants as C
from src.exception import CustomException
from src.logger import logging
from src.utils import indicator_bound_violations

REQUIRED_COLUMNS = {
    "owid_co2": ["iso_code", "country", "year", "co2_per_capita", "ghg_per_capita", "gdp", "population"],
    "owid_energy": ["iso_code", "country", "year", "renewables_share_elec"],
    "un_m49_regions": ["alpha-3", "region", "sub-region", "intermediate-region"],
}


class DataAuditError(ValueError):
    """A raw input violates an assumption the pipeline relies on."""


class DataIngestion:
    """Loads and audits the raw inputs and builds the analysis table. After a run,
    `self.audit` holds the data-audit summary that train_pipeline writes to the manifest."""

    def __init__(self):
        self.audit = {}

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion")
        try:
            co2 = pd.read_csv(C.RAW_CO2_PATH)
            energy = pd.read_csv(C.RAW_ENERGY_PATH)
            regions_raw = pd.read_csv(C.RAW_REGIONS_PATH, keep_default_na=False)
            for name, df in [("owid_co2", co2), ("owid_energy", energy), ("un_m49_regions", regions_raw)]:
                self._require_columns(name, df)

            co2_real = self._filter_real_countries(co2)
            energy_real = self._filter_real_countries(energy)
            for name, df in [("owid_co2", co2_real), ("owid_energy", energy_real)]:
                self._require_unique(name, df, ["iso_code", "year"])
            self._require_unique("un_m49_regions", regions_raw, ["alpha-3"])

            co2_year = co2_real[co2_real["year"] == C.REFERENCE_YEAR]
            energy_year = energy_real[energy_real["year"] == C.REFERENCE_YEAR]

            merged = co2_year[
                ["iso_code", "country", "co2_per_capita", "ghg_per_capita", "gdp", "population"]
            ].merge(
                energy_year[["iso_code", "renewables_share_elec"]], on="iso_code", how="inner", validate="one_to_one"
            )
            merged["gdp_per_capita"] = merged["gdp"] / merged["population"]

            slopes = self._compute_trend_slopes(co2_real, set(merged["iso_code"]))
            merged = merged.merge(slopes, on="iso_code", how="left", validate="one_to_one")
            merged = merged.merge(self._load_regions(regions_raw), on="iso_code", how="left", validate="one_to_one")
            unmapped = merged.loc[merged["region"].isna(), "country"].tolist()
            if unmapped:
                logging.warning(f"No UN M49 region for {unmapped}; they will use the global median if imputed")

            indicator_cols = list(C.INDICATOR_DIRECTIONS)
            violations = indicator_bound_violations(merged, allow_missing=True)
            if violations:
                raise DataAuditError(
                    f"{len(violations)} indicator value(s) at {C.REFERENCE_YEAR} are non-finite or out of range "
                    f"(see INDICATOR_BOUNDS in src/constants.py): " + "; ".join(violations[:10])
                )

            C.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            merged.to_csv(C.DATA_PATH, index=False)
            logging.info(f"Saved raw merged indicator table ({len(merged)} countries) to {C.DATA_PATH}")

            missing = merged[indicator_cols].isna()
            keep = missing.sum(axis=1) <= C.MAX_MISSING_INDICATORS
            retained = merged[keep].copy()
            dropped = [
                {"iso_code": row["iso_code"], "country": row["country"],
                 "missing_indicators": [c for c in indicator_cols if missing.at[idx, c]]}
                for idx, row in merged[~keep].iterrows()
            ]
            logging.info(f"Dropped {len(dropped)} countries missing >{C.MAX_MISSING_INDICATORS} indicators: {dropped}")

            for col in indicator_cols:
                retained[f"{col}_imputed"] = retained[col].isna()
                filled = impute_by_region(retained, col, C.IMPUTATION_GROUP_LEVELS, C.MIN_IMPUTATION_GROUP_OBS)
                retained[col] = filled["value"]
                retained[f"{col}_impute_source"] = filled["source"]
                retained[f"{col}_impute_min"] = filled["peer_min"]
                retained[f"{col}_impute_max"] = filled["peer_max"]

            retained.to_csv(C.TRAIN_PATH, index=False)
            logging.info(f"Saved final analysis-ready table ({len(retained)} countries) to {C.TRAIN_PATH}")

            test_df = retained[retained["country"].isin(C.ANCHOR_COUNTRIES)].copy()
            test_df.to_csv(C.TEST_PATH, index=False)
            logging.info(f"Saved {len(test_df)}-country anchor verification set to {C.TEST_PATH}")

            def label(df, iso_codes):
                names = df.drop_duplicates("iso_code").set_index("iso_code")["country"]
                return [f"{iso} ({names[iso]})" for iso in sorted(iso_codes)]

            self.audit = {
                "checks_passed": [
                    "required columns present in every raw file",
                    "no duplicate (iso_code, year) rows among real countries",
                    "UN M49 alpha-3 codes unique",
                    "CO2/energy, trend and region merges are one-to-one",
                    "indicator values finite and within INDICATOR_BOUNDS",
                ],
                "raw_rows": {"owid_co2": len(co2), "owid_energy": len(energy), "un_m49_regions": len(regions_raw)},
                "real_countries_at_reference_year": {
                    "owid_co2": int(co2_year["iso_code"].nunique()),
                    "owid_energy": int(energy_year["iso_code"].nunique()),
                    "merged": len(merged),
                },
                "unmatched_in_merge": {
                    "owid_co2_only": label(co2_year, set(co2_year["iso_code"]) - set(energy_year["iso_code"])),
                    "owid_energy_only": label(energy_year, set(energy_year["iso_code"]) - set(co2_year["iso_code"])),
                },
                "missing_before_imputation": {c: int(missing[c].sum()) for c in indicator_cols},
                "indicator_ranges": {
                    c: {"min": float(merged[c].min()), "max": float(merged[c].max()), "non_missing": int(merged[c].notna().sum())}
                    for c in indicator_cols
                },
                "dropped_countries": dropped,
                "unmapped_regions": unmapped,
            }
            return C.TRAIN_PATH, C.TEST_PATH

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _require_columns(name, df):
        absent = [c for c in REQUIRED_COLUMNS[name] if c not in df.columns]
        if absent:
            raise DataAuditError(f"{name}: required column(s) missing: {absent}")

    @staticmethod
    def _require_unique(name, df, keys):
        dupes = df[df.duplicated(keys, keep=False)]
        if not dupes.empty:
            examples = dupes[keys].drop_duplicates().head(5).to_dict(orient="records")
            raise DataAuditError(f"{name}: {len(dupes)} rows share a {tuple(keys)} key, e.g. {examples}")

    @staticmethod
    def _load_regions(regions_raw):
        regions = regions_raw.rename(
            columns={
                "alpha-3": "iso_code",
                "sub-region": "sub_region",
                "intermediate-region": "intermediate_region",
            }
        )[["iso_code", "region", "sub_region", "intermediate_region"]]
        regions = regions.replace("", pd.NA)
        for iso, fields in C.REGION_OVERRIDES.items():
            for field, value in fields.items():
                regions.loc[regions["iso_code"] == iso, field] = value
        return regions

    @staticmethod
    def _filter_real_countries(df):
        real = df[df["iso_code"].astype(str).str.match(C.REAL_ISO_REGEX, na=False)]
        return real[~real["iso_code"].isin(C.EXCLUDE_ISO)]

    @staticmethod
    def _compute_trend_slopes(co2_real, iso_codes):
        window = co2_real[co2_real["year"].between(C.TREND_START_YEAR, C.TREND_END_YEAR)]
        rows = []
        for iso in sorted(iso_codes):
            sub = window.loc[window["iso_code"] == iso, ["year", "co2_per_capita"]].dropna()
            slope = slope_pct = None
            if len(sub) >= C.MIN_TREND_YEARS:
                slope = stats.linregress(sub["year"], sub["co2_per_capita"]).slope
                # Relative (%/yr) version: not an index input, only used by the
                # sensitivity analysis to test the absolute slope for level bias.
                mean_level = sub["co2_per_capita"].mean()
                slope_pct = slope / mean_level * 100 if mean_level > 0 else None
            rows.append({"iso_code": iso, "co2pc_trend_slope": slope, "co2pc_trend_pct": slope_pct})
        return pd.DataFrame(rows)


def impute_by_region(df, col, group_levels, min_obs):
    """Fill missing `col` values with the median of the most specific group
    (in `group_levels` order) that has at least `min_obs` observed values,
    falling back to the global median. Medians use observed values only, never
    earlier imputations.

    Returns a DataFrame aligned with `df`: `value` (filled), `source` (human-readable
    group), and `peer_min`/`peer_max` - the observed range of the group the value
    came from, used by the imputation-sensitivity check (NaN where not imputed)."""
    out = pd.DataFrame({"value": df[col], "source": pd.Series(pd.NA, index=df.index, dtype="object"),
                        "peer_min": float("nan"), "peer_max": float("nan")})
    observed = df[col].notna()
    for idx in df.index[~observed]:
        for level in group_levels:
            group = df.at[idx, level]
            if pd.isna(group):
                continue
            peers = df.loc[observed & (df[level] == group), col]
            if len(peers) >= min_obs:
                source = f"{group} median (n={len(peers)})"
                break
        else:
            peers = df.loc[observed, col]
            source = f"global median (n={len(peers)})"
        out.loc[idx, ["value", "source", "peer_min", "peer_max"]] = [peers.median(), source, peers.min(), peers.max()]
    out["value"] = out["value"].astype(float)
    return out


if __name__ == "__main__":
    DataIngestion().initiate_data_ingestion()
