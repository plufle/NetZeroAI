from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_CO2_PATH = ROOT_DIR / "data" / "raw" / "owid_co2_data.csv"
RAW_ENERGY_PATH = ROOT_DIR / "data" / "raw" / "owid_energy_data.csv"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATA_PATH = ARTIFACTS_DIR / "data.csv"
TRAIN_PATH = ARTIFACTS_DIR / "train.csv"
TEST_PATH = ARTIFACTS_DIR / "test.csv"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
FINAL_TABLE_PATH = ARTIFACTS_DIR / "netzero_indicators_2022.csv"
MANIFEST_PATH = ARTIFACTS_DIR / "pipeline_manifest.json"

REFERENCE_YEAR = 2022          # 2023 has zero gdp data in both raw datasets
TREND_START_YEAR = 2013
TREND_END_YEAR = 2022
MIN_TREND_YEARS = 8            # need >=8 of 10 non-null years to fit a trend slope

REAL_ISO_REGEX = r"^[A-Z]{3}$"
EXCLUDE_ISO = {"ATA"}           # Antarctica passes the ISO3 regex but isn't a country
MAX_MISSING_INDICATORS = 1      # drop a country missing more than 1 of the 5 indicators

INDICATOR_DIRECTIONS = {
    "co2_per_capita": "lower_better",
    "ghg_per_capita": "lower_better",
    "co2pc_trend_slope": "lower_better",        # more negative = faster decarbonization
    "renewables_share_elec": "higher_better",
    "gdp_per_capita": "higher_better",
}

DIMENSIONS = {
    "emissions_intensity": ["co2_per_capita", "ghg_per_capita"],
    "decarbonization_trajectory": ["co2pc_trend_slope"],
    "energy_transition": ["renewables_share_elec"],
    "economic_capacity": ["gdp_per_capita"],
}

DIMENSION_WEIGHTS = {k: 0.25 for k in DIMENSIONS}

N_CLUSTERS_DEFAULT = 4
RANDOM_STATE = 42
TIER_LABELS = ["Lagging", "Emerging", "Advancing", "Leader"]

# Curated set used as the "test" fixture: not a supervised holdout (there's no
# label to predict out-of-sample here), but a fixed, human-checkable set of
# countries used to sanity-check that the pipeline produces plausible results
# (see verification section in CLAUDE.md / the frozen plan).
ANCHOR_COUNTRIES = [
    "Norway", "Iceland", "Sweden", "France", "Germany",
    "United States", "China", "India", "Nigeria", "Ethiopia",
    "Qatar", "Saudi Arabia",
]
