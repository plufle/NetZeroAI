from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_CO2_PATH = ROOT_DIR / "data" / "raw" / "owid_co2_data.csv"
RAW_ENERGY_PATH = ROOT_DIR / "data" / "raw" / "owid_energy_data.csv"
# UN M49 regions, via github.com/lukes/ISO-3166-Countries-with-Regional-Codes
# (all/all.csv @ 145f1ad). Neither OWID file has a region column.
RAW_REGIONS_PATH = ROOT_DIR / "data" / "raw" / "un_m49_regions.csv"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATA_PATH = ARTIFACTS_DIR / "data.csv"
TRAIN_PATH = ARTIFACTS_DIR / "train.csv"
TEST_PATH = ARTIFACTS_DIR / "test.csv"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
FINAL_TABLE_PATH = ARTIFACTS_DIR / "netzero_indicators_2022.csv"
MANIFEST_PATH = ARTIFACTS_DIR / "pipeline_manifest.json"

# Bump whenever the artifact schema changes; app.py rebuilds stale artifacts.
PIPELINE_VERSION = 3

REFERENCE_YEAR = 2022          # 2023 has zero gdp data in both raw datasets
TREND_START_YEAR = 2013
TREND_END_YEAR = 2022
MIN_TREND_YEARS = 8            # need >=8 of 10 non-null years to fit a trend slope

REAL_ISO_REGEX = r"^[A-Z]{3}$"
EXCLUDE_ISO = {"ATA"}           # Antarctica passes the ISO3 regex but isn't a country
MAX_MISSING_INDICATORS = 1      # drop a country missing more than 1 of the 5 indicators

# Missing values are filled with the median of the most specific UN M49 group
# that has at least MIN_IMPUTATION_GROUP_OBS observed values, falling back to
# the global cohort median. Ordered most -> least specific.
IMPUTATION_GROUP_LEVELS = ["intermediate_region", "sub_region", "region"]
MIN_IMPUTATION_GROUP_OBS = 5
# UN M49 leaves Taiwan's region blank; assigned by geography.
REGION_OVERRIDES = {"TWN": {"region": "Asia", "sub_region": "Eastern Asia"}}

INDICATOR_DIRECTIONS = {
    "co2_per_capita": "lower_better",
    "ghg_per_capita": "lower_better",
    "co2pc_trend_slope": "lower_better",        # more negative = faster decarbonization
    "renewables_share_elec": "higher_better",
    "gdp_per_capita": "higher_better",
}

# Physically valid ranges (inclusive; None = unbounded). Enforced on the ingested 2022 data
# (a violation fails the build) and on what-if inputs. Every value must also be finite.
# ghg_per_capita has no lower bound: OWID's GHG series includes land use, so net forest
# sinks make it legitimately negative (488 country-years in the raw file). The trend slope
# can be any finite number.
INDICATOR_BOUNDS = {
    "co2_per_capita": {"min": 0.0, "max": None},
    "ghg_per_capita": {"min": None, "max": None},
    "co2pc_trend_slope": {"min": None, "max": None},
    "renewables_share_elec": {"min": 0.0, "max": 100.0},
    "gdp_per_capita": {"min": 1.0, "max": None},       # $1 floor: must be positive
}

DIMENSIONS = {
    "emissions_intensity": ["co2_per_capita", "ghg_per_capita"],
    "decarbonization_trajectory": ["co2pc_trend_slope"],
    "energy_transition": ["renewables_share_elec"],
    "economic_capacity": ["gdp_per_capita"],
}

DIMENSION_WEIGHTS = {k: 0.25 for k in DIMENSIONS}

N_CLUSTERS_DEFAULT = 4
K_RANGE = range(2, 8)
RANDOM_STATE = 42
TIER_LABELS = ["Lagging", "Emerging", "Advancing", "Leader"]

# Weight-uncertainty Monte Carlo: each dimension weight is drawn uniformly
# from [1 - SPREAD, 1 + SPREAD] x its baseline weight, then renormalized.
N_WEIGHT_DRAWS = 1000
WEIGHT_DRAW_SPREAD = 0.5
# KMeans stability: refit on N_BOOTSTRAP random subsamples of this fraction.
N_BOOTSTRAP = 50
BOOTSTRAP_FRACTION = 0.8
# Chance baseline for tier/segment agreement: shuffle segment labels, then re-order the
# shuffled segments by mean index score exactly as model_trainer does.
N_PERMUTATIONS = 1000

# Curated set used as the "test" fixture: not a supervised holdout (there's no
# label to predict out-of-sample here), but a fixed, human-checkable set of
# countries used to sanity-check that the pipeline produces plausible results
# (see verification section in CLAUDE.md / the frozen plan).
ANCHOR_COUNTRIES = [
    "Norway", "Iceland", "Sweden", "France", "Germany",
    "United States", "China", "India", "Nigeria", "Ethiopia",
    "Qatar", "Saudi Arabia",
]
