"""Unit tests for the scoring, imputation and validation helpers (no pipeline run)."""
import numpy as np
import pandas as pd
import pytest

from src import constants as C
from src.components.data_ingestion import DataIngestion, impute_by_region
from src.components.data_transformation import PercentileNormalizer, add_dimension_scores, composite_score
from src.components.index_validation import IndexValidation, quartile_tiers, rank_desc
from src.fingerprint import build_fingerprint
from src.utils import indicator_bound_violations


class TestPercentileNormalizer:
    def make(self):
        ref = pd.DataFrame({"good": [1.0, 2.0, 3.0, 4.0], "bad": [1.0, 2.0, 3.0, 4.0]})
        return PercentileNormalizer({"good": "higher_better", "bad": "lower_better"}).fit(ref)

    def test_direction(self):
        out = self.make().transform(pd.DataFrame({"good": [1.0, 4.0], "bad": [1.0, 4.0]}))
        assert out["good_norm"].tolist() == [25.0, 100.0]
        assert out["bad_norm"].tolist() == [75.0, 0.0]

    def test_unseen_values_stay_in_range(self):
        out = self.make().transform(pd.DataFrame({"good": [-100.0, 1e9], "bad": [-100.0, 1e9]}))
        assert out["good_norm"].tolist() == [0.0, 100.0]
        assert out["bad_norm"].tolist() == [100.0, 0.0]

    def test_monotonic(self):
        values = np.linspace(0, 5, 50)
        out = self.make().transform(pd.DataFrame({"good": values, "bad": values}))
        assert out["good_norm"].is_monotonic_increasing
        assert out["bad_norm"].is_monotonic_decreasing

    def test_outlier_does_not_compress_others(self):
        # The reason percentile beats min-max: one huge value doesn't squash the rest.
        ref = pd.DataFrame({"x": [1.0, 2.0, 3.0, 1000.0]})
        out = PercentileNormalizer({"x": "higher_better"}).fit(ref).transform(ref)
        assert out["x_norm"].tolist() == [25.0, 50.0, 75.0, 100.0]


class TestComposite:
    def frame(self):
        scores = {f"{d}_score": [10.0 * (i + 1), 80.0] for i, d in enumerate(C.DIMENSIONS)}
        return pd.DataFrame(scores)

    def test_baseline_is_equal_weighted_mean(self):
        df = self.frame()
        expected = df[[f"{d}_score" for d in C.DIMENSIONS]].mean(axis=1)
        assert np.allclose(composite_score(df), expected)

    def test_weights_are_renormalized(self):
        df = self.frame()
        doubled = {d: 2 for d in C.DIMENSIONS}
        assert np.allclose(composite_score(df, doubled), composite_score(df))

    def test_single_dimension_weight(self):
        df = self.frame()
        only_first = {d: (1 if i == 0 else 0) for i, d in enumerate(C.DIMENSIONS)}
        first = list(C.DIMENSIONS)[0]
        assert np.allclose(composite_score(df, only_first), df[f"{first}_score"])

    def test_dimension_scores_average_their_indicators(self):
        row = {f"{ind}_norm": 0.0 for ind in C.INDICATOR_DIRECTIONS}
        row.update({"co2_per_capita_norm": 40.0, "ghg_per_capita_norm": 60.0})
        out = add_dimension_scores(pd.DataFrame([row]))
        assert out["emissions_intensity_score"].iloc[0] == 50.0

    def test_baseline_weights_sum_to_one(self):
        assert sum(C.DIMENSION_WEIGHTS.values()) == pytest.approx(1.0)
        assert set(C.DIMENSION_WEIGHTS) == set(C.DIMENSIONS)


class TestImputeByRegion:
    def frame(self):
        rows = []
        # 5 observed in sub-region A1 (values 10..14), 2 observed in A2, 1 region-only group.
        rows += [{"sub_region": "A1", "region": "A", "x": v} for v in [10, 11, 12, 13, 14]]
        rows += [{"sub_region": "A2", "region": "A", "x": v} for v in [100, 200]]
        rows += [{"sub_region": "B1", "region": "B", "x": 1000}]
        rows += [
            {"sub_region": "A1", "region": "A", "x": np.nan},     # -> A1 median
            {"sub_region": "A2", "region": "A", "x": np.nan},     # A2 too small -> region A median
            {"sub_region": "B1", "region": "B", "x": np.nan},     # B too small -> global median
            {"sub_region": None, "region": None, "x": np.nan},    # unmapped -> global median
        ]
        return pd.DataFrame(rows)

    def impute(self, df):
        return impute_by_region(df, "x", ["sub_region", "region"], min_obs=5)

    def test_hierarchy_and_fallback(self):
        df = self.frame()
        out = self.impute(df)
        observed = df["x"].dropna()
        assert out["value"].iloc[8] == 12                                   # A1 median
        assert out["value"].iloc[9] == pd.Series([10, 11, 12, 13, 14, 100, 200]).median()
        assert out["value"].iloc[10] == observed.median()
        assert out["value"].iloc[11] == observed.median()
        assert out["source"].iloc[8] == "A1 median (n=5)"
        assert out["source"].iloc[10].startswith("global median")

    def test_peer_range_is_the_source_groups_observed_range(self):
        out = self.impute(self.frame())
        assert (out["peer_min"].iloc[8], out["peer_max"].iloc[8]) == (10, 14)        # A1
        assert (out["peer_min"].iloc[9], out["peer_max"].iloc[9]) == (10, 200)       # region A
        assert (out["peer_min"].iloc[10], out["peer_max"].iloc[10]) == (10, 1000)    # global

    def test_observed_values_untouched(self):
        df = self.frame()
        out = self.impute(df)
        observed = df["x"].notna()
        assert out.loc[observed, "value"].equals(df.loc[observed, "x"].astype(float))
        assert out.loc[observed, "source"].isna().all()
        assert out.loc[observed, ["peer_min", "peer_max"]].isna().all().all()

    def test_medians_ignore_earlier_imputations(self):
        # Two missing values in the same group must get the same (observed-only) median.
        df = self.frame()
        df.loc[len(df)] = {"sub_region": "A1", "region": "A", "x": np.nan}
        out = self.impute(df)
        assert out["value"].iloc[8] == out["value"].iloc[-1] == 12


class TestIndicatorBounds:
    def valid(self, **overrides):
        row = {"country": "X", "co2_per_capita": 5.0, "ghg_per_capita": 6.0, "co2pc_trend_slope": -0.1,
               "renewables_share_elec": 40.0, "gdp_per_capita": 20000.0}
        return pd.DataFrame([row | overrides])

    def test_valid_and_boundary_values_pass(self):
        assert indicator_bound_violations(self.valid()) == []
        assert indicator_bound_violations(self.valid(co2_per_capita=0.0, renewables_share_elec=100.0, gdp_per_capita=1.0)) == []
        # GHG includes land use, so net sinks are legitimately negative.
        assert indicator_bound_violations(self.valid(ghg_per_capita=-3.0)) == []

    @pytest.mark.parametrize("field,value", [
        ("co2_per_capita", -1.0), ("renewables_share_elec", 150.0), ("renewables_share_elec", -0.1),
        ("gdp_per_capita", -500.0), ("gdp_per_capita", 0.0), ("co2pc_trend_slope", np.inf),
        ("ghg_per_capita", np.nan), ("co2_per_capita", "abc"),
    ])
    def test_invalid_values_are_reported(self, field, value):
        problems = indicator_bound_violations(self.valid(**{field: value}))
        assert len(problems) == 1 and field in problems[0]

    def test_missing_allowed_only_when_requested(self):
        df = self.valid(gdp_per_capita=np.nan)
        assert indicator_bound_violations(df, allow_missing=True) == []
        assert len(indicator_bound_violations(df)) == 1


class TestPermutationBaseline:
    def test_ordering_step_raises_chance_agreement_above_naive(self):
        # Tiers are score quartiles; random segments re-numbered by mean score still
        # line up with tiers slightly more often than the naive 25%.
        rng = np.random.default_rng(0)
        scores = rng.uniform(0, 100, 200)
        tiers = pd.qcut(scores, 4, labels=False)
        clusters = rng.permutation(np.repeat(np.arange(4), 50))
        exact, within = IndexValidation._permutation_baseline(tiers, clusters, scores)
        assert len(exact) == C.N_PERMUTATIONS
        assert 25 < exact.mean() < 35
        assert within.mean() > exact.mean()


class TestFingerprint:
    def test_config_change_changes_fingerprint(self, monkeypatch):
        before = build_fingerprint()
        monkeypatch.setattr(C, "DIMENSION_WEIGHTS", {**C.DIMENSION_WEIGHTS, "economic_capacity": 0.5})
        after = build_fingerprint()
        assert after["config_sha256"] != before["config_sha256"]
        assert after["fingerprint"] != before["fingerprint"]
        assert after["inputs"] == before["inputs"]

    def test_artifact_paths_do_not_affect_fingerprint(self, monkeypatch, tmp_path):
        before = build_fingerprint()["fingerprint"]
        monkeypatch.setattr(C, "MODEL_PATH", tmp_path / "model.pkl")
        assert build_fingerprint()["fingerprint"] == before


class TestValidationHelpers:
    def test_quartile_tiers_match_qcut(self):
        rng = np.random.default_rng(0)
        scores = rng.uniform(0, 100, 194)
        expected = pd.qcut(scores, 4, labels=False)
        assert np.array_equal(quartile_tiers(scores), expected)

    def test_quartile_tiers_columnwise(self):
        rng = np.random.default_rng(1)
        scores = rng.uniform(0, 100, (194, 5))
        tiers = quartile_tiers(scores)
        for j in range(scores.shape[1]):
            assert np.array_equal(tiers[:, j], pd.qcut(scores[:, j], 4, labels=False))

    def test_rank_desc(self):
        assert rank_desc([10, 30, 20, 30]).tolist() == [4, 1, 3, 1]

    def test_rank_desc_ignores_float_noise(self):
        assert rank_desc([46.005154639175259, 46.005154639175252, 40.0]).tolist() == [1, 1, 3]


def test_real_country_filter():
    df = pd.DataFrame({"iso_code": ["USA", "ATA", "OWID_WRL", None, "FRA"], "country": list("abcde")})
    kept = DataIngestion._filter_real_countries(df)["iso_code"].tolist()
    assert kept == ["USA", "FRA"]
