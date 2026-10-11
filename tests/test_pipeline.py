"""End-to-end checks on a fresh pipeline run (see the `built` fixture in conftest.py)."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src import constants as C
from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.fingerprint import artifact_paths, stale_reasons
from src.pipeline.predict_pipeline import CustomCountryData, InvalidIndicatorInput, PredictPipeline
from src.pipeline.train_pipeline import TrainPipeline
from tests.conftest import redirect_artifacts

INDICATORS = list(C.INDICATOR_DIRECTIONS)
SCORE_COLS = [f"{d}_score" for d in C.DIMENSIONS] + ["netzero_index_score"]

# Documented in CLAUDE.md: the 11 territories dropped for missing >1 indicator.
DOCUMENTED_DROPS = {
    "Bermuda", "Curacao", "Faroe Islands", "French Polynesia", "Greenland", "Micronesia (country)",
    "Montserrat", "New Caledonia", "Niue", "Saint Pierre and Miquelon", "Tuvalu",
}


class TestDataDecisions:
    def test_country_counts(self, built):
        raw_merge = pd.read_csv(built["dir"] / C.DATA_PATH.name)
        assert len(raw_merge) == 205
        assert len(built["final"]) == 194
        assert set(raw_merge["country"]) - set(built["final"]["country"]) == DOCUMENTED_DROPS

    def test_no_missing_values_after_imputation(self, built):
        assert built["final"][INDICATORS + SCORE_COLS].notna().all().all()

    def test_every_country_has_a_region(self, built):
        assert built["final"]["region"].notna().all()
        assert built["final"].set_index("iso_code").at["TWN", "sub_region"] == "Eastern Asia"

    def test_imputation_sources_recorded(self, built):
        final = built["final"]
        for col in INDICATORS:
            imputed = final[f"{col}_imputed"]
            assert final.loc[imputed, f"{col}_impute_source"].notna().all()
            assert final.loc[~imputed, f"{col}_impute_source"].isna().all()
        details = built["manifest"]["imputation_details"]
        assert len(details) == sum(built["manifest"]["imputed_counts"].values())

    def test_regional_imputation_choices(self, built):
        final = built["final"].set_index("country")
        assert final.at["Somalia", "gdp_per_capita_impute_source"].startswith("Eastern Africa median")
        # Pacific sub-regions have no observed GDP peers -> global fallback, not Australia/NZ.
        assert final.at["Fiji", "gdp_per_capita_impute_source"].startswith("global median")


class TestScores:
    def test_scores_in_range(self, built):
        scores = built["final"][SCORE_COLS + [f"{i}_norm" for i in INDICATORS]]
        assert ((scores >= 0) & (scores <= 100)).all().all()

    def test_tiers_are_balanced_quartiles(self, built):
        counts = built["final"]["tier"].value_counts()
        assert set(counts.index) == set(C.TIER_LABELS)
        assert counts.min() >= 45 and counts.max() <= 52

    def test_tiers_follow_score_order(self, built):
        means = built["final"].groupby("tier")["netzero_index_score"].mean()
        assert means.reindex(C.TIER_LABELS).is_monotonic_increasing

    def test_anchor_plausibility(self, built):
        tier = built["final"].set_index("country")["tier"]
        for country in ["Norway", "Sweden", "Iceland", "Denmark", "Switzerland"]:
            assert tier[country] == "Leader", country
        for country in ["Saudi Arabia", "Brunei", "Turkmenistan"]:
            assert tier[country] == "Lagging", country

    def test_rank_matches_score(self, built):
        final = built["final"].sort_values("rank")
        assert final["rank"].iloc[0] == 1
        # Ties share a rank; compare at the same precision rank_desc uses.
        assert final["netzero_index_score"].round(9).is_monotonic_decreasing
        tied = final[final.duplicated("rank", keep=False)]
        assert (tied.groupby("rank")["netzero_index_score"].agg(lambda s: s.max() - s.min()) < 1e-9).all()


class TestPredictPipeline:
    def test_reproduces_training_results(self, built):
        """Scoring each real country's raw values through PredictPipeline must
        give exactly the scores, tiers and segments the training run produced."""
        final = built["final"]
        predicted = PredictPipeline().predict(final[["country"] + INDICATORS].copy())
        assert np.allclose(predicted["netzero_index_score"], final["netzero_index_score"], rtol=0, atol=1e-9)
        assert (predicted["tier"].astype(str) == final["tier"]).all()
        assert (predicted["cluster_label"] == final["cluster_label"]).all()

    def test_extreme_hypotheticals_get_a_tier(self, built):
        # Scores beyond the cohort's min/max must still land in the outer tiers.
        best = CustomCountryData(0.01, 0.01, -2.0, 100.0, 500_000.0, country="Best").to_dataframe()
        worst = CustomCountryData(80.0, 90.0, 3.0, 0.0, 100.0, country="Worst").to_dataframe()
        result = PredictPipeline().predict(pd.concat([best, worst], ignore_index=True))
        assert result["tier"].astype(str).tolist() == ["Leader", "Lagging"]

    def test_physically_impossible_inputs_are_rejected(self, built):
        # The reported case: negative emissions, >100% renewables and negative GDP used to score 75.0 / Leader.
        bad = CustomCountryData(-1.0, -1.0, -0.1, 150.0, -500.0, country="Impossible").to_dataframe()
        with pytest.raises(InvalidIndicatorInput) as err:
            PredictPipeline().predict(bad)
        message = str(err.value)
        for field in ["co2_per_capita", "renewables_share_elec", "gdp_per_capita"]:
            assert field in message
        assert "ghg_per_capita" not in message        # net-negative GHG (land-use sink) is allowed

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_non_finite_inputs_are_rejected(self, built, value):
        with pytest.raises(InvalidIndicatorInput):
            PredictPipeline().predict(CustomCountryData(5.0, 6.0, value, 40.0, 20000.0).to_dataframe())

    def test_missing_fitted_artifacts_give_a_clear_error(self, built, tmp_path):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(C, "MODEL_PATH", tmp_path / "model.pkl")
            with pytest.raises(FileNotFoundError, match="train_pipeline"):
                PredictPipeline().predict(CustomCountryData(5.0, 6.0, -0.1, 40.0, 20000.0).to_dataframe())


class TestFreshness:
    def copy_build(self, built, mp, tmp_path):
        shutil.copytree(built["dir"], tmp_path / "artifacts")
        redirect_artifacts(mp, tmp_path / "artifacts")

    def test_fresh_build_has_no_stale_reasons(self, built):
        assert stale_reasons() == []

    def test_missing_pickle_is_detected(self, built, tmp_path):
        with pytest.MonkeyPatch.context() as mp:
            self.copy_build(built, mp, tmp_path)
            C.MODEL_PATH.unlink()
            assert stale_reasons() == ["missing artifacts: model.pkl"]

    def test_replaced_artifact_is_detected(self, built, tmp_path):
        with pytest.MonkeyPatch.context() as mp:
            self.copy_build(built, mp, tmp_path)
            C.PREPROCESSOR_PATH.write_bytes(C.PREPROCESSOR_PATH.read_bytes() + b"\0")
            assert stale_reasons() == ["artifact modified or from a different build: preprocessor.pkl"]

    def test_changed_configuration_is_detected(self, built):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(C, "DIMENSION_WEIGHTS", {**C.DIMENSION_WEIGHTS, "economic_capacity": 0.4})
            assert stale_reasons() == ["configuration in src/constants.py changed"]

    def test_changed_raw_input_is_detected(self, built, tmp_path):
        modified = tmp_path / C.RAW_REGIONS_PATH.name
        modified.write_bytes(C.RAW_REGIONS_PATH.read_bytes().replace(b"Western Africa", b"West Africa", 1))
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(C, "RAW_REGIONS_PATH", modified)
            assert stale_reasons() == ["raw input changed: un_m49_regions.csv"]


class TestDataAudit:
    def test_audit_recorded(self, built):
        audit = built["manifest"]["data_audit"]
        assert audit["real_countries_at_reference_year"]["merged"] == 205
        assert {d["country"] for d in audit["dropped_countries"]} == DOCUMENTED_DROPS
        assert all(len(d["missing_indicators"]) > C.MAX_MISSING_INDICATORS for d in audit["dropped_countries"])
        build = built["manifest"]["build"]
        assert set(build["inputs"]) == {"owid_co2", "owid_energy", "un_m49_regions"}
        assert set(build["artifact_sha256"]) == {p.name for p in artifact_paths().values()}

    def test_duplicate_iso_year_rows_fail_the_build(self, built, tmp_path):
        energy = pd.read_csv(C.RAW_ENERGY_PATH)
        duplicated = pd.concat([energy, energy[(energy["iso_code"] == "FRA") & (energy["year"] == 2022)]])
        path = tmp_path / "energy_dup.csv"
        duplicated.to_csv(path, index=False)
        with pytest.MonkeyPatch.context() as mp:
            redirect_artifacts(mp, tmp_path / "artifacts")
            mp.setattr(C, "RAW_ENERGY_PATH", path)
            with pytest.raises(CustomException, match=r"owid_energy: 2 rows share a \('iso_code', 'year'\) key"):
                DataIngestion().initiate_data_ingestion()

    def test_out_of_range_values_fail_the_build(self, built, tmp_path):
        energy = pd.read_csv(C.RAW_ENERGY_PATH)
        energy.loc[(energy["iso_code"] == "FRA") & (energy["year"] == 2022), "renewables_share_elec"] = 140.0
        path = tmp_path / "energy_bad.csv"
        energy.to_csv(path, index=False)
        with pytest.MonkeyPatch.context() as mp:
            redirect_artifacts(mp, tmp_path / "artifacts")
            mp.setattr(C, "RAW_ENERGY_PATH", path)
            with pytest.raises(CustomException, match=r"France: renewables_share_elec = 140\.0"):
                DataIngestion().initiate_data_ingestion()

    def test_missing_required_column_fails_the_build(self, built, tmp_path):
        path = tmp_path / "regions_bad.csv"
        pd.read_csv(C.RAW_REGIONS_PATH, keep_default_na=False).drop(columns="sub-region").to_csv(path, index=False)
        with pytest.MonkeyPatch.context() as mp:
            redirect_artifacts(mp, tmp_path / "artifacts")
            mp.setattr(C, "RAW_REGIONS_PATH", path)
            with pytest.raises(CustomException, match=r"un_m49_regions: required column\(s\) missing: \['sub-region'\]"):
                DataIngestion().initiate_data_ingestion()


class TestValidation:
    def test_manifest_contents(self, built):
        m = built["manifest"]
        assert m["pipeline_version"] == C.PIPELINE_VERSION
        for key in ["build", "data_audit", "sensitivity", "tier_cluster_agreement", "cluster_stability",
                    "pca_loadings", "imputation_details"]:
            assert key in m, key
        assert len(m["sensitivity"]["weight_scenarios"]) == 2 * len(C.DIMENSIONS)

    def test_weight_bands_bracket_baseline(self, built):
        final = built["final"]
        assert (final["weight_rank_p05"] <= final["weight_rank_p95"]).all()
        inside = (final["weight_rank_p05"] <= final["rank"]) & (final["rank"] <= final["weight_rank_p95"])
        assert inside.mean() >= 0.95
        assert final["weight_tier_stability"].between(0, 1).all()

    def test_imputation_sensitivity(self, built):
        final = built["final"]
        imputed = final[[f"{c}_imputed" for c in INDICATORS]].any(axis=1)
        summary = built["manifest"]["sensitivity"]["imputation"]
        assert summary["n_countries"] == imputed.sum()
        # Only imputed countries get a range, and the range always contains the baseline.
        assert final.loc[~imputed, "impute_rank_best"].isna().all()
        rows = final[imputed]
        assert ((rows["impute_rank_best"] <= rows["rank"]) & (rows["rank"] <= rows["impute_rank_worst"])).all()
        assert ((rows["impute_score_low"] <= rows["netzero_index_score"] + 1e-9)
                & (rows["netzero_index_score"] <= rows["impute_score_high"] + 1e-9)).all()

    def test_global_median_scenario_is_a_real_alternative(self, built):
        scenarios = {s["scenario"]: s for s in built["manifest"]["sensitivity"]["method_alternatives"]}
        assert scenarios["Global-median imputation"]["spearman_vs_baseline"] < 1.0

    def test_tier_cluster_agreement_beats_permutation_baseline(self, built):
        agreement = built["manifest"]["tier_cluster_agreement"]
        null = agreement["permutation_baseline"]
        # The ordering step lifts chance agreement above the naive 25%.
        assert 25 < null["exact_agreement_mean_pct"] < agreement["exact_agreement_pct"]
        assert null["exact_agreement_p_value"] < 0.01
        assert agreement["adjusted_rand_index"] > 0

    def test_selected_k_is_reasonably_stable(self, built):
        assert built["manifest"]["cluster_stability"]["selected_k"]["mean_ari"] > 0.8


def test_pipeline_is_deterministic(built, tmp_path):
    """A second, independent run must reproduce the scored table exactly."""
    with pytest.MonkeyPatch.context() as mp:
        redirect_artifacts(mp, tmp_path)
        TrainPipeline().run()
        rerun = pd.read_csv(C.FINAL_TABLE_PATH)
        with open(C.MANIFEST_PATH) as f:
            manifest = json.load(f)
    pd.testing.assert_frame_equal(rerun, built["final"])
    for key in ["sensitivity", "tier_cluster_agreement", "cluster_stability", "elbow_silhouette_table"]:
        assert manifest[key] == built["manifest"][key], key
