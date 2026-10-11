"""Smoke test: every dashboard page renders without an exception, against the
artifacts built by the `built` fixture."""
import re
import sys

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import constants as C

PAGES = [
    "Overview / Ranking", "Country Detail", "Scenario Lab",
    "Robustness & Validation", "PCA & Clusters", "Methodology & Limitations",
]


@pytest.fixture(scope="module")
def app(built):
    at = AppTest.from_file(str(C.ROOT_DIR / "app.py"), default_timeout=120).run()
    assert not at.exception
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(app, page):
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, [e.value for e in app.exception]


def test_overview_rank_ranges_are_whole_numbers(app):
    # Nullable rank columns come back from the CSV as floats; the table must still show "1-65", not "1.0-65.0".
    app.sidebar.radio[0].set_value("Overview / Ranking").run()
    table = app.dataframe[0].value
    imputation = table.loc[table["imputation_band"] != "", "imputation_band"]
    assert len(imputation) == 31
    for band in pd.concat([imputation, table["weight_band"]]):
        assert re.fullmatch(r"\d+-\d+", band), band


def test_whatif_rejects_impossible_values(app):
    """The reported case: -1 emissions, 150% renewables, negative GDP must not produce a score."""
    app.sidebar.radio[0].set_value("Scenario Lab").run()
    inputs = {"co2_per_capita": -1.0, "ghg_per_capita": -1.0, "renewables_share_elec": 150.0, "gdp_per_capita": -500.0}
    for indicator, value in inputs.items():
        widget = app.number_input(key=f"in_{indicator}_India")
        bounds = C.INDICATOR_BOUNDS[indicator]                          # UI bounds match the pipeline's
        assert widget.min == (bounds["min"] if bounds["min"] is not None else -sys.float_info.max)
        assert widget.max == (bounds["max"] if bounds["max"] is not None else sys.float_info.max)
        widget.set_value(value)
    app.run()
    labels = [m.label for m in app.metric]
    assert "Would rank" not in labels, "an impossible country was scored"
    errors = " ".join(e.value for e in app.error)
    for field in ["co2_per_capita", "renewables_share_elec", "gdp_per_capita"]:
        assert field in errors


def test_baseline_weights_change_nothing(app):
    app.sidebar.radio[0].set_value("Scenario Lab").run()
    for d in C.DIMENSIONS:  # reset explicitly; the app instance is shared across tests
        app.slider(key=f"w_{d}").set_value(int(C.DIMENSION_WEIGHTS[d] * 100))
    app.run()
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Countries changing tier"] == "0"
    assert metrics["Mean absolute rank change"] == "0.0"


def test_reweighting_matches_pipeline_scenario(app, built):
    """Zeroing a dimension's slider must reproduce the pipeline's 'Drop <dimension>' scenario."""
    app.sidebar.radio[0].set_value("Scenario Lab").run()
    for d in C.DIMENSIONS:
        app.slider(key=f"w_{d}").set_value(0 if d == "economic_capacity" else int(C.DIMENSION_WEIGHTS[d] * 100))
    app.run()
    spearman = float(next(m.value for m in app.metric if m.label == "Rank correlation with baseline"))
    scenario = next(
        s for s in built["manifest"]["sensitivity"]["weight_scenarios"] if s["scenario"] == "Drop Economic Capacity"
    )
    assert spearman == pytest.approx(scenario["spearman_vs_baseline"], abs=1e-3)
