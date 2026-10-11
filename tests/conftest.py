import json

import pandas as pd
import pytest

from src import constants as C
from src.pipeline.train_pipeline import TrainPipeline

ARTIFACT_PATHS = [
    "DATA_PATH", "TRAIN_PATH", "TEST_PATH", "PREPROCESSOR_PATH", "MODEL_PATH", "FINAL_TABLE_PATH", "MANIFEST_PATH",
]


def redirect_artifacts(mp, directory):
    """Point every artifact path in src.constants at `directory`, so tests never
    touch the real artifacts/ folder."""
    mp.setattr(C, "ARTIFACTS_DIR", directory)
    for attr in ARTIFACT_PATHS:
        mp.setattr(C, attr, directory / getattr(C, attr).name)


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    """Run the full pipeline once into a temp dir. Artifact paths stay
    redirected for the whole session, so PredictPipeline and app.py tests read
    these artifacts."""
    out = tmp_path_factory.mktemp("artifacts")
    with pytest.MonkeyPatch.context() as mp:
        redirect_artifacts(mp, out)
        TrainPipeline().run()
        final = pd.read_csv(C.FINAL_TABLE_PATH)
        with open(C.MANIFEST_PATH) as f:
            manifest = json.load(f)
        yield {"final": final, "manifest": manifest, "dir": out}
