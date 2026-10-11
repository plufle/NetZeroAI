"""Build fingerprint and artifact freshness checks.

A build is identified by the SHA-256 of every raw input file plus a hash of every
configuration constant in src.constants (paths excluded, so redirecting artifacts in
tests doesn't change it). The manifest records that fingerprint and the hash of each
artifact the build wrote; `stale_reasons()` compares both against what is on disk now.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from src import constants as C

GENERATED_ARTIFACTS = ["DATA_PATH", "TRAIN_PATH", "TEST_PATH", "PREPROCESSOR_PATH", "MODEL_PATH", "FINAL_TABLE_PATH"]


def raw_input_paths():
    return {"owid_co2": C.RAW_CO2_PATH, "owid_energy": C.RAW_ENERGY_PATH, "un_m49_regions": C.RAW_REGIONS_PATH}


def artifact_paths():
    """Every artifact a build writes, except the manifest itself."""
    return {getattr(C, attr).name: getattr(C, attr) for attr in GENERATED_ARTIFACTS}


@lru_cache(maxsize=64)
def _sha256(path, mtime_ns, size):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path):
    """SHA-256 of a file, memoized on (path, mtime, size) so the dashboard can re-check
    freshness on every rerun without re-reading 23 MB of raw data."""
    stat = Path(path).stat()
    return _sha256(str(path), stat.st_mtime_ns, stat.st_size)


def _jsonable(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, (range, tuple)):
        return list(value)
    raise TypeError(f"unserializable config value: {value!r}")


def config_snapshot():
    """Every UPPER_CASE constant except paths - picks up new settings automatically."""
    return {
        name: getattr(C, name)
        for name in sorted(dir(C))
        if name.isupper() and not isinstance(getattr(C, name), Path)
    }


def _hash_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=_jsonable).encode()).hexdigest()


def build_fingerprint():
    inputs = {
        name: {"file": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for name, path in raw_input_paths().items()
    }
    config_sha256 = _hash_json(config_snapshot())
    return {
        "fingerprint": _hash_json({"inputs": {k: v["sha256"] for k, v in inputs.items()}, "config": config_sha256}),
        "inputs": inputs,
        "config_sha256": config_sha256,
    }


def stale_reasons():
    """Why the current artifacts can't be trusted (empty list = fresh)."""
    missing = [name for name, path in artifact_paths().items() if not path.exists()]
    if not C.MANIFEST_PATH.exists():
        missing.append(C.MANIFEST_PATH.name)
    if missing:
        return [f"missing artifacts: {', '.join(missing)}"]

    with open(C.MANIFEST_PATH) as f:
        manifest = json.load(f)
    if manifest.get("pipeline_version") != C.PIPELINE_VERSION:
        return [f"pipeline version changed ({manifest.get('pipeline_version')} -> {C.PIPELINE_VERSION})"]

    reasons = []
    recorded = manifest.get("build", {})
    current = build_fingerprint()
    for name, info in current["inputs"].items():
        if recorded.get("inputs", {}).get(name, {}).get("sha256") != info["sha256"]:
            reasons.append(f"raw input changed: {info['file']}")
    if recorded.get("config_sha256") != current["config_sha256"]:
        reasons.append("configuration in src/constants.py changed")

    recorded_artifacts = recorded.get("artifact_sha256", {})
    for name, path in artifact_paths().items():
        if recorded_artifacts.get(name) != file_sha256(path):
            reasons.append(f"artifact modified or from a different build: {name}")
    return reasons
