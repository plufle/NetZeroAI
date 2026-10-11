import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import constants as C
from src.exception import CustomException


def describe_bounds(indicator):
    lo, hi = C.INDICATOR_BOUNDS[indicator]["min"], C.INDICATOR_BOUNDS[indicator]["max"]
    if lo is None and hi is None:
        return "any finite number"
    if hi is None:
        return f">= {lo:g}"
    if lo is None:
        return f"<= {hi:g}"
    return f"between {lo:g} and {hi:g}"


def indicator_bound_violations(df, indicators=None, label_col="country", allow_missing=False):
    """Human-readable problems for values that are non-numeric, non-finite or outside
    C.INDICATOR_BOUNDS. Missing values are reported unless `allow_missing` (ingestion
    tolerates gaps that are imputed later; what-if inputs must be complete)."""
    problems = []
    for col in indicators or C.INDICATOR_DIRECTIONS:
        lo, hi = C.INDICATOR_BOUNDS[col]["min"], C.INDICATOR_BOUNDS[col]["max"]
        values = pd.to_numeric(df[col], errors="coerce")
        bad = values.isna() & df[col].notna()                     # non-numeric
        if not allow_missing:
            bad |= values.isna()
        bad |= np.isinf(values)
        if lo is not None:
            bad |= values < lo
        if hi is not None:
            bad |= values > hi
        for idx in df.index[bad]:
            label = df.at[idx, label_col] if label_col in df.columns else idx
            value = df.at[idx, col]
            value = value.item() if hasattr(value, "item") else value      # np.float64(1.0) -> 1.0
            problems.append(f"{label}: {col} = {value!r} (must be {describe_bounds(col)})")
    return problems


def save_object(file_path, obj):
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e, sys)
