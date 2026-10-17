from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

DEFAULT_CONTAMINATION = 0.08
MAX_FEATURES = 12
MIN_TRAINING_ROWS = 8
RANDOM_STATE = 42


class PipelineValidationError(ValueError):
    pass


@dataclass(frozen=True)
class PreparedData:
    frame: pd.DataFrame
    features: list[str]
    missing_counts: pd.Series
    numeric_candidates: list[str]


def _validate_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise PipelineValidationError("The uploaded object is not a valid table.")
    if frame.empty:
        raise PipelineValidationError("The dataset has no rows.")
    if frame.shape[1] == 0:
        raise PipelineValidationError("The dataset has no columns.")


def prepare_data(frame: pd.DataFrame, features: Iterable[str] | None = None) -> PreparedData:
    _validate_frame(frame)
    clean = frame.copy()
    clean.columns = [str(column).strip() for column in clean.columns]
    if any(not column for column in clean.columns):
        raise PipelineValidationError("Every column must have a name.")
    if clean.columns.duplicated().any():
        raise PipelineValidationError("Column names must be unique.")

    numeric_candidates = clean.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_candidates:
        raise PipelineValidationError("At least one numeric column is required for anomaly detection.")

    selected = numeric_candidates if features is None else list(features)
    invalid = [feature for feature in selected if feature not in numeric_candidates]
    if invalid:
        raise PipelineValidationError(f"Select numeric columns only: {', '.join(invalid)}")
    if not selected:
        raise PipelineValidationError("Select at least one numeric feature.")
    if len(selected) > MAX_FEATURES:
        raise PipelineValidationError(f"Select at most {MAX_FEATURES} numeric features for a stable dashboard.")

    selected_frame = clean[selected].replace([np.inf, -np.inf], np.nan)
    missing_counts = selected_frame.isna().sum().sort_values(ascending=False)
    usable = selected_frame.notna().any(axis=1)
    if int(usable.sum()) < MIN_TRAINING_ROWS:
        raise PipelineValidationError(
            f"At least {MIN_TRAINING_ROWS} rows with a numeric value are required; found {int(usable.sum())}."
        )
    clean[selected] = selected_frame
    return PreparedData(clean, selected, missing_counts, numeric_candidates)


def train_isolation_forest(
    prepared: PreparedData,
    contamination: float = DEFAULT_CONTAMINATION,
) -> tuple[pd.DataFrame, Pipeline]:
    if not 0.001 <= contamination <= 0.5:
        raise PipelineValidationError("Contamination must be between 0.1% and 50%.")
    if len(prepared.frame) < MIN_TRAINING_ROWS:
        raise PipelineValidationError(f"At least {MIN_TRAINING_ROWS} rows are required for training.")

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "detector",
                IsolationForest(
                    contamination=contamination,
                    random_state=RANDOM_STATE,
                    n_estimators=250,
                    n_jobs=1,
                ),
            ),
        ]
    )
    values = prepared.frame[prepared.features]
    model.fit(values)
    predictions = model.predict(values)
    raw_scores = model.decision_function(values)
    results = prepared.frame.copy()
    results["anomaly_score"] = -raw_scores
    results["prediction"] = np.where(predictions == -1, "Anomaly", "Normal")
    results["is_anomaly"] = predictions == -1
    return results, model
