import numpy as np
import pandas as pd
import pytest

from anomalyx.pipeline import PipelineValidationError, prepare_data, train_isolation_forest


def make_frame(rows: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "signal_a": rng.normal(0, 1, rows),
            "signal_b": rng.normal(10, 2, rows),
            "label": ["batch-a"] * rows,
        }
    )


def test_prepare_data_selects_numeric_features_and_tracks_missing_values():
    frame = make_frame()
    frame.loc[2, "signal_a"] = np.nan
    prepared = prepare_data(frame, ["signal_a", "signal_b"])
    assert prepared.features == ["signal_a", "signal_b"]
    assert prepared.missing_counts["signal_a"] == 1
    assert prepared.numeric_candidates == ["signal_a", "signal_b"]


def test_prediction_pipeline_retains_rows_and_marks_outliers():
    frame = make_frame()
    frame.loc[0, ["signal_a", "signal_b"]] = [12, 40]
    frame.loc[1, "signal_a"] = np.nan
    prepared = prepare_data(frame, ["signal_a", "signal_b"])
    results, model = train_isolation_forest(prepared, contamination=0.1)
    assert len(results) == len(frame)
    assert {"anomaly_score", "prediction", "is_anomaly"}.issubset(results.columns)
    assert results["is_anomaly"].dtype == bool
    assert results["prediction"].isin(["Normal", "Anomaly"]).all()
    assert int(results["is_anomaly"].sum()) >= 1
    assert model.named_steps["detector"].random_state == 42


def test_invalid_inputs_raise_clear_validation_errors():
    with pytest.raises(PipelineValidationError, match="numeric column"):
        prepare_data(pd.DataFrame({"name": ["a", "b"]}))
    with pytest.raises(PipelineValidationError, match="At least 8"):
        prepare_data(pd.DataFrame({"value": range(3)}))
    with pytest.raises(PipelineValidationError, match="between 0.1% and 50%"):
        train_isolation_forest(prepare_data(make_frame()), contamination=0.8)
