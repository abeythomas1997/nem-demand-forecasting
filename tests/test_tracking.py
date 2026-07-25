# tests/test_tracking.py

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.tracking.experiment_tracker import (
    EXPERIMENT_COLUMNS,
    append_experiment_record,
    create_experiment_record,
    track_experiment,
)
from src.tracking.model_manifest import (
    create_model_manifest,
)


@pytest.fixture
def training_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "INTERVAL_DATETIME": pd.date_range(
                start="2025-01-01 00:00:00",
                periods=6,
                freq="30min",
            ),
            "temperature": [
                18.0,
                19.0,
                20.0,
                21.0,
                22.0,
                23.0,
            ],
            "demand_lag_1": [
                4500.0,
                4550.0,
                4600.0,
                4650.0,
                4700.0,
                4750.0,
            ],
            "OPERATIONAL_DEMAND": [
                4520.0,
                4570.0,
                4620.0,
                4670.0,
                4720.0,
                4770.0,
            ],
        }
    )


@pytest.fixture
def feature_columns() -> list[str]:
    return [
        "temperature",
        "demand_lag_1",
    ]


@pytest.fixture
def validation_metrics() -> dict[str, float]:
    return {
        "MAE": 56.508,
        "RMSE": 80.125,
        "MAPE": 1.08,
        "R2": 0.991,
    }


@pytest.fixture
def test_metrics() -> dict[str, float]:
    return {
        "MAE": 61.039,
        "RMSE": 86.846,
        "MAPE": 1.173,
        "R2": 0.990,
    }


def test_create_experiment_record(
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    record = create_experiment_record(
        model_name="Tuned XGBoost",
        selected_model="tuned_xgboost",
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        best_cv_mae=58.25,
        feature_count=34,
        training_rows=12028,
        validation_rows=2577,
        test_rows=2579,
        parameters={
            "n_estimators": 500,
            "max_depth": 6,
        },
        run_id="test-run-001",
    )

    assert record["run_id"] == "test-run-001"
    assert record["model_name"] == "Tuned XGBoost"
    assert (
        record["selected_model"]
        == "tuned_xgboost"
    )
    assert record[
        "validation_mae"
    ] == pytest.approx(
        56.508
    )
    assert record["test_mae"] == pytest.approx(
        61.039
    )
    assert record["test_rmse"] == pytest.approx(
        86.846
    )
    assert record["test_mape"] == pytest.approx(
        1.173
    )
    assert record["test_r2"] == pytest.approx(
        0.990
    )
    assert record["best_cv_mae"] == pytest.approx(
        58.25
    )
    assert record["feature_count"] == 34
    assert record["training_rows"] == 12028
    assert record["validation_rows"] == 2577
    assert record["test_rows"] == 2579

    parameters = json.loads(
        record["parameters"]
    )

    assert parameters["n_estimators"] == 500
    assert parameters["max_depth"] == 6


def test_create_experiment_record_supports_lowercase_metrics() -> None:
    record = create_experiment_record(
        model_name="XGBoost",
        selected_model="xgboost",
        validation_metrics={
            "mae": 60.0,
        },
        test_metrics={
            "mae": 62.0,
            "rmse": 88.0,
            "mape": 1.2,
            "r2": 0.98,
        },
        best_cv_mae=59.0,
        feature_count=10,
        training_rows=100,
        validation_rows=20,
        test_rows=20,
    )

    assert record["validation_mae"] == 60.0
    assert record["test_mae"] == 62.0
    assert record["test_rmse"] == 88.0
    assert record["test_mape"] == 1.2
    assert record["test_r2"] == 0.98


def test_create_experiment_record_rejects_invalid_feature_count(
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    with pytest.raises(
        ValueError,
        match="Feature count",
    ):
        create_experiment_record(
            model_name="XGBoost",
            selected_model="xgboost",
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            best_cv_mae=58.0,
            feature_count=0,
            training_rows=100,
            validation_rows=20,
            test_rows=20,
        )


def test_create_experiment_record_rejects_invalid_row_count(
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    with pytest.raises(
        ValueError,
        match="Dataset row counts",
    ):
        create_experiment_record(
            model_name="XGBoost",
            selected_model="xgboost",
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            best_cv_mae=58.0,
            feature_count=10,
            training_rows=0,
            validation_rows=20,
            test_rows=20,
        )


def test_append_experiment_record_creates_csv(
    tmp_path: Path,
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    output_path = (
        tmp_path
        / "tracking"
        / "experiment_history.csv"
    )

    record = create_experiment_record(
        model_name="Tuned XGBoost",
        selected_model="tuned_xgboost",
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        best_cv_mae=58.25,
        feature_count=34,
        training_rows=12028,
        validation_rows=2577,
        test_rows=2579,
        run_id="test-run-001",
    )

    history = append_experiment_record(
        record=record,
        output_path=output_path,
    )

    assert output_path.exists()
    assert len(history) == 1
    assert list(history.columns) == EXPERIMENT_COLUMNS
    assert history.loc[
        0,
        "run_id",
    ] == "test-run-001"


def test_append_experiment_record_adds_new_run(
    tmp_path: Path,
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    output_path = (
        tmp_path
        / "experiment_history.csv"
    )

    first_record = create_experiment_record(
        model_name="Random Forest",
        selected_model="random_forest",
        validation_metrics={
            "MAE": 61.002,
        },
        test_metrics={
            "MAE": 65.0,
            "RMSE": 90.0,
            "MAPE": 1.3,
            "R2": 0.98,
        },
        best_cv_mae=62.0,
        feature_count=34,
        training_rows=12028,
        validation_rows=2577,
        test_rows=2579,
        run_id="run-001",
    )

    second_record = create_experiment_record(
        model_name="Tuned XGBoost",
        selected_model="tuned_xgboost",
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        best_cv_mae=58.25,
        feature_count=34,
        training_rows=12028,
        validation_rows=2577,
        test_rows=2579,
        run_id="run-002",
    )

    append_experiment_record(
        record=first_record,
        output_path=output_path,
    )

    history = append_experiment_record(
        record=second_record,
        output_path=output_path,
    )

    assert len(history) == 2
    assert list(
        history["run_id"]
    ) == [
        "run-001",
        "run-002",
    ]


def test_track_experiment_runs_end_to_end(
    tmp_path: Path,
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    output_path = (
        tmp_path
        / "experiment_history.csv"
    )

    record = track_experiment(
        model_name="Tuned XGBoost",
        selected_model="tuned_xgboost",
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        best_cv_mae=58.25,
        feature_count=34,
        training_rows=12028,
        validation_rows=2577,
        test_rows=2579,
        output_path=output_path,
        parameters={
            "learning_rate": 0.05,
        },
        run_id="tracked-run",
    )

    assert output_path.exists()
    assert record["run_id"] == "tracked-run"

    saved_history = pd.read_csv(
        output_path
    )

    assert len(saved_history) == 1
    assert saved_history.loc[
        0,
        "selected_model",
    ] == "tuned_xgboost"


def test_create_model_manifest(
    tmp_path: Path,
    training_dataframe: pd.DataFrame,
    feature_columns: list[str],
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    model_path = (
        tmp_path
        / "models"
        / "final_xgboost_model.json"
    )

    feature_columns_path = (
        tmp_path
        / "models"
        / "feature_columns.joblib"
    )

    output_path = (
        tmp_path
        / "tracking"
        / "model_manifest.json"
    )

    manifest = create_model_manifest(
        model_name="tuned_xgboost",
        model_path=model_path,
        feature_columns_path=(
            feature_columns_path
        ),
        feature_columns=feature_columns,
        target_column="OPERATIONAL_DEMAND",
        timestamp_column="INTERVAL_DATETIME",
        forecast_horizon="30 minutes ahead",
        training_dataframe=(
            training_dataframe
        ),
        validation_metrics=(
            validation_metrics
        ),
        test_metrics=test_metrics,
        model_parameters={
            "n_estimators": 500,
            "max_depth": 6,
        },
        output_path=output_path,
    )

    assert output_path.exists()
    assert manifest[
        "model"
    ]["name"] == "tuned_xgboost"
    assert manifest[
        "forecast"
    ]["target_column"] == (
        "OPERATIONAL_DEMAND"
    )
    assert manifest[
        "training_data"
    ]["row_count"] == 6
    assert manifest[
        "training_data"
    ]["feature_count"] == 2
    assert manifest["features"] == feature_columns
    assert manifest[
        "performance"
    ]["test"]["MAE"] == pytest.approx(
        61.039
    )

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved_manifest = json.load(
            file
        )

    assert (
        saved_manifest["manifest_version"]
        == "1.0"
    )
    assert (
        saved_manifest["project"]["region"]
        == "VIC1"
    )


def test_model_manifest_rejects_empty_training_dataframe(
    tmp_path: Path,
    feature_columns: list[str],
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        create_model_manifest(
            model_name="tuned_xgboost",
            model_path=(
                tmp_path / "model.json"
            ),
            feature_columns_path=(
                tmp_path
                / "features.joblib"
            ),
            feature_columns=feature_columns,
            target_column="OPERATIONAL_DEMAND",
            timestamp_column="INTERVAL_DATETIME",
            forecast_horizon="30 minutes ahead",
            training_dataframe=pd.DataFrame(),
            validation_metrics=(
                validation_metrics
            ),
            test_metrics=test_metrics,
            model_parameters={},
            output_path=(
                tmp_path
                / "manifest.json"
            ),
        )


def test_model_manifest_rejects_missing_timestamp_column(
    tmp_path: Path,
    training_dataframe: pd.DataFrame,
    feature_columns: list[str],
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    invalid_dataframe = (
        training_dataframe.drop(
            columns=[
                "INTERVAL_DATETIME"
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="missing INTERVAL_DATETIME",
    ):
        create_model_manifest(
            model_name="tuned_xgboost",
            model_path=(
                tmp_path / "model.json"
            ),
            feature_columns_path=(
                tmp_path
                / "features.joblib"
            ),
            feature_columns=feature_columns,
            target_column="OPERATIONAL_DEMAND",
            timestamp_column="INTERVAL_DATETIME",
            forecast_horizon="30 minutes ahead",
            training_dataframe=(
                invalid_dataframe
            ),
            validation_metrics=(
                validation_metrics
            ),
            test_metrics=test_metrics,
            model_parameters={},
            output_path=(
                tmp_path
                / "manifest.json"
            ),
        )