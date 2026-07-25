# tests/test_inference.py

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest
from xgboost import XGBRegressor

from src.inference.predict import (
    generate_predictions,
    load_feature_columns,
    load_forecasting_model,
    predict_from_csv,
    save_predictions,
    validate_inference_data,
)


@pytest.fixture
def feature_columns() -> list[str]:
    return [
        "temperature",
        "demand_lag_1",
    ]


@pytest.fixture
def training_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temperature": [
                18.0,
                20.0,
                22.0,
                24.0,
                26.0,
                28.0,
            ],
            "demand_lag_1": [
                4500.0,
                4600.0,
                4700.0,
                4800.0,
                4900.0,
                5000.0,
            ],
            "target": [
                4550.0,
                4650.0,
                4750.0,
                4850.0,
                4950.0,
                5050.0,
            ],
        }
    )


@pytest.fixture
def inference_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "INTERVAL_DATETIME": [
                "2026-01-01 00:00:00",
                "2026-01-01 00:30:00",
            ],
            "temperature": [
                21.0,
                23.0,
            ],
            "demand_lag_1": [
                4680.0,
                4780.0,
            ],
        }
    )


@pytest.fixture
def saved_model_files(
    tmp_path: Path,
    feature_columns: list[str],
    training_dataframe: pd.DataFrame,
) -> tuple[Path, Path]:
    model = XGBRegressor(
        n_estimators=5,
        max_depth=2,
        learning_rate=0.1,
        random_state=42,
        n_jobs=1,
    )

    model.fit(
        training_dataframe[feature_columns],
        training_dataframe["target"],
    )

    model_path = (
        tmp_path
        / "test_model.json"
    )

    feature_columns_path = (
        tmp_path
        / "feature_columns.joblib"
    )

    model.save_model(
        model_path
    )

    joblib.dump(
        feature_columns,
        feature_columns_path,
    )

    return (
        model_path,
        feature_columns_path,
    )


def test_load_feature_columns(
    saved_model_files: tuple[Path, Path],
    feature_columns: list[str],
) -> None:
    _, feature_columns_path = (
        saved_model_files
    )

    loaded_columns = load_feature_columns(
        feature_columns_path
    )

    assert loaded_columns == feature_columns


def test_load_forecasting_model(
    saved_model_files: tuple[Path, Path],
) -> None:
    model_path, _ = saved_model_files

    model = load_forecasting_model(
        model_path
    )

    assert isinstance(
        model,
        XGBRegressor,
    )


def test_validate_inference_data_returns_features_in_saved_order(
    inference_dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    validated = validate_inference_data(
        dataframe=inference_dataframe,
        feature_columns=feature_columns,
    )

    assert list(
        validated.columns
    ) == feature_columns

    assert len(validated) == 2


def test_validate_inference_data_rejects_missing_feature(
    inference_dataframe: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="missing required features",
    ):
        validate_inference_data(
            dataframe=inference_dataframe,
            feature_columns=[
                "temperature",
                "missing_feature",
            ],
        )


def test_validate_inference_data_rejects_invalid_numeric_values(
    inference_dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    invalid_dataframe = (
        inference_dataframe.copy()
    )

    invalid_dataframe.loc[
        0,
        "temperature",
    ] = "invalid"

    with pytest.raises(
        ValueError,
        match=(
            "missing or non-numeric values"
        ),
    ):
        validate_inference_data(
            dataframe=invalid_dataframe,
            feature_columns=feature_columns,
        )


def test_generate_predictions_returns_expected_columns(
    inference_dataframe: pd.DataFrame,
    saved_model_files: tuple[Path, Path],
) -> None:
    model_path, feature_columns_path = (
        saved_model_files
    )

    predictions = generate_predictions(
        dataframe=inference_dataframe,
        model_path=model_path,
        feature_columns_path=(
            feature_columns_path
        ),
    )

    assert list(
        predictions.columns
    ) == [
        "INTERVAL_DATETIME",
        "PREDICTED_OPERATIONAL_DEMAND",
    ]

    assert len(predictions) == len(
        inference_dataframe
    )

    assert predictions[
        "PREDICTED_OPERATIONAL_DEMAND"
    ].notna().all()


def test_generate_predictions_without_timestamp_column(
    inference_dataframe: pd.DataFrame,
    saved_model_files: tuple[Path, Path],
) -> None:
    model_path, feature_columns_path = (
        saved_model_files
    )

    feature_only_dataframe = (
        inference_dataframe.drop(
            columns=[
                "INTERVAL_DATETIME"
            ]
        )
    )

    predictions = generate_predictions(
        dataframe=feature_only_dataframe,
        model_path=model_path,
        feature_columns_path=(
            feature_columns_path
        ),
    )

    assert list(
        predictions.columns
    ) == [
        "PREDICTED_OPERATIONAL_DEMAND"
    ]

    assert len(predictions) == 2


def test_generate_predictions_rejects_invalid_timestamp(
    inference_dataframe: pd.DataFrame,
    saved_model_files: tuple[Path, Path],
) -> None:
    model_path, feature_columns_path = (
        saved_model_files
    )

    invalid_dataframe = (
        inference_dataframe.copy()
    )

    invalid_dataframe.loc[
        0,
        "INTERVAL_DATETIME",
    ] = "not-a-date"

    with pytest.raises(
        ValueError,
        match="invalid timestamp values",
    ):
        generate_predictions(
            dataframe=invalid_dataframe,
            model_path=model_path,
            feature_columns_path=(
                feature_columns_path
            ),
        )


def test_save_predictions_creates_csv(
    tmp_path: Path,
) -> None:
    predictions = pd.DataFrame(
        {
            "PREDICTED_OPERATIONAL_DEMAND": [
                4500.0,
                4600.0,
            ]
        }
    )

    output_path = (
        tmp_path
        / "predictions"
        / "forecast.csv"
    )

    saved_path = save_predictions(
        predictions=predictions,
        output_path=output_path,
    )

    assert saved_path == output_path
    assert output_path.exists()

    saved_dataframe = pd.read_csv(
        output_path
    )

    pd.testing.assert_frame_equal(
        saved_dataframe,
        predictions,
    )


def test_predict_from_csv_runs_end_to_end(
    tmp_path: Path,
    inference_dataframe: pd.DataFrame,
    saved_model_files: tuple[Path, Path],
) -> None:
    model_path, feature_columns_path = (
        saved_model_files
    )

    input_path = (
        tmp_path
        / "inference_input.csv"
    )

    output_path = (
        tmp_path
        / "inference_output.csv"
    )

    inference_dataframe.to_csv(
        input_path,
        index=False,
    )

    predictions = predict_from_csv(
        input_path=input_path,
        output_path=output_path,
        model_path=model_path,
        feature_columns_path=(
            feature_columns_path
        ),
    )

    assert output_path.exists()
    assert len(predictions) == 2

    saved_predictions = pd.read_csv(
        output_path
    )

    assert list(
        saved_predictions.columns
    ) == [
        "INTERVAL_DATETIME",
        "PREDICTED_OPERATIONAL_DEMAND",
    ]


def test_load_feature_columns_rejects_missing_file(
    tmp_path: Path,
) -> None:
    missing_path = (
        tmp_path
        / "missing_features.joblib"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Feature-column file",
    ):
        load_feature_columns(
            missing_path
        )


def test_load_forecasting_model_rejects_missing_file(
    tmp_path: Path,
) -> None:
    missing_path = (
        tmp_path
        / "missing_model.json"
    )

    with pytest.raises(
        FileNotFoundError,
        match="XGBoost model file",
    ):
        load_forecasting_model(
            missing_path
        )