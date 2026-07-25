# src/inference/predict.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from xgboost import XGBRegressor


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "final_xgboost_model.json"
)

DEFAULT_FEATURE_COLUMNS_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "feature_columns.joblib"
)

DEFAULT_TIMESTAMP_COLUMN = "INTERVAL_DATETIME"
DEFAULT_PREDICTION_COLUMN = "PREDICTED_OPERATIONAL_DEMAND"


def _validate_file(
    file_path: Path,
    description: str,
) -> None:
    if not file_path.exists():
        raise FileNotFoundError(
            f"{description} was not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"{description} is not a file: {file_path}"
        )


def load_feature_columns(
    feature_columns_path: Path | str = (
        DEFAULT_FEATURE_COLUMNS_PATH
    ),
) -> list[str]:
    feature_columns_path = Path(
        feature_columns_path
    )

    _validate_file(
        file_path=feature_columns_path,
        description="Feature-column file",
    )

    feature_columns = joblib.load(
        feature_columns_path
    )

    if not isinstance(
        feature_columns,
        (list, tuple),
    ):
        raise TypeError(
            "Saved feature columns must be a list "
            "or tuple."
        )

    feature_columns = [
        str(column)
        for column in feature_columns
    ]

    if not feature_columns:
        raise ValueError(
            "Saved feature-column list is empty."
        )

    if len(feature_columns) != len(
        set(feature_columns)
    ):
        raise ValueError(
            "Saved feature-column list contains "
            "duplicate columns."
        )

    return feature_columns


def load_forecasting_model(
    model_path: Path | str = (
        DEFAULT_MODEL_PATH
    ),
) -> XGBRegressor:
    model_path = Path(model_path)

    _validate_file(
        file_path=model_path,
        description="XGBoost model file",
    )

    model = XGBRegressor()

    model.load_model(
        model_path
    )

    return model


def validate_inference_data(
    dataframe: pd.DataFrame,
    feature_columns: Iterable[str],
) -> pd.DataFrame:
    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "Inference input must be a pandas "
            "DataFrame."
        )

    if dataframe.empty:
        raise ValueError(
            "Inference input contains no rows."
        )

    feature_columns = list(
        feature_columns
    )

    missing_columns = [
        column
        for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Inference input is missing required "
            f"features: {missing_columns}"
        )

    feature_data = dataframe[
        feature_columns
    ].copy()

    for column in feature_columns:
        feature_data[column] = pd.to_numeric(
            feature_data[column],
            errors="coerce",
        )

    invalid_columns = (
        feature_data.columns[
            feature_data.isna().any()
        ].tolist()
    )

    if invalid_columns:
        raise ValueError(
            "Inference features contain missing or "
            "non-numeric values in columns: "
            f"{invalid_columns}"
        )

    return feature_data


def generate_predictions(
    dataframe: pd.DataFrame,
    model_path: Path | str = (
        DEFAULT_MODEL_PATH
    ),
    feature_columns_path: Path | str = (
        DEFAULT_FEATURE_COLUMNS_PATH
    ),
    timestamp_column: str = (
        DEFAULT_TIMESTAMP_COLUMN
    ),
    prediction_column: str = (
        DEFAULT_PREDICTION_COLUMN
    ),
) -> pd.DataFrame:
    feature_columns = load_feature_columns(
        feature_columns_path=(
            feature_columns_path
        )
    )

    model = load_forecasting_model(
        model_path=model_path
    )

    feature_data = validate_inference_data(
        dataframe=dataframe,
        feature_columns=feature_columns,
    )

    predictions = model.predict(
        feature_data
    )

    result = pd.DataFrame(
        {
            prediction_column: predictions,
        },
        index=dataframe.index,
    )

    if timestamp_column in dataframe.columns:
        timestamps = pd.to_datetime(
            dataframe[timestamp_column],
            errors="coerce",
        )

        if timestamps.isna().any():
            raise ValueError(
                f"{timestamp_column} contains "
                "invalid timestamp values."
            )

        result.insert(
            0,
            timestamp_column,
            timestamps,
        )

    return result.reset_index(
        drop=True
    )


def save_predictions(
    predictions: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        output_path,
        index=False,
    )

    return output_path


def predict_from_csv(
    input_path: Path | str,
    output_path: Path | str,
    model_path: Path | str = (
        DEFAULT_MODEL_PATH
    ),
    feature_columns_path: Path | str = (
        DEFAULT_FEATURE_COLUMNS_PATH
    ),
    timestamp_column: str = (
        DEFAULT_TIMESTAMP_COLUMN
    ),
    prediction_column: str = (
        DEFAULT_PREDICTION_COLUMN
    ),
) -> pd.DataFrame:
    input_path = Path(
        input_path
    )

    _validate_file(
        file_path=input_path,
        description="Inference CSV file",
    )

    dataframe = pd.read_csv(
        input_path
    )

    predictions = generate_predictions(
        dataframe=dataframe,
        model_path=model_path,
        feature_columns_path=(
            feature_columns_path
        ),
        timestamp_column=timestamp_column,
        prediction_column=prediction_column,
    )

    save_predictions(
        predictions=predictions,
        output_path=output_path,
    )

    return predictions