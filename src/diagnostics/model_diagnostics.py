# src/diagnostics/model_diagnostics.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _convert_to_serializable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


def _save_json(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_data = {
        key: _convert_to_serializable(value)
        for key, value in data.items()
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            serializable_data,
            file,
            indent=4,
            ensure_ascii=False,
        )


def _calculate_percentage_error(
    actual: pd.Series,
    absolute_error: pd.Series,
) -> pd.Series:
    percentage_error = pd.Series(
        np.nan,
        index=actual.index,
        dtype=float,
    )

    non_zero_actual_mask = actual != 0

    percentage_error.loc[non_zero_actual_mask] = (
        absolute_error.loc[non_zero_actual_mask]
        / actual.loc[non_zero_actual_mask].abs()
        * 100
    )

    return percentage_error


def _validate_inputs(
    timestamps: pd.Series,
    actual_values: pd.Series,
    predicted_values: pd.Series,
) -> None:
    row_counts = {
        len(timestamps),
        len(actual_values),
        len(predicted_values),
    }

    if len(row_counts) != 1:
        raise ValueError(
            "timestamps, actual_values and predicted_values "
            "must contain the same number of rows."
        )

    if len(actual_values) == 0:
        raise ValueError(
            "Model diagnostics cannot be generated from an empty dataset."
        )

    invalid_timestamp_count = timestamps.isna().sum()

    if invalid_timestamp_count > 0:
        raise ValueError(
            f"Found {invalid_timestamp_count} invalid timestamps."
        )

    actual_array = actual_values.to_numpy(dtype=float)
    prediction_array = predicted_values.to_numpy(dtype=float)

    if not np.isfinite(actual_array).all():
        raise ValueError(
            "Actual values contain missing or non-finite values."
        )

    if not np.isfinite(prediction_array).all():
        raise ValueError(
            "Predicted values contain missing or non-finite values."
        )


def generate_model_diagnostics(
    timestamps: pd.Series | np.ndarray | list,
    actual_values: pd.Series | np.ndarray | list,
    predicted_values: pd.Series | np.ndarray | list,
    output_directory: str | Path,
    worst_prediction_count: int = 50,
) -> dict[str, Any]:
    """
    Generate residual and prediction diagnostics for the final test dataset.

    Residual definition:
        residual = actual - prediction

    Positive residual:
        The model underpredicted demand.

    Negative residual:
        The model overpredicted demand.
    """

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    timestamp_series = pd.to_datetime(
        pd.Series(timestamps).reset_index(drop=True),
        errors="coerce",
    )

    actual_series = pd.to_numeric(
        pd.Series(actual_values).reset_index(drop=True),
        errors="coerce",
    )

    prediction_series = pd.to_numeric(
        pd.Series(predicted_values).reset_index(drop=True),
        errors="coerce",
    )

    _validate_inputs(
        timestamps=timestamp_series,
        actual_values=actual_series,
        predicted_values=prediction_series,
    )

    residual = actual_series - prediction_series
    absolute_error = residual.abs()

    percentage_error = _calculate_percentage_error(
        actual=actual_series,
        absolute_error=absolute_error,
    )

    diagnostics_dataframe = pd.DataFrame(
        {
            "timestamp": timestamp_series,
            "actual": actual_series,
            "prediction": prediction_series,
            "residual": residual,
            "absolute_error": absolute_error,
            "percentage_error": percentage_error,
        }
    )

    diagnostics_dataframe = diagnostics_dataframe.sort_values(
        by="timestamp"
    ).reset_index(drop=True)

    valid_percentage_errors = diagnostics_dataframe[
        "percentage_error"
    ].dropna()

    mean_absolute_percentage_error = (
        float(valid_percentage_errors.mean())
        if not valid_percentage_errors.empty
        else None
    )

    residual_summary = {
        "stage": "model_diagnostics",
        "status": "PASS",
        "row_count": len(diagnostics_dataframe),
        "residual_definition": "actual_minus_prediction",
        "mean_residual": float(
            diagnostics_dataframe["residual"].mean()
        ),
        "median_residual": float(
            diagnostics_dataframe["residual"].median()
        ),
        "residual_standard_deviation": float(
            diagnostics_dataframe["residual"].std(ddof=1)
        ),
        "minimum_residual": float(
            diagnostics_dataframe["residual"].min()
        ),
        "maximum_residual": float(
            diagnostics_dataframe["residual"].max()
        ),
        "percentile_05": float(
            diagnostics_dataframe["residual"].quantile(0.05)
        ),
        "percentile_25": float(
            diagnostics_dataframe["residual"].quantile(0.25)
        ),
        "percentile_75": float(
            diagnostics_dataframe["residual"].quantile(0.75)
        ),
        "percentile_95": float(
            diagnostics_dataframe["residual"].quantile(0.95)
        ),
        "residual_skewness": float(
            diagnostics_dataframe["residual"].skew()
        ),
        "mean_absolute_error": float(
            diagnostics_dataframe["absolute_error"].mean()
        ),
        "root_mean_squared_error": float(
            np.sqrt(
                np.mean(
                    np.square(
                        diagnostics_dataframe["residual"]
                    )
                )
            )
        ),
        "mean_absolute_percentage_error": (
            mean_absolute_percentage_error
        ),
        "underprediction_count": int(
            (diagnostics_dataframe["residual"] > 0).sum()
        ),
        "overprediction_count": int(
            (diagnostics_dataframe["residual"] < 0).sum()
        ),
        "exact_prediction_count": int(
            (diagnostics_dataframe["residual"] == 0).sum()
        ),
        "underprediction_percentage": float(
            (
                diagnostics_dataframe["residual"] > 0
            ).mean()
            * 100
        ),
        "overprediction_percentage": float(
            (
                diagnostics_dataframe["residual"] < 0
            ).mean()
            * 100
        ),
        "largest_absolute_error": float(
            diagnostics_dataframe["absolute_error"].max()
        ),
        "percentage_error_missing_count": int(
            diagnostics_dataframe["percentage_error"].isna().sum()
        ),
    }

    worst_prediction_count = min(
        max(int(worst_prediction_count), 1),
        len(diagnostics_dataframe),
    )

    worst_predictions = (
        diagnostics_dataframe.sort_values(
            by="absolute_error",
            ascending=False,
        )
        .head(worst_prediction_count)
        .reset_index(drop=True)
    )

    diagnostics_output_path = (
        output_directory / "prediction_diagnostics.csv"
    )

    worst_predictions_output_path = (
        output_directory / "worst_predictions.csv"
    )

    summary_output_path = (
        output_directory / "residual_summary.json"
    )

    diagnostics_dataframe.to_csv(
        diagnostics_output_path,
        index=False,
    )

    worst_predictions.to_csv(
        worst_predictions_output_path,
        index=False,
    )

    _save_json(
        data=residual_summary,
        output_path=summary_output_path,
    )

    return residual_summary