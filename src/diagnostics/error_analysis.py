# src/diagnostics/error_analysis.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def _calculate_group_metrics(group: pd.DataFrame) -> pd.Series:
    actual = group["actual"].to_numpy(dtype=float)
    predicted = group["prediction"].to_numpy(dtype=float)

    residual = actual - predicted
    absolute_error = np.abs(residual)
    squared_error = np.square(residual)

    non_zero_actual = actual != 0
    percentage_error = np.full(actual.shape, np.nan, dtype=float)

    percentage_error[non_zero_actual] = (
        absolute_error[non_zero_actual]
        / np.abs(actual[non_zero_actual])
        * 100
    )

    return pd.Series(
        {
            "count": len(group),
            "mae": float(np.mean(absolute_error)),
            "median_absolute_error": float(np.median(absolute_error)),
            "rmse": float(np.sqrt(np.mean(squared_error))),
            "mape": (
                float(np.nanmean(percentage_error))
                if np.any(non_zero_actual)
                else None
            ),
            "mean_residual": float(np.mean(residual)),
            "underprediction_percentage": float(
                np.mean(residual > 0) * 100
            ),
            "overprediction_percentage": float(
                np.mean(residual < 0) * 100
            ),
        }
    )


def _prepare_prediction_data(
    timestamps: pd.Series,
    actual_values: pd.Series,
    predicted_values: pd.Series,
) -> pd.DataFrame:
    prediction_data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                timestamps,
                errors="coerce",
            ),
            "actual": pd.to_numeric(
                actual_values,
                errors="coerce",
            ),
            "prediction": pd.to_numeric(
                predicted_values,
                errors="coerce",
            ),
        }
    )

    invalid_rows = prediction_data.isna().any(axis=1)

    if invalid_rows.any():
        invalid_count = int(invalid_rows.sum())
        raise ValueError(
            "Time-based error analysis received "
            f"{invalid_count} rows containing invalid timestamps "
            "or missing numeric values."
        )

    if prediction_data.empty:
        raise ValueError(
            "Time-based error analysis received no prediction rows."
        )

    if prediction_data["timestamp"].duplicated().any():
        duplicate_count = int(
            prediction_data["timestamp"].duplicated().sum()
        )
        raise ValueError(
            "Time-based error analysis received "
            f"{duplicate_count} duplicate timestamps."
        )

    prediction_data = prediction_data.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    prediction_data["hour"] = prediction_data[
        "timestamp"
    ].dt.hour

    prediction_data["day_of_week"] = prediction_data[
        "timestamp"
    ].dt.day_name()

    prediction_data["day_number"] = prediction_data[
        "timestamp"
    ].dt.dayofweek

    prediction_data["month"] = prediction_data[
        "timestamp"
    ].dt.month_name()

    prediction_data["month_number"] = prediction_data[
        "timestamp"
    ].dt.month

    prediction_data["period_type"] = np.where(
        prediction_data["day_number"] < 5,
        "weekday",
        "weekend",
    )

    return prediction_data


def _build_hourly_error_report(
    prediction_data: pd.DataFrame,
) -> pd.DataFrame:
    hourly_error = (
        prediction_data.groupby(
            "hour",
            observed=True,
            sort=True,
        )
        .apply(
            _calculate_group_metrics,
            include_groups=False,
        )
        .reset_index()
    )

    hourly_error["hour_label"] = hourly_error["hour"].map(
        lambda hour: f"{int(hour):02d}:00"
    )

    column_order = [
        "hour",
        "hour_label",
        "count",
        "mae",
        "median_absolute_error",
        "rmse",
        "mape",
        "mean_residual",
        "underprediction_percentage",
        "overprediction_percentage",
    ]

    return hourly_error[column_order]


def _build_day_of_week_error_report(
    prediction_data: pd.DataFrame,
) -> pd.DataFrame:
    day_of_week_error = (
        prediction_data.groupby(
            ["day_number", "day_of_week"],
            observed=True,
            sort=True,
        )
        .apply(
            _calculate_group_metrics,
            include_groups=False,
        )
        .reset_index()
        .sort_values("day_number")
        .reset_index(drop=True)
    )

    column_order = [
        "day_number",
        "day_of_week",
        "count",
        "mae",
        "median_absolute_error",
        "rmse",
        "mape",
        "mean_residual",
        "underprediction_percentage",
        "overprediction_percentage",
    ]

    return day_of_week_error[column_order]


def _build_month_error_report(
    prediction_data: pd.DataFrame,
) -> pd.DataFrame:
    month_error = (
        prediction_data.groupby(
            ["month_number", "month"],
            observed=True,
            sort=True,
        )
        .apply(
            _calculate_group_metrics,
            include_groups=False,
        )
        .reset_index()
        .sort_values("month_number")
        .reset_index(drop=True)
    )

    column_order = [
        "month_number",
        "month",
        "count",
        "mae",
        "median_absolute_error",
        "rmse",
        "mape",
        "mean_residual",
        "underprediction_percentage",
        "overprediction_percentage",
    ]

    return month_error[column_order]


def _build_weekday_vs_weekend_report(
    prediction_data: pd.DataFrame,
) -> dict[str, Any]:
    grouped_metrics = (
        prediction_data.groupby(
            "period_type",
            observed=True,
            sort=False,
        )
        .apply(
            _calculate_group_metrics,
            include_groups=False,
        )
    )

    report: dict[str, Any] = {
        "stage": "weekday_vs_weekend_error_analysis",
        "status": "PASS",
        "total_row_count": int(len(prediction_data)),
    }

    for period_type in ["weekday", "weekend"]:
        if period_type not in grouped_metrics.index:
            report[period_type] = {
                "count": 0,
                "mae": None,
                "median_absolute_error": None,
                "rmse": None,
                "mape": None,
                "mean_residual": None,
                "underprediction_percentage": None,
                "overprediction_percentage": None,
            }
            continue

        period_metrics = grouped_metrics.loc[period_type]

        report[period_type] = {
            "count": int(period_metrics["count"]),
            "mae": float(period_metrics["mae"]),
            "median_absolute_error": float(
                period_metrics["median_absolute_error"]
            ),
            "rmse": float(period_metrics["rmse"]),
            "mape": (
                float(period_metrics["mape"])
                if pd.notna(period_metrics["mape"])
                else None
            ),
            "mean_residual": float(
                period_metrics["mean_residual"]
            ),
            "underprediction_percentage": float(
                period_metrics[
                    "underprediction_percentage"
                ]
            ),
            "overprediction_percentage": float(
                period_metrics[
                    "overprediction_percentage"
                ]
            ),
        }

    return report


def generate_time_based_error_analysis(
    timestamps: pd.Series,
    actual_values: pd.Series,
    predicted_values: pd.Series,
    output_directory: Path | str,
) -> dict[str, Any]:
    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_data = _prepare_prediction_data(
        timestamps=timestamps,
        actual_values=actual_values,
        predicted_values=predicted_values,
    )

    hourly_error = _build_hourly_error_report(
        prediction_data
    )

    day_of_week_error = _build_day_of_week_error_report(
        prediction_data
    )

    month_error = _build_month_error_report(
        prediction_data
    )

    weekday_vs_weekend = (
        _build_weekday_vs_weekend_report(
            prediction_data
        )
    )

    hourly_output_path = (
        output_directory / "hourly_error.csv"
    )

    day_of_week_output_path = (
        output_directory / "day_of_week_error.csv"
    )

    month_output_path = (
        output_directory / "month_error.csv"
    )

    weekday_weekend_output_path = (
        output_directory / "weekday_vs_weekend.json"
    )

    hourly_error.to_csv(
        hourly_output_path,
        index=False,
    )

    day_of_week_error.to_csv(
        day_of_week_output_path,
        index=False,
    )

    month_error.to_csv(
        month_output_path,
        index=False,
    )

    with weekday_weekend_output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            weekday_vs_weekend,
            output_file,
            indent=4,
        )

    return {
        "stage": "time_based_error_analysis",
        "status": "PASS",
        "row_count": int(len(prediction_data)),
        "hour_group_count": int(len(hourly_error)),
        "day_of_week_group_count": int(
            len(day_of_week_error)
        ),
        "month_group_count": int(len(month_error)),
        "outputs": {
            "hourly_error": str(hourly_output_path),
            "day_of_week_error": str(
                day_of_week_output_path
            ),
            "month_error": str(month_output_path),
            "weekday_vs_weekend": str(
                weekday_weekend_output_path
            ),
        },
    }