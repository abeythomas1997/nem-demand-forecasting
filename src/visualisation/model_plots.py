# src/visualisation/model_plots.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_PREDICTION_COLUMNS = {
    "timestamp",
    "actual",
    "prediction",
    "residual",
    "absolute_error",
}

REQUIRED_ERROR_COLUMNS = {
    "mae",
}


def _validate_file(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required visualisation input was not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Visualisation input is not a file: {file_path}"
        )


def _validate_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    file_name: str,
) -> None:
    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"{file_name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def _save_plot(
    output_path: Path,
) -> None:
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def _load_prediction_diagnostics(
    file_path: Path,
) -> pd.DataFrame:
    _validate_file(file_path)

    dataframe = pd.read_csv(file_path)

    _validate_columns(
        dataframe=dataframe,
        required_columns=REQUIRED_PREDICTION_COLUMNS,
        file_name=file_path.name,
    )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        errors="coerce",
    )

    numeric_columns = [
        "actual",
        "prediction",
        "residual",
        "absolute_error",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    required_data_columns = [
        "timestamp",
        *numeric_columns,
    ]

    if dataframe[
        required_data_columns
    ].isna().any().any():
        raise ValueError(
            f"{file_path.name} contains missing or invalid values."
        )

    if dataframe.empty:
        raise ValueError(
            f"{file_path.name} contains no prediction rows."
        )

    return (
        dataframe
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def _load_error_report(
    file_path: Path,
    grouping_columns: list[str],
) -> pd.DataFrame:
    _validate_file(file_path)

    dataframe = pd.read_csv(file_path)

    required_columns = (
        REQUIRED_ERROR_COLUMNS
        | set(grouping_columns)
    )

    _validate_columns(
        dataframe=dataframe,
        required_columns=required_columns,
        file_name=file_path.name,
    )

    dataframe["mae"] = pd.to_numeric(
        dataframe["mae"],
        errors="coerce",
    )

    if dataframe["mae"].isna().any():
        raise ValueError(
            f"{file_path.name} contains invalid MAE values."
        )

    if dataframe.empty:
        raise ValueError(
            f"{file_path.name} contains no error-analysis rows."
        )

    return dataframe


def _plot_actual_vs_prediction(
    prediction_data: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(15, 6))

    plt.plot(
        prediction_data["timestamp"],
        prediction_data["actual"],
        label="Actual demand",
        linewidth=1.2,
    )

    plt.plot(
        prediction_data["timestamp"],
        prediction_data["prediction"],
        label="Predicted demand",
        linewidth=1.2,
        alpha=0.85,
    )

    plt.title(
        "Actual vs Predicted Operational Demand"
    )
    plt.xlabel("Timestamp")
    plt.ylabel("Operational demand (MW)")
    plt.legend()
    plt.grid(alpha=0.25)

    _save_plot(output_path)


def _plot_actual_vs_prediction_scatter(
    prediction_data: pd.DataFrame,
    output_path: Path,
) -> None:
    actual = prediction_data["actual"]
    prediction = prediction_data["prediction"]

    minimum_value = float(
        min(
            actual.min(),
            prediction.min(),
        )
    )

    maximum_value = float(
        max(
            actual.max(),
            prediction.max(),
        )
    )

    plt.figure(figsize=(8, 8))

    plt.scatter(
        actual,
        prediction,
        alpha=0.35,
        s=18,
    )

    plt.plot(
        [
            minimum_value,
            maximum_value,
        ],
        [
            minimum_value,
            maximum_value,
        ],
        linestyle="--",
        linewidth=1.5,
        label="Ideal prediction",
    )

    plt.title(
        "Actual vs Predicted Demand"
    )
    plt.xlabel("Actual demand (MW)")
    plt.ylabel("Predicted demand (MW)")
    plt.xlim(
        minimum_value,
        maximum_value,
    )
    plt.ylim(
        minimum_value,
        maximum_value,
    )
    plt.legend()
    plt.grid(alpha=0.25)

    _save_plot(output_path)


def _plot_residual_histogram(
    prediction_data: pd.DataFrame,
    output_path: Path,
) -> None:
    mean_residual = float(
        prediction_data["residual"].mean()
    )

    plt.figure(figsize=(10, 6))

    plt.hist(
        prediction_data["residual"],
        bins=40,
        edgecolor="black",
        alpha=0.8,
    )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=1.5,
        label="Zero residual",
    )

    plt.axvline(
        mean_residual,
        linestyle=":",
        linewidth=1.5,
        label=(
            f"Mean residual: "
            f"{mean_residual:.1f} MW"
        ),
    )

    plt.title("Residual Distribution")
    plt.xlabel(
        "Residual: actual − prediction (MW)"
    )
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(
        axis="y",
        alpha=0.25,
    )

    _save_plot(output_path)


def _plot_residual_time_series(
    prediction_data: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(15, 6))

    plt.plot(
        prediction_data["timestamp"],
        prediction_data["residual"],
        linewidth=0.9,
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1.3,
    )

    plt.title(
        "Forecast Residuals Over Time"
    )
    plt.xlabel("Timestamp")
    plt.ylabel(
        "Residual: actual − prediction (MW)"
    )
    plt.grid(alpha=0.25)

    _save_plot(output_path)


def _plot_hourly_error(
    hourly_error: pd.DataFrame,
    output_path: Path,
) -> None:
    hourly_error = (
        hourly_error
        .sort_values("hour")
        .reset_index(drop=True)
    )

    if "hour_label" in hourly_error.columns:
        labels = hourly_error[
            "hour_label"
        ].astype(str)
    else:
        labels = hourly_error["hour"].map(
            lambda value: f"{int(value):02d}:00"
        )

    plt.figure(figsize=(13, 6))

    plt.bar(
        labels,
        hourly_error["mae"],
    )

    plt.title(
        "Mean Absolute Error by Hour"
    )
    plt.xlabel("Hour of day")
    plt.ylabel("MAE (MW)")
    plt.xticks(rotation=45)
    plt.grid(
        axis="y",
        alpha=0.25,
    )

    _save_plot(output_path)


def _plot_day_of_week_error(
    day_of_week_error: pd.DataFrame,
    output_path: Path,
) -> None:
    if "day_number" in day_of_week_error.columns:
        day_of_week_error = (
            day_of_week_error
            .sort_values("day_number")
            .reset_index(drop=True)
        )

    plt.figure(figsize=(10, 6))

    plt.bar(
        day_of_week_error["day_of_week"],
        day_of_week_error["mae"],
    )

    plt.title(
        "Mean Absolute Error by Day of Week"
    )
    plt.xlabel("Day of week")
    plt.ylabel("MAE (MW)")
    plt.xticks(rotation=30)
    plt.grid(
        axis="y",
        alpha=0.25,
    )

    _save_plot(output_path)


def _plot_month_error(
    month_error: pd.DataFrame,
    output_path: Path,
) -> None:
    if "month_number" in month_error.columns:
        month_error = (
            month_error
            .sort_values("month_number")
            .reset_index(drop=True)
        )

    plt.figure(figsize=(10, 6))

    plt.bar(
        month_error["month"],
        month_error["mae"],
    )

    plt.title(
        "Mean Absolute Error by Month"
    )
    plt.xlabel("Month")
    plt.ylabel("MAE (MW)")
    plt.xticks(rotation=30)
    plt.grid(
        axis="y",
        alpha=0.25,
    )

    _save_plot(output_path)


def generate_model_visualisations(
    diagnostics_directory: Path | str,
    output_directory: Path | str,
) -> dict[str, Any]:
    diagnostics_directory = Path(
        diagnostics_directory
    )

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_path = (
        diagnostics_directory
        / "prediction_diagnostics.csv"
    )

    hourly_error_path = (
        diagnostics_directory
        / "hourly_error.csv"
    )

    day_of_week_error_path = (
        diagnostics_directory
        / "day_of_week_error.csv"
    )

    month_error_path = (
        diagnostics_directory
        / "month_error.csv"
    )

    prediction_data = (
        _load_prediction_diagnostics(
            prediction_path
        )
    )

    hourly_error = _load_error_report(
        file_path=hourly_error_path,
        grouping_columns=["hour"],
    )

    day_of_week_error = _load_error_report(
        file_path=day_of_week_error_path,
        grouping_columns=["day_of_week"],
    )

    month_error = _load_error_report(
        file_path=month_error_path,
        grouping_columns=["month"],
    )

    output_paths = {
        "actual_vs_prediction": (
            output_directory
            / "actual_vs_prediction.png"
        ),
        "actual_vs_prediction_scatter": (
            output_directory
            / "actual_vs_prediction_scatter.png"
        ),
        "residual_histogram": (
            output_directory
            / "residual_histogram.png"
        ),
        "residual_time_series": (
            output_directory
            / "residual_time_series.png"
        ),
        "hourly_error": (
            output_directory
            / "hourly_error.png"
        ),
        "day_of_week_error": (
            output_directory
            / "day_of_week_error.png"
        ),
        "month_error": (
            output_directory
            / "month_error.png"
        ),
    }

    _plot_actual_vs_prediction(
        prediction_data=prediction_data,
        output_path=output_paths[
            "actual_vs_prediction"
        ],
    )

    _plot_actual_vs_prediction_scatter(
        prediction_data=prediction_data,
        output_path=output_paths[
            "actual_vs_prediction_scatter"
        ],
    )

    _plot_residual_histogram(
        prediction_data=prediction_data,
        output_path=output_paths[
            "residual_histogram"
        ],
    )

    _plot_residual_time_series(
        prediction_data=prediction_data,
        output_path=output_paths[
            "residual_time_series"
        ],
    )

    _plot_hourly_error(
        hourly_error=hourly_error,
        output_path=output_paths[
            "hourly_error"
        ],
    )

    _plot_day_of_week_error(
        day_of_week_error=day_of_week_error,
        output_path=output_paths[
            "day_of_week_error"
        ],
    )

    _plot_month_error(
        month_error=month_error,
        output_path=output_paths[
            "month_error"
        ],
    )

    return {
        "stage": "model_visualisation",
        "status": "PASS",
        "plot_count": len(output_paths),
        "outputs": {
            plot_name: str(plot_path)
            for plot_name, plot_path
            in output_paths.items()
        },
    }