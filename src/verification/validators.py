from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class PipelineValidationError(RuntimeError):
    """Raised when a critical pipeline quality check fails."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def save_json_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            _json_safe(report),
            file,
            indent=4,
            ensure_ascii=False,
        )


def _require_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    stage: str,
) -> list[str]:
    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]
    if missing_columns:
        raise PipelineValidationError(
            f"{stage}: missing required columns: {missing_columns}"
        )
    return missing_columns


def _timestamp_summary(
    dataframe: pd.DataFrame,
    timestamp_column: str,
) -> dict:
    timestamps = pd.to_datetime(
        dataframe[timestamp_column],
        errors="coerce",
    )

    valid_timestamps = timestamps.dropna().sort_values()
    differences = valid_timestamps.diff().dropna()

    interval_counts = {
        str(interval): int(count)
        for interval, count in differences.value_counts().head(10).items()
    }

    return {
        "invalid_timestamp_count": int(timestamps.isna().sum()),
        "duplicate_timestamp_count": int(timestamps.duplicated().sum()),
        "is_monotonic_increasing": bool(timestamps.is_monotonic_increasing),
        "minimum_timestamp": (
            valid_timestamps.min()
            if not valid_timestamps.empty
            else None
        ),
        "maximum_timestamp": (
            valid_timestamps.max()
            if not valid_timestamps.empty
            else None
        ),
        "most_common_intervals": interval_counts,
    }


def validate_demand_data(
    dataframe: pd.DataFrame,
    timestamp_column: str,
    target_column: str,
    output_path: Path,
) -> dict:
    _require_columns(
        dataframe,
        [timestamp_column, target_column],
        "Demand validation",
    )

    timestamp_summary = _timestamp_summary(
        dataframe,
        timestamp_column,
    )

    target = pd.to_numeric(
        dataframe[target_column],
        errors="coerce",
    )

    report = {
        "stage": "demand",
        "status": "PASS",
        "row_count": int(len(dataframe)),
        "column_count": int(dataframe.shape[1]),
        "timestamp": timestamp_summary,
        "target": {
            "missing_or_non_numeric_count": int(target.isna().sum()),
            "infinite_count": int(
                np.isinf(target.dropna().to_numpy()).sum()
            ),
            "non_positive_count": int((target.dropna() <= 0).sum()),
            "minimum": (
                float(target.min())
                if target.notna().any()
                else None
            ),
            "maximum": (
                float(target.max())
                if target.notna().any()
                else None
            ),
            "mean": (
                float(target.mean())
                if target.notna().any()
                else None
            ),
        },
    }

    failures = []
    if report["row_count"] == 0:
        failures.append("Demand dataframe is empty.")
    if timestamp_summary["invalid_timestamp_count"] > 0:
        failures.append("Invalid demand timestamps found.")
    if timestamp_summary["duplicate_timestamp_count"] > 0:
        failures.append("Duplicate demand timestamps found.")
    if report["target"]["missing_or_non_numeric_count"] > 0:
        failures.append("Missing or non-numeric demand targets found.")
    if report["target"]["infinite_count"] > 0:
        failures.append("Infinite demand targets found.")

    if failures:
        report["status"] = "FAIL"
        report["failures"] = failures

    save_json_report(report, output_path)

    if failures:
        raise PipelineValidationError(
            "Demand validation failed. "
            f"See {output_path}"
        )

    return report


def validate_weather_data(
    dataframe: pd.DataFrame,
    timestamp_column: str,
    output_path: Path,
) -> dict:
    _require_columns(
        dataframe,
        [timestamp_column],
        "Weather validation",
    )

    timestamp_summary = _timestamp_summary(
        dataframe,
        timestamp_column,
    )

    weather_columns = [
        column
        for column in dataframe.columns
        if column != timestamp_column
    ]

    missing_by_column = {
        column: int(dataframe[column].isna().sum())
        for column in weather_columns
        if dataframe[column].isna().any()
    }

    numeric_columns = dataframe[weather_columns].select_dtypes(
        include=[np.number]
    ).columns.tolist()

    infinite_by_column = {}
    for column in numeric_columns:
        count = int(
            np.isinf(
                pd.to_numeric(
                    dataframe[column],
                    errors="coerce",
                ).dropna().to_numpy()
            ).sum()
        )
        if count:
            infinite_by_column[column] = count

    report = {
        "stage": "weather",
        "status": "PASS",
        "row_count": int(len(dataframe)),
        "column_count": int(dataframe.shape[1]),
        "weather_feature_count": int(len(weather_columns)),
        "timestamp": timestamp_summary,
        "missing_value_count": int(
            dataframe[weather_columns].isna().sum().sum()
        ),
        "missing_values_by_column": missing_by_column,
        "infinite_values_by_column": infinite_by_column,
    }

    failures = []
    if report["row_count"] == 0:
        failures.append("Weather dataframe is empty.")
    if timestamp_summary["invalid_timestamp_count"] > 0:
        failures.append("Invalid weather timestamps found.")
    if timestamp_summary["duplicate_timestamp_count"] > 0:
        failures.append("Duplicate weather timestamps found.")
    if infinite_by_column:
        failures.append("Infinite weather values found.")

    if failures:
        report["status"] = "FAIL"
        report["failures"] = failures

    save_json_report(report, output_path)

    if failures:
        raise PipelineValidationError(
            "Weather validation failed. "
            f"See {output_path}"
        )

    return report


def validate_demand_weather_merge(
    demand_dataframe: pd.DataFrame,
    weather_dataframe: pd.DataFrame,
    merged_dataframe: pd.DataFrame,
    timestamp_column: str,
    output_path: Path,
    detail_directory: Path,
    minimum_match_rate: float = 0.99,
) -> dict:
    for name, dataframe in [
        ("Demand", demand_dataframe),
        ("Weather", weather_dataframe),
        ("Merged", merged_dataframe),
    ]:
        _require_columns(
            dataframe,
            [timestamp_column],
            f"{name} merge validation",
        )

    demand_timestamps = pd.to_datetime(
        demand_dataframe[timestamp_column],
        errors="coerce",
    )
    weather_timestamps = pd.to_datetime(
        weather_dataframe[timestamp_column],
        errors="coerce",
    )
    merged_timestamps = pd.to_datetime(
        merged_dataframe[timestamp_column],
        errors="coerce",
    )

    demand_set = set(demand_timestamps.dropna())
    weather_set = set(weather_timestamps.dropna())

    unmatched_demand = sorted(demand_set - weather_set)
    extra_weather = sorted(weather_set - demand_set)
    matched_timestamp_count = len(demand_set & weather_set)

    unique_demand_count = len(demand_set)
    match_rate = (
        matched_timestamp_count / unique_demand_count
        if unique_demand_count
        else 0.0
    )

    weather_columns = [
        "temperature",
        "relative_humidity",
        "apparent_temperature",
        "cloud_cover",
        "wind_speed",
        "solar_radiation",
        "precipitation",
        "is_raining",
    ]
    merged_weather_columns = [
        column
        for column in weather_columns
        if column in merged_dataframe.columns
    ]

    missing_weather_after_merge = {
        column: int(merged_dataframe[column].isna().sum())
        for column in merged_weather_columns
        if merged_dataframe[column].isna().any()
    }

    row_difference = int(
        len(merged_dataframe) - len(demand_dataframe)
    )

    report = {
        "stage": "demand_weather_merge",
        "status": "PASS",
        "demand_rows_before_merge": int(len(demand_dataframe)),
        "weather_rows_before_merge": int(len(weather_dataframe)),
        "merged_rows": int(len(merged_dataframe)),
        "row_difference_vs_demand": row_difference,
        "unique_demand_timestamps": int(unique_demand_count),
        "unique_weather_timestamps": int(len(weather_set)),
        "matched_timestamp_count": int(matched_timestamp_count),
        "unmatched_demand_timestamp_count": int(len(unmatched_demand)),
        "extra_weather_timestamp_count": int(len(extra_weather)),
        "match_rate": float(match_rate),
        "minimum_required_match_rate": float(minimum_match_rate),
        "duplicate_merged_timestamp_count": int(
            merged_timestamps.duplicated().sum()
        ),
        "invalid_merged_timestamp_count": int(
            merged_timestamps.isna().sum()
        ),
        "weather_columns_expected": weather_columns,
        "weather_columns_present_after_merge": merged_weather_columns,
        "missing_weather_value_count_after_merge": int(
            sum(missing_weather_after_merge.values())
        ),
        "missing_weather_values_by_column_after_merge": (
            missing_weather_after_merge
        ),
    }

    detail_directory.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {timestamp_column: unmatched_demand}
    ).to_csv(
        detail_directory / "unmatched_demand_timestamps.csv",
        index=False,
    )

    pd.DataFrame(
        {timestamp_column: extra_weather}
    ).to_csv(
        detail_directory / "extra_weather_timestamps.csv",
        index=False,
    )

    if merged_weather_columns:
        missing_rows = merged_dataframe.loc[
            merged_dataframe[merged_weather_columns]
            .isna()
            .any(axis=1)
        ].copy()

        missing_rows.to_csv(
            detail_directory
            / "merged_rows_with_missing_weather.csv",
            index=False,
        )

    failures = []
    warnings = []

    if match_rate < minimum_match_rate:
        failures.append(
            "Demand-weather timestamp match rate is below "
            f"{minimum_match_rate:.2%}."
        )
    if report["duplicate_merged_timestamp_count"] > 0:
        failures.append(
            "Duplicate timestamps were created by the merge."
        )
    if report["invalid_merged_timestamp_count"] > 0:
        failures.append(
            "Invalid timestamps exist after the merge."
        )
    if row_difference > 0:
        failures.append(
            "The merge expanded the number of demand rows."
        )
    if row_difference < 0:
        warnings.append(
            "The merge removed demand rows."
        )
    if len(merged_weather_columns) != len(weather_columns):
        failures.append(
            "Not all expected weather columns are present "
            "after the merge."
        )
    if report["missing_weather_value_count_after_merge"] > 0:
        warnings.append(
            "Missing weather values exist after the merge."
        )
    if unmatched_demand:
        warnings.append(
            "Some demand timestamps have no exact weather match."
        )

    if warnings:
        report["warnings"] = warnings
    if failures:
        report["status"] = "FAIL"
        report["failures"] = failures
    elif warnings:
        report["status"] = "PASS_WITH_WARNINGS"

    save_json_report(report, output_path)

    if failures:
        raise PipelineValidationError(
            "Demand-weather merge validation failed. "
            f"See {output_path}"
        )

    return report


def validate_training_dataset(
    dataframe: pd.DataFrame,
    timestamp_column: str,
    target_column: str,
    output_path: Path,
) -> dict:
    _require_columns(
        dataframe,
        [timestamp_column, target_column],
        "Training dataset validation",
    )

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in [timestamp_column, target_column]
    ]

    non_numeric_features = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(dataframe[column])
    ]

    numeric_frame = dataframe[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    missing_by_column = {
        column: int(dataframe[column].isna().sum())
        for column in dataframe.columns
        if dataframe[column].isna().any()
    }

    infinite_by_column = {}
    for column in numeric_frame.columns:
        count = int(
            np.isinf(
                numeric_frame[column].dropna().to_numpy()
            ).sum()
        )
        if count:
            infinite_by_column[column] = count

    timestamp_summary = _timestamp_summary(
        dataframe,
        timestamp_column,
    )

    report = {
        "stage": "training_dataset",
        "status": "PASS",
        "row_count": int(len(dataframe)),
        "column_count": int(dataframe.shape[1]),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "target_in_features": bool(
            target_column in feature_columns
        ),
        "timestamp_in_features": bool(
            timestamp_column in feature_columns
        ),
        "non_numeric_features": non_numeric_features,
        "missing_value_count": int(
            dataframe.isna().sum().sum()
        ),
        "missing_values_by_column": missing_by_column,
        "infinite_values_by_column": infinite_by_column,
        "timestamp": timestamp_summary,
    }

    failures = []
    if report["row_count"] == 0:
        failures.append("Training dataset is empty.")
    if report["feature_count"] == 0:
        failures.append("No model features were created.")
    if non_numeric_features:
        failures.append(
            "Non-numeric modelling features were found."
        )
    if report["missing_value_count"] > 0:
        failures.append(
            "Missing values remain in the training dataset."
        )
    if infinite_by_column:
        failures.append(
            "Infinite values remain in the training dataset."
        )
    if timestamp_summary["invalid_timestamp_count"] > 0:
        failures.append(
            "Invalid timestamps remain in the training dataset."
        )
    if timestamp_summary["duplicate_timestamp_count"] > 0:
        failures.append(
            "Duplicate timestamps remain in the training dataset."
        )
    if not timestamp_summary["is_monotonic_increasing"]:
        failures.append(
            "Training timestamps are not chronologically sorted."
        )

    if failures:
        report["status"] = "FAIL"
        report["failures"] = failures

    save_json_report(report, output_path)

    if failures:
        raise PipelineValidationError(
            "Training dataset validation failed. "
            f"See {output_path}"
        )

    return report


def validate_chronological_split(
    full_dataframe: pd.DataFrame,
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    feature_columns: list[str],
    timestamp_column: str,
    target_column: str,
    output_path: Path,
) -> dict:
    for name, dataframe in [
        ("Full", full_dataframe),
        ("Train", train_dataframe),
        ("Validation", validation_dataframe),
        ("Test", test_dataframe),
    ]:
        _require_columns(
            dataframe,
            [timestamp_column, target_column],
            f"{name} split validation",
        )

    train_time = pd.to_datetime(
        train_dataframe[timestamp_column],
        errors="coerce",
    )
    validation_time = pd.to_datetime(
        validation_dataframe[timestamp_column],
        errors="coerce",
    )
    test_time = pd.to_datetime(
        test_dataframe[timestamp_column],
        errors="coerce",
    )

    total_split_rows = (
        len(train_dataframe)
        + len(validation_dataframe)
        + len(test_dataframe)
    )

    train_set = set(train_time.dropna())
    validation_set = set(validation_time.dropna())
    test_set = set(test_time.dropna())

    overlap_counts = {
        "train_validation": int(
            len(train_set & validation_set)
        ),
        "train_test": int(len(train_set & test_set)),
        "validation_test": int(
            len(validation_set & test_set)
        ),
    }

    chronological_boundaries = {
        "train_start": (
            train_time.min()
            if len(train_time)
            else None
        ),
        "train_end": (
            train_time.max()
            if len(train_time)
            else None
        ),
        "validation_start": (
            validation_time.min()
            if len(validation_time)
            else None
        ),
        "validation_end": (
            validation_time.max()
            if len(validation_time)
            else None
        ),
        "test_start": (
            test_time.min()
            if len(test_time)
            else None
        ),
        "test_end": (
            test_time.max()
            if len(test_time)
            else None
        ),
    }

    train_before_validation = bool(
        len(train_time)
        and len(validation_time)
        and train_time.max() < validation_time.min()
    )
    validation_before_test = bool(
        len(validation_time)
        and len(test_time)
        and validation_time.max() < test_time.min()
    )

    missing_feature_columns = [
        column
        for column in feature_columns
        if column not in full_dataframe.columns
    ]

    report = {
        "stage": "chronological_split",
        "status": "PASS",
        "full_row_count": int(len(full_dataframe)),
        "train_row_count": int(len(train_dataframe)),
        "validation_row_count": int(
            len(validation_dataframe)
        ),
        "test_row_count": int(len(test_dataframe)),
        "split_row_total": int(total_split_rows),
        "row_total_matches_full_dataset": bool(
            total_split_rows == len(full_dataframe)
        ),
        "feature_count": int(len(feature_columns)),
        "missing_feature_columns": missing_feature_columns,
        "target_excluded_from_features": bool(
            target_column not in feature_columns
        ),
        "timestamp_excluded_from_features": bool(
            timestamp_column not in feature_columns
        ),
        "timestamp_overlap_counts": overlap_counts,
        "train_before_validation": train_before_validation,
        "validation_before_test": validation_before_test,
        "boundaries": chronological_boundaries,
    }

    failures = []
    if not report["row_total_matches_full_dataset"]:
        failures.append(
            "Train, validation and test rows do not sum "
            "to the full dataset."
        )
    if any(overlap_counts.values()):
        failures.append(
            "Timestamp overlap exists between data splits."
        )
    if not train_before_validation:
        failures.append(
            "Training data does not end before validation data."
        )
    if not validation_before_test:
        failures.append(
            "Validation data does not end before test data."
        )
    if missing_feature_columns:
        failures.append(
            "Some selected feature columns do not exist."
        )
    if not report["target_excluded_from_features"]:
        failures.append(
            "Target leakage: target is included in features."
        )
    if not report["timestamp_excluded_from_features"]:
        failures.append(
            "Raw timestamp is included in model features."
        )
    if min(
        len(train_dataframe),
        len(validation_dataframe),
        len(test_dataframe),
    ) == 0:
        failures.append("At least one split is empty.")

    if failures:
        report["status"] = "FAIL"
        report["failures"] = failures

    save_json_report(report, output_path)

    if failures:
        raise PipelineValidationError(
            "Chronological split validation failed. "
            f"See {output_path}"
        )

    return report


def validate_model_outputs(
    comparison_dataframe: pd.DataFrame,
    test_predictions_dataframe: pd.DataFrame,
    test_metrics: dict,
    selected_model: str,
    output_path: Path,
) -> dict:
    required_comparison_columns = [
        "model",
        "MAE",
        "RMSE",
        "MAPE",
        "R2",
    ]

    missing_comparison_columns = [
        column
        for column in required_comparison_columns
        if column not in comparison_dataframe.columns
    ]

    finite_metric_checks = {}
    for metric_name, metric_value in test_metrics.items():
        try:
            finite_metric_checks[metric_name] = bool(
                np.isfinite(float(metric_value))
            )
        except (TypeError, ValueError):
            finite_metric_checks[metric_name] = False

    prediction_columns = [
        column
        for column in ["actual", "predicted", "residual"]
        if column in test_predictions_dataframe.columns
    ]

    non_finite_prediction_counts = {}
    for column in prediction_columns:
        values = pd.to_numeric(
            test_predictions_dataframe[column],
            errors="coerce",
        )
        count = int(
            (~np.isfinite(values.to_numpy())).sum()
        )
        if count:
            non_finite_prediction_counts[column] = count

    report = {
        "stage": "model_outputs",
        "status": "PASS",
        "selected_model": selected_model,
        "selected_model_present_in_comparison": bool(
            selected_model
            in comparison_dataframe.get(
                "model",
                pd.Series(dtype=str),
            ).astype(str).tolist()
        ),
        "comparison_row_count": int(
            len(comparison_dataframe)
        ),
        "missing_comparison_columns": (
            missing_comparison_columns
        ),
        "test_prediction_row_count": int(
            len(test_predictions_dataframe)
        ),
        "test_prediction_missing_value_count": int(
            test_predictions_dataframe.isna().sum().sum()
        ),
        "non_finite_prediction_counts": (
            non_finite_prediction_counts
        ),
        "finite_test_metrics": finite_metric_checks,
    }

    failures = []
    if missing_comparison_columns:
        failures.append(
            "Model comparison output is missing columns."
        )
    if not report["selected_model_present_in_comparison"]:
        failures.append(
            "Selected model is absent from comparison table."
        )
    if report["test_prediction_row_count"] == 0:
        failures.append("Test predictions are empty.")
    if report["test_prediction_missing_value_count"] > 0:
        failures.append(
            "Missing values exist in test predictions."
        )
    if non_finite_prediction_counts:
        failures.append(
            "Non-finite test prediction values were found."
        )
    if not all(finite_metric_checks.values()):
        failures.append(
            "One or more final test metrics are non-finite."
        )

    if failures:
        report["status"] = "FAIL"
        report["failures"] = failures

    save_json_report(report, output_path)

    if failures:
        raise PipelineValidationError(
            "Model output validation failed. "
            f"See {output_path}"
        )

    return report
