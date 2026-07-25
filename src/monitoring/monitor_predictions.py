# src/monitoring/monitor_predictions.py

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_missing_value_report(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    missing_rate_threshold: float = 0.01,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for feature in feature_columns:
        if feature not in dataframe.columns:
            records.append(
                {
                    "feature": feature,
                    "missing_count": len(dataframe),
                    "row_count": len(dataframe),
                    "missing_rate": 1.0,
                    "threshold": missing_rate_threshold,
                    "status": "FAIL",
                    "reason": "Feature is missing from the current dataframe.",
                }
            )
            continue

        missing_count = int(dataframe[feature].isna().sum())
        row_count = int(len(dataframe))

        missing_rate = (
            missing_count / row_count
            if row_count > 0
            else 1.0
        )

        status = (
            "FAIL"
            if missing_rate > missing_rate_threshold
            else "PASS"
        )

        records.append(
            {
                "feature": feature,
                "missing_count": missing_count,
                "row_count": row_count,
                "missing_rate": float(missing_rate),
                "threshold": missing_rate_threshold,
                "status": status,
                "reason": (
                    "Missing rate exceeded threshold."
                    if status == "FAIL"
                    else ""
                ),
            }
        )

    return pd.DataFrame(records)


def calculate_psi(
    reference_series: pd.Series,
    current_series: pd.Series,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    reference_numeric = pd.to_numeric(
        reference_series,
        errors="coerce",
    )

    current_numeric = pd.to_numeric(
        current_series,
        errors="coerce",
    )

    reference_missing_rate = float(
        reference_numeric.isna().mean()
    )

    current_missing_rate = float(
        current_numeric.isna().mean()
    )

    reference_clean = (
        reference_numeric
        .dropna()
        .astype(float)
    )

    current_clean = (
        current_numeric
        .dropna()
        .astype(float)
    )

    if reference_clean.empty and current_clean.empty:
        return 0.0

    if reference_clean.empty or current_clean.empty:
        return float("inf")

    quantiles = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    bin_edges = np.unique(
        reference_clean.quantile(
            quantiles
        ).to_numpy()
    )

    if len(bin_edges) < 2:
        reference_value = float(
            reference_clean.iloc[0]
        )

        current_difference_rate = float(
            (
                ~np.isclose(
                    current_clean.to_numpy(),
                    reference_value,
                )
            ).mean()
        )

        reference_distribution = np.array(
            [
                max(
                    1.0 - reference_missing_rate,
                    epsilon,
                ),
                max(
                    reference_missing_rate,
                    epsilon,
                ),
            ]
        )

        current_distribution = np.array(
            [
                max(
                    (
                        1.0
                        - current_missing_rate
                        - current_difference_rate
                    ),
                    epsilon,
                ),
                max(
                    current_missing_rate
                    + current_difference_rate,
                    epsilon,
                ),
            ]
        )

    else:
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        reference_counts, _ = np.histogram(
            reference_clean,
            bins=bin_edges,
        )

        current_counts, _ = np.histogram(
            current_clean,
            bins=bin_edges,
        )

        reference_non_missing_rate = (
            1.0 - reference_missing_rate
        )

        current_non_missing_rate = (
            1.0 - current_missing_rate
        )

        reference_distribution = (
            reference_counts
            / max(
                reference_counts.sum(),
                1,
            )
        ) * reference_non_missing_rate

        current_distribution = (
            current_counts
            / max(
                current_counts.sum(),
                1,
            )
        ) * current_non_missing_rate

        reference_distribution = np.append(
            reference_distribution,
            reference_missing_rate,
        )

        current_distribution = np.append(
            current_distribution,
            current_missing_rate,
        )

        reference_distribution = np.clip(
            reference_distribution,
            epsilon,
            None,
        )

        current_distribution = np.clip(
            current_distribution,
            epsilon,
            None,
        )

    psi_values = (
        current_distribution
        - reference_distribution
    ) * np.log(
        current_distribution
        / reference_distribution
    )

    return float(np.sum(psi_values))


def calculate_feature_drift(
    reference_dataframe: pd.DataFrame,
    current_dataframe: pd.DataFrame,
    feature_columns: list[str],
    psi_warning_threshold: float = 0.10,
    psi_failure_threshold: float = 0.25,
    psi_bins: int = 10,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for feature in feature_columns:
        if feature not in reference_dataframe.columns:
            records.append(
                {
                    "feature": feature,
                    "psi": float("inf"),
                    "warning_threshold": psi_warning_threshold,
                    "failure_threshold": psi_failure_threshold,
                    "status": "FAIL",
                    "reason": (
                        "Feature is missing from the "
                        "reference dataframe."
                    ),
                }
            )
            continue

        if feature not in current_dataframe.columns:
            records.append(
                {
                    "feature": feature,
                    "psi": float("inf"),
                    "warning_threshold": psi_warning_threshold,
                    "failure_threshold": psi_failure_threshold,
                    "status": "FAIL",
                    "reason": (
                        "Feature is missing from the "
                        "current dataframe."
                    ),
                }
            )
            continue

        psi = calculate_psi(
            reference_series=(
                reference_dataframe[feature]
            ),
            current_series=(
                current_dataframe[feature]
            ),
            bins=psi_bins,
        )

        if not np.isfinite(psi):
            status = "FAIL"
        elif psi >= psi_failure_threshold:
            status = "FAIL"
        elif psi >= psi_warning_threshold:
            status = "WARNING"
        else:
            status = "PASS"

        records.append(
            {
                "feature": feature,
                "psi": float(psi),
                "warning_threshold": (
                    psi_warning_threshold
                ),
                "failure_threshold": (
                    psi_failure_threshold
                ),
                "status": status,
                "reason": (
                    "PSI exceeded failure threshold."
                    if status == "FAIL"
                    else (
                        "PSI exceeded warning threshold."
                        if status == "WARNING"
                        else ""
                    )
                ),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            by="psi",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def calculate_prediction_monitoring_metrics(
    prediction_dataframe: pd.DataFrame,
    actual_column: str,
    prediction_column: str,
    mae_threshold: float,
) -> dict[str, Any]:
    valid_predictions = (
        prediction_dataframe[
            [
                actual_column,
                prediction_column,
            ]
        ]
        .dropna()
        .copy()
    )

    if valid_predictions.empty:
        return {
            "row_count": 0,
            "mae": None,
            "rmse": None,
            "mape": None,
            "mae_threshold": mae_threshold,
            "status": "FAIL",
            "reason": (
                "No valid actual and prediction "
                "pairs were available."
            ),
        }

    actual = valid_predictions[
        actual_column
    ].astype(float)

    prediction = valid_predictions[
        prediction_column
    ].astype(float)

    mae = float(
        mean_absolute_error(
            actual,
            prediction,
        )
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                actual,
                prediction,
            )
        )
    )

    non_zero_mask = actual != 0

    if non_zero_mask.any():
        mape = float(
            np.mean(
                np.abs(
                    (
                        actual[non_zero_mask]
                        - prediction[non_zero_mask]
                    )
                    / actual[non_zero_mask]
                )
            )
            * 100.0
        )
    else:
        mape = None

    status = (
        "FAIL"
        if mae > mae_threshold
        else "PASS"
    )

    return {
        "row_count": int(
            len(valid_predictions)
        ),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "mae_threshold": mae_threshold,
        "status": status,
        "reason": (
            "MAE exceeded threshold."
            if status == "FAIL"
            else ""
        ),
    }


def evaluate_monitoring_status(
    missing_value_report: pd.DataFrame,
    feature_drift_report: pd.DataFrame,
    prediction_metrics: dict[str, Any],
    maximum_drift_failures: int = 5,
) -> dict[str, Any]:
    missing_failure_count = int(
        (
            missing_value_report["status"]
            == "FAIL"
        ).sum()
    )

    drift_failure_count = int(
        (
            feature_drift_report["status"]
            == "FAIL"
        ).sum()
    )

    drift_warning_count = int(
        (
            feature_drift_report["status"]
            == "WARNING"
        ).sum()
    )

    missing_value_status = (
        "FAIL"
        if missing_failure_count > 0
        else "PASS"
    )

    performance_status = str(
        prediction_metrics.get(
            "status",
            "FAIL",
        )
    )

    if (
        drift_failure_count
        > maximum_drift_failures
    ):
        feature_drift_status = "FAIL"

    elif (
        drift_failure_count > 0
        or drift_warning_count > 0
    ):
        feature_drift_status = "WARNING"

    else:
        feature_drift_status = "PASS"

    hard_failure_exists = (
        missing_value_status == "FAIL"
        or performance_status == "FAIL"
        or feature_drift_status == "FAIL"
    )

    warning_exists = (
        feature_drift_status == "WARNING"
    )

    if hard_failure_exists:
        overall_status = "FAIL"
    elif warning_exists:
        overall_status = "WARNING"
    else:
        overall_status = "PASS"

    reasons: list[str] = []

    if missing_failure_count > 0:
        reasons.append(
            f"{missing_failure_count} features exceeded "
            "the missing-value threshold."
        )

    if (
        drift_failure_count
        > maximum_drift_failures
    ):
        reasons.append(
            f"{drift_failure_count} features exceeded "
            "the PSI failure threshold."
        )

    elif drift_failure_count > 0:
        reasons.append(
            f"{drift_failure_count} features exceeded "
            "the PSI failure threshold, but the allowed "
            f"monitoring limit is {maximum_drift_failures}."
        )

    if drift_warning_count > 0:
        reasons.append(
            f"{drift_warning_count} features reached "
            "the PSI warning threshold."
        )

    if performance_status == "FAIL":
        reasons.append(
            "Prediction performance exceeded "
            "the configured MAE threshold."
        )

    return {
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "status": overall_status,
        "missing_value_status": (
            missing_value_status
        ),
        "feature_drift_status": (
            feature_drift_status
        ),
        "performance_status": (
            performance_status
        ),
        "missing_failure_count": (
            missing_failure_count
        ),
        "drift_failure_count": (
            drift_failure_count
        ),
        "drift_warning_count": (
            drift_warning_count
        ),
        "maximum_drift_failures": (
            maximum_drift_failures
        ),
        "mae_threshold": (
            prediction_metrics.get(
                "mae_threshold"
            )
        ),
        "current_mae": (
            prediction_metrics.get(
                "mae"
            )
        ),
        "reasons": reasons,
    }


def save_monitoring_outputs(
    output_directory: str | Path,
    missing_value_report: pd.DataFrame,
    feature_drift_report: pd.DataFrame,
    prediction_metrics: dict[str, Any],
    monitoring_summary: dict[str, Any],
) -> None:
    output_path = Path(
        output_directory
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    missing_value_report.to_csv(
        output_path
        / "missing_value_monitoring.csv",
        index=False,
    )

    feature_drift_report.to_csv(
        output_path
        / "feature_drift_monitoring.csv",
        index=False,
    )

    with (
        output_path
        / "prediction_monitoring_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            prediction_metrics,
            file,
            indent=4,
            allow_nan=False,
        )

    with (
        output_path
        / "monitoring_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            monitoring_summary,
            file,
            indent=4,
            allow_nan=False,
        )


def run_monitoring(
    reference_dataframe: pd.DataFrame,
    current_dataframe: pd.DataFrame,
    feature_columns: list[str],
    output_directory: str | Path,
    prediction_dataframe: pd.DataFrame,
    actual_column: str,
    prediction_column: str,
    missing_rate_threshold: float = 0.01,
    psi_warning_threshold: float = 0.10,
    psi_failure_threshold: float = 0.25,
    psi_bins: int = 10,
    mae_threshold: float = 100.0,
    maximum_drift_failures: int = 5,
) -> dict[str, Any]:
    missing_value_report = (
        calculate_missing_value_report(
            dataframe=current_dataframe,
            feature_columns=feature_columns,
            missing_rate_threshold=(
                missing_rate_threshold
            ),
        )
    )

    feature_drift_report = (
        calculate_feature_drift(
            reference_dataframe=(
                reference_dataframe
            ),
            current_dataframe=(
                current_dataframe
            ),
            feature_columns=feature_columns,
            psi_warning_threshold=(
                psi_warning_threshold
            ),
            psi_failure_threshold=(
                psi_failure_threshold
            ),
            psi_bins=psi_bins,
        )
    )

    prediction_metrics = (
        calculate_prediction_monitoring_metrics(
            prediction_dataframe=(
                prediction_dataframe
            ),
            actual_column=actual_column,
            prediction_column=(
                prediction_column
            ),
            mae_threshold=mae_threshold,
        )
    )

    monitoring_summary = (
        evaluate_monitoring_status(
            missing_value_report=(
                missing_value_report
            ),
            feature_drift_report=(
                feature_drift_report
            ),
            prediction_metrics=(
                prediction_metrics
            ),
            maximum_drift_failures=(
                maximum_drift_failures
            ),
        )
    )

    save_monitoring_outputs(
        output_directory=(
            output_directory
        ),
        missing_value_report=(
            missing_value_report
        ),
        feature_drift_report=(
            feature_drift_report
        ),
        prediction_metrics=(
            prediction_metrics
        ),
        monitoring_summary=(
            monitoring_summary
        ),
    )

    return monitoring_summary