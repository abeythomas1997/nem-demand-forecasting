# src/tracking/experiment_tracker.py

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd


EXPERIMENT_COLUMNS = [
    "run_id",
    "run_timestamp_utc",
    "model_name",
    "selected_model",
    "validation_mae",
    "test_mae",
    "test_rmse",
    "test_mape",
    "test_r2",
    "best_cv_mae",
    "feature_count",
    "training_rows",
    "validation_rows",
    "test_rows",
    "parameters",
]


def _get_metric(
    metrics: dict[str, Any],
    metric_name: str,
) -> float | None:
    possible_names = [
        metric_name,
        metric_name.upper(),
        metric_name.lower(),
    ]

    for name in possible_names:
        if name in metrics:
            value = metrics[name]

            if value is None:
                return None

            return float(value)

    return None


def _serialise_parameters(
    parameters: dict[str, Any] | None,
) -> str:
    if not parameters:
        return "{}"

    serialisable_parameters = {}

    for key, value in parameters.items():
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (TypeError, ValueError):
                pass

        serialisable_parameters[
            str(key)
        ] = value

    return json.dumps(
        serialisable_parameters,
        sort_keys=True,
    )


def create_experiment_record(
    model_name: str,
    selected_model: str,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    best_cv_mae: float,
    feature_count: int,
    training_rows: int,
    validation_rows: int,
    test_rows: int,
    parameters: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    if feature_count <= 0:
        raise ValueError(
            "Feature count must be greater than zero."
        )

    row_counts = {
        "training_rows": training_rows,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
    }

    invalid_row_counts = [
        name
        for name, value in row_counts.items()
        if value <= 0
    ]

    if invalid_row_counts:
        raise ValueError(
            "Dataset row counts must be greater "
            f"than zero: {invalid_row_counts}"
        )

    return {
        "run_id": (
            run_id
            or datetime.now(
                timezone.utc
            ).strftime("%Y%m%dT%H%M%SZ")
            + "_"
            + uuid4().hex[:8]
        ),
        "run_timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_name": model_name,
        "selected_model": selected_model,
        "validation_mae": _get_metric(
            validation_metrics,
            "MAE",
        ),
        "test_mae": _get_metric(
            test_metrics,
            "MAE",
        ),
        "test_rmse": _get_metric(
            test_metrics,
            "RMSE",
        ),
        "test_mape": _get_metric(
            test_metrics,
            "MAPE",
        ),
        "test_r2": _get_metric(
            test_metrics,
            "R2",
        ),
        "best_cv_mae": float(
            best_cv_mae
        ),
        "feature_count": int(
            feature_count
        ),
        "training_rows": int(
            training_rows
        ),
        "validation_rows": int(
            validation_rows
        ),
        "test_rows": int(
            test_rows
        ),
        "parameters": (
            _serialise_parameters(
                parameters
            )
        ),
    }


def append_experiment_record(
    record: dict[str, Any],
    output_path: Path | str,
) -> pd.DataFrame:
    output_path = Path(
        output_path
    )

    missing_columns = [
        column
        for column in EXPERIMENT_COLUMNS
        if column not in record
    ]

    if missing_columns:
        raise ValueError(
            "Experiment record is missing "
            f"columns: {missing_columns}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    new_record_df = pd.DataFrame(
        [
            {
                column: record[column]
                for column in EXPERIMENT_COLUMNS
            }
        ]
    )

    if output_path.exists():
        existing_df = pd.read_csv(
            output_path
        )

        for column in EXPERIMENT_COLUMNS:
            if column not in existing_df.columns:
                existing_df[column] = pd.NA

        existing_df = existing_df[
            EXPERIMENT_COLUMNS
        ]

        experiment_history = pd.concat(
            [
                existing_df,
                new_record_df,
            ],
            ignore_index=True,
        )
    else:
        experiment_history = (
            new_record_df
        )

    experiment_history.to_csv(
        output_path,
        index=False,
    )

    return experiment_history


def track_experiment(
    model_name: str,
    selected_model: str,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    best_cv_mae: float,
    feature_count: int,
    training_rows: int,
    validation_rows: int,
    test_rows: int,
    output_path: Path | str,
    parameters: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    record = create_experiment_record(
        model_name=model_name,
        selected_model=selected_model,
        validation_metrics=(
            validation_metrics
        ),
        test_metrics=test_metrics,
        best_cv_mae=best_cv_mae,
        feature_count=feature_count,
        training_rows=training_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        parameters=parameters,
        run_id=run_id,
    )

    append_experiment_record(
        record=record,
        output_path=output_path,
    )

    return record