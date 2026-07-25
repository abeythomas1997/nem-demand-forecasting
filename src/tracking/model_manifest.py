# src/tracking/model_manifest.py

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn
import xgboost


def _make_json_serialisable(
    value: Any,
) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(key): _make_json_serialisable(
                item
            )
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _make_json_serialisable(item)
            for item in value
        ]

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    return value


def create_model_manifest(
    model_name: str,
    model_path: Path | str,
    feature_columns_path: Path | str,
    feature_columns: list[str],
    target_column: str,
    timestamp_column: str,
    forecast_horizon: str,
    training_dataframe: pd.DataFrame,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    model_parameters: dict[str, Any] | None,
    output_path: Path | str,
) -> dict[str, Any]:
    if training_dataframe.empty:
        raise ValueError(
            "Training dataframe cannot be empty "
            "when creating the model manifest."
        )

    if timestamp_column not in training_dataframe.columns:
        raise ValueError(
            f"Training dataframe is missing "
            f"{timestamp_column}."
        )

    if not feature_columns:
        raise ValueError(
            "Feature-column list cannot be empty."
        )

    timestamps = pd.to_datetime(
        training_dataframe[timestamp_column],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise ValueError(
            f"{timestamp_column} contains invalid "
            "timestamp values."
        )

    model_path = Path(model_path)
    feature_columns_path = Path(
        feature_columns_path
    )
    output_path = Path(output_path)

    manifest = {
        "manifest_version": "1.0",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "project": {
            "name": "energy-vic",
            "region": "VIC1",
            "problem_type": (
                "operational_demand_forecasting"
            ),
        },
        "model": {
            "name": model_name,
            "framework": "xgboost",
            "model_path": str(model_path),
            "feature_columns_path": str(
                feature_columns_path
            ),
            "parameters": (
                model_parameters or {}
            ),
        },
        "forecast": {
            "target_column": target_column,
            "timestamp_column": (
                timestamp_column
            ),
            "forecast_horizon": (
                forecast_horizon
            ),
        },
        "training_data": {
            "row_count": int(
                len(training_dataframe)
            ),
            "feature_count": int(
                len(feature_columns)
            ),
            "start_timestamp": (
                timestamps.min().isoformat()
            ),
            "end_timestamp": (
                timestamps.max().isoformat()
            ),
        },
        "features": feature_columns,
        "performance": {
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "environment": {
            "python_version": (
                platform.python_version()
            ),
            "platform": platform.platform(),
            "pandas_version": pd.__version__,
            "scikit_learn_version": (
                sklearn.__version__
            ),
            "xgboost_version": (
                xgboost.__version__
            ),
        },
    }

    serialisable_manifest = (
        _make_json_serialisable(
            manifest
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serialisable_manifest,
            file,
            indent=4,
        )

    return serialisable_manifest