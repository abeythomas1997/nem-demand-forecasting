# src/monitoring/should_retrain.py

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MAX_MAE = 100.0
DEFAULT_MAX_DRIFT_FAILURES = 5
DEFAULT_MAX_MISSING_FAILURES = 0


def should_retrain_model(
    monitoring_summary: dict[str, Any],
    maximum_mae: float = DEFAULT_MAX_MAE,
    maximum_drift_failures: int = (
        DEFAULT_MAX_DRIFT_FAILURES
    ),
    maximum_missing_failures: int = (
        DEFAULT_MAX_MISSING_FAILURES
    ),
) -> dict[str, Any]:
    if maximum_mae < 0:
        raise ValueError(
            "Maximum MAE cannot be negative."
        )

    if maximum_drift_failures < 0:
        raise ValueError(
            "Maximum drift failures cannot be negative."
        )

    if maximum_missing_failures < 0:
        raise ValueError(
            "Maximum missing failures cannot be negative."
        )

    required_fields = [
        "missing_failure_count",
        "drift_failure_count",
        "performance_status",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in monitoring_summary
    ]

    if missing_fields:
        raise ValueError(
            "Monitoring summary is missing required "
            f"fields: {missing_fields}"
        )

    missing_failure_count = int(
        monitoring_summary[
            "missing_failure_count"
        ]
    )

    drift_failure_count = int(
        monitoring_summary[
            "drift_failure_count"
        ]
    )

    current_mae = monitoring_summary.get(
        "current_mae"
    )

    performance_status = str(
        monitoring_summary[
            "performance_status"
        ]
    )

    reasons: list[str] = []

    if (
        missing_failure_count
        > maximum_missing_failures
    ):
        reasons.append(
            "Missing-value failures exceeded "
            "the allowed threshold."
        )

    if (
        drift_failure_count
        > maximum_drift_failures
    ):
        reasons.append(
            "Feature-drift failures exceeded "
            "the allowed threshold."
        )

    if current_mae is not None:
        current_mae = float(
            current_mae
        )

        if current_mae > maximum_mae:
            reasons.append(
                f"Current MAE {current_mae:.3f} "
                f"exceeded maximum MAE "
                f"{maximum_mae:.3f}."
            )

    elif performance_status == "FAIL":
        reasons.append(
            "Model performance monitoring failed."
        )

    retraining_required = bool(
        reasons
    )

    return {
        "stage": "retraining_decision",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "retraining_required": (
            retraining_required
        ),
        "decision": (
            "RETRAIN"
            if retraining_required
            else "KEEP_CURRENT_MODEL"
        ),
        "thresholds": {
            "maximum_mae": float(
                maximum_mae
            ),
            "maximum_drift_failures": int(
                maximum_drift_failures
            ),
            "maximum_missing_failures": int(
                maximum_missing_failures
            ),
        },
        "observed": {
            "current_mae": current_mae,
            "drift_failure_count": (
                drift_failure_count
            ),
            "missing_failure_count": (
                missing_failure_count
            ),
            "performance_status": (
                performance_status
            ),
        },
        "reasons": (
            reasons
            if reasons
            else [
                "Monitoring results are within "
                "the configured thresholds."
            ]
        ),
    }


def save_retraining_decision(
    decision: dict[str, Any],
    output_path: Path | str,
) -> Path:
    output_path = Path(
        output_path
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
            decision,
            file,
            indent=4,
            allow_nan=False,
        )

    return output_path


def evaluate_and_save_retraining_decision(
    monitoring_summary: dict[str, Any],
    output_path: Path | str,
    maximum_mae: float = DEFAULT_MAX_MAE,
    maximum_drift_failures: int = (
        DEFAULT_MAX_DRIFT_FAILURES
    ),
    maximum_missing_failures: int = (
        DEFAULT_MAX_MISSING_FAILURES
    ),
) -> dict[str, Any]:
    decision = should_retrain_model(
        monitoring_summary=(
            monitoring_summary
        ),
        maximum_mae=maximum_mae,
        maximum_drift_failures=(
            maximum_drift_failures
        ),
        maximum_missing_failures=(
            maximum_missing_failures
        ),
    )

    saved_path = save_retraining_decision(
        decision=decision,
        output_path=output_path,
    )

    return {
        **decision,
        "output_path": str(
            saved_path
        ),
    }