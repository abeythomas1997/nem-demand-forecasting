# tests/test_monitoring.py

from __future__ import annotations

import pandas as pd
import pytest

from src.monitoring.monitor_predictions import (
    calculate_feature_drift,
    calculate_missing_value_report,
    calculate_prediction_monitoring_metrics,
    evaluate_monitoring_status,
)
from src.monitoring.should_retrain import (
    should_retrain_model,
)


FEATURE_COLUMNS = [
    "temperature",
    "demand_lag_1",
]


@pytest.fixture
def reference_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temperature": [
                18.0,
                20.0,
                22.0,
                24.0,
                26.0,
            ],
            "demand_lag_1": [
                4500.0,
                4600.0,
                4700.0,
                4800.0,
                4900.0,
            ],
        }
    )


@pytest.fixture
def stable_current_dataframe(
    reference_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    return reference_dataframe.copy()


@pytest.fixture
def drifted_current_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temperature": [
                35.0,
                37.0,
                39.0,
                41.0,
                43.0,
            ],
            "demand_lag_1": [
                7000.0,
                7200.0,
                7400.0,
                7600.0,
                7800.0,
            ],
        }
    )


def test_missing_value_report_passes_when_no_missing_values(
    stable_current_dataframe: pd.DataFrame,
) -> None:
    report = calculate_missing_value_report(
        dataframe=stable_current_dataframe,
        feature_columns=FEATURE_COLUMNS,
        missing_rate_threshold=0.01,
    )

    assert len(report) == 2
    assert report["missing_count"].sum() == 0
    assert set(report["status"]) == {"PASS"}


def test_missing_value_report_detects_failure() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [
                1.0,
                None,
                3.0,
                None,
            ],
            "feature_b": [
                10.0,
                11.0,
                12.0,
                13.0,
            ],
        }
    )

    report = calculate_missing_value_report(
        dataframe=dataframe,
        feature_columns=[
            "feature_a",
            "feature_b",
        ],
        missing_rate_threshold=0.25,
    )

    feature_a_status = report.loc[
        report["feature"] == "feature_a",
        "status",
    ].iloc[0]

    feature_b_status = report.loc[
        report["feature"] == "feature_b",
        "status",
    ].iloc[0]

    assert feature_a_status == "FAIL"
    assert feature_b_status == "PASS"


def test_feature_drift_passes_for_stable_data(
    reference_dataframe: pd.DataFrame,
    stable_current_dataframe: pd.DataFrame,
) -> None:
    report = calculate_feature_drift(
        reference_dataframe=reference_dataframe,
        current_dataframe=stable_current_dataframe,
        feature_columns=FEATURE_COLUMNS,
        psi_warning_threshold=0.10,
        psi_failure_threshold=0.25,
        psi_bins=5,
    )

    assert len(report) == 2
    assert set(report["status"]) == {"PASS"}
    assert (report["psi"] < 0.10).all()


def test_feature_drift_detects_shifted_data(
    reference_dataframe: pd.DataFrame,
    drifted_current_dataframe: pd.DataFrame,
) -> None:
    report = calculate_feature_drift(
        reference_dataframe=reference_dataframe,
        current_dataframe=drifted_current_dataframe,
        feature_columns=FEATURE_COLUMNS,
        psi_warning_threshold=0.10,
        psi_failure_threshold=0.25,
        psi_bins=5,
    )

    assert len(report) == 2
    assert (report["status"] == "FAIL").all()
    assert (report["psi"] >= 0.25).all()


def test_prediction_monitoring_metrics_are_correct() -> None:
    dataframe = pd.DataFrame(
        {
            "actual": [
                100.0,
                200.0,
                300.0,
            ],
            "prediction": [
                90.0,
                210.0,
                290.0,
            ],
        }
    )

    metrics = calculate_prediction_monitoring_metrics(
        prediction_dataframe=dataframe,
        actual_column="actual",
        prediction_column="prediction",
        mae_threshold=100.0,
    )

    assert metrics["row_count"] == 3
    assert metrics["mae"] == pytest.approx(10.0)
    assert metrics["rmse"] == pytest.approx(10.0)
    assert metrics["mape"] == pytest.approx(
        6.111111,
        rel=1e-5,
    )
    assert metrics["status"] == "PASS"
    assert metrics["mae_threshold"] == 100.0


def test_monitoring_status_passes_when_all_checks_pass(
    reference_dataframe: pd.DataFrame,
    stable_current_dataframe: pd.DataFrame,
) -> None:
    missing_report = calculate_missing_value_report(
        dataframe=stable_current_dataframe,
        feature_columns=FEATURE_COLUMNS,
        missing_rate_threshold=0.01,
    )

    drift_report = calculate_feature_drift(
        reference_dataframe=reference_dataframe,
        current_dataframe=stable_current_dataframe,
        feature_columns=FEATURE_COLUMNS,
        psi_warning_threshold=0.10,
        psi_failure_threshold=0.25,
        psi_bins=5,
    )

    prediction_metrics = {
        "row_count": 10,
        "mae": 60.0,
        "rmse": 75.0,
        "mape": 2.5,
        "mae_threshold": 100.0,
        "status": "PASS",
        "reason": "",
    }

    summary = evaluate_monitoring_status(
        missing_value_report=missing_report,
        feature_drift_report=drift_report,
        prediction_metrics=prediction_metrics,
        maximum_drift_failures=5,
    )

    assert summary["status"] == "PASS"
    assert summary["missing_value_status"] == "PASS"
    assert summary["feature_drift_status"] == "PASS"
    assert summary["performance_status"] == "PASS"
    assert summary["missing_failure_count"] == 0
    assert summary["drift_failure_count"] == 0


def test_monitoring_status_fails_when_mae_is_high(
    reference_dataframe: pd.DataFrame,
    stable_current_dataframe: pd.DataFrame,
) -> None:
    missing_report = calculate_missing_value_report(
        dataframe=stable_current_dataframe,
        feature_columns=FEATURE_COLUMNS,
        missing_rate_threshold=0.01,
    )

    drift_report = calculate_feature_drift(
        reference_dataframe=reference_dataframe,
        current_dataframe=stable_current_dataframe,
        feature_columns=FEATURE_COLUMNS,
        psi_warning_threshold=0.10,
        psi_failure_threshold=0.25,
        psi_bins=5,
    )

    prediction_metrics = {
        "row_count": 10,
        "mae": 150.0,
        "rmse": 170.0,
        "mape": 6.0,
        "mae_threshold": 100.0,
        "status": "FAIL",
        "reason": "MAE exceeded threshold.",
    }

    summary = evaluate_monitoring_status(
        missing_value_report=missing_report,
        feature_drift_report=drift_report,
        prediction_metrics=prediction_metrics,
        maximum_drift_failures=5,
    )

    assert summary["status"] == "FAIL"
    assert summary["performance_status"] == "FAIL"
    assert summary["current_mae"] == 150.0


def test_retraining_not_required_for_healthy_model() -> None:
    monitoring_summary = {
        "missing_failure_count": 0,
        "drift_failure_count": 1,
        "performance_status": "PASS",
        "current_mae": 61.0,
    }

    decision = should_retrain_model(
        monitoring_summary=monitoring_summary,
        maximum_mae=100.0,
        maximum_drift_failures=5,
        maximum_missing_failures=0,
    )

    assert decision["retraining_required"] is False
    assert decision["decision"] == "KEEP_CURRENT_MODEL"


def test_retraining_required_when_mae_exceeds_threshold() -> None:
    monitoring_summary = {
        "missing_failure_count": 0,
        "drift_failure_count": 0,
        "performance_status": "FAIL",
        "current_mae": 125.0,
    }

    decision = should_retrain_model(
        monitoring_summary=monitoring_summary,
        maximum_mae=100.0,
        maximum_drift_failures=5,
        maximum_missing_failures=0,
    )

    assert decision["retraining_required"] is True
    assert decision["decision"] == "RETRAIN"


def test_retraining_required_when_drift_failures_exceed_limit() -> None:
    monitoring_summary = {
        "missing_failure_count": 0,
        "drift_failure_count": 7,
        "performance_status": "PASS",
        "current_mae": 60.0,
    }

    decision = should_retrain_model(
        monitoring_summary=monitoring_summary,
        maximum_mae=100.0,
        maximum_drift_failures=5,
        maximum_missing_failures=0,
    )

    assert decision["retraining_required"] is True
    assert decision["decision"] == "RETRAIN"


def test_feature_drift_reports_missing_feature(
    reference_dataframe: pd.DataFrame,
    stable_current_dataframe: pd.DataFrame,
) -> None:
    report = calculate_feature_drift(
        reference_dataframe=reference_dataframe,
        current_dataframe=stable_current_dataframe,
        feature_columns=[
            "temperature",
            "missing_feature",
        ],
    )

    missing_feature_report = report.loc[
        report["feature"] == "missing_feature"
    ].iloc[0]

    assert missing_feature_report["status"] == "FAIL"
    assert missing_feature_report["psi"] == float("inf")
    assert (
        missing_feature_report["reason"]
        == "Feature is missing from the reference dataframe."
    )