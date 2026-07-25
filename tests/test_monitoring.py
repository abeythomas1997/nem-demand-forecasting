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
def stable_current_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temperature": [
                18.5,
                20.5,
                22.5,
                24.5,
                26.5,
            ],
            "demand_lag_1": [
                4520.0,
                4620.0,
                4720.0,
                4820.0,
                4920.0,
            ],
        }
    )


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
        feature_columns=[
            "temperature",
            "demand_lag_1",
        ],
        mean_drift_threshold=0.20,
        std_drift_threshold=0.20,
    )

    assert len(report) == 2
    assert set(report["status"]) == {"PASS"}


def test_feature_drift_detects_shifted_data(
    reference_dataframe: pd.DataFrame,
    drifted_current_dataframe: pd.DataFrame,
) -> None:
    report = calculate_feature_drift(
        reference_dataframe=reference_dataframe,
        current_dataframe=drifted_current_dataframe,
        feature_columns=[
            "temperature",
            "demand_lag_1",
        ],
        mean_drift_threshold=0.20,
        std_drift_threshold=0.20,
    )

    assert (
        report["status"] == "FAIL"
    ).all()

    assert (
        report["mean_relative_change"]
        > 0.20
    ).all()


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
        dataframe=dataframe,
        actual_column="actual",
        prediction_column="prediction",
    )

    assert metrics["row_count"] == 3
    assert metrics["mae"] == pytest.approx(
        10.0
    )
    assert metrics["rmse"] == pytest.approx(
        10.0
    )
    assert metrics[
        "mean_residual"
    ] == pytest.approx(
        10.0 / 3.0
    )
    assert metrics[
        "maximum_absolute_error"
    ] == pytest.approx(
        10.0
    )


def test_monitoring_status_passes_when_all_checks_pass(
    reference_dataframe: pd.DataFrame,
    stable_current_dataframe: pd.DataFrame,
) -> None:
    missing_report = calculate_missing_value_report(
        dataframe=stable_current_dataframe,
        missing_rate_threshold=0.01,
    )

    drift_report = calculate_feature_drift(
        reference_dataframe=reference_dataframe,
        current_dataframe=stable_current_dataframe,
        feature_columns=[
            "temperature",
            "demand_lag_1",
        ],
        mean_drift_threshold=0.20,
        std_drift_threshold=0.20,
    )

    prediction_metrics = {
        "mae": 60.0,
    }

    summary = evaluate_monitoring_status(
        missing_report=missing_report,
        drift_report=drift_report,
        prediction_metrics=prediction_metrics,
        mae_threshold=100.0,
    )

    assert summary["status"] == "PASS"
    assert (
        summary["missing_value_status"]
        == "PASS"
    )
    assert (
        summary["feature_drift_status"]
        == "PASS"
    )
    assert (
        summary["performance_status"]
        == "PASS"
    )


def test_monitoring_status_fails_when_mae_is_high(
    reference_dataframe: pd.DataFrame,
    stable_current_dataframe: pd.DataFrame,
) -> None:
    missing_report = calculate_missing_value_report(
        dataframe=stable_current_dataframe,
        missing_rate_threshold=0.01,
    )

    drift_report = calculate_feature_drift(
        reference_dataframe=reference_dataframe,
        current_dataframe=stable_current_dataframe,
        feature_columns=[
            "temperature",
            "demand_lag_1",
        ],
        mean_drift_threshold=0.20,
        std_drift_threshold=0.20,
    )

    prediction_metrics = {
        "mae": 150.0,
    }

    summary = evaluate_monitoring_status(
        missing_report=missing_report,
        drift_report=drift_report,
        prediction_metrics=prediction_metrics,
        mae_threshold=100.0,
    )

    assert summary["status"] == "FAIL"
    assert (
        summary["performance_status"]
        == "FAIL"
    )


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

    assert (
        decision["retraining_required"]
        is False
    )
    assert (
        decision["decision"]
        == "KEEP_CURRENT_MODEL"
    )


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

    assert (
        decision["retraining_required"]
        is True
    )
    assert (
        decision["decision"]
        == "RETRAIN"
    )


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

    assert (
        decision["retraining_required"]
        is True
    )


def test_feature_drift_raises_for_missing_feature(
    reference_dataframe: pd.DataFrame,
    stable_current_dataframe: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        calculate_feature_drift(
            reference_dataframe=(
                reference_dataframe
            ),
            current_dataframe=(
                stable_current_dataframe
            ),
            feature_columns=[
                "temperature",
                "missing_feature",
            ],
        )