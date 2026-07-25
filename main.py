# main.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from xgboost import XGBRegressor

from src.diagnostics import (
    generate_model_diagnostics,
    generate_time_based_error_analysis,
)
from src.features.build_calendar_features import (
    add_calendar_features,
)
from src.features.build_cyclical_features import (
    add_cyclical_features,
)
from src.features.build_lag_features import (
    add_lag_features,
)
from src.features.build_rolling_features import (
    add_rolling_features,
)
from src.features.build_weather_features import (
    build_weather_features,
)
from src.inference import (
    generate_predictions,
    save_predictions,
)
from src.ingestion.load_historical_operational_demand import (
    load_historical_operational_demand,
)
from src.ingestion.weather import (
    load_historical_weather,
)
from src.modelling.baselines import (
    generate_baseline_predictions,
)
from src.modelling.linear_models import (
    train_linear_regression,
    train_ridge_regression,
)
from src.modelling.metrics import (
    calculate_regression_metrics,
    evaluate_prediction_dataframe,
)
from src.modelling.preprocessing import (
    standard_scale_features,
)
from src.modelling.split_data import (
    split_time_series_data,
)
from src.modelling.tree_models import (
    get_feature_importance,
    train_random_forest,
)
from src.modelling.tune_xgboost import (
    tune_xgboost_regressor,
)
from src.modelling.xgboost_model import (
    get_xgboost_feature_importance,
)
from src.monitoring import (
    evaluate_and_save_retraining_decision,
    run_monitoring,
)
from src.processing.clean_aemo import (
    clean_operational_demand,
    validate_operational_demand,
)
from src.processing.clean_weather import (
    clean_weather_data,
)
from src.processing.merge_demand_weather import (
    merge_demand_weather,
)
from src.processing.prepare_model_data import (
    prepare_model_data,
)
from src.processing.prepare_training_data import (
    prepare_training_data,
)
from src.tracking import (
    create_model_manifest,
    track_experiment,
)
from src.verification.validators import (
    PipelineValidationError,
    save_json_report,
    validate_chronological_split,
    validate_demand_data,
    validate_demand_weather_merge,
    validate_model_outputs,
    validate_training_dataset,
    validate_weather_data,
)
from src.visualisation import (
    generate_model_visualisations,
)


PROJECT_ROOT = Path(__file__).resolve().parent

ARCHIVE_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "aemo"
    / "operational_demand"
    / "archive"
)

OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"
MODEL_DIRECTORY = PROJECT_ROOT / "models"

METRICS_DIRECTORY = (
    OUTPUT_DIRECTORY
    / "metrics"
)

PREDICTIONS_DIRECTORY = (
    OUTPUT_DIRECTORY
    / "predictions"
)

TUNING_DIRECTORY = (
    OUTPUT_DIRECTORY
    / "tuning"
)

FEATURE_IMPORTANCE_DIRECTORY = (
    OUTPUT_DIRECTORY
    / "feature_importance"
)

VERIFICATION_DIRECTORY = (
    OUTPUT_DIRECTORY
    / "verification"
)

VERIFICATION_DETAILS_DIRECTORY = (
    VERIFICATION_DIRECTORY
    / "details"
)

DIAGNOSTICS_DIRECTORY = (
    OUTPUT_DIRECTORY
    / "diagnostics"
)

PLOTS_DIRECTORY = (
    OUTPUT_DIRECTORY
    / "plots"
)

TRACKING_DIRECTORY = (
    OUTPUT_DIRECTORY
    / "tracking"
)

MONITORING_DIRECTORY = (
    OUTPUT_DIRECTORY
    / "monitoring"
)

MODEL_PATH = (
    MODEL_DIRECTORY
    / "final_xgboost_model.json"
)

FEATURE_COLUMNS_PATH = (
    MODEL_DIRECTORY
    / "feature_columns.joblib"
)

EXPERIMENT_HISTORY_PATH = (
    TRACKING_DIRECTORY
    / "experiment_history.csv"
)

MODEL_MANIFEST_PATH = (
    TRACKING_DIRECTORY
    / "model_manifest.json"
)

RETRAINING_DECISION_PATH = (
    MONITORING_DIRECTORY
    / "retraining_decision.json"
)

INFERENCE_OUTPUT_PATH = (
    PREDICTIONS_DIRECTORY
    / "inference_predictions.csv"
)


REGION = "VIC1"
TARGET_COLUMN = "OPERATIONAL_DEMAND"
TIMESTAMP_COLUMN = "INTERVAL_DATETIME"

WEATHER_START_DATE = "2025-06-01"
WEATHER_END_DATE = "2026-06-01"
WEATHER_LATITUDE = -37.8136
WEATHER_LONGITUDE = 144.9631
WEATHER_TIMEZONE = "Australia/Brisbane"

FORECAST_HORIZON = "30 minutes ahead"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

MINIMUM_WEATHER_MATCH_RATE = 0.99

TUNING_ITERATIONS = 25
TUNING_SPLITS = 4
RANDOM_STATE = 42

MISSING_RATE_THRESHOLD = 0.01

PSI_WARNING_THRESHOLD = 0.10
PSI_FAILURE_THRESHOLD = 0.25
PSI_BINS = 10

MAE_THRESHOLD = 100.0

MAXIMUM_MONITORING_DRIFT_FAILURES = 5

MAXIMUM_RETRAINING_MAE = 100.0
MAXIMUM_DRIFT_FAILURES = 5
MAXIMUM_MISSING_FAILURES = 0


MONITORED_FEATURE_CANDIDATES = [
    "temperature",
    "apparent_temperature",
    "relative_humidity",
    "humidity",
    "wind_speed",
    "cloud_cover",
    "precipitation",
    "rain",
    "solar_radiation",
]

def make_output_directories() -> None:
    directories = [
        METRICS_DIRECTORY,
        PREDICTIONS_DIRECTORY,
        TUNING_DIRECTORY,
        FEATURE_IMPORTANCE_DIRECTORY,
        VERIFICATION_DIRECTORY,
        VERIFICATION_DETAILS_DIRECTORY,
        DIAGNOSTICS_DIRECTORY,
        PLOTS_DIRECTORY,
        TRACKING_DIRECTORY,
        MONITORING_DIRECTORY,
        MODEL_DIRECTORY,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def save_json(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            default=str,
        )


def prepare_demand_data() -> tuple[
    pd.DataFrame,
    dict,
]:
    raw_df = (
        load_historical_operational_demand(
            ARCHIVE_DIRECTORY
        )
    )

    clean_df = clean_operational_demand(
        raw_df,
        region=REGION,
    )

    notebook_validation_report = (
        validate_operational_demand(
            clean_df
        )
    )

    validate_demand_data(
        dataframe=clean_df,
        timestamp_column=TIMESTAMP_COLUMN,
        target_column=TARGET_COLUMN,
        output_path=(
            VERIFICATION_DIRECTORY
            / "01_demand_validation.json"
        ),
    )

    model_df = prepare_model_data(
        clean_df
    )

    feature_df = add_calendar_features(
        model_df
    )

    feature_df = add_lag_features(
        feature_df
    )

    feature_df = add_rolling_features(
        feature_df
    )

    feature_df = add_cyclical_features(
        feature_df
    )

    return (
        feature_df,
        notebook_validation_report,
    )


def prepare_weather_data() -> pd.DataFrame:
    raw_df = load_historical_weather(
        start_date=WEATHER_START_DATE,
        end_date=WEATHER_END_DATE,
        latitude=WEATHER_LATITUDE,
        longitude=WEATHER_LONGITUDE,
        timezone=WEATHER_TIMEZONE,
    )

    clean_df = clean_weather_data(
        raw_df
    )

    feature_df = build_weather_features(
        clean_df,
        heating_base_temperature=18.0,
        cooling_base_temperature=22.0,
        high_temperature_threshold=30.0,
        high_humidity_threshold=80.0,
    )

    validate_weather_data(
        dataframe=feature_df,
        timestamp_column=TIMESTAMP_COLUMN,
        output_path=(
            VERIFICATION_DIRECTORY
            / "02_weather_validation.json"
        ),
    )

    return feature_df


def build_training_dataset() -> tuple[
    pd.DataFrame,
    dict,
]:
    (
        demand_df,
        notebook_validation_report,
    ) = prepare_demand_data()

    weather_df = prepare_weather_data()

    merged_df = merge_demand_weather(
        demand_df,
        weather_df,
    )

    validate_demand_weather_merge(
        demand_dataframe=demand_df,
        weather_dataframe=weather_df,
        merged_dataframe=merged_df,
        timestamp_column=TIMESTAMP_COLUMN,
        output_path=(
            VERIFICATION_DIRECTORY
            / "03_merge_validation.json"
        ),
        detail_directory=(
            VERIFICATION_DETAILS_DIRECTORY
        ),
        minimum_match_rate=(
            MINIMUM_WEATHER_MATCH_RATE
        ),
    )

    training_df = prepare_training_data(
        merged_df
    )

    validate_training_dataset(
        dataframe=training_df,
        timestamp_column=TIMESTAMP_COLUMN,
        target_column=TARGET_COLUMN,
        output_path=(
            VERIFICATION_DIRECTORY
            / "04_training_dataset_validation.json"
        ),
    )

    return (
        training_df,
        notebook_validation_report,
    )


def split_model_data(
    training_df: pd.DataFrame,
) -> dict[str, Any]:
    (
        train_df,
        validation_df,
        test_df,
    ) = split_time_series_data(
        training_df,
        train_ratio=TRAIN_RATIO,
        validation_ratio=VALIDATION_RATIO,
    )

    feature_columns = [
        column
        for column in training_df.columns
        if column
        not in [
            TIMESTAMP_COLUMN,
            TARGET_COLUMN,
        ]
    ]

    validate_chronological_split(
        full_dataframe=training_df,
        train_dataframe=train_df,
        validation_dataframe=validation_df,
        test_dataframe=test_df,
        feature_columns=feature_columns,
        timestamp_column=TIMESTAMP_COLUMN,
        target_column=TARGET_COLUMN,
        output_path=(
            VERIFICATION_DIRECTORY
            / "05_split_validation.json"
        ),
    )

    return {
        "train_df": train_df,
        "validation_df": validation_df,
        "test_df": test_df,
        "feature_columns": feature_columns,
        "X_train": train_df[
            feature_columns
        ].copy(),
        "y_train": train_df[
            TARGET_COLUMN
        ].copy(),
        "X_validation": validation_df[
            feature_columns
        ].copy(),
        "y_validation": validation_df[
            TARGET_COLUMN
        ].copy(),
        "X_test": test_df[
            feature_columns
        ].copy(),
        "y_test": test_df[
            TARGET_COLUMN
        ].copy(),
    }


def metric_row(
    model_name: str,
    y_true: pd.Series,
    predictions: Any,
) -> dict[str, Any]:
    return {
        "model": model_name,
        **calculate_regression_metrics(
            y_true=y_true,
            y_pred=predictions,
        ),
    }


def train_validation_candidates(
    data: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
]:
    X_train = data["X_train"]
    y_train = data["y_train"]

    X_validation = data[
        "X_validation"
    ]

    y_validation = data[
        "y_validation"
    ]

    rows: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}

    baseline_predictions = (
        generate_baseline_predictions(
            data["validation_df"]
        )
    )

    baseline_metrics = (
        evaluate_prediction_dataframe(
            baseline_predictions
        )
    )

    rows.extend(
        baseline_metrics.to_dict(
            orient="records"
        )
    )

    (
        X_train_scaled,
        X_validation_scaled,
        _,
        scaler,
    ) = standard_scale_features(
        X_train,
        X_validation,
        data["X_test"],
    )

    (
        linear_model,
        linear_predictions,
    ) = train_linear_regression(
        X_train=X_train_scaled,
        y_train=y_train,
        X_validation=(
            X_validation_scaled
        ),
    )

    (
        ridge_model,
        ridge_predictions,
    ) = train_ridge_regression(
        X_train=X_train_scaled,
        y_train=y_train,
        X_validation=(
            X_validation_scaled
        ),
        alpha=1.0,
    )

    rows.append(
        metric_row(
            "linear_regression",
            y_validation,
            linear_predictions,
        )
    )

    rows.append(
        metric_row(
            "ridge_regression",
            y_validation,
            ridge_predictions,
        )
    )

    (
        random_forest_model,
        random_forest_predictions,
    ) = train_random_forest(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    rows.append(
        metric_row(
            "random_forest",
            y_validation,
            random_forest_predictions,
        )
    )

    default_xgboost_model = (
        XGBRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )

    default_xgboost_model.fit(
        X_train,
        y_train,
    )

    default_xgboost_predictions = (
        default_xgboost_model.predict(
            X_validation
        )
    )

    rows.append(
        metric_row(
            "default_xgboost",
            y_validation,
            default_xgboost_predictions,
        )
    )

    (
        tuned_xgboost_model,
        best_parameters,
        cv_results,
        search,
    ) = tune_xgboost_regressor(
        X_train=X_train,
        y_train=y_train,
        n_iter=TUNING_ITERATIONS,
        n_splits=TUNING_SPLITS,
        random_state=RANDOM_STATE,
        search_n_jobs=-1,
        verbose=1,
    )

    tuned_xgboost_predictions = (
        tuned_xgboost_model.predict(
            X_validation
        )
    )

    rows.append(
        metric_row(
            "tuned_xgboost",
            y_validation,
            tuned_xgboost_predictions,
        )
    )

    comparison_df = (
        pd.DataFrame(rows)
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    validation_predictions_df = (
        pd.DataFrame(
            {
                TIMESTAMP_COLUMN: (
                    data[
                        "validation_df"
                    ][
                        TIMESTAMP_COLUMN
                    ].reset_index(
                        drop=True
                    )
                ),
                "actual": (
                    y_validation.reset_index(
                        drop=True
                    )
                ),
                "default_xgboost": (
                    default_xgboost_predictions
                ),
                "tuned_xgboost": (
                    tuned_xgboost_predictions
                ),
            }
        )
    )

    artifacts.update(
        {
            "scaler": scaler,
            "linear_model": linear_model,
            "ridge_model": ridge_model,
            "random_forest_model": (
                random_forest_model
            ),
            "default_xgboost_model": (
                default_xgboost_model
            ),
            "tuned_xgboost_model": (
                tuned_xgboost_model
            ),
            "best_parameters": (
                best_parameters
            ),
            "best_cv_mae": float(
                -search.best_score_
            ),
            "validation_predictions": (
                validation_predictions_df
            ),
        }
    )

    return (
        comparison_df,
        artifacts,
        cv_results,
    )


def select_xgboost_model(
    comparison_df: pd.DataFrame,
    best_parameters: dict[str, Any],
) -> tuple[
    str,
    dict[str, Any] | None,
]:
    xgboost_comparison = (
        comparison_df[
            comparison_df["model"].isin(
                [
                    "default_xgboost",
                    "tuned_xgboost",
                ]
            )
        ]
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    if xgboost_comparison.empty:
        raise ValueError(
            "No XGBoost candidates were found "
            "in the model comparison."
        )

    selected_model = str(
        xgboost_comparison.loc[
            0,
            "model",
        ]
    )

    selected_parameters = (
        best_parameters
        if selected_model
        == "tuned_xgboost"
        else None
    )

    return (
        selected_model,
        selected_parameters,
    )


def get_selected_validation_metrics(
    comparison_df: pd.DataFrame,
    selected_model: str,
) -> dict[str, float]:
    selected_rows = comparison_df[
        comparison_df["model"]
        == selected_model
    ]

    if selected_rows.empty:
        raise ValueError(
            "Selected model was not found "
            "in validation results."
        )

    metric_columns = [
        column
        for column in [
            "MAE",
            "RMSE",
            "MAPE",
            "R2",
        ]
        if column in selected_rows.columns
    ]

    selected_row = (
        selected_rows.iloc[0]
    )

    return {
        column: float(
            selected_row[column]
        )
        for column in metric_columns
    }


def build_final_training_dataframe(
    data: dict[str, Any],
) -> pd.DataFrame:
    return pd.concat(
        [
            data["train_df"],
            data["validation_df"],
        ],
        ignore_index=True,
    )


def retrain_and_test(
    data: dict[str, Any],
    selected_model: str,
    selected_parameters: (
        dict[str, Any] | None
    ),
) -> tuple[
    XGBRegressor,
    pd.DataFrame,
    dict[str, float],
]:
    X_final_train = pd.concat(
        [
            data["X_train"],
            data["X_validation"],
        ],
        ignore_index=True,
    )

    y_final_train = pd.concat(
        [
            data["y_train"],
            data["y_validation"],
        ],
        ignore_index=True,
    )

    if (
        selected_model
        == "tuned_xgboost"
    ):
        final_model = XGBRegressor(
            **(
                selected_parameters
                or {}
            )
        )
    else:
        final_model = XGBRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    final_model.fit(
        X_final_train,
        y_final_train,
    )

    predictions = final_model.predict(
        data["X_test"]
    )

    test_metrics = (
        calculate_regression_metrics(
            y_true=data["y_test"],
            y_pred=predictions,
        )
    )

    test_predictions_df = (
        pd.DataFrame(
            {
                TIMESTAMP_COLUMN: (
                    data["test_df"][
                        TIMESTAMP_COLUMN
                    ].reset_index(
                        drop=True
                    )
                ),
                "actual": (
                    data[
                        "y_test"
                    ].reset_index(
                        drop=True
                    )
                ),
                "predicted": (
                    predictions
                ),
            }
        )
    )

    test_predictions_df[
        "residual"
    ] = (
        test_predictions_df[
            "actual"
        ]
        - test_predictions_df[
            "predicted"
        ]
    )

    return (
        final_model,
        test_predictions_df,
        test_metrics,
    )


def save_training_outputs(
    training_df: pd.DataFrame,
    notebook_validation_report: dict,
    comparison_df: pd.DataFrame,
    artifacts: dict[str, Any],
    cv_results: pd.DataFrame,
    selected_model: str,
    selected_parameters: (
        dict[str, Any] | None
    ),
    final_model: XGBRegressor,
    test_predictions_df: pd.DataFrame,
    test_metrics: dict[str, float],
    feature_columns: list[str],
) -> None:
    comparison_df.to_csv(
        METRICS_DIRECTORY
        / "validation_model_comparison.csv",
        index=False,
    )

    artifacts[
        "validation_predictions"
    ].to_csv(
        PREDICTIONS_DIRECTORY
        / "xgboost_validation_predictions.csv",
        index=False,
    )

    cv_results.to_csv(
        TUNING_DIRECTORY
        / "xgboost_cv_results.csv",
        index=False,
    )

    test_predictions_df.to_csv(
        PREDICTIONS_DIRECTORY
        / "final_test_predictions.csv",
        index=False,
    )

    training_df.to_csv(
        OUTPUT_DIRECTORY
        / "training_dataset.csv",
        index=False,
    )

    save_json_report(
        notebook_validation_report,
        VERIFICATION_DIRECTORY
        / "00_existing_demand_validation.json",
    )

    save_json(
        artifacts["best_parameters"],
        TUNING_DIRECTORY
        / "best_xgboost_parameters.json",
    )

    save_json(
        {
            "selected_model": (
                selected_model
            ),
            "selected_parameters": (
                selected_parameters
            ),
            "best_cross_validation_MAE": (
                artifacts["best_cv_mae"]
            ),
        },
        METRICS_DIRECTORY
        / "model_selection.json",
    )

    save_json(
        test_metrics,
        METRICS_DIRECTORY
        / "final_test_metrics.json",
    )

    final_importance = (
        get_xgboost_feature_importance(
            model=final_model,
            feature_names=feature_columns,
        )
    )

    final_importance.to_csv(
        FEATURE_IMPORTANCE_DIRECTORY
        / "final_xgboost_feature_importance.csv",
        index=False,
    )

    random_forest_importance = (
        get_feature_importance(
            model=artifacts[
                "random_forest_model"
            ],
            feature_names=feature_columns,
        )
    )

    random_forest_importance.to_csv(
        FEATURE_IMPORTANCE_DIRECTORY
        / "random_forest_feature_importance.csv",
        index=False,
    )

    final_model.save_model(
        MODEL_PATH
    )

    joblib.dump(
        feature_columns,
        FEATURE_COLUMNS_PATH,
    )


def run_saved_model_inference(
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    inference_predictions = (
        generate_predictions(
            dataframe=test_df,
            model_path=MODEL_PATH,
            feature_columns_path=(
                FEATURE_COLUMNS_PATH
            ),
            timestamp_column=(
                TIMESTAMP_COLUMN
            ),
            prediction_column=(
                "predicted"
            ),
        )
    )

    save_predictions(
        predictions=(
            inference_predictions
        ),
        output_path=(
            INFERENCE_OUTPUT_PATH
        ),
    )

    return inference_predictions


def run_model_tracking(
    training_df: pd.DataFrame,
    data: dict[str, Any],
    comparison_df: pd.DataFrame,
    selected_model: str,
    selected_parameters: (
        dict[str, Any] | None
    ),
    best_cv_mae: float,
    test_metrics: dict[str, float],
) -> dict[str, Any]:
    validation_metrics = (
        get_selected_validation_metrics(
            comparison_df=(
                comparison_df
            ),
            selected_model=(
                selected_model
            ),
        )
    )

    experiment_record = (
        track_experiment(
            model_name=selected_model,
            selected_model=(
                selected_model
            ),
            validation_metrics=(
                validation_metrics
            ),
            test_metrics=test_metrics,
            best_cv_mae=best_cv_mae,
            feature_count=len(
                data["feature_columns"]
            ),
            training_rows=len(
                data["train_df"]
            ),
            validation_rows=len(
                data["validation_df"]
            ),
            test_rows=len(
                data["test_df"]
            ),
            output_path=(
                EXPERIMENT_HISTORY_PATH
            ),
            parameters=(
                selected_parameters
            ),
        )
    )

    final_training_df = (
        build_final_training_dataframe(
            data
        )
    )

    manifest = create_model_manifest(
        model_name=selected_model,
        model_path=MODEL_PATH,
        feature_columns_path=(
            FEATURE_COLUMNS_PATH
        ),
        feature_columns=(
            data["feature_columns"]
        ),
        target_column=TARGET_COLUMN,
        timestamp_column=(
            TIMESTAMP_COLUMN
        ),
        forecast_horizon=(
            FORECAST_HORIZON
        ),
        training_dataframe=(
            final_training_df
        ),
        validation_metrics=(
            validation_metrics
        ),
        test_metrics=test_metrics,
        model_parameters=(
            selected_parameters
            or {}
        ),
        output_path=(
            MODEL_MANIFEST_PATH
        ),
    )

    return {
        "experiment_record": (
            experiment_record
        ),
        "manifest": manifest,
        "source_dataset_rows": int(
            len(training_df)
        ),
    }

def get_monitored_features(
    dataframe: pd.DataFrame,
) -> list[str]:
    monitored_features = [
        feature
        for feature in MONITORED_FEATURE_CANDIDATES
        if feature in dataframe.columns
    ]

    if not monitored_features:
        raise ValueError(
            "No configured monitoring features were found "
            "in the current dataframe."
        )

    return monitored_features



def run_model_monitoring(
    data: dict[str, Any],
    inference_predictions: pd.DataFrame,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    reference_dataframe = build_final_training_dataframe(
        data
    )

    current_dataframe = data["test_df"]

    monitored_features = get_monitored_features(
        current_dataframe
    )

    prediction_monitoring_df = pd.DataFrame(
        {
            "actual": (
                data["y_test"]
                .reset_index(drop=True)
            ),
            "prediction": (
                inference_predictions["predicted"]
                .reset_index(drop=True)
            ),
        }
    )

    monitoring_result = run_monitoring(
        reference_dataframe=reference_dataframe,
        current_dataframe=current_dataframe,
        feature_columns=monitored_features,
        output_directory=MONITORING_DIRECTORY,
        prediction_dataframe=prediction_monitoring_df,
        actual_column="actual",
        prediction_column="prediction",
        missing_rate_threshold=MISSING_RATE_THRESHOLD,
        psi_warning_threshold=PSI_WARNING_THRESHOLD,
        psi_failure_threshold=PSI_FAILURE_THRESHOLD,
        psi_bins=PSI_BINS,
        mae_threshold=MAE_THRESHOLD,
        maximum_drift_failures=(
            MAXIMUM_MONITORING_DRIFT_FAILURES
        ),
    )

    monitoring_result["monitored_features"] = (
        monitored_features
    )

    retraining_decision = (
        evaluate_and_save_retraining_decision(
            monitoring_summary=monitoring_result,
            output_path=RETRAINING_DECISION_PATH,
            maximum_mae=MAXIMUM_RETRAINING_MAE,
            maximum_drift_failures=(
                MAXIMUM_DRIFT_FAILURES
            ),
            maximum_missing_failures=(
                MAXIMUM_MISSING_FAILURES
            ),
        )
    )

    return (
        monitoring_result,
        retraining_decision,
    )

def save_pipeline_summary(
    training_df: pd.DataFrame,
    data: dict[str, Any],
    comparison_df: pd.DataFrame,
    selected_model: str,
    best_cv_mae: float,
    test_metrics: dict[str, float],
    diagnostics_summary: dict[str, Any],
    time_based_error_summary: dict[str, Any],
    visualisation_summary: dict[str, Any],
    tracking_summary: dict[str, Any],
    monitoring_summary: dict[str, Any],
    retraining_decision: dict[str, Any],
) -> None:
    report = {
        "status": "PASS",
        "dataset_rows": int(
            len(training_df)
        ),
        "feature_count": int(
            len(data["feature_columns"])
        ),
        "split_rows": {
            "train": int(
                len(data["train_df"])
            ),
            "validation": int(
                len(
                    data[
                        "validation_df"
                    ]
                )
            ),
            "test": int(
                len(data["test_df"])
            ),
        },
        "selected_model": (
            selected_model
        ),
        "best_tuning_cv_mae": float(
            best_cv_mae
        ),
        "validation_models": (
            comparison_df.to_dict(
                orient="records"
            )
        ),
        "final_test_metrics": (
            test_metrics
        ),
        "model_artifacts": {
            "model": str(
                MODEL_PATH
            ),
            "feature_columns": str(
                FEATURE_COLUMNS_PATH
            ),
            "manifest": str(
                MODEL_MANIFEST_PATH
            ),
            "experiment_history": str(
                EXPERIMENT_HISTORY_PATH
            ),
        },
        "inference_output": str(
            INFERENCE_OUTPUT_PATH
        ),
        "diagnostics": (
            diagnostics_summary
        ),
        "time_based_error_analysis": (
            time_based_error_summary
        ),
        "visualisation": (
            visualisation_summary
        ),
        "tracking": (
            tracking_summary
        ),
        "monitoring": (
            monitoring_summary
        ),
        "retraining_decision": (
            retraining_decision
        ),
        "verification_directory": str(
            VERIFICATION_DIRECTORY
        ),
    }

    save_json_report(
        report,
        VERIFICATION_DIRECTORY
        / "07_pipeline_summary.json",
    )


def main() -> None:
    make_output_directories()

    try:
        print(
            "[1/13] Preparing and validating data..."
        )

        (
            training_df,
            notebook_validation_report,
        ) = build_training_dataset()

        print(
            "[2/13] Creating chronological splits..."
        )

        data = split_model_data(
            training_df
        )

        print(
            "[3/13] Training model candidates..."
        )

        (
            comparison_df,
            artifacts,
            cv_results,
        ) = train_validation_candidates(
            data
        )

        print(
            "[4/13] Selecting champion model..."
        )

        (
            selected_model,
            selected_parameters,
        ) = select_xgboost_model(
            comparison_df,
            artifacts[
                "best_parameters"
            ],
        )

        print(
            "[5/13] Retraining and testing champion..."
        )

        (
            final_model,
            test_predictions_df,
            test_metrics,
        ) = retrain_and_test(
            data,
            selected_model,
            selected_parameters,
        )

        print(
            "[6/13] Verifying model outputs..."
        )

        validate_model_outputs(
            comparison_dataframe=(
                comparison_df
            ),
            test_predictions_dataframe=(
                test_predictions_df
            ),
            test_metrics=test_metrics,
            selected_model=(
                selected_model
            ),
            output_path=(
                VERIFICATION_DIRECTORY
                / "06_model_output_validation.json"
            ),
        )

        print(
            "[7/13] Saving models and outputs..."
        )

        save_training_outputs(
            training_df=training_df,
            notebook_validation_report=(
                notebook_validation_report
            ),
            comparison_df=comparison_df,
            artifacts=artifacts,
            cv_results=cv_results,
            selected_model=(
                selected_model
            ),
            selected_parameters=(
                selected_parameters
            ),
            final_model=final_model,
            test_predictions_df=(
                test_predictions_df
            ),
            test_metrics=test_metrics,
            feature_columns=(
                data["feature_columns"]
            ),
        )

        print(
            "[8/13] Running saved-model inference..."
        )

        inference_predictions = (
            run_saved_model_inference(
                data["test_df"]
            )
        )

        print(
            "[9/13] Generating diagnostics..."
        )

        diagnostics_summary = (
            generate_model_diagnostics(
                timestamps=(
                    test_predictions_df[
                        TIMESTAMP_COLUMN
                    ]
                ),
                actual_values=(
                    test_predictions_df[
                        "actual"
                    ]
                ),
                predicted_values=(
                    test_predictions_df[
                        "predicted"
                    ]
                ),
                output_directory=(
                    DIAGNOSTICS_DIRECTORY
                ),
                worst_prediction_count=50,
            )
        )

        print(
            "[10/13] Analysing time-based errors..."
        )

        time_based_error_summary = (
            generate_time_based_error_analysis(
                timestamps=(
                    test_predictions_df[
                        TIMESTAMP_COLUMN
                    ]
                ),
                actual_values=(
                    test_predictions_df[
                        "actual"
                    ]
                ),
                predicted_values=(
                    test_predictions_df[
                        "predicted"
                    ]
                ),
                output_directory=(
                    DIAGNOSTICS_DIRECTORY
                ),
            )
        )

        print(
            "[11/13] Generating visualisations..."
        )

        visualisation_summary = (
            generate_model_visualisations(
                diagnostics_directory=(
                    DIAGNOSTICS_DIRECTORY
                ),
                output_directory=(
                    PLOTS_DIRECTORY
                ),
            )
        )

        print(
            "[12/13] Tracking and monitoring model..."
        )

        tracking_summary = (
            run_model_tracking(
                training_df=training_df,
                data=data,
                comparison_df=(
                    comparison_df
                ),
                selected_model=(
                    selected_model
                ),
                selected_parameters=(
                    selected_parameters
                ),
                best_cv_mae=(
                    artifacts[
                        "best_cv_mae"
                    ]
                ),
                test_metrics=(
                    test_metrics
                ),
            )
        )

        (
            monitoring_summary,
            retraining_decision,
        ) = run_model_monitoring(
            data=data,
            inference_predictions=(
                inference_predictions
            ),
        )

        print(
            "[13/13] Saving pipeline summary..."
        )

        save_pipeline_summary(
            training_df=training_df,
            data=data,
            comparison_df=(
                comparison_df
            ),
            selected_model=(
                selected_model
            ),
            best_cv_mae=(
                artifacts[
                    "best_cv_mae"
                ]
            ),
            test_metrics=test_metrics,
            diagnostics_summary=(
                diagnostics_summary
            ),
            time_based_error_summary=(
                time_based_error_summary
            ),
            visualisation_summary=(
                visualisation_summary
            ),
            tracking_summary=(
                tracking_summary
            ),
            monitoring_summary=(
                monitoring_summary
            ),
            retraining_decision=(
                retraining_decision
            ),
        )

        print(
            "\nPipeline completed successfully."
        )

        print(
            f"Selected model: "
            f"{selected_model}"
        )

        print(
            f"Test MAE: "
            f"{test_metrics['MAE']:.3f} MW"
        )

        print(
            f"Monitoring status: "
            f"{monitoring_summary['status']}"
        )

        print(
            f"Retraining decision: "
            f"{retraining_decision['decision']}"
        )

        print(
            f"Outputs: {OUTPUT_DIRECTORY}"
        )

    except PipelineValidationError as error:
        print(
            "\nPipeline stopped because a "
            "verification check failed."
        )

        print(
            str(error)
        )

        raise

    except Exception as error:
        print(
            "\nPipeline failed."
        )

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        raise


if __name__ == "__main__":
    main()