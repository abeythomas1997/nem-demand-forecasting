from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBRegressor

from src.diagnostics import (
    generate_model_diagnostics,
    generate_time_based_error_analysis,
)
from src.features.build_calendar_features import add_calendar_features
from src.features.build_cyclical_features import add_cyclical_features
from src.features.build_lag_features import add_lag_features
from src.features.build_rolling_features import add_rolling_features
from src.features.build_weather_features import build_weather_features
from src.ingestion.load_historical_operational_demand import (
    load_historical_operational_demand,
)
from src.ingestion.weather import load_historical_weather
from src.modelling.baselines import generate_baseline_predictions
from src.modelling.linear_models import (
    train_linear_regression,
    train_ridge_regression,
)
from src.modelling.metrics import (
    calculate_regression_metrics,
    evaluate_prediction_dataframe,
)
from src.modelling.preprocessing import standard_scale_features
from src.modelling.split_data import split_time_series_data
from src.modelling.tree_models import (
    get_feature_importance,
    train_random_forest,
)
from src.modelling.tune_xgboost import tune_xgboost_regressor
from src.modelling.xgboost_model import get_xgboost_feature_importance
from src.processing.clean_aemo import (
    clean_operational_demand,
    validate_operational_demand,
)
from src.processing.clean_weather import clean_weather_data
from src.processing.merge_demand_weather import merge_demand_weather
from src.processing.prepare_model_data import prepare_model_data
from src.processing.prepare_training_data import prepare_training_data
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
from src.visualisation import generate_model_visualisations


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

REGION = "VIC1"
TARGET_COLUMN = "OPERATIONAL_DEMAND"
TIMESTAMP_COLUMN = "INTERVAL_DATETIME"

WEATHER_START_DATE = "2025-06-01"
WEATHER_END_DATE = "2026-06-01"
WEATHER_LATITUDE = -37.8136
WEATHER_LONGITUDE = 144.9631
WEATHER_TIMEZONE = "Australia/Brisbane"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

MINIMUM_WEATHER_MATCH_RATE = 0.99

TUNING_ITERATIONS = 25
TUNING_SPLITS = 4
RANDOM_STATE = 42


def make_output_directories() -> None:
    directories = [
        OUTPUT_DIRECTORY / "metrics",
        OUTPUT_DIRECTORY / "predictions",
        OUTPUT_DIRECTORY / "tuning",
        OUTPUT_DIRECTORY / "feature_importance",
        VERIFICATION_DIRECTORY,
        VERIFICATION_DETAILS_DIRECTORY,
        DIAGNOSTICS_DIRECTORY,
        PLOTS_DIRECTORY,
        MODEL_DIRECTORY,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def prepare_demand_data() -> tuple[pd.DataFrame, dict]:
    raw_df = load_historical_operational_demand(
        ARCHIVE_DIRECTORY
    )

    clean_df = clean_operational_demand(
        raw_df,
        region=REGION,
    )

    notebook_validation_report = (
        validate_operational_demand(clean_df)
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

    model_df = prepare_model_data(clean_df)

    feature_df = add_calendar_features(model_df)
    feature_df = add_lag_features(feature_df)
    feature_df = add_rolling_features(feature_df)
    feature_df = add_cyclical_features(feature_df)

    return feature_df, notebook_validation_report


def prepare_weather_data() -> pd.DataFrame:
    raw_df = load_historical_weather(
        start_date=WEATHER_START_DATE,
        end_date=WEATHER_END_DATE,
        latitude=WEATHER_LATITUDE,
        longitude=WEATHER_LONGITUDE,
        timezone=WEATHER_TIMEZONE,
    )

    clean_df = clean_weather_data(raw_df)

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


def build_training_dataset() -> tuple[pd.DataFrame, dict]:
    demand_df, notebook_validation_report = (
        prepare_demand_data()
    )

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

    return training_df, notebook_validation_report


def split_model_data(
    training_df: pd.DataFrame,
) -> dict:
    train_df, validation_df, test_df = (
        split_time_series_data(
            training_df,
            train_ratio=TRAIN_RATIO,
            validation_ratio=VALIDATION_RATIO,
        )
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
    predictions,
) -> dict:
    return {
        "model": model_name,
        **calculate_regression_metrics(
            y_true=y_true,
            y_pred=predictions,
        ),
    }


def train_validation_candidates(
    data: dict,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    X_train = data["X_train"]
    y_train = data["y_train"]

    X_validation = data["X_validation"]
    y_validation = data["y_validation"]

    rows: list[dict] = []
    artifacts: dict = {}

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

    linear_model, linear_predictions = (
        train_linear_regression(
            X_train=X_train_scaled,
            y_train=y_train,
            X_validation=X_validation_scaled,
        )
    )

    ridge_model, ridge_predictions = (
        train_ridge_regression(
            X_train=X_train_scaled,
            y_train=y_train,
            X_validation=X_validation_scaled,
            alpha=1.0,
        )
    )

    rows.append(
        metric_row(
            model_name="linear_regression",
            y_true=y_validation,
            predictions=linear_predictions,
        )
    )

    rows.append(
        metric_row(
            model_name="ridge_regression",
            y_true=y_validation,
            predictions=ridge_predictions,
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
            model_name="random_forest",
            y_true=y_validation,
            predictions=random_forest_predictions,
        )
    )

    default_xgboost_model = XGBRegressor(
        random_state=RANDOM_STATE,
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
            model_name="default_xgboost",
            y_true=y_validation,
            predictions=(
                default_xgboost_predictions
            ),
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
            model_name="tuned_xgboost",
            y_true=y_validation,
            predictions=(
                tuned_xgboost_predictions
            ),
        )
    )

    comparison_df = (
        pd.DataFrame(rows)
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    validation_predictions_df = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: (
                data["validation_df"][
                    TIMESTAMP_COLUMN
                ].reset_index(drop=True)
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

    return comparison_df, artifacts, cv_results


def select_xgboost_model(
    comparison_df: pd.DataFrame,
    best_parameters: dict,
) -> tuple[str, dict | None]:
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


def retrain_and_test(
    data: dict,
    selected_model: str,
    selected_parameters: dict | None,
) -> tuple[
    XGBRegressor,
    pd.DataFrame,
    dict,
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

    if selected_model == "tuned_xgboost":
        if selected_parameters is None:
            raise ValueError(
                "Tuned XGBoost was selected, but no "
                "tuned parameters were provided."
            )

        final_model = XGBRegressor(
            **selected_parameters,
            random_state=RANDOM_STATE,
        )
    else:
        final_model = XGBRegressor(
            random_state=RANDOM_STATE,
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

    test_predictions_df = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: (
                data["test_df"][
                    TIMESTAMP_COLUMN
                ].reset_index(drop=True)
            ),
            "actual": (
                data["y_test"].reset_index(
                    drop=True
                )
            ),
            "predicted": predictions,
        }
    )

    test_predictions_df["residual"] = (
        test_predictions_df["actual"]
        - test_predictions_df["predicted"]
    )

    return (
        final_model,
        test_predictions_df,
        test_metrics,
    )


def save_outputs(
    training_df: pd.DataFrame,
    notebook_validation_report: dict,
    comparison_df: pd.DataFrame,
    artifacts: dict,
    cv_results: pd.DataFrame,
    selected_model: str,
    selected_parameters: dict | None,
    final_model: XGBRegressor,
    test_predictions_df: pd.DataFrame,
    test_metrics: dict,
    feature_columns: list[str],
) -> None:
    metrics_directory = (
        OUTPUT_DIRECTORY
        / "metrics"
    )

    predictions_directory = (
        OUTPUT_DIRECTORY
        / "predictions"
    )

    tuning_directory = (
        OUTPUT_DIRECTORY
        / "tuning"
    )

    feature_importance_directory = (
        OUTPUT_DIRECTORY
        / "feature_importance"
    )

    comparison_df.to_csv(
        metrics_directory
        / "validation_model_comparison.csv",
        index=False,
    )

    artifacts[
        "validation_predictions"
    ].to_csv(
        predictions_directory
        / "xgboost_validation_predictions.csv",
        index=False,
    )

    cv_results.to_csv(
        tuning_directory
        / "xgboost_cv_results.csv",
        index=False,
    )

    test_predictions_df.to_csv(
        predictions_directory
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

    with (
        tuning_directory
        / "best_xgboost_parameters.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            artifacts["best_parameters"],
            file,
            indent=4,
        )

    with (
        metrics_directory
        / "model_selection.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "selected_model": (
                    selected_model
                ),
                "selected_parameters": (
                    selected_parameters
                ),
                "best_cross_validation_MAE": (
                    artifacts[
                        "best_cv_mae"
                    ]
                ),
            },
            file,
            indent=4,
        )

    with (
        metrics_directory
        / "final_test_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            test_metrics,
            file,
            indent=4,
        )

    final_importance = (
        get_xgboost_feature_importance(
            model=final_model,
            feature_names=feature_columns,
        )
    )

    final_importance.to_csv(
        feature_importance_directory
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
        feature_importance_directory
        / "random_forest_feature_importance.csv",
        index=False,
    )

    final_model.save_model(
        MODEL_DIRECTORY
        / "final_xgboost_model.json"
    )

    joblib.dump(
        feature_columns,
        MODEL_DIRECTORY
        / "feature_columns.joblib",
    )


def save_pipeline_summary(
    training_df: pd.DataFrame,
    data: dict,
    comparison_df: pd.DataFrame,
    selected_model: str,
    best_cv_mae: float,
    test_metrics: dict,
    visualisation: dict,
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
                len(data["validation_df"])
            ),
            "test": int(
                len(data["test_df"])
            ),
        },
        "selected_model": selected_model,
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
        "output_directories": {
            "verification": str(
                VERIFICATION_DIRECTORY
            ),
            "diagnostics": str(
                DIAGNOSTICS_DIRECTORY
            ),
            "plots": str(
                PLOTS_DIRECTORY
            ),
            "models": str(
                MODEL_DIRECTORY
            ),
        },
        "visualisation": visualisation,
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
            "[1/10] Preparing and validating data..."
        )

        (
            training_df,
            notebook_validation_report,
        ) = build_training_dataset()

        print(
            "[2/10] Creating and validating splits..."
        )

        data = split_model_data(
            training_df
        )

        print(
            "[3/10] Training validation candidates..."
        )

        (
            comparison_df,
            artifacts,
            cv_results,
        ) = train_validation_candidates(
            data
        )

        print(
            "[4/10] Selecting XGBoost model..."
        )

        (
            selected_model,
            selected_parameters,
        ) = select_xgboost_model(
            comparison_df=comparison_df,
            best_parameters=(
                artifacts[
                    "best_parameters"
                ]
            ),
        )

        print(
            "[5/10] Retraining selected model..."
        )

        (
            final_model,
            test_predictions_df,
            test_metrics,
        ) = retrain_and_test(
            data=data,
            selected_model=selected_model,
            selected_parameters=(
                selected_parameters
            ),
        )

        print(
            "[6/10] Verifying model outputs..."
        )

        validate_model_outputs(
            comparison_dataframe=(
                comparison_df
            ),
            test_predictions_dataframe=(
                test_predictions_df
            ),
            test_metrics=test_metrics,
            selected_model=selected_model,
            output_path=(
                VERIFICATION_DIRECTORY
                / "06_model_output_validation.json"
            ),
        )

        print(
            "[7/10] Saving model outputs..."
        )

        save_outputs(
            training_df=training_df,
            notebook_validation_report=(
                notebook_validation_report
            ),
            comparison_df=comparison_df,
            artifacts=artifacts,
            cv_results=cv_results,
            selected_model=selected_model,
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
            "[8/10] Generating model diagnostics..."
        )

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

        print(
            "[9/10] Generating model visualisations..."
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
            "[10/10] Saving pipeline summary..."
        )

        save_pipeline_summary(
            training_df=training_df,
            data=data,
            comparison_df=comparison_df,
            selected_model=selected_model,
            best_cv_mae=(
                artifacts["best_cv_mae"]
            ),
            test_metrics=test_metrics,
            visualisation=(
                visualisation_summary
            ),
        )

        print(
            "\nPipeline completed successfully."
        )

        print(
            "Verification reports: "
            f"{VERIFICATION_DIRECTORY}"
        )

        print(
            "Diagnostics: "
            f"{DIAGNOSTICS_DIRECTORY}"
        )

        print(
            "Plots: "
            f"{PLOTS_DIRECTORY}"
        )

        print(
            "Models: "
            f"{MODEL_DIRECTORY}"
        )

    except PipelineValidationError as error:
        print(
            "\nPipeline stopped because a "
            "verification check failed."
        )

        print(str(error))

        raise


if __name__ == "__main__":
    main()