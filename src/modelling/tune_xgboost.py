# src/modelling/tune_xgboost.py

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit,
)
from xgboost import XGBRegressor


def tune_xgboost_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 25,
    n_splits: int = 4,
    random_state: int = 42,
    search_n_jobs: int = -1,
    verbose: int = 1,
) -> tuple[
    XGBRegressor,
    dict,
    pd.DataFrame,
    RandomizedSearchCV,
]:
    """
    Tune XGBoost using expanding-window time-series cross-validation.

    Only the training dataset is used during hyperparameter search.
    Validation and test datasets remain untouched.
    """

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")

    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    if X_train.empty or y_train.empty:
        raise ValueError("Training data cannot be empty.")

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train must contain the same number of rows."
        )

    if X_train.isna().any().any():
        raise ValueError("X_train contains missing values.")

    if y_train.isna().any():
        raise ValueError("y_train contains missing values.")

    if n_iter <= 0:
        raise ValueError("n_iter must be greater than zero.")

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    base_model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=1,
    )

    parameter_distributions = {
        "n_estimators": [
            300,
            500,
            700,
            900,
            1200,
        ],
        "learning_rate": [
            0.01,
            0.025,
            0.05,
            0.075,
            0.1,
        ],
        "max_depth": [
            3,
            4,
            5,
            6,
            7,
            8,
        ],
        "min_child_weight": [
            1,
            3,
            5,
            7,
            10,
        ],
        "subsample": [
            0.7,
            0.8,
            0.9,
            1.0,
        ],
        "colsample_bytree": [
            0.7,
            0.8,
            0.9,
            1.0,
        ],
        "gamma": [
            0.0,
            0.05,
            0.1,
            0.2,
            0.5,
        ],
        "reg_alpha": [
            0.0,
            0.01,
            0.1,
            0.5,
            1.0,
        ],
        "reg_lambda": [
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
        ],
    }

    time_series_cv = TimeSeriesSplit(
        n_splits=n_splits,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=parameter_distributions,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=time_series_cv,
        refit=True,
        random_state=random_state,
        n_jobs=search_n_jobs,
        verbose=verbose,
        return_train_score=True,
    )

    search.fit(
        X_train,
        y_train,
    )

    cv_results = pd.DataFrame(
        search.cv_results_
    )

    cv_results["mean_validation_MAE"] = (
        -cv_results["mean_test_score"]
    )

    cv_results["std_validation_MAE"] = (
        cv_results["std_test_score"]
    )

    cv_results["mean_training_MAE"] = (
        -cv_results["mean_train_score"]
    )

    result_columns = [
        "rank_test_score",
        "mean_validation_MAE",
        "std_validation_MAE",
        "mean_training_MAE",
        "mean_fit_time",
        "params",
    ]

    cv_results = (
        cv_results[result_columns]
        .sort_values("rank_test_score")
        .reset_index(drop=True)
    )

    best_model = search.best_estimator_
    best_parameters = search.best_params_

    return (
        best_model,
        best_parameters,
        cv_results,
        search,
    )