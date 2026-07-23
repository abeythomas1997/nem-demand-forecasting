# src/modelling/xgboost_model.py

from __future__ import annotations

import numpy as np
import pandas as pd

from xgboost import XGBRegressor


def _validate_xgboost_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
) -> None:
    """
    Validate training and validation data for XGBoost.
    """

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")

    if not isinstance(X_validation, pd.DataFrame):
        raise TypeError("X_validation must be a pandas DataFrame.")

    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    if X_train.empty:
        raise ValueError("X_train is empty.")

    if X_validation.empty:
        raise ValueError("X_validation is empty.")

    if y_train.empty:
        raise ValueError("y_train is empty.")

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train must contain the same number of rows."
        )

    if list(X_train.columns) != list(X_validation.columns):
        raise ValueError(
            "X_train and X_validation must contain the same columns "
            "in the same order."
        )

    if X_train.isna().any().any():
        raise ValueError("X_train contains missing values.")

    if X_validation.isna().any().any():
        raise ValueError("X_validation contains missing values.")

    if y_train.isna().any():
        raise ValueError("y_train contains missing values.")


def train_xgboost_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    min_child_weight: float = 1.0,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[XGBRegressor, np.ndarray]:
    """
    Train an XGBoost regressor and predict the validation set.
    """

    _validate_xgboost_data(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
    )

    if n_estimators <= 0:
        raise ValueError("n_estimators must be greater than zero.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero.")

    if max_depth <= 0:
        raise ValueError("max_depth must be greater than zero.")

    if min_child_weight < 0:
        raise ValueError(
            "min_child_weight must be greater than or equal to zero."
        )

    if not 0 < subsample <= 1:
        raise ValueError("subsample must be between 0 and 1.")

    if not 0 < colsample_bytree <= 1:
        raise ValueError(
            "colsample_bytree must be between 0 and 1."
        )

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        tree_method="hist",
        random_state=random_state,
        n_jobs=n_jobs,
    )

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    validation_predictions = model.predict(
        X_validation
    )

    return model, np.asarray(
        validation_predictions,
        dtype=float,
    )


def get_xgboost_feature_importance(
    model: XGBRegressor,
    feature_names: list[str] | pd.Index,
) -> pd.DataFrame:
    """
    Return XGBoost feature importances in descending order.
    """

    feature_names = list(feature_names)

    if len(feature_names) != len(model.feature_importances_):
        raise ValueError(
            "Number of feature names does not match model features."
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )

    return (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )