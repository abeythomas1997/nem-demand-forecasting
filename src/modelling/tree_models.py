# src/modelling/tree_models.py

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor


def _validate_tree_model_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
) -> None:
    """
    Validate training and validation inputs for tree-based models.
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


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    n_estimators: int = 300,
    max_depth: int | None = 20,
    min_samples_leaf: int = 2,
    max_features: str | float | int | None = 1.0,
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[RandomForestRegressor, np.ndarray]:
    """
    Train a Random Forest regressor and predict the validation set.
    """

    _validate_tree_model_data(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
    )

    if n_estimators <= 0:
        raise ValueError("n_estimators must be greater than zero.")

    if max_depth is not None and max_depth <= 0:
        raise ValueError("max_depth must be greater than zero or None.")

    if min_samples_leaf <= 0:
        raise ValueError("min_samples_leaf must be greater than zero.")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    model.fit(
        X_train,
        y_train,
    )

    validation_predictions = model.predict(
        X_validation
    )

    return model, np.asarray(
        validation_predictions,
        dtype=float,
    )


def get_feature_importance(
    model: RandomForestRegressor,
    feature_names: list[str] | pd.Index,
) -> pd.DataFrame:
    """
    Return Random Forest feature importances in descending order.
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