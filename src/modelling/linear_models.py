# src/modelling/linear_models.py

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.base import RegressorMixin


def _validate_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
) -> None:
    """
    Validate model training inputs.
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


def predict_model(
    model: RegressorMixin,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Generate predictions using a fitted regression model.
    """

    if not hasattr(model, "predict"):
        raise TypeError("model must provide a predict method.")

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if X.empty:
        raise ValueError("X is empty.")

    predictions = model.predict(X)

    return np.asarray(predictions, dtype=float)


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
) -> tuple[LinearRegression, np.ndarray]:
    """
    Train a Linear Regression model and predict the validation set.
    """

    _validate_training_data(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
    )

    model = LinearRegression()

    model.fit(
        X_train,
        y_train,
    )

    validation_predictions = predict_model(
        model,
        X_validation,
    )

    return model, validation_predictions


def train_ridge_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    alpha: float = 1.0,
) -> tuple[Ridge, np.ndarray]:
    """
    Train a Ridge Regression model and predict the validation set.
    """

    _validate_training_data(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
    )

    if alpha < 0:
        raise ValueError("alpha must be greater than or equal to zero.")

    model = Ridge(
        alpha=alpha,
    )

    model.fit(
        X_train,
        y_train,
    )

    validation_predictions = predict_model(
        model,
        X_validation,
    )

    return model, validation_predictions