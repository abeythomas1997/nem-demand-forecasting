from __future__ import annotations

import pandas as pd

from sklearn.preprocessing import StandardScaler


def standard_scale_features(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    StandardScaler,
]:
    """
    Fit a StandardScaler using only the training data.

    The fitted scaler is then applied to the validation and
    test datasets to prevent data leakage.

    Returns
    -------
    X_train_scaled
    X_validation_scaled
    X_test_scaled
    scaler
    """

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a DataFrame.")

    if not isinstance(X_validation, pd.DataFrame):
        raise TypeError("X_validation must be a DataFrame.")

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a DataFrame.")

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_validation_scaled = pd.DataFrame(
        scaler.transform(X_validation),
        columns=X_validation.columns,
        index=X_validation.index,
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    return (
        X_train_scaled,
        X_validation_scaled,
        X_test_scaled,
        scaler,
    )