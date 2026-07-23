from __future__ import annotations

import pandas as pd


TIMESTAMP_COLUMN = "INTERVAL_DATETIME"


def split_time_series_data(
    data: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    timestamp_column: str = TIMESTAMP_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time-series data chronologically into train, validation, and test sets.

    The data is never shuffled.

    Parameters
    ----------
    data:
        Final modelling DataFrame.

    train_ratio:
        Proportion assigned to the training set.

    validation_ratio:
        Proportion assigned to the validation set.

    timestamp_column:
        Name of the datetime column.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        train_df, validation_df, test_df
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("data is empty.")

    if timestamp_column not in data.columns:
        raise ValueError(
            f"Timestamp column '{timestamp_column}' was not found."
        )

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")

    test_ratio = 1 - train_ratio - validation_ratio

    if test_ratio <= 0:
        raise ValueError(
            "train_ratio and validation_ratio must leave data for testing."
        )

    split_df = data.copy()

    split_df[timestamp_column] = pd.to_datetime(
        split_df[timestamp_column],
        errors="raise",
    )

    split_df = (
        split_df
        .sort_values(timestamp_column)
        .reset_index(drop=True)
    )

    duplicate_count = split_df[timestamp_column].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Data contains {duplicate_count} duplicate timestamps."
        )

    total_rows = len(split_df)

    train_end = int(total_rows * train_ratio)
    validation_end = train_end + int(
        total_rows * validation_ratio
    )

    train_df = split_df.iloc[:train_end].copy()
    validation_df = split_df.iloc[
        train_end:validation_end
    ].copy()
    test_df = split_df.iloc[validation_end:].copy()

    if train_df.empty or validation_df.empty or test_df.empty:
        raise ValueError(
            "One or more datasets are empty after splitting."
        )

    if (
        train_df[timestamp_column].max()
        >= validation_df[timestamp_column].min()
    ):
        raise ValueError(
            "Training and validation periods overlap."
        )

    if (
        validation_df[timestamp_column].max()
        >= test_df[timestamp_column].min()
    ):
        raise ValueError(
            "Validation and test periods overlap."
        )

    return train_df, validation_df, test_df