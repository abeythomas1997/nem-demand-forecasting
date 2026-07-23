from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


DEFAULT_LAGS = (1, 2, 48, 336)


def add_lag_features(
    df: pd.DataFrame,
    lags: Sequence[int] = DEFAULT_LAGS,
    target_column: str = "OPERATIONAL_DEMAND",
) -> pd.DataFrame:
    """
    Create lag features from the operational demand target.

    For 30-minute data:
    - lag_1   = demand 30 minutes earlier
    - lag_2   = demand 1 hour earlier
    - lag_48  = demand at the same time on the previous day
    - lag_336 = demand at the same time one week earlier

    Parameters
    ----------
    df:
        Chronologically sorted DataFrame.
    lags:
        Positive row offsets used to create lag features.
    target_column:
        Column used to generate the lag values.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional lag columns.
    """

    required_columns = {
        "INTERVAL_DATETIME",
        target_column,
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if not lags:
        raise ValueError("At least one lag must be provided.")

    invalid_lags = [
        lag
        for lag in lags
        if not isinstance(lag, int) or lag <= 0
    ]

    if invalid_lags:
        raise ValueError(
            f"Lags must be positive integers: {invalid_lags}"
        )

    lag_df = df.copy()

    lag_df = lag_df.sort_values(
        by="INTERVAL_DATETIME"
    ).reset_index(drop=True)

    for lag in lags:
        lag_df[f"lag_{lag}"] = lag_df[target_column].shift(lag)

    return lag_df