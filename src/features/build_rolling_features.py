from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


DEFAULT_WINDOWS = (4, 48)


def add_rolling_features(
    df: pd.DataFrame,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    target_column: str = "OPERATIONAL_DEMAND",
) -> pd.DataFrame:
    """
    Create rolling statistical features from past demand values.

    For 30-minute data:
    - window 4  = previous 2 hours
    - window 48 = previous 24 hours

    Rolling calculations use only previous observations and exclude
    the current row to prevent target leakage.
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

    if not windows:
        raise ValueError("At least one rolling window must be provided.")

    invalid_windows = [
        window
        for window in windows
        if not isinstance(window, int) or window <= 0
    ]

    if invalid_windows:
        raise ValueError(
            f"Rolling windows must be positive integers: {invalid_windows}"
        )

    rolling_df = df.copy()

    rolling_df = rolling_df.sort_values(
        by="INTERVAL_DATETIME"
    ).reset_index(drop=True)

    # Shift first so the current target value is never included.
    past_demand = rolling_df[target_column].shift(1)

    for window in windows:
        rolling_window = past_demand.rolling(
            window=window,
            min_periods=window,
        )

        rolling_df[f"rolling_mean_{window}"] = rolling_window.mean()
        rolling_df[f"rolling_std_{window}"] = rolling_window.std()
        rolling_df[f"rolling_min_{window}"] = rolling_window.min()
        rolling_df[f"rolling_max_{window}"] = rolling_window.max()

    return rolling_df