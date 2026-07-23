from __future__ import annotations

import numpy as np
import pandas as pd


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert calendar variables into cyclical sine and cosine features.

    This helps the model understand that:
    - 23:30 is close to 00:00
    - Sunday is close to Monday
    - December is close to January
    """

    required_columns = {
        "hour",
        "minute",
        "day_of_week",
        "month",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    cyclical_df = df.copy()

    # Convert hour and minute into one continuous time-of-day value.
    cyclical_df["time_of_day"] = (
        cyclical_df["hour"]
        + cyclical_df["minute"] / 60
    )

    cyclical_df["time_sin"] = np.sin(
        2 * np.pi * cyclical_df["time_of_day"] / 24
    )

    cyclical_df["time_cos"] = np.cos(
        2 * np.pi * cyclical_df["time_of_day"] / 24
    )

    cyclical_df["day_of_week_sin"] = np.sin(
        2 * np.pi * cyclical_df["day_of_week"] / 7
    )

    cyclical_df["day_of_week_cos"] = np.cos(
        2 * np.pi * cyclical_df["day_of_week"] / 7
    )

    cyclical_df["month_sin"] = np.sin(
        2 * np.pi * (cyclical_df["month"] - 1) / 12
    )

    cyclical_df["month_cos"] = np.cos(
        2 * np.pi * (cyclical_df["month"] - 1) / 12
    )

    return cyclical_df