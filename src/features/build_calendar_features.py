from __future__ import annotations

import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create calendar-based features from INTERVAL_DATETIME.

    Parameters
    ----------
    df:
        Prepared operational demand DataFrame containing
        INTERVAL_DATETIME and OPERATIONAL_DEMAND.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional calendar features.
    """

    required_columns = {
        "INTERVAL_DATETIME",
        "OPERATIONAL_DEMAND",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    feature_df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(
        feature_df["INTERVAL_DATETIME"]
    ):
        feature_df["INTERVAL_DATETIME"] = pd.to_datetime(
            feature_df["INTERVAL_DATETIME"],
            errors="coerce",
        )

    if feature_df["INTERVAL_DATETIME"].isna().any():
        raise ValueError(
            "INTERVAL_DATETIME contains invalid or missing datetime values."
        )

    feature_df["year"] = feature_df["INTERVAL_DATETIME"].dt.year
    feature_df["month"] = feature_df["INTERVAL_DATETIME"].dt.month
    feature_df["day"] = feature_df["INTERVAL_DATETIME"].dt.day
    feature_df["day_of_week"] = (
        feature_df["INTERVAL_DATETIME"].dt.dayofweek
    )
    feature_df["hour"] = feature_df["INTERVAL_DATETIME"].dt.hour
    feature_df["minute"] = feature_df["INTERVAL_DATETIME"].dt.minute

    feature_df["is_weekend"] = (
        feature_df["day_of_week"] >= 5
    ).astype(int)

    feature_df = feature_df.sort_values(
        by="INTERVAL_DATETIME"
    ).reset_index(drop=True)

    return feature_df