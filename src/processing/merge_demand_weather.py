# src/processing/merge_demand_weather.py

from __future__ import annotations

import pandas as pd


TIMESTAMP_COLUMN = "INTERVAL_DATETIME"

WEATHER_COLUMNS = [
    "temperature",
    "relative_humidity",
    "apparent_temperature",
    "cloud_cover",
    "wind_speed",
    "solar_radiation",
    "precipitation",
    "is_raining",
]


def _validate_demand_data(
    demand_df: pd.DataFrame,
) -> None:
    """
    Validate the demand feature DataFrame before merging.
    """

    if not isinstance(demand_df, pd.DataFrame):
        raise TypeError("demand_df must be a pandas DataFrame.")

    if demand_df.empty:
        raise ValueError("demand_df is empty.")

    if TIMESTAMP_COLUMN not in demand_df.columns:
        raise ValueError(
            f"demand_df must contain '{TIMESTAMP_COLUMN}'."
        )

    if demand_df[TIMESTAMP_COLUMN].isna().any():
        raise ValueError(
            "demand_df contains missing timestamps."
        )

    duplicate_count = demand_df[
        TIMESTAMP_COLUMN
    ].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(
            f"demand_df contains {duplicate_count} duplicate timestamps."
        )


def _validate_weather_data(
    weather_df: pd.DataFrame,
) -> None:
    """
    Validate the cleaned half-hourly weather DataFrame.
    """

    if not isinstance(weather_df, pd.DataFrame):
        raise TypeError("weather_df must be a pandas DataFrame.")

    if weather_df.empty:
        raise ValueError("weather_df is empty.")

    required_columns = {
        TIMESTAMP_COLUMN,
        *WEATHER_COLUMNS,
    }

    missing_columns = required_columns.difference(
        weather_df.columns
    )

    if missing_columns:
        raise ValueError(
            "weather_df is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if weather_df[TIMESTAMP_COLUMN].isna().any():
        raise ValueError(
            "weather_df contains missing timestamps."
        )

    duplicate_count = weather_df[
        TIMESTAMP_COLUMN
    ].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(
            f"weather_df contains {duplicate_count} duplicate timestamps."
        )


def merge_demand_weather(
    demand_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge demand features with cleaned weather data.

    Demand timestamps are used as the master timeline, so no demand
    observations should be removed.

    Parameters
    ----------
    demand_df:
        Demand DataFrame containing INTERVAL_DATETIME and demand features.

    weather_df:
        Clean half-hourly weather DataFrame.

    Returns
    -------
    pd.DataFrame
        Demand and weather data merged on INTERVAL_DATETIME.
    """

    _validate_demand_data(demand_df)
    _validate_weather_data(weather_df)

    demand = demand_df.copy()
    weather = weather_df.copy()

    demand[TIMESTAMP_COLUMN] = pd.to_datetime(
        demand[TIMESTAMP_COLUMN],
        errors="raise",
    )

    weather[TIMESTAMP_COLUMN] = pd.to_datetime(
        weather[TIMESTAMP_COLUMN],
        errors="raise",
    )

    demand = (
        demand
        .sort_values(TIMESTAMP_COLUMN)
        .reset_index(drop=True)
    )

    weather = (
        weather[
            [TIMESTAMP_COLUMN] + WEATHER_COLUMNS
        ]
        .sort_values(TIMESTAMP_COLUMN)
        .reset_index(drop=True)
    )

    demand_row_count = len(demand)

    merged_df = demand.merge(
        weather,
        on=TIMESTAMP_COLUMN,
        how="left",
        validate="one_to_one",
    )

    if len(merged_df) != demand_row_count:
        raise RuntimeError(
            "Demand row count changed during the weather merge."
        )

    duplicate_count = merged_df[
        TIMESTAMP_COLUMN
    ].duplicated().sum()

    if duplicate_count > 0:
        raise RuntimeError(
            "Merged data contains duplicate timestamps."
        )

    missing_weather_counts = merged_df[
        WEATHER_COLUMNS
    ].isna().sum()

    unmatched_rows = merged_df[
        WEATHER_COLUMNS
    ].isna().all(axis=1).sum()

    if unmatched_rows > 0:
        raise ValueError(
            f"{unmatched_rows} demand rows did not match weather data.\n"
            f"Missing weather values:\n{missing_weather_counts}"
        )

    partial_missing = missing_weather_counts[
        missing_weather_counts > 0
    ]

    if not partial_missing.empty:
        raise ValueError(
            "Merged data contains partially missing weather values:\n"
            f"{partial_missing}"
        )

    merged_df = (
        merged_df
        .sort_values(TIMESTAMP_COLUMN)
        .reset_index(drop=True)
    )

    return merged_df