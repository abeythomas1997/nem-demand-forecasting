# src/features/build_weather_features.py

from __future__ import annotations

import pandas as pd


TIMESTAMP_COLUMN = "INTERVAL_DATETIME"

REQUIRED_WEATHER_COLUMNS = [
    TIMESTAMP_COLUMN,
    "temperature",
    "relative_humidity",
    "apparent_temperature",
    "precipitation",
    "solar_radiation",
    "is_raining",
]


def _validate_weather_columns(
    weather_df: pd.DataFrame,
) -> None:
    """
    Validate columns required for weather feature engineering.
    """

    if not isinstance(weather_df, pd.DataFrame):
        raise TypeError("weather_df must be a pandas DataFrame.")

    if weather_df.empty:
        raise ValueError("weather_df is empty.")

    missing_columns = set(REQUIRED_WEATHER_COLUMNS).difference(
        weather_df.columns
    )

    if missing_columns:
        raise ValueError(
            "weather_df is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def build_weather_features(
    weather_df: pd.DataFrame,
    heating_base_temperature: float = 18.0,
    cooling_base_temperature: float = 22.0,
    high_temperature_threshold: float = 30.0,
    high_humidity_threshold: float = 80.0,
) -> pd.DataFrame:
    """
    Add derived weather features to cleaned half-hourly weather data.

    Parameters
    ----------
    weather_df:
        Clean half-hourly weather DataFrame.

    heating_base_temperature:
        Temperature below which heating demand may increase.

    cooling_base_temperature:
        Temperature above which cooling demand may increase.

    high_temperature_threshold:
        Threshold used to create the high-temperature indicator.

    high_humidity_threshold:
        Threshold used to create the high-humidity indicator.

    Returns
    -------
    pd.DataFrame
        Weather DataFrame containing original and derived features.
    """

    _validate_weather_columns(weather_df)

    featured_df = weather_df.copy()

    featured_df[TIMESTAMP_COLUMN] = pd.to_datetime(
        featured_df[TIMESTAMP_COLUMN],
        errors="raise",
    )

    featured_df = (
        featured_df
        .sort_values(TIMESTAMP_COLUMN)
        .reset_index(drop=True)
    )

    duplicate_count = featured_df[
        TIMESTAMP_COLUMN
    ].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(
            f"weather_df contains {duplicate_count} duplicate timestamps."
        )

    # Heating requirement:
    # Larger values indicate colder conditions.
    featured_df["heating_degree"] = (
        heating_base_temperature
        - featured_df["temperature"]
    ).clip(lower=0)

    # Cooling requirement:
    # Larger values indicate hotter conditions.
    featured_df["cooling_degree"] = (
        featured_df["temperature"]
        - cooling_base_temperature
    ).clip(lower=0)

    # Non-linear temperature effect.
    featured_df["temperature_squared"] = (
        featured_df["temperature"] ** 2
    )

    # Difference between measured and apparent temperature.
    featured_df["apparent_temperature_difference"] = (
        featured_df["apparent_temperature"]
        - featured_df["temperature"]
    )

    # Half-hourly data:
    # 2 rows = 1 hour
    # 6 rows = 3 hours
    featured_df["temperature_change_1h"] = (
        featured_df["temperature"].diff(periods=2)
    )

    featured_df["temperature_change_3h"] = (
        featured_df["temperature"].diff(periods=6)
    )

    # Humidity-temperature interaction.
    featured_df["temperature_humidity_interaction"] = (
        featured_df["temperature"]
        * featured_df["relative_humidity"]
    )

    # Binary extreme-weather indicators.
    featured_df["is_high_temperature"] = (
        featured_df["temperature"]
        >= high_temperature_threshold
    ).astype("int8")

    featured_df["is_high_humidity"] = (
        featured_df["relative_humidity"]
        >= high_humidity_threshold
    ).astype("int8")

    featured_df["is_hot_and_humid"] = (
        (
            featured_df["is_high_temperature"] == 1
        )
        & (
            featured_df["is_high_humidity"] == 1
        )
    ).astype("int8")

    # Daylight / solar-generation proxy.
    featured_df["is_daylight"] = (
        featured_df["solar_radiation"] > 0
    ).astype("int8")

    # Rain intensity groups.
    featured_df["is_moderate_rain"] = (
        featured_df["precipitation"] >= 1.0
    ).astype("int8")

    featured_df["is_heavy_rain"] = (
        featured_df["precipitation"] >= 5.0
    ).astype("int8")

    return featured_df