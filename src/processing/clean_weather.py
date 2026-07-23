from __future__ import annotations

import pandas as pd


TIMESTAMP_COLUMN = "INTERVAL_DATETIME"

COLUMN_RENAME_MAP = {
    "time": TIMESTAMP_COLUMN,
    "temperature_2m": "temperature",
    "relative_humidity_2m": "relative_humidity",
    "apparent_temperature": "apparent_temperature",
    "precipitation": "precipitation",
    "cloud_cover": "cloud_cover",
    "wind_speed_10m": "wind_speed",
    "shortwave_radiation_instant": "solar_radiation",
}

CONTINUOUS_COLUMNS = [
    "temperature",
    "relative_humidity",
    "apparent_temperature",
    "cloud_cover",
    "wind_speed",
    "solar_radiation",
]

REQUIRED_RAW_COLUMNS = list(COLUMN_RENAME_MAP.keys())


def _validate_raw_weather(weather_df: pd.DataFrame) -> None:
    """
    Validate the raw hourly weather DataFrame.
    """

    if not isinstance(weather_df, pd.DataFrame):
        raise TypeError("weather_df must be a pandas DataFrame.")

    if weather_df.empty:
        raise ValueError("weather_df is empty.")

    missing_columns = set(REQUIRED_RAW_COLUMNS).difference(
        weather_df.columns
    )

    if missing_columns:
        raise ValueError(
            "Raw weather data is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def _validate_hourly_weather(weather_df: pd.DataFrame) -> None:
    """
    Validate cleaned hourly weather observations.
    """

    duplicate_count = weather_df[TIMESTAMP_COLUMN].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Weather data contains {duplicate_count} duplicate timestamps."
        )

    if weather_df[TIMESTAMP_COLUMN].isna().any():
        raise ValueError(
            "Weather data contains invalid or missing timestamps."
        )

    weather_columns = CONTINUOUS_COLUMNS + ["precipitation"]

    non_numeric_columns = [
        column
        for column in weather_columns
        if not pd.api.types.is_numeric_dtype(weather_df[column])
    ]

    if non_numeric_columns:
        raise TypeError(
            "The following weather columns are not numeric: "
            f"{non_numeric_columns}"
        )

    missing_values = weather_df[weather_columns].isna().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        raise ValueError(
            "Hourly weather data contains missing values:\n"
            f"{missing_values}"
        )

    time_differences = weather_df[
        TIMESTAMP_COLUMN
    ].diff().dropna()

    unexpected_differences = time_differences[
        time_differences != pd.Timedelta(hours=1)
    ]

    if not unexpected_differences.empty:
        raise ValueError(
            "Hourly weather timestamps are not continuous. "
            f"Found {len(unexpected_differences)} unexpected intervals."
        )


def _resample_continuous_variables(
    hourly_weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert continuous hourly variables to 30-minute frequency using
    time-based linear interpolation.
    """

    continuous_df = (
        hourly_weather_df[
            [TIMESTAMP_COLUMN] + CONTINUOUS_COLUMNS
        ]
        .set_index(TIMESTAMP_COLUMN)
        .resample("30min")
        .interpolate(method="time")
        .reset_index()
    )

    return continuous_df


def _expand_hourly_precipitation(
    hourly_weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align each hourly precipitation observation with the two 30-minute
    intervals covered by that hour.

    An observation recorded at 05:00 is assigned to:

        04:30
        05:00

    The precipitation amount is not linearly interpolated.
    """

    precipitation_df = hourly_weather_df[
        [TIMESTAMP_COLUMN, "precipitation"]
    ].copy()

    interval_end_df = precipitation_df.copy()

    first_half_df = precipitation_df.copy()
    first_half_df[TIMESTAMP_COLUMN] = (
        first_half_df[TIMESTAMP_COLUMN]
        - pd.Timedelta(minutes=30)
    )

    half_hourly_precipitation = pd.concat(
        [first_half_df, interval_end_df],
        ignore_index=True,
    )

    half_hourly_precipitation = (
        half_hourly_precipitation
        .sort_values(TIMESTAMP_COLUMN)
        .drop_duplicates(
            subset=[TIMESTAMP_COLUMN],
            keep="last",
        )
        .reset_index(drop=True)
    )

    half_hourly_precipitation["is_raining"] = (
        half_hourly_precipitation["precipitation"] > 0
    ).astype("int8")

    return half_hourly_precipitation


def _validate_half_hourly_weather(
    weather_df: pd.DataFrame,
) -> None:
    """
    Validate the final half-hourly weather dataset.
    """

    duplicate_count = weather_df[TIMESTAMP_COLUMN].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(
            "Half-hourly weather data contains duplicate timestamps."
        )

    missing_values = weather_df.isna().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        raise ValueError(
            "Half-hourly weather data contains missing values:\n"
            f"{missing_values}"
        )

    time_differences = weather_df[
        TIMESTAMP_COLUMN
    ].diff().dropna()

    unexpected_differences = time_differences[
        time_differences != pd.Timedelta(minutes=30)
    ]

    if not unexpected_differences.empty:
        raise ValueError(
            "Half-hourly weather timestamps are not continuous. "
            f"Found {len(unexpected_differences)} unexpected intervals."
        )


def clean_weather_data(
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean hourly Open-Meteo weather data and convert it to a
    merge-ready 30-minute dataset.

    Parameters
    ----------
    weather_df:
        Raw hourly weather DataFrame returned by
        load_historical_weather().

    Returns
    -------
    pd.DataFrame
        Clean half-hourly weather dataset.
    """

    _validate_raw_weather(weather_df)

    cleaned_df = weather_df[REQUIRED_RAW_COLUMNS].copy()

    cleaned_df = cleaned_df.rename(
        columns=COLUMN_RENAME_MAP
    )

    cleaned_df[TIMESTAMP_COLUMN] = pd.to_datetime(
        cleaned_df[TIMESTAMP_COLUMN],
        errors="raise",
    )

    numeric_columns = CONTINUOUS_COLUMNS + ["precipitation"]

    cleaned_df[numeric_columns] = cleaned_df[
        numeric_columns
    ].apply(
        pd.to_numeric,
        errors="raise",
    )

    cleaned_df = (
        cleaned_df
        .sort_values(TIMESTAMP_COLUMN)
        .reset_index(drop=True)
    )

    _validate_hourly_weather(cleaned_df)

    continuous_df = _resample_continuous_variables(
        cleaned_df
    )

    precipitation_df = _expand_hourly_precipitation(
        cleaned_df
    )

    half_hourly_weather_df = continuous_df.merge(
        precipitation_df,
        on=TIMESTAMP_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    half_hourly_weather_df = (
        half_hourly_weather_df
        .sort_values(TIMESTAMP_COLUMN)
        .reset_index(drop=True)
    )

    _validate_half_hourly_weather(
        half_hourly_weather_df
    )

    return half_hourly_weather_df