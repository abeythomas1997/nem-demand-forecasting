# src/ingestion/weather.py

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import requests


OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DEFAULT_HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation_instant",
]


def load_historical_weather(
    start_date: str | date,
    end_date: str | date,
    latitude: float = -37.8136,
    longitude: float = 144.9631,
    timezone: str = "Australia/Melbourne",
    hourly_variables: list[str] | None = None,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Download hourly historical weather data from Open-Meteo.

    Parameters
    ----------
    start_date:
        First date to retrieve, in YYYY-MM-DD format.

    end_date:
        Last date to retrieve, in YYYY-MM-DD format.

    latitude:
        Location latitude. Defaults to Melbourne.

    longitude:
        Location longitude. Defaults to Melbourne.

    timezone:
        Timezone used for returned timestamps.

    hourly_variables:
        Weather variables requested from the API.

    timeout:
        Maximum number of seconds to wait for the API response.

    Returns
    -------
    pd.DataFrame
        Raw hourly weather data returned by Open-Meteo.
    """

    start_date = pd.Timestamp(start_date).date()
    end_date = pd.Timestamp(end_date).date()

    if start_date > end_date:
        raise ValueError(
            "start_date must be earlier than or equal to end_date."
        )

    if not -90 <= latitude <= 90:
        raise ValueError("latitude must be between -90 and 90.")

    if not -180 <= longitude <= 180:
        raise ValueError("longitude must be between -180 and 180.")

    variables = hourly_variables or DEFAULT_HOURLY_VARIABLES

    if not variables:
        raise ValueError("At least one hourly weather variable is required.")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(variables),
        "timezone": timezone,
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "timeformat": "iso8601",
    }

    try:
        response = requests.get(
            OPEN_METEO_ARCHIVE_URL,
            params=params,
            timeout=timeout,
        )

        response.raise_for_status()

    except requests.Timeout as exc:
        raise RuntimeError(
            "The Open-Meteo request timed out."
        ) from exc

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Unable to retrieve historical weather data: {exc}"
        ) from exc

    try:
        response_data: dict[str, Any] = response.json()

    except ValueError as exc:
        raise RuntimeError(
            "Open-Meteo returned an invalid JSON response."
        ) from exc

    if response_data.get("error"):
        reason = response_data.get(
            "reason",
            "Unknown Open-Meteo API error.",
        )
        raise RuntimeError(reason)

    hourly_data = response_data.get("hourly")

    if not isinstance(hourly_data, dict):
        raise RuntimeError(
            "The API response does not contain hourly weather data."
        )

    if "time" not in hourly_data:
        raise RuntimeError(
            "The hourly weather response does not contain timestamps."
        )

    weather_df = pd.DataFrame(hourly_data)

    if weather_df.empty:
        raise RuntimeError(
            "The API returned an empty hourly weather dataset."
        )

    expected_columns = {"time", *variables}
    missing_columns = expected_columns.difference(weather_df.columns)

    if missing_columns:
        raise RuntimeError(
            "The API response is missing expected columns: "
            f"{sorted(missing_columns)}"
        )

    weather_df.attrs["latitude"] = response_data.get("latitude")
    weather_df.attrs["longitude"] = response_data.get("longitude")
    weather_df.attrs["elevation"] = response_data.get("elevation")
    weather_df.attrs["timezone"] = response_data.get("timezone")
    weather_df.attrs["timezone_abbreviation"] = response_data.get(
        "timezone_abbreviation"
    )
    weather_df.attrs["utc_offset_seconds"] = response_data.get(
        "utc_offset_seconds"
    )
    weather_df.attrs["hourly_units"] = response_data.get(
        "hourly_units",
        {},
    )

    return weather_df