from __future__ import annotations

import pandas as pd


NUMERIC_COLUMNS = [
    "OPERATIONAL_DEMAND",
    "OPERATIONAL_DEMAND_ADJUSTMENT",
    "WDR_ESTIMATE",
]


def clean_operational_demand(
    df: pd.DataFrame,
    region: str = "VIC1",
) -> pd.DataFrame:
    """
    Clean AEMO actual operational demand data for one region.

    Parameters
    ----------
    df:
        Raw DataFrame returned by load_aemo_directory().
    region:
        AEMO region identifier, such as VIC1, NSW1 or QLD1.

    Returns
    -------
    pd.DataFrame
        Cleaned operational demand data sorted by interval datetime.
    """

    required_columns = {
        "REGIONID",
        "INTERVAL_DATETIME",
        "OPERATIONAL_DEMAND",
        "OPERATIONAL_DEMAND_ADJUSTMENT",
        "WDR_ESTIMATE",
        "LASTCHANGED",
        "SOURCE_FILE",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    clean_df = df.copy()

    # Keep only the selected AEMO region.
    clean_df = clean_df.loc[
        clean_df["REGIONID"].eq(region)
    ].copy()

    if clean_df.empty:
        available_regions = sorted(
            df["REGIONID"].dropna().unique().tolist()
        )

        raise ValueError(
            f"No records found for region '{region}'. "
            f"Available regions: {available_regions}"
        )

    # Convert datetime columns.
    clean_df["INTERVAL_DATETIME"] = pd.to_datetime(
        clean_df["INTERVAL_DATETIME"],
        format="%Y/%m/%d %H:%M:%S",
        errors="coerce",
    )

    clean_df["LASTCHANGED"] = pd.to_datetime(
        clean_df["LASTCHANGED"],
        format="%Y/%m/%d %H:%M:%S",
        errors="coerce",
    )

    # Convert demand columns from object/string to numeric.
    for column in NUMERIC_COLUMNS:
        clean_df[column] = pd.to_numeric(
            clean_df[column],
            errors="coerce",
        )

    # Remove exact duplicate rows.
    clean_df = clean_df.drop_duplicates()

    # If the same region and interval appear more than once,
    # keep the most recently updated record.
    clean_df = clean_df.sort_values(
        by=["INTERVAL_DATETIME", "LASTCHANGED"]
    )

    clean_df = clean_df.drop_duplicates(
        subset=["REGIONID", "INTERVAL_DATETIME"],
        keep="last",
    )

    # Sort chronologically.
    clean_df = clean_df.sort_values(
        by="INTERVAL_DATETIME"
    ).reset_index(drop=True)

    # Validate important converted columns.
    validation_columns = [
        "INTERVAL_DATETIME",
        "OPERATIONAL_DEMAND",
    ]

    null_counts = clean_df[validation_columns].isna().sum()
    invalid_columns = null_counts[null_counts > 0]

    if not invalid_columns.empty:
        raise ValueError(
            "Invalid or missing values found after conversion:\n"
            f"{invalid_columns.to_string()}"
        )

    return clean_df


def find_missing_intervals(
    df: pd.DataFrame,
    frequency: str = "30min",
) -> pd.DatetimeIndex:
    """
    Find missing timestamps in a cleaned operational demand DataFrame.
    """

    if "INTERVAL_DATETIME" not in df.columns:
        raise ValueError(
            "DataFrame must contain an INTERVAL_DATETIME column."
        )

    if df.empty:
        return pd.DatetimeIndex([])

    actual_intervals = pd.DatetimeIndex(
        df["INTERVAL_DATETIME"].dropna().unique()
    )

    expected_intervals = pd.date_range(
        start=actual_intervals.min(),
        end=actual_intervals.max(),
        freq=frequency,
    )

    return expected_intervals.difference(actual_intervals)


def validate_operational_demand(
    df: pd.DataFrame,
    frequency: str = "30min",
) -> dict:
    """
    Return basic data-quality validation results.
    """

    missing_intervals = find_missing_intervals(
        df,
        frequency=frequency,
    )

    duplicate_count = df.duplicated(
        subset=["REGIONID", "INTERVAL_DATETIME"]
    ).sum()

    return {
        "rows": len(df),
        "start_datetime": df["INTERVAL_DATETIME"].min(),
        "end_datetime": df["INTERVAL_DATETIME"].max(),
        "duplicate_intervals": int(duplicate_count),
        "missing_interval_count": len(missing_intervals),
        "missing_intervals": missing_intervals.tolist(),
        "missing_values": df.isna().sum().to_dict(),
    }