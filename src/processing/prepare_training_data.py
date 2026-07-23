import pandas as pd


def prepare_training_data(
    df: pd.DataFrame,
    datetime_col: str = "INTERVAL_DATETIME",
    target_col: str = "OPERATIONAL_DEMAND",
) -> pd.DataFrame:
    """
    Prepare the engineered dataset for modelling.

    Steps:
    1. Validate required columns.
    2. Convert the datetime column.
    3. Sort observations chronologically.
    4. Check for duplicate timestamps.
    5. Remove warm-up rows containing missing feature values.
    6. Reset the dataframe index.
    """

    required_columns = {datetime_col, target_col}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    prepared_df = df.copy()

    prepared_df[datetime_col] = pd.to_datetime(
        prepared_df[datetime_col],
        errors="raise",
    )

    prepared_df = prepared_df.sort_values(
        datetime_col
    ).reset_index(drop=True)

    duplicate_count = prepared_df[datetime_col].duplicated().sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate timestamps."
        )

    prepared_df = prepared_df.dropna().reset_index(drop=True)

    if prepared_df.empty:
        raise ValueError(
            "No rows remain after removing missing values."
        )

    if prepared_df.isna().any().any():
        raise ValueError(
            "Missing values remain in the prepared dataset."
        )

    return prepared_df