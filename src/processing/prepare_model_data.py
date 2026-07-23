from __future__ import annotations

import pandas as pd


COLUMNS_TO_KEEP = [
    "INTERVAL_DATETIME",
    "OPERATIONAL_DEMAND",
]


def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns required for initial feature engineering.

    Parameters
    ----------
    df:
        Cleaned operational demand DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame containing datetime and operational demand.
    """

    missing_columns = [
        column
        for column in COLUMNS_TO_KEEP
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    model_df = df[COLUMNS_TO_KEEP].copy()

    model_df = model_df.sort_values(
        by="INTERVAL_DATETIME"
    ).reset_index(drop=True)

    return model_df
    