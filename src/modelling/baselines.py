from __future__ import annotations

import pandas as pd


TARGET_COLUMN = "OPERATIONAL_DEMAND"

BASELINE_COLUMNS = {
    "previous_interval": "lag_1",
    "previous_day": "lag_48",
    "previous_week": "lag_336",
}


def generate_baseline_predictions(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """
    Generate baseline demand forecasts using existing lag features.

    Parameters
    ----------
    data:
        Validation or test DataFrame containing the target and lag columns.

    target_column:
        Name of the target demand column.

    Returns
    -------
    pd.DataFrame
        DataFrame containing actual demand and baseline predictions.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("data is empty.")

    required_columns = {
        target_column,
        *BASELINE_COLUMNS.values(),
    }

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    prediction_df = pd.DataFrame(
        index=data.index
    )

    prediction_df["actual"] = data[target_column]

    for baseline_name, lag_column in BASELINE_COLUMNS.items():
        prediction_df[baseline_name] = data[lag_column]

    if prediction_df.isna().any().any():
        raise ValueError(
            "Baseline predictions contain missing values."
        )

    return prediction_df.reset_index(drop=True)