from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def calculate_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """
    Calculate regression forecasting metrics.
    """

    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    if len(y_true_array) != len(y_pred_array):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    if len(y_true_array) == 0:
        raise ValueError("Prediction arrays cannot be empty.")

    mae = mean_absolute_error(
        y_true_array,
        y_pred_array,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true_array,
            y_pred_array,
        )
    )

    non_zero_mask = y_true_array != 0

    if not non_zero_mask.all():
        raise ValueError(
            "MAPE cannot be calculated because y_true contains zero."
        )

    mape = (
        np.mean(
            np.abs(
                (
                    y_true_array
                    - y_pred_array
                )
                / y_true_array
            )
        )
        * 100
    )

    r2 = r2_score(
        y_true_array,
        y_pred_array,
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2),
    }


def evaluate_prediction_dataframe(
    prediction_df: pd.DataFrame,
    actual_column: str = "actual",
) -> pd.DataFrame:
    """
    Evaluate all prediction columns against the actual column.
    """

    if actual_column not in prediction_df.columns:
        raise ValueError(
            f"Actual column '{actual_column}' was not found."
        )

    prediction_columns = [
        column
        for column in prediction_df.columns
        if column != actual_column
    ]

    if not prediction_columns:
        raise ValueError(
            "No prediction columns were found."
        )

    metric_rows = []

    for prediction_column in prediction_columns:
        metrics = calculate_regression_metrics(
            y_true=prediction_df[actual_column],
            y_pred=prediction_df[prediction_column],
        )

        metric_rows.append(
            {
                "model": prediction_column,
                **metrics,
            }
        )

    return (
        pd.DataFrame(metric_rows)
        .sort_values("MAE")
        .reset_index(drop=True)
    )