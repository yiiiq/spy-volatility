"""
Utility functions: random seed, train/test split, evaluation metrics.
"""

import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Single source of truth for the random seed
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def split_train_test(
    df: pd.DataFrame, split_date: str = "2022-01-01"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train and test sets using a time-based cutoff.

    Args:
        df: DataFrame with a DatetimeIndex.
        split_date: All rows before this date go to train; on/after go to test.

    Returns:
        Tuple of (train_df, test_df).
    """
    train = df[df.index < split_date]
    test = df[df.index >= split_date]
    print(f"Train: {len(train)} rows ({train.index[0].date()} – {train.index[-1].date()})")
    print(f"Test:  {len(test)} rows  ({test.index[0].date()} – {test.index[-1].date()})")
    return train, test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression evaluation metrics.

    Args:
        y_true: Array of true values.
        y_pred: Array of predicted values.

    Returns:
        Dictionary with MAE, RMSE, R2, MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # Avoid division by zero in MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}
