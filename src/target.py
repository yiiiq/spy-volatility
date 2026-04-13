"""
Target engineering for volatility prediction.

Target matches the final-project volatility setup:
next-day rolling volatility of log returns.

rolling_vol_t = std(r_{t-window+1}, ..., r_t)
target_t = rolling_vol_{t+1}
"""

import numpy as np
import pandas as pd

TARGET_COL = "target_vol"


def make_target(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    Compute next-day rolling volatility target from log returns.

    Args:
        df: DataFrame with 'Close' column.
        window: Look-back window for rolling volatility.

    Returns:
        Series named 'target_vol' with NaN rows dropped.
    """
    log_return = np.log(df["Close"] / df["Close"].shift(1))
    rolling_vol = log_return.rolling(window=window).std()
    target = rolling_vol.shift(-1)

    target.name = TARGET_COL
    target = target.dropna()

    print("\n--- Target summary statistics ---")
    print(target.describe().to_string())
    print()

    return target
