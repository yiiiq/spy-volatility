"""
Target engineering: next-day realized volatility as abs(ret_1d.shift(-1)).
"""

import pandas as pd


TARGET_COL = "target_vol"


def make_target(df: pd.DataFrame) -> pd.Series:
    """
    Compute the prediction target: absolute next-day return.

    target_vol = abs(Close.pct_change().shift(-1))

    Args:
        df: DataFrame with at least a 'Close' column.

    Returns:
        Series named 'target_vol' with NaN rows dropped.
    """
    ret_1d = df["Close"].pct_change()
    target = ret_1d.shift(-1).abs()
    target.name = TARGET_COL
    target = target.dropna()

    print("\n--- Target summary statistics ---")
    print(target.describe().to_string())
    print()

    return target
