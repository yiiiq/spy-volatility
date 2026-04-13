"""
Data loading: download SPY OHLCV data via yfinance or reload from disk.
"""

import os
import pandas as pd
import yfinance as yf

DATA_PATH = "data/spy.csv"
START_DATE = "2010-01-01"
COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def load_data() -> pd.DataFrame:
    """
    Load SPY daily OHLCV data.

    If data/spy.csv already exists it is read from disk; otherwise the data is
    downloaded via yfinance and saved for future runs.

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume] and a
        DatetimeIndex, with no missing values.
    """
    if os.path.exists(DATA_PATH):
        print(f"Loading data from {DATA_PATH} …")
        df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    else:
        print(f"Downloading SPY data from {START_DATE} via yfinance …")
        raw = yf.download("SPY", start=START_DATE, auto_adjust=True, progress=False)
        # Flatten multi-level columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[COLUMNS].copy()
        os.makedirs("data", exist_ok=True)
        df.to_csv(DATA_PATH)
        print(f"Saved to {DATA_PATH}")

    df = df[COLUMNS].dropna()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    print(f"Loaded {len(df)} rows  ({df.index[0].date()} – {df.index[-1].date()})")
    return df
