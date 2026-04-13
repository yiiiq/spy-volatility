"""
Feature engineering: create all predictive features from raw OHLCV data.
"""

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature set from raw OHLCV columns.

    All features use only current and past information to avoid look-ahead bias.

    Args:
        df: DataFrame with columns [Open, High, Low, Close, Volume].

    Returns:
        DataFrame containing only the engineered feature columns (NaN rows
        dropped).
    """
    feat = pd.DataFrame(index=df.index)

    # --- Returns and momentum ---
    feat["ret_1d"] = df["Close"].pct_change(1)
    feat["ret_3d"] = df["Close"].pct_change(3)
    feat["ret_5d"] = df["Close"].pct_change(5)
    feat["ret_10d"] = df["Close"].pct_change(10)
    feat["mom_5"] = df["Close"] / df["Close"].shift(5) - 1
    feat["mom_10"] = df["Close"] / df["Close"].shift(10) - 1

    # --- Trend features ---
    ma_5 = df["Close"].rolling(5).mean()
    ma_20 = df["Close"].rolling(20).mean()
    ma_50 = df["Close"].rolling(50).mean()
    feat["ma_5"] = ma_5
    feat["ma_20"] = ma_20
    feat["ma_50"] = ma_50
    feat["price_to_ma20"] = df["Close"] / ma_20
    feat["price_to_ma50"] = df["Close"] / ma_50
    feat["ma_ratio_5_20"] = ma_5 / ma_20

    # --- Volatility features ---
    feat["vol_5"] = feat["ret_1d"].rolling(5).std()
    feat["vol_10"] = feat["ret_1d"].rolling(10).std()
    feat["vol_20"] = feat["ret_1d"].rolling(20).std()
    feat["vol_ratio"] = feat["vol_5"] / feat["vol_20"]

    # --- Range and candle features ---
    feat["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
    feat["co_return"] = (df["Close"] - df["Open"]) / df["Open"]
    feat["oc_gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    # --- Volume features ---
    feat["vol_chg"] = df["Volume"].pct_change()
    vol_ma20 = df["Volume"].rolling(20).mean()
    feat["vol_ma20"] = vol_ma20
    feat["volume_spike"] = df["Volume"] / vol_ma20

    # --- RSI (14-day) ---
    feat["rsi"] = _compute_rsi(df["Close"], period=14)

    feat = feat.dropna()
    print(f"Features shape after dropna: {feat.shape}")
    return feat


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index.

    Args:
        close: Series of closing prices.
        period: Look-back window (default 14).

    Returns:
        RSI series aligned with the input index.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi
