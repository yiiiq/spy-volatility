"""
Utility functions: random seed, train/test split, evaluation metrics.
"""

import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

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


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    max_features: int = 10,
    corr_threshold: float = 0.95,
    keep_prefixes: tuple[str, ...] = ("rv_lag",),
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.Series]:
    """
    Select a compact feature set using train-only XGBoost importance.

    The selector always keeps features whose names start with keep_prefixes,
    averages feature importances across time-series CV folds, and removes
    near-duplicate columns based on pairwise correlation.

    Args:
        X_train: Training feature matrix.
        y_train: Training target.
        X_test: Test feature matrix.
        max_features: Maximum number of features to keep.
        corr_threshold: Drop candidates with abs correlation above this value
            against any already-selected feature.
        keep_prefixes: Feature-name prefixes that should always be kept.

    Returns:
        Filtered X_train, filtered X_test, selected feature names, and the
        averaged feature importance scores.
    """
    from xgboost import XGBRegressor

    if X_train.empty:
        raise ValueError("X_train is empty; cannot run feature selection.")

    mandatory = [
        col for col in X_train.columns if any(col.startswith(prefix) for prefix in keep_prefixes)
    ]

    tscv = TimeSeriesSplit(n_splits=5)
    mean_importance = pd.Series(0.0, index=X_train.columns, dtype=float)

    for fold_idx, (fit_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=SEED + fold_idx,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        mean_importance += pd.Series(model.feature_importances_, index=X_train.columns)

    mean_importance /= tscv.get_n_splits()
    ranked = mean_importance.sort_values(ascending=False)
    corr = X_train.corr().abs()

    selected: list[str] = []
    for col in mandatory:
        if col not in selected:
            selected.append(col)

    for col in ranked.index:
        if col in selected:
            continue
        if len(selected) >= max_features:
            break
        if any(corr.loc[col, kept] > corr_threshold for kept in selected):
            continue
        selected.append(col)

    if not selected:
        selected = ranked.head(min(max_features, len(ranked))).index.tolist()

    print("\n--- Feature selection ---")
    print(f"Mandatory kept features: {mandatory}")
    print(f"Selected {len(selected)} features: {selected}")
    print("Top averaged importances:")
    for feature, score in ranked.head(max_features + len(mandatory)).items():
        print(f"  {feature:<25} {score:.4f}")

    return X_train[selected], X_test[selected], selected, ranked
