"""
XGBoost regressor: training, hyperparameter tuning, and evaluation.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

from src.utils import compute_metrics, SEED


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[np.ndarray, dict, XGBRegressor]:
    """
    Tune and train an XGBoost regressor, then evaluate on the test set.

    Hyperparameter search uses RandomizedSearchCV with time-series-aware
    cross-validation (no shuffle).

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test:  Test features.
        y_test:  Test target.

    Returns:
        Tuple of (predictions array, metrics dict, fitted model).
    """
    param_dist = {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [1, 1.5, 2],
    }

    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )

    # Time-series CV: use TimeSeriesSplit so later folds are always "future"
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=30,
        scoring="neg_mean_squared_error",
        cv=tscv,
        random_state=SEED,
        n_jobs=-1,
        verbose=0,
    )

    print("Tuning XGBoost (RandomizedSearchCV, 30 iterations) …")
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")

    model = search.best_estimator_
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test.values, y_pred)
    print("\n--- XGBoost test metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    return y_pred, metrics, model
