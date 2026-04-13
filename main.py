"""
main.py – end-to-end pipeline for the SPY volatility prediction project.

Run with:
    python main.py
"""

import os
import numpy as np
import pandas as pd

from src.utils import set_seed, split_train_test, compute_metrics, select_features
from src.data import load_data
from src.features import add_features
from src.target import make_target, TARGET_COL
from src.model_xgb import train_xgb
from src.model_lstm import train_lstm
from src.plots import (
    plot_target_hist,
    plot_feature_importance,
    plot_predictions,
    plot_lstm_loss,
    plot_metric_comparison,
)

SPLIT_DATE = "2022-01-01"
METRICS_PATH = "results/metrics.csv"
MAX_FEATURES = 10


def main() -> None:
    # ------------------------------------------------------------------
    # 0. Reproducibility
    # ------------------------------------------------------------------
    set_seed()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    raw = load_data()

    # ------------------------------------------------------------------
    # 2. Create features
    # ------------------------------------------------------------------
    features = add_features(raw)

    # ------------------------------------------------------------------
    # 3. Create target
    # ------------------------------------------------------------------
    target = make_target(raw)

    # Align features and target on the same index
    combined = features.join(target, how="inner").dropna()
    X = combined.drop(columns=[TARGET_COL])
    y = combined[TARGET_COL]

    print(f"\nAligned dataset: {len(combined)} rows, {X.shape[1]} features")

    # ------------------------------------------------------------------
    # 4. Plot target distribution (uses full aligned target)
    # ------------------------------------------------------------------
    plot_target_hist(y)

    # ------------------------------------------------------------------
    # 5. Train/test split
    # ------------------------------------------------------------------
    print("\n--- Train/test split ---")
    train_df, test_df = split_train_test(combined, SPLIT_DATE)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    X_train, X_test, selected_features, _ = select_features(
        X_train,
        y_train,
        X_test,
        max_features=MAX_FEATURES,
    )

    # Naive baseline: predict today's volatility = yesterday's realized volatility
    # target_vol is already on the combined index, so shift(1) gives last-known value.
    baseline_pred = combined[TARGET_COL].shift(1).reindex(y_test.index).values
    # First test row may be NaN if the prior day is not in the test index;
    # fill with the last training target value instead.
    if np.isnan(baseline_pred[0]):
        baseline_pred[0] = y_train.iloc[-1]
    baseline_metrics = compute_metrics(y_test.values, baseline_pred)

    # ------------------------------------------------------------------
    # 6. Train XGBoost
    # ------------------------------------------------------------------
    print("\n=== XGBoost ===")
    xgb_pred, xgb_metrics, xgb_model = train_xgb(X_train, y_train, X_test, y_test)

    # ------------------------------------------------------------------
    # 7. Train LSTM
    # ------------------------------------------------------------------
    print("\n=== LSTM ===")
    lstm_pred, lstm_metrics, lstm_model, train_losses, val_losses, y_test_lstm = train_lstm(
        X_train, y_train, X_test, y_test
    )

    # ------------------------------------------------------------------
    # 8. Save metrics
    # ------------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    metrics_df = pd.DataFrame(
        {"Baseline": baseline_metrics, "XGBoost": xgb_metrics, "LSTM": lstm_metrics}
    ).T
    metrics_df.index.name = "Model"
    metrics_df.to_csv(METRICS_PATH)
    print(f"\nMetrics saved to {METRICS_PATH}")
    print(metrics_df.to_string())

    # ------------------------------------------------------------------
    # 9. Save plots
    # ------------------------------------------------------------------
    print("\n--- Saving plots ---")

    plot_feature_importance(xgb_model, selected_features)

    plot_predictions(
        index=X_test.index,
        y_true=y_test.values,
        y_pred=xgb_pred,
        model_name="XGBoost",
        filename="predictions_xgb.png",
    )

    # LSTM predictions are offset by SEQ_LEN rows
    from src.model_lstm import SEQ_LEN
    lstm_index = X_test.index[SEQ_LEN:]
    plot_predictions(
        index=lstm_index,
        y_true=y_test_lstm,
        y_pred=lstm_pred,
        model_name="LSTM",
        filename="predictions_lstm.png",
    )

    plot_lstm_loss(train_losses, val_losses)
    plot_predictions(
        index=X_test.index,
        y_true=y_test.values,
        y_pred=baseline_pred,
        model_name="Baseline (t-1)",
        filename="predictions_baseline.png",
    )

    plot_metric_comparison("MAE", baseline_metrics, xgb_metrics, lstm_metrics)
    plot_metric_comparison("RMSE", baseline_metrics, xgb_metrics, lstm_metrics)
    plot_metric_comparison("R2", baseline_metrics, xgb_metrics, lstm_metrics)

    print("\nDone. All results saved to results/")


if __name__ == "__main__":
    main()
