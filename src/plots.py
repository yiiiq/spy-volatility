"""
Plotting: generate and save all required result plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe for scripts
import matplotlib.pyplot as plt

PLOTS_DIR = "results/plots"


def _savefig(filename: str) -> None:
    """Save the current figure and close it."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")


def plot_target_hist(target: pd.Series) -> None:
    """Histogram of target volatility values."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(target, bins=80, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.set_title("Distribution of Next-Day Realized Volatility (SPY)")
    ax.set_xlabel("Volatility (|daily return|)")
    ax.set_ylabel("Count")
    _savefig("target_vol_hist.png")


def plot_feature_importance(model, feature_names: list[str]) -> None:
    """Horizontal bar chart of XGBoost feature importances."""
    importances = model.feature_importances_
    idx = np.argsort(importances)
    top_n = min(20, len(idx))
    idx = idx[-top_n:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color="steelblue",
    )
    ax.set_title("XGBoost Feature Importances (top 20)")
    ax.set_xlabel("Importance")
    _savefig("feature_importance_xgb.png")


def plot_predictions(
    index: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    filename: str,
) -> None:
    """Line plot of actual vs predicted volatility on the test set."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(index, y_true, label="Actual", color="steelblue", linewidth=0.9)
    ax.plot(index, y_pred, label="Predicted", color="tomato", linewidth=0.9, alpha=0.85)
    ax.set_title(f"{model_name}: Actual vs Predicted Next-Day Volatility (Test Set)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.legend()
    _savefig(filename)


def plot_lstm_loss(train_losses: list[float], val_losses: list[float]) -> None:
    """Training and validation loss curves for the LSTM."""
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, label="Train Loss", color="steelblue")
    ax.plot(epochs, val_losses, label="Val Loss", color="tomato")
    ax.set_title("LSTM Training & Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    _savefig("lstm_loss_curve.png")


def plot_metrics_comparison(
    metrics_baseline: dict, metrics_xgb: dict, metrics_lstm: dict
) -> None:
    """Bar chart comparing MAE, RMSE, and R2 across baseline and both models."""
    metrics_to_plot = ["MAE", "RMSE", "R2"]
    x = np.arange(len(metrics_to_plot))
    width = 0.25

    bl_vals  = [metrics_baseline[m] for m in metrics_to_plot]
    xgb_vals = [metrics_xgb[m]      for m in metrics_to_plot]
    lstm_vals = [metrics_lstm[m]    for m in metrics_to_plot]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars0 = ax.bar(x - width,       bl_vals,   width, label="Baseline (t-1)", color="#aaaaaa")
    bars1 = ax.bar(x,               xgb_vals,  width, label="XGBoost",        color="steelblue")
    bars2 = ax.bar(x + width,       lstm_vals, width, label="LSTM",            color="tomato")

    ax.set_title("Model Comparison: MAE, RMSE, R2")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot)
    ax.legend()

    # Annotate bars
    for bar in list(bars0) + list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(
            f"{h:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    _savefig("metrics_comparison.png")
