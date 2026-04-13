# SPY Volatility Prediction

This project predicts next-day SPY volatility from daily OHLCV market data using three approaches:

- a naive persistence baseline
- an XGBoost regressor with time-series-aware hyperparameter tuning
- a PyTorch LSTM trained on rolling feature sequences


## Objective

The goal is to forecast the next trading day's realized volatility for SPY, the S&P 500 ETF. This is framed as a supervised regression problem using only information available up to time $t$ to predict volatility at time $t+1$.

In code, the target is defined as:

$$
	ext{target}_t = \sigma_{t+1}^{(5)}
$$

where $\sigma_{t+1}^{(5)}$ is the 5-day rolling standard deviation of log returns, shifted forward by one day.

## Repository Layout

```text
main.py               End-to-end pipeline entry point
README.md             Project documentation
data/
  spy.csv             Cached SPY OHLCV data
results/
  metrics.csv         Evaluation metrics by model
  plots/              Saved charts and diagnostic figures
src/
  data.py             Download/load SPY OHLCV data from yfinance
  features.py         Feature engineering
  model_lstm.py       PyTorch LSTM training and inference
  model_xgb.py        XGBoost training and hyperparameter search
  plots.py            Plot generation
  target.py           Target construction
  utils.py            Seeding, split logic, metrics, feature selection
```

## Data

- Instrument: SPY
- Frequency: Daily
- Source: yfinance
- Start date: 2010-01-01
- Raw fields used: Open, High, Low, Close, Volume


## Methodology

The main pipeline in `main.py` follows these steps:

1. Set a global random seed for reproducibility.
2. Load cached or freshly downloaded SPY data.
3. Create engineered features from OHLCV history.
4. Create the next-day volatility target.
5. Align features and target on a common date index.
6. Split the dataset chronologically at 2022-01-01.
7. Run train-only feature selection and keep at most 10 features.
8. Train and evaluate the baseline, XGBoost, and LSTM models.
9. Save metrics and plots under `results/`.

This setup is explicitly time-series-aware. The train/test split is chronological, and the XGBoost tuning step uses `TimeSeriesSplit` rather than shuffled cross-validation.


## Feature Set

The feature engineering code in `src/features.py` builds 23 features from price and volume history.

### Returns and momentum

- `ret_1d`, `ret_3d`, `ret_5d`, `ret_10d`
- `mom_5`, `mom_10`

### Trend features

- `ma_5`, `ma_20`, `ma_50`

### Volatility features

- `vol_5`, `vol_10`, `vol_20`
- `vol_ratio`
- `rv_lag1`, `rv_lag2`, `rv_lag5`

### Range and candle features

- `hl_range`
- `co_return`
- `oc_gap`

### Volume features

- `vol_chg`
- `vol_ma20`
- `volume_spike`

### Oscillator feature

- `rsi`

All features are derived from current and past values only. After feature construction, rows with insufficient rolling history are dropped.

## Feature Selection

Before model training, the script reduces the input space using `select_features` in `src/utils.py`.

The selection logic:

- computes XGBoost feature importance on training data only
- averages importance across 5 `TimeSeriesSplit` folds
- keeps lagged realized-volatility features by default (`rv_lag*`)
- removes highly correlated candidates above a correlation threshold
- retains at most 10 features for downstream models

This keeps the final training set compact and helps both the tree model and the LSTM focus on the most informative signals.

## Models

### Baseline

The baseline predicts that tomorrow's volatility will equal the most recently observed volatility. This is a strong persistence benchmark for volatility forecasting and is important because many complex models fail to beat it consistently.

### XGBoost

The XGBoost model:

- uses `XGBRegressor`
- tunes hyperparameters with `RandomizedSearchCV`
- evaluates 30 random parameter combinations
- uses 5-fold `TimeSeriesSplit`
- optimizes negative mean squared error

The search includes:

- number of trees
- tree depth
- learning rate
- subsampling
- column subsampling
- child weight
- L1 and L2 regularization

### LSTM

The LSTM model in `src/model_lstm.py` uses:

- sequence length: 20
- hidden size: 64
- number of layers: 2
- dropout: 0.2
- batch size: 64
- optimizer: Adam
- learning rate: 1e-3
- early stopping patience: 10
- max epochs: 100

Features are standardized using training-set statistics only. The validation split is chronological, and the test predictions are aligned to account for the 20-step input window.

## Evaluation Setup

- Train/test split date: `2022-01-01`
- Metrics: MAE, RMSE, R2, MAPE
- Leakage control: past-only features, chronological split, time-series CV

The baseline, XGBoost, and LSTM are all evaluated on the same held-out test period.

## Current Results

The current benchmark in `results/metrics.csv` is:

| Model | MAE | RMSE | R2 | MAPE |
|-------|-----|------|----|------|
| Baseline | 0.001677 | 0.002964 | 0.7642 | 20.83 |
| XGBoost | 0.001852 | 0.003136 | 0.7361 | 22.64 |
| LSTM | 0.002299 | 0.003316 | 0.7094 | 31.14 |

At the moment, the naive baseline outperforms both learned models on the saved test results. That is not unusual in volatility prediction and is a useful finding rather than a failure: it suggests the feature set and model classes still leave room for improvement relative to a strong persistence benchmark.

## Generated Outputs

Running the pipeline writes the following artifacts under `results/`:

### Metrics

- `metrics.csv`: summary table of evaluation metrics by model

### Plots

- `target_vol_hist.png`: distribution of the target variable
- `feature_importance_xgb.png`: top XGBoost feature importances
- `predictions_baseline.png`: baseline predictions vs actuals
- `predictions_xgb.png`: XGBoost predictions vs actuals
- `predictions_lstm.png`: LSTM predictions vs actuals
- `lstm_loss_curve.png`: LSTM training and validation loss
- `metric_mae.png`: MAE comparison across models
- `metric_rmse.png`: RMSE comparison across models
- `metric_r2.png`: R2 comparison across models

Depending on prior runs, there may also be older plot files in `results/plots/` from earlier versions of the project.

## Installation

This repository does not currently include a `requirements.txt`, so install dependencies manually.

```bash
pip install yfinance xgboost scikit-learn torch matplotlib pandas numpy
```

## How To Run

From the repository root:

```bash
python main.py
```
