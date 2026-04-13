# SPY Volatility Prediction

Predict next-day realized volatility of the S&P 500 ETF (SPY) using XGBoost and a PyTorch LSTM, benchmarked against a naive baseline.

## Project structure

```
main.py               # end-to-end pipeline entry point
src/
  data.py             # download / load SPY OHLCV data via yfinance
  features.py         # feature engineering (returns, MA, RSI, volume, …)
  target.py           # target = abs(ret_1d.shift(-1))
  model_xgb.py        # XGBoost regressor with RandomizedSearchCV tuning
  model_lstm.py       # PyTorch LSTM regressor trained from scratch
  plots.py            # all result plots
  utils.py            # seed, train/test split, evaluation metrics
data/
  spy.csv             # cached SPY data (auto-generated on first run)
results/
  metrics.csv         # MAE, RMSE, R², MAPE for all models
  plots/              # saved figures
```

## Setup

```bash
pip install yfinance xgboost scikit-learn torch matplotlib pandas numpy
```

## Run

```bash
python main.py
```

On first run SPY data is downloaded from 2010-01-01 and saved to `data/spy.csv`. Subsequent runs use the cached file.

## Models

| Model | Description |
|-------|-------------|
| Baseline | Yesterday's realized volatility (naïve t-1 predictor) |
| XGBoost | Gradient-boosted trees tuned with `RandomizedSearchCV` (30 iterations, `TimeSeriesSplit`) |
| LSTM | 2-layer PyTorch LSTM with early stopping (patience 10) |

## Features (23 total)

- Daily / multi-day returns and momentum (1d, 3d, 5d, 10d)
- Moving averages and price-to-MA ratios (5, 20, 50-day)
- Rolling volatility and vol ratio (5, 10, 20-day)
- High-low range, close-open return, overnight gap
- Volume change and volume spike vs 20-day average
- RSI (14-day)

## Evaluation

Time-based train/test split at **2022-01-01**. No data leakage — all features and the baseline predictor use only past information.

Metrics reported: MAE, RMSE, R², MAPE.
