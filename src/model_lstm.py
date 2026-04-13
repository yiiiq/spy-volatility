"""
PyTorch LSTM regressor: model definition, dataset, training, and evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.utils import compute_metrics, SEED

# Prevent thread-contention deadlocks on macOS (Accelerate / OpenMP interaction).
# Must be set before any DataLoader or LSTM forward pass.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Training hyper-parameters
SEQ_LEN = 20
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10          # early stopping patience
LR = 1e-3
VAL_FRACTION = 0.15    # fraction of train data used as validation


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """Rolling-window sequence dataset for the LSTM."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN):
        """
        Args:
            X: Feature matrix, shape (T, num_features).
            y: Target vector, shape (T,).
            seq_len: Number of past time steps per sample.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.y) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_seq = self.X[idx : idx + self.seq_len]      # (seq_len, features)
        y_val = self.y[idx + self.seq_len]             # scalar
        return x_seq, y_val


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LSTMRegressor(nn.Module):
    """Single-output LSTM regressor trained from scratch."""

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]   # take the last time step
        return self.head(last).squeeze(1)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _scale(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardise features using training mean/std (no data leakage)."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def _run_epoch(
    model: LSTMRegressor,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    """Run one epoch; if optimizer is None, run in eval mode."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    with torch.set_grad_enabled(training):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_lstm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[np.ndarray, dict, LSTMRegressor, list[float], list[float]]:
    """
    Train a PyTorch LSTM regressor with early stopping and evaluate on test.

    Features are standardised using training statistics only.

    Args:
        X_train: Training features DataFrame.
        y_train: Training target Series.
        X_test:  Test features DataFrame.
        y_test:  Test target Series.

    Returns:
        Tuple of (predictions array, metrics dict, trained model,
                  train_losses list, val_losses list).
    """
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"LSTM training on device: {device}")

    # Standardise
    Xtr_np, Xte_np = _scale(X_train.values, X_test.values)
    ytr_np = y_train.values.astype(np.float32)
    yte_np = y_test.values.astype(np.float32)

    # Train / val split (chronological)
    n_val = max(SEQ_LEN + 1, int(len(Xtr_np) * VAL_FRACTION))
    Xtr, Xval = Xtr_np[:-n_val], Xtr_np[-n_val:]
    ytr, yval = ytr_np[:-n_val], ytr_np[-n_val:]

    train_ds = SequenceDataset(Xtr, ytr)
    val_ds = SequenceDataset(Xval, yval)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=False)

    model = LSTMRegressor(input_size=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    print(f"Training LSTM (max {MAX_EPOCHS} epochs, patience {PATIENCE}) …", flush=True)
    try:
        for epoch in range(1, MAX_EPOCHS + 1):
            tr_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
            vl_loss = _run_epoch(model, val_loader, criterion, None, device)
            train_losses.append(tr_loss)
            val_losses.append(vl_loss)

            marker = "*" if vl_loss < best_val_loss else " "
            print(
                f"  Epoch {epoch:3d}{marker} | train {tr_loss:.6f} | val {vl_loss:.6f}",
                flush=True,
            )

            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}", flush=True)
                break
    except KeyboardInterrupt:
        print("\n  Training interrupted by user.", flush=True)

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict on test set
    test_ds = SequenceDataset(Xte_np, yte_np)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            preds.append(model(X_batch.to(device)).cpu().numpy())
    y_pred = np.concatenate(preds)

    # Align y_test to match (first SEQ_LEN rows are consumed as context)
    y_test_aligned = yte_np[SEQ_LEN:]

    metrics = compute_metrics(y_test_aligned, y_pred)
    print("\n--- LSTM test metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    return y_pred, metrics, model, train_losses, val_losses, y_test_aligned
