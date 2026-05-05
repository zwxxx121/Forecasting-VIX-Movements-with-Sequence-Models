from __future__ import annotations

import copy
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

def make_time_splits(index: pd.Index, train_frac: float = 0.80, val_frac: float = 0.10):
    """Return train, validation, and test date indexes using chronological split."""
    dates = pd.Index(index)
    n = len(dates)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return dates[:train_end], dates[train_end:val_end], dates[val_end:]


def prepare_tabular_features(
    master: pd.DataFrame,
    target_col: str = "target_vix_spike_10d",
    exclude_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Keep numeric features and remove target/leakage columns."""
    if exclude_cols is None:
        exclude_cols = [
            target_col,
            "target_vix_spike",
            "vix_future_max_10d_change",
            "vix_future_max_5d_change",
            "vix_future_max_15d_change",
            "vix_future_max_20d_change",
        ]
    model_df = master.replace([np.inf, -np.inf], np.nan).dropna().copy()
    numeric_cols = model_df.select_dtypes(include=[np.number, "bool"]).columns
    feature_cols = [col for col in numeric_cols if col not in set(exclude_cols)]
    X_df = model_df[feature_cols].astype(float)
    y = model_df[target_col].astype(int)
    return X_df, y


def scale_features_train_only(X_df: pd.DataFrame, train_dates: Sequence[pd.Timestamp]):
    """Fit StandardScaler on train dates and transform the full feature frame."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(X_df.loc[train_dates])
    X_scaled = pd.DataFrame(scaler.transform(X_df), index=X_df.index, columns=X_df.columns)
    return X_scaled, scaler

def make_lstm_sequences(
    X_df: pd.DataFrame,
    y_series: pd.Series,
    seq_len: int = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert aligned X/y into rolling windows for sequence models."""
    X_values = X_df.values
    y_values = y_series.values
    dates = X_df.index
    X_seq, y_seq, target_dates = [], [], []
    for i in range(seq_len, len(X_df)):
        X_seq.append(X_values[i - seq_len : i])
        y_seq.append(y_values[i])
        target_dates.append(dates[i])
    return (
        np.array(X_seq, dtype=np.float32),
        np.array(y_seq, dtype=np.float32),
        np.array(target_dates),
    )


def split_sequences_by_dates(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    seq_dates: np.ndarray,
    train_dates: Sequence[pd.Timestamp],
    val_dates: Sequence[pd.Timestamp],
):
    """Split sequence arrays according to target dates."""
    train_last = pd.Timestamp(train_dates[-1])
    val_last = pd.Timestamp(val_dates[-1])
    seq_dates = pd.to_datetime(seq_dates)
    train_mask = seq_dates <= train_last
    val_mask = (seq_dates > train_last) & (seq_dates <= val_last)
    test_mask = seq_dates > val_last
    return (
        X_seq[train_mask], y_seq[train_mask],
        X_seq[val_mask], y_seq[val_mask],
        X_seq[test_mask], y_seq[test_mask],
        np.array(seq_dates[test_mask]),
    )


def make_vix_sequence_dataset_class():
    """Return a proper torch Dataset subclass for VIX sequences."""
    import torch
    from torch.utils.data import Dataset

    class VIXSequenceDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    return VIXSequenceDataset


def make_dataloaders(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    batch_size: int = 64,
    shuffle_train: bool = True,
):
    """Build PyTorch DataLoaders for train/val/test sequence arrays."""
    from torch.utils.data import DataLoader

    DatasetCls = make_vix_sequence_dataset_class()
    train_loader = DataLoader(DatasetCls(X_train, y_train), batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(DatasetCls(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(DatasetCls(X_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
