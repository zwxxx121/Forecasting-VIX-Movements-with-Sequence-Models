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

from .common import get_torch_device

def build_fomc_attention_targets(
    fomc_df: pd.DataFrame,
    vix_series: pd.Series,
    horizon: int = 10,
    emb_dir: str | Path = "fomc_token_embeddings",
    target_col: str = "fomc_target_10d",
) -> pd.DataFrame:
    """Create event-level FOMC target rows using future VIX log return."""
    emb_dir = Path(emb_dir)
    rows = []
    vix_series = vix_series.dropna().sort_index()
    vix_dates = vix_series.index

    for fomc_date in fomc_df.index:
        fomc_date = pd.Timestamp(fomc_date)
        pos = vix_dates.searchsorted(fomc_date)
        if pos >= len(vix_dates) or pos + horizon >= len(vix_dates):
            continue

        date_str = fomc_date.strftime("%Y%m%d")
        if not (emb_dir / f"{date_str}_tokens.npy").exists() or not (emb_dir / f"{date_str}_mask.npy").exists():
            continue

        start_date = vix_dates[pos]
        future_date = vix_dates[pos + horizon]
        vix_today = float(vix_series.loc[start_date])
        vix_future = float(vix_series.loc[future_date])
        if vix_today <= 0 or vix_future <= 0:
            continue

        rows.append(
            {
                "fomc_date": fomc_date,
                "vix_start_date": start_date,
                "vix_future_date": future_date,
                target_col: np.log(vix_future / vix_today),
            }
        )
    return pd.DataFrame(rows).set_index("fomc_date").sort_index()


class FOMCTokenDataset:  # inherits Dataset dynamically to avoid torch import at module import
    """Dataset for saved FOMC token embeddings."""

    def __init__(
        self,
        dates: Sequence[pd.Timestamp],
        target_df: pd.DataFrame,
        emb_dir: str | Path = "fomc_token_embeddings",
        target_col: str = "fomc_target_10d_scaled",
    ):
        from torch.utils.data import Dataset

        if not isinstance(self, Dataset):
            pass
        self.dates = list(dates)
        self.target_df = target_df
        self.emb_dir = Path(emb_dir)
        self.target_col = target_col

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, idx: int):
        import torch

        date = pd.Timestamp(self.dates[idx])
        date_str = date.strftime("%Y%m%d")
        tokens = np.load(self.emb_dir / f"{date_str}_tokens.npy").astype(np.float32)
        mask = np.load(self.emb_dir / f"{date_str}_mask.npy").astype(np.float32)
        y = np.float32(self.target_df.loc[date, self.target_col])
        return torch.tensor(tokens), torch.tensor(mask), torch.tensor([y])


def make_fomc_token_dataset_class():
    """Return a proper torch Dataset subclass for FOMCTokenDataset."""
    from torch.utils.data import Dataset

    class _FOMCTokenDataset(FOMCTokenDataset, Dataset):
        pass

    return _FOMCTokenDataset


def make_fomc_attention_model_class():
    """Return FOMCTokenAttentionRegressor class after torch is available."""
    import torch
    import torch.nn as nn

    class FOMCTokenAttentionRegressor(nn.Module):
        def __init__(self, token_dim=768, hidden_dim=128, proj_dim=48, dropout=0.20):
            super().__init__()
            self.scorer = nn.Sequential(
                nn.Linear(token_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1, bias=False),
            )
            self.proj = nn.Sequential(
                nn.Linear(token_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.head = nn.Sequential(
                nn.Linear(proj_dim, 24),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(24, 1),
            )

        def pool(self, token_embeddings, attention_mask):
            scores = self.scorer(token_embeddings)
            mask = attention_mask.unsqueeze(-1)
            scores = scores.masked_fill(mask == 0, -1e9)
            weights = torch.softmax(scores, dim=1)
            context = (weights * token_embeddings).sum(dim=1)
            emb = self.proj(context)
            return emb, weights.squeeze(-1)

        def forward(self, token_embeddings, attention_mask):
            emb, weights = self.pool(token_embeddings, attention_mask)
            pred = self.head(emb)
            return pred, emb, weights

    return FOMCTokenAttentionRegressor


def train_fomc_attention_model(
    model,
    train_loader,
    val_loader,
    epochs: int = 300,
    patience: int = 30,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device=None,
):
    """Train token-attention regressor with Huber loss and early stopping."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = device or get_torch_device()
    model = model.to(device)
    criterion = nn.HuberLoss(delta=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, n_train = 0.0, 0
        for tokens, mask, y in train_loader:
            tokens, mask, y = tokens.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            pred, _, _ = model(tokens, mask)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * tokens.size(0)
            n_train += tokens.size(0)
        train_loss /= max(n_train, 1)

        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for tokens, mask, y in val_loader:
                tokens, mask, y = tokens.to(device), mask.to(device), y.to(device)
                pred, _, _ = model(tokens, mask)
                loss = criterion(pred, y)
                val_loss += loss.item() * tokens.size(0)
                n_val += tokens.size(0)
        val_loss /= max(n_val, 1)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def extract_attention_embeddings(
    model,
    dates: Sequence[pd.Timestamp] | pd.DataFrame | pd.Series | pd.Index,
    emb_dir: str | Path = "fomc_token_embeddings",
    proj_dim: int = 48,
    device=None,
) -> pd.DataFrame:
    """Run trained token-attention model and return stmt_attn_* features.

    Parameters
    ----------
    model:
        Trained token-attention model.
    dates:
        Either an iterable of FOMC dates or a DataFrame/Series whose index is the
        FOMC date index. Passing the full ``fomc_df`` is supported.
    emb_dir:
        Directory containing ``YYYYMMDD_tokens.npy`` and ``YYYYMMDD_mask.npy``.
    proj_dim:
        Number of projected embedding dimensions to return.
    device:
        Torch device. Defaults to MPS/CUDA/CPU auto-selection.
    """
    import torch

    if isinstance(dates, (pd.DataFrame, pd.Series)):
        date_iter = dates.index
    elif isinstance(dates, pd.Index):
        date_iter = dates
    else:
        date_iter = dates

    device = device or get_torch_device()
    model = model.to(device)
    model.eval()
    rows = []
    emb_dir = Path(emb_dir)

    with torch.no_grad():
        for date in date_iter:
            date = pd.Timestamp(date)
            date_str = date.strftime("%Y%m%d")
            token_path = emb_dir / f"{date_str}_tokens.npy"
            mask_path = emb_dir / f"{date_str}_mask.npy"
            if not token_path.exists() or not mask_path.exists():
                # Skip dates whose token embeddings were not generated.
                continue
            tokens = np.load(token_path).astype(np.float32)
            mask = np.load(mask_path).astype(np.float32)
            tokens_t = torch.tensor(tokens).unsqueeze(0).to(device)
            mask_t = torch.tensor(mask).unsqueeze(0).to(device)
            _, emb, _ = model(tokens_t, mask_t)
            values = emb.cpu().numpy().ravel()
            row = {"date": date}
            row.update({f"stmt_attn_{i + 1}": values[i] for i in range(min(proj_dim, len(values)))})
            rows.append(row)

    if not rows:
        cols = [f"stmt_attn_{i + 1}" for i in range(proj_dim)]
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="date"))
    return pd.DataFrame(rows).set_index("date").sort_index()
