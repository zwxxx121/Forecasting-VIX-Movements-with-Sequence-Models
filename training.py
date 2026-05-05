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
from .metrics import binary_metrics

def predict_proba_loader(model, loader, device=None) -> Tuple[np.ndarray, np.ndarray]:
    """Return sigmoid probabilities and true labels for a PyTorch DataLoader."""
    import torch

    device = device or get_torch_device()
    model.eval()
    probs, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            batch_probs = torch.sigmoid(logits).cpu().numpy().ravel()
            probs.append(batch_probs)
            actuals.append(y_batch.cpu().numpy().ravel())
    probs = np.concatenate(probs)
    actuals = np.concatenate(actuals)
    unique_vals = np.unique(actuals)
    if not set(unique_vals).issubset({0.0, 1.0}):
        raise ValueError(f"Labels are not binary. Found: {unique_vals[:20]}")
    return probs, actuals.astype(int)


def run_epoch(model, loader, criterion, optimizer=None, train: bool = True, device=None) -> float:
    """Run one train/eval epoch for binary classification."""
    import torch

    device = device or get_torch_device()
    model.train(mode=train)

    total_loss, n = 0.0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).float()

        if y_batch.ndim == 1:
            y_batch = y_batch.view(-1, 1)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(X_batch)

        if logits.ndim == 1:
            logits = logits.view(-1, 1)

        loss = criterion(logits, y_batch)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        n += X_batch.size(0)

    return total_loss / max(n, 1)


def train_one_config(
    model_class,
    model_kwargs: Dict,
    train_loader,
    val_loader,
    pos_weight_value: float = 1.5,
    lr: float = 5e-5,
    weight_decay: float = 1e-4,
    epochs: int = 250,
    patience: int = 25,
    device=None,
    max_bce_ratio: float = 1.20,
    max_bce_abs_delta: float = 0.05,
):
    """
    Train binary classifier with early stopping on constrained PR-AUC.

    The model checkpoint is selected by validation PR-AUC only if validation BCE
    remains close to the best validation BCE seen so far.

    This prevents selecting overconfident models with exploding BCE.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = device or get_torch_device()

    model = model_class(**model_kwargs).to(device)

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
    )

    best_state = None
    best_score = -np.inf
    best_val_loss_seen = np.inf
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            train=True,
            device=device,
        )

        val_loss = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            train=False,
            device=device,
        )

        scheduler.step(val_loss)

        val_prob, val_true = predict_proba_loader(model, val_loader, device=device)
        val_metrics = binary_metrics(val_true, val_prob, threshold=0.5)

        val_pr_auc = val_metrics["pr_auc"]
        val_roc_auc = val_metrics["roc_auc"]

        best_val_loss_seen = min(best_val_loss_seen, val_loss)

        allowed_val_loss = max(
            best_val_loss_seen * max_bce_ratio,
            best_val_loss_seen + max_bce_abs_delta,
        )

        bce_ok = val_loss <= allowed_val_loss

        # Main score: PR-AUC, but only eligible if BCE is controlled.
        # If BCE is not controlled, do not allow checkpoint replacement.
        score = val_pr_auc if bce_ok and not np.isnan(val_pr_auc) else -np.inf

        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss_seen": best_val_loss_seen,
                "allowed_val_loss": allowed_val_loss,
                "bce_ok": bce_ok,
                "lr": current_lr,
                "val_precision_05": val_metrics["precision"],
                "val_recall_05": val_metrics["recall"],
                "val_f1_05": val_metrics["f1"],
                "val_roc_auc": val_roc_auc,
                "val_pr_auc": val_pr_auc,
                "val_pred_positive_rate_05": val_metrics["positive_rate_pred"],
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train BCE: {train_loss:.5f} | "
            f"Val BCE: {val_loss:.5f} | "
            f"Best Val BCE: {best_val_loss_seen:.5f} | "
            f"BCE OK: {bce_ok} | "
            f"Val PR-AUC: {val_pr_auc:.4f} | "
            f"Val ROC-AUC: {val_roc_auc:.4f} | "
            f"F1@0.5: {val_metrics['f1']:.4f} | "
            f"Pred+@0.5: {val_metrics['positive_rate_pred']:.3f} | "
            f"LR: {current_lr:.2e}"
        )

        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping on constrained Val PR-AUC.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(history)