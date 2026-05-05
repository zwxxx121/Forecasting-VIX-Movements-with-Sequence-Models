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

from .metrics import binary_metrics

def evaluate_baselines(y_train, y_val, y_test) -> pd.DataFrame:
    """Simple binary baselines for context."""
    rows = []
    y_train = np.asarray(y_train).astype(int)
    for name, pred_val, pred_test in [
        ("always_zero", np.zeros_like(y_val), np.zeros_like(y_test)),
        ("always_one", np.ones_like(y_val), np.ones_like(y_test)),
        ("train_base_rate", np.full_like(y_val, y_train.mean(), dtype=float), np.full_like(y_test, y_train.mean(), dtype=float)),
    ]:
        rows.append({"model": name, "split": "val", **binary_metrics(y_val, pred_val, threshold=0.5)})
        rows.append({"model": name, "split": "test", **binary_metrics(y_test, pred_test, threshold=0.5)})
    return pd.DataFrame(rows)


def extract_last_timestep(loader) -> Tuple[np.ndarray, np.ndarray]:
    """Use only the most recent timestep from each sequence batch."""
    X_list, y_list = [], []
    for X_batch, y_batch in loader:
        X_list.append(X_batch[:, -1, :].numpy())
        y_list.append(y_batch.numpy().ravel())
    return np.concatenate(X_list), np.concatenate(y_list)


def extract_flat_sequence(loader) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten full sequence windows into 2D tabular rows."""
    X_list, y_list = [], []
    for X_batch, y_batch in loader:
        batch = X_batch.numpy()
        X_list.append(batch.reshape(batch.shape[0], -1))
        y_list.append(y_batch.numpy().ravel())
    return np.concatenate(X_list), np.concatenate(y_list)


def train_xgboost_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    params: Optional[Dict] = None,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 30,
    verbose_eval: int = 20,
):
    """Train an XGBoost binary classifier for spike prediction."""
    import xgboost as xgb

    if params is None:
        pos = max(float(np.sum(y_train == 1)), 1.0)
        neg = max(float(np.sum(y_train == 0)), 1.0)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "learning_rate": 0.03,
            "max_depth": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 10,
            "reg_alpha": 0.01,
            "reg_lambda": 1.0,
            "scale_pos_weight": neg / pos,
            "seed": 42,
        }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
    )
    return model


def evaluate_xgboost_classifier(model, X, y, threshold: float = 0.5) -> Dict[str, object]:
    """Evaluate XGBoost binary classifier with the shared metric function."""
    import xgboost as xgb

    prob = model.predict(xgb.DMatrix(X))
    return binary_metrics(y, prob, threshold=threshold)
