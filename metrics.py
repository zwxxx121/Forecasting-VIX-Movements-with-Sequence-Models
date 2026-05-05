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

def safe_auc_metrics(y_true, prob) -> Tuple[float, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    return roc_auc_score(y_true, prob), average_precision_score(y_true, prob)


def binary_metrics(y_true, prob, threshold: float = 0.5) -> Dict[str, object]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob)
    pred = (prob >= threshold).astype(int)
    roc_auc, pr_auc = safe_auc_metrics(y_true, prob)
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_pred": float(np.mean(pred)),
        "confusion_matrix": confusion_matrix(y_true, pred),
    }


def print_metrics(title: str, metrics: Dict[str, object]) -> None:
    """Pretty-print metrics dictionary."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for key in [
        "threshold", "accuracy", "precision", "recall", "f1", "roc_auc",
        "pr_auc", "positive_rate_true", "positive_rate_pred",
    ]:
        value = metrics[key]
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])


def threshold_by_alert_rate(prob, alert_rate: float = 0.15) -> float:
    """Threshold that flags the top alert_rate fraction of predictions."""
    return float(np.quantile(prob, 1.0 - alert_rate))


def find_best_threshold(
    y_true,
    prob,
    metric: str = "f1",
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, object]]:
    """Search thresholds and maximize a chosen metric from binary_metrics."""
    thresholds = thresholds if thresholds is not None else np.linspace(0.05, 0.95, 181)
    metrics = [binary_metrics(y_true, prob, threshold=float(t)) for t in thresholds]
    metrics = [m for m in metrics if not pd.isna(m.get(metric, np.nan))]
    if not metrics:
        raise ValueError(f"No valid metrics found for metric={metric}")
    best = sorted(metrics, key=lambda m: m[metric], reverse=True)[0]
    return best["threshold"], best


def find_best_threshold_with_precision_floor(
    y_true,
    prob,
    min_precision: float = 0.25,
    min_recall: float = 0.10,
) -> Tuple[float, Dict[str, object]]:
    """Best-F1 threshold among candidates satisfying precision/recall floors."""
    thresholds = np.linspace(0.05, 0.95, 181)
    candidates = []
    for t in thresholds:
        m = binary_metrics(y_true, prob, threshold=float(t))
        if m["precision"] >= min_precision and m["recall"] >= min_recall:
            candidates.append(m)
    if not candidates:
        all_metrics = [binary_metrics(y_true, prob, threshold=float(t)) for t in thresholds]
        best = sorted(all_metrics, key=lambda m: (m["precision"], m["f1"]), reverse=True)[0]
        return best["threshold"], best
    best = sorted(candidates, key=lambda m: m["f1"], reverse=True)[0]
    return best["threshold"], best


def find_best_threshold_by_f1_limited_alert_rate(
    y_true,
    prob,
    max_alert_rate: float = 0.30,
) -> Tuple[float, Dict[str, object]]:
    """Best-F1 threshold while limiting predicted positive rate."""
    thresholds = np.linspace(0.05, 0.95, 181)
    candidates = []
    for t in thresholds:
        m = binary_metrics(y_true, prob, threshold=float(t))
        if m["positive_rate_pred"] <= max_alert_rate:
            candidates.append(m)
    if not candidates:
        raise ValueError("No threshold satisfied max_alert_rate.")
    best = sorted(candidates, key=lambda m: m["f1"], reverse=True)[0]
    return best["threshold"], best
