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

def build_daily_fomc_features(
    daily_dates: pd.DatetimeIndex,
    fomc_df: pd.DataFrame,
    score_cols: Optional[Sequence[str]] = None,
    attn_proj_dim: Optional[int] = 48,
) -> pd.DataFrame:
    """Forward-fill event-level FOMC text features to business-day frequency."""
    fomc_df = fomc_df.copy().sort_index()
    if score_cols is None:
        base_cols = [
            "hawkish_score",
            "dovish_score",
            "uncertainty_score",
            "inflation_score",
            "labor_score",
            "policy_encoded",
            "text_length",
            "cosine_change_from_prev",
            "language_shift",
            "is_sep",
        ]
        attn_cols = []
        if attn_proj_dim:
            attn_cols = [f"stmt_attn_{j + 1}" for j in range(attn_proj_dim)]
        score_cols = [col for col in base_cols + attn_cols if col in fomc_df.columns]

    last_features = {col: 0.0 for col in score_cols}
    last_fomc_date = None
    fomc_dates = list(fomc_df.index)
    rows = []

    for day in pd.DatetimeIndex(daily_dates):
        is_fomc_day = int(day in fomc_df.index)
        if is_fomc_day:
            last_features = {col: fomc_df.loc[day, col] for col in score_cols}
            last_fomc_date = day

        days_since = (day - last_fomc_date).days if last_fomc_date is not None else np.nan
        future = [d for d in fomc_dates if d > day]
        days_to_next = (future[0] - day).days if future else np.nan

        rows.append(
            {
                "date": day,
                "is_fomc_day": is_fomc_day,
                "days_since_last_fomc": days_since,
                "days_to_next_fomc": days_to_next,
                **last_features,
            }
        )
    out = pd.DataFrame(rows).set_index("date")
    if "is_sep" in out.columns:
        out["is_sep"] = out["is_sep"].astype(bool)
    return out


def fetch_google_trends(
    keywords: Sequence[str],
    start: str = "2006-01-01",
    end: str = "2026-04-30",
    geo: str = "US",
    sleep_sec: float = 2.0,
) -> pd.DataFrame:
    """Fetch weekly Google Trends data for one or more keywords."""
    from pytrends.request import TrendReq

    pytrends = TrendReq(hl="en-US", tz=360)
    pytrends.build_payload(kw_list=list(keywords), timeframe=f"{start} {end}", geo=geo)
    if sleep_sec:
        time.sleep(sleep_sec)
    data = pytrends.interest_over_time()
    if "isPartial" in data.columns:
        data = data.drop(columns=["isPartial"])
    data.index = pd.to_datetime(data.index).tz_localize(None)
    return data


def trends_weekly_to_daily(
    trends_weekly: pd.DataFrame,
    daily_dates: pd.DatetimeIndex,
    prefix: str = "trend_",
) -> pd.DataFrame:
    """Forward-fill weekly Google Trends data to business-day dates."""
    out = trends_weekly.copy()
    out.columns = [f"{prefix}{str(col).lower().replace(' ', '_')}" for col in out.columns]
    return out.reindex(pd.DatetimeIndex(daily_dates)).ffill()


def download_yfinance_close(
    ticker: str,
    start: str = "2006-01-01",
    end: str = "2026-04-30",
    name: Optional[str] = None,
) -> pd.Series:
    """Download adjusted close/close data from yfinance."""
    import yfinance as yf

    close = yf.download(ticker, start=start, end=end)["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.name = name or ticker
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def build_master_dataset(
    daily_fomc: pd.DataFrame,
    vix: pd.Series,
    google_trends_daily: Optional[pd.DataFrame] = None,
    extra_features: Optional[pd.DataFrame] = None,
    ffill: bool = True,
) -> pd.DataFrame:
    """Merge daily FOMC features, VIX, Google Trends, and optional features."""
    master = daily_fomc.copy()
    vix = vix.copy()
    vix.name = vix.name or "^VIX"
    master = master.join(vix.reindex(master.index), how="left")
    if google_trends_daily is not None:
        master = master.join(google_trends_daily.reindex(master.index), how="left")
    if extra_features is not None:
        master = master.join(extra_features.reindex(master.index), how="left")
    return master.ffill() if ffill else master


def compute_forward_spike(vix: pd.Series, window: int = 10) -> pd.Series:
    """
    Future max log change over the next `window` trading days, excluding today.

    For each t: log(max(VIX[t+1 : t+window]) / VIX[t]).
    """
    future_max = vix.shift(-1).rolling(window=window).max().shift(-(window - 1))
    return np.log(future_max / vix)


def add_vix_spike_target(
    master: pd.DataFrame,
    vix_col: str = "^VIX",
    window: int = 10,
    quantile: float = 0.85,
    target_col: str = "target_vix_spike_10d",
    train_end: Optional[pd.Timestamp | str] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Add future max VIX log-change and binary spike label.

    If train_end is provided, the threshold is fit only on dates <= train_end,
    which avoids full-sample threshold leakage.
    """
    out = master.copy()
    change_col = f"vix_future_max_{window}d_change"
    out[change_col] = compute_forward_spike(out[vix_col], window=window)

    threshold_source = out[change_col]
    if train_end is not None:
        threshold_source = threshold_source.loc[: pd.Timestamp(train_end)]
    threshold = float(threshold_source.quantile(quantile))

    out[target_col] = (out[change_col] > threshold).astype(int)
    out = out.dropna(subset=[target_col, change_col])
    return out, threshold
