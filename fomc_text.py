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

from .constants import BASE_FOMC_URL, BOILERPLATE_PATTERNS, LEXICONS, RATE_DECISION_PATTERNS, DECISION_MAP, FOMC_MEETINGS

def fetch_statement(date_str: str, timeout: int = 10) -> str:
    """Fetch one FOMC statement from the Federal Reserve website."""
    import requests
    from bs4 import BeautifulSoup

    url = BASE_FOMC_URL.format(date_str)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    content = (
        soup.find("div", {"class": "col-xs-12 col-sm-8 col-md-8"})
        or soup.find("div", {"id": "content"})
        or soup.find("td", {"class": "content"})
        or soup.find("body")
    )
    return content.get_text(separator=" ", strip=True) if content else ""


def build_fomc_dataframe(
    meetings: Sequence[Tuple[str, bool]] = FOMC_MEETINGS,
    sleep_sec: float = 0.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch all configured FOMC statements and return an indexed DataFrame."""
    records = []
    for date_str, is_sep in meetings:
        if verbose:
            print(f"Fetching {date_str}...")
        try:
            text = fetch_statement(date_str)
        except Exception as exc:
            if verbose:
                print(f"  failed: {exc}")
            text = ""
        records.append({"date": pd.to_datetime(date_str), "is_sep": is_sep, "raw_text": text})
        if sleep_sec:
            time.sleep(sleep_sec)

    return pd.DataFrame(records).set_index("date").sort_index()


def clean_statement(text: str) -> str:
    """Remove standard boilerplate and normalize whitespace."""
    text = "" if pd.isna(text) else str(text)
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def add_clean_text(df: pd.DataFrame, raw_col: str = "raw_text") -> pd.DataFrame:
    """Add clean_text and text_length columns."""
    out = df.copy()
    out["clean_text"] = out[raw_col].apply(clean_statement)
    out["text_length"] = out["clean_text"].apply(lambda text: len(text.split()))
    return out


def score_text(text: str, keywords: Sequence[str]) -> float:
    """Keyword/phrase hit rate normalized by token count."""
    tokens = str(text).lower().split()
    total = len(tokens)
    if total == 0:
        return 0.0
    hits = 0
    for kw in keywords:
        parts = kw.lower().split()
        n = len(parts)
        hits += sum(1 for i in range(len(tokens) - n + 1) if tokens[i : i + n] == parts)
    return hits / total


def detect_policy_decision(text: str) -> str:
    """Classify statement as hike/cut/hold/unknown from simple regex rules."""
    text_lower = str(text).lower()
    for decision, patterns in RATE_DECISION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return decision
    return "unknown"


def add_lexicon_features(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    lexicons: Dict[str, Sequence[str]] = LEXICONS,
) -> pd.DataFrame:
    """Add hawkish/dovish/uncertainty/inflation/labor scores and policy encoding."""
    out = df.copy()
    for label, keywords in lexicons.items():
        out[f"{label}_score"] = out[text_col].apply(lambda text: score_text(text, keywords))
    out["policy_decision"] = out[text_col].apply(detect_policy_decision)
    out["policy_encoded"] = out["policy_decision"].map(DECISION_MAP).fillna(0).astype(int)
    return out
