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

def load_finbert(model_name: str = "ProsusAI/finbert"):
    """Load FinBERT tokenizer/model lazily."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def embed_text(
    text: str,
    tokenizer,
    model,
    strategy: str = "mean",
    max_length: int = 512,
) -> np.ndarray:
    """Embed one text using CLS or mean-pool FinBERT hidden states."""
    import torch

    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state
    if strategy == "cls":
        emb = hidden[:, 0, :]
    elif strategy == "mean":
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb = (hidden * mask).sum(1) / mask.sum(1)
    else:
        raise ValueError("strategy must be 'cls' or 'mean'")
    return emb.squeeze(0).cpu().numpy()


def add_finbert_embedding_features(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    n_components: int = 10,
    strategy: str = "mean",
    max_length: int = 512,
    tokenizer=None,
    model=None,
) -> pd.DataFrame:
    """
    Add raw embedding object column, cosine similarity/change, PCA components,
    and PCA component changes.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity

    if tokenizer is None or model is None:
        tokenizer, model = load_finbert()

    out = df.copy()
    embeddings = [
        embed_text(text, tokenizer=tokenizer, model=model, strategy=strategy, max_length=max_length)
        for text in out[text_col]
    ]
    embedding_matrix = np.stack(embeddings)
    out["embedding"] = embeddings

    cos_sims = [1.0]
    for i in range(1, len(embedding_matrix)):
        sim = cosine_similarity(
            embedding_matrix[i - 1].reshape(1, -1),
            embedding_matrix[i].reshape(1, -1),
        )[0, 0]
        cos_sims.append(float(sim))
    out["cosine_change_from_prev"] = cos_sims
    out["language_shift"] = 1.0 - out["cosine_change_from_prev"]

    n_components = min(n_components, len(out), embedding_matrix.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    stmt_pca = pca.fit_transform(embedding_matrix)
    for i in range(n_components):
        out[f"stmt_pca_{i + 1}"] = stmt_pca[:, i]
        out[f"change_pca_{i + 1}"] = out[f"stmt_pca_{i + 1}"].diff().fillna(0.0)
    return out


def save_token_embeddings(
    df: pd.DataFrame,
    out_dir: str | Path = "fomc_token_embeddings",
    text_col: str = "clean_text",
    max_length: int = 128,
    tokenizer=None,
    model=None,
) -> None:
    """Save one token-embedding matrix and attention mask per FOMC date."""
    import torch

    if tokenizer is None or model is None:
        tokenizer, model = load_finbert()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for date, row in df.iterrows():
        inputs = tokenizer(
            row[text_col],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        tokens = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        mask = inputs["attention_mask"].squeeze(0).cpu().numpy()
        date_str = pd.Timestamp(date).strftime("%Y%m%d")
        np.save(out_path / f"{date_str}_tokens.npy", tokens)
        np.save(out_path / f"{date_str}_mask.npy", mask)
