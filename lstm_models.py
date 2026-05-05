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

def make_lstm_model_classes():
    """Return VIXLSTMMLP, Attention, VIXLSTMMLP_ATT, and VIXLSTM_TBPTT classes."""
    import torch
    import torch.nn as nn

    class VIXLSTMMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim=32, num_layers=1, mlp_hidden_dim=16, dropout=0.10):
            super().__init__()
            self.input_norm = nn.LayerNorm(input_dim)
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_dim, 1),
            )

        def forward(self, x):
            x = self.input_norm(x)
            out, _ = self.lstm(x)
            last_hidden = out[:, -1, :]
            return self.head(last_hidden)

    class Attention(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.score = nn.Linear(hidden_dim, 1, bias=False)

        def forward(self, lstm_out):
            # lstm_out: (batch, seq_len, hidden_dim)
            scores = self.score(lstm_out)              # (batch, seq_len, 1)
            weights = torch.softmax(scores, dim=1)     # (batch, seq_len, 1)
            context = (weights * lstm_out).sum(dim=1)  # (batch, hidden_dim)
            return context, weights


    class VIXLSTMMLP_ATT(nn.Module):
        def __init__(
            self,
            input_dim,
            hidden_dim=64,
            num_layers=2,
            mlp_hidden_dim=32,
            dropout=0.3,
        ):
            super().__init__()

            self.input_norm = nn.LayerNorm(input_dim)

            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

            self.attention = Attention(hidden_dim)
            self.dropout = nn.Dropout(dropout)

            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, mlp_hidden_dim),
                nn.LayerNorm(mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim // 2, 1),
            )

        def forward(self, x, return_attention=False):
            x = self.input_norm(x)

            lstm_out, (h_n, c_n) = self.lstm(x)

            context, attn_weights = self.attention(lstm_out)
            h_last = h_n[-1]

            combined = torch.cat([context, h_last], dim=-1)
            combined = self.dropout(combined)

            pred = self.mlp(combined)

            if return_attention:
                return pred, attn_weights
            return pred

    class VIXLSTM_TBPTT(nn.Module):
        def __init__(self, input_dim, hidden_dim=32, num_layers=1, mlp_hidden_dim=16, dropout=0.20, tbptt_step=None):
            super().__init__()
            self.tbptt_step = tbptt_step
            self.input_norm = nn.LayerNorm(input_dim)
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_dim, 1),
            )

        def forward(self, x):
            x = self.input_norm(x)
            if self.tbptt_step is None:
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])
            h = None
            out = None
            for start in range(0, x.size(1), self.tbptt_step):
                chunk = x[:, start : start + self.tbptt_step, :]
                out, h = self.lstm(chunk, h)
                h = tuple(t.detach() for t in h)
            return self.head(out[:, -1, :])

    return VIXLSTMMLP, Attention, VIXLSTMMLP_ATT, VIXLSTM_TBPTT

