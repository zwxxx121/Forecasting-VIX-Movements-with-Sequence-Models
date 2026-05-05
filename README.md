# vix_fomc_pkg

This package splits `vix_fomc_pipeline.py` into smaller files so the notebook can stay clean.

Recommended notebook import:

```python
from vix_fomc_pkg import *
```

Module map:

- `common.py`: seed/device helpers
- `constants.py`: FOMC dates, URL patterns, lexicons
- `fomc_text.py`: FOMC fetching, cleaning, lexicon features
- `finbert_features.py`: FinBERT embedding and token-embedding export helpers
- `fomc_attention.py`: token-level FOMC attention embedding model
- `market_features.py`: daily FOMC calendar features, Google Trends, yfinance, VIX target
- `sequence_data.py`: chronological split, scaling, sequence construction, DataLoaders
- `lstm_models.py`: LSTM model class factory
- `metrics.py`: classification metrics and threshold search
- `training.py`: PyTorch prediction/training loops
- `baselines.py`: baseline and XGBoost helpers

You can also import selectively, e.g.:

```python
from vix_fomc_pkg.market_features import add_vix_spike_target
from vix_fomc_pkg.sequence_data import make_lstm_sequences
```
