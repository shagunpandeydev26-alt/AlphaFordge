"""Shared, fully-offline fixtures for the test suite.

No network or yfinance access: every fixture builds an in-memory DataFrame so the
suite runs deterministically in CI.
"""

# Ensure `wrds` is importable before finrl tries to import it at module level.
# finrl 0.3.7 eagerly imports wrds via its train → data_processor chain, but
# wrds is an optional academic-database package not available in CI.  A stub
# in sys.modules satisfies the import guard so the rest of the stack loads.
import sys
from unittest.mock import MagicMock

if "wrds" not in sys.modules:
    sys.modules["wrds"] = MagicMock()

import numpy as np
import pandas as pd
import pytest

# The technical-indicator columns the trading env expects when a DataFrame is
# passed directly (so FeatureEngineer / data download are never invoked).
_INDICATOR_COLS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
    "turbulence",
]


def make_price_data(close: np.ndarray, start: str = "2023-01-02") -> pd.DataFrame:
    """Build a single-stock processed DataFrame the env can consume directly.

    Includes OHLCV, ticker, a benchmark column and neutral technical indicators,
    on a business-day index (so it is not flagged as fabricated data).
    """
    close = np.asarray(close, dtype=float)
    n = len(close)
    dates = pd.bdate_range(start, periods=n)
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000.0,
            "tic": "TST",
            "close_benchmark": close,
        }
    )
    for col in _INDICATOR_COLS:
        df[col] = 0.0
    # A few indicators read more naturally as price-level / neutral defaults.
    df["rsi_30"] = 50.0
    df["boll_ub"] = df["close"]
    df["boll_lb"] = df["close"]
    df["close_30_sma"] = df["close"]
    df["close_60_sma"] = df["close"]
    return df


@pytest.fixture
def rising_data() -> pd.DataFrame:
    """A monotonically rising price series."""
    return make_price_data(np.linspace(100.0, 160.0, 40))


@pytest.fixture
def crash_data() -> pd.DataFrame:
    """A series that rises then crashes hard (for drawdown checks)."""
    up = np.linspace(100.0, 150.0, 20)
    down = np.linspace(150.0, 60.0, 20)
    return make_price_data(np.concatenate([up, down]))


class StubModel:
    """Minimal stand-in for a Stable-Baselines3 model.

    Implements the only method the Backtester uses (``predict``) so ``run_agent``
    can be exercised without training a real model.
    """

    def __init__(self, action: float = 0.0):
        self.action = action

    def predict(self, obs, deterministic: bool = True):
        return np.array([self.action], dtype=np.float32), None


@pytest.fixture
def stub_model() -> StubModel:
    return StubModel(action=0.0)
