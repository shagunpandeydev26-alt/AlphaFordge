"""
Out-of-sample backtesting and baseline comparison for RL trading agents.

This module runs three strategies over an identical out-of-sample window with the
same starting cash and the same transaction costs:

* ``agent``        - the trained PPO model's actions
* ``buy_and_hold`` - spend all cash on day 1 (paying the 0.1% buy cost once) and hold
* ``random``       - random actions under a fixed seed (a "is it learning anything?"
                     sanity check)

All three are then scored with :func:`src.utils.metrics.calculate_trading_metrics`,
using the buy-and-hold equity curve as the benchmark so beta / Treynor /
information-ratio / excess-return populate against the obvious alternative.

The heavy RL stack (finrl / stable-baselines3) is imported lazily so that the
pure-Python helpers (``run_buy_and_hold``, the out-of-sample guard, fabricated-data
detection) can be imported and tested without those dependencies installed.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.metrics import calculate_trading_metrics, create_performance_report

# Transaction costs charged by the environment (0.1% per side). Baselines MUST pay
# the same costs as the agent for the comparison to be fair.
BUY_COST_PCT = 0.001
SELL_COST_PCT = 0.001

# Default location of trained models and their sidecar metadata.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


class Backtester:
    """Runs strategies over one fixed ``(data, initial_amount)`` context.

    Every runner returns a ``dict`` with at least ``portfolio_values`` (the equity
    curve) and ``actions``. The same ``initial_amount``, the same data window and
    the same transaction costs are used for all three strategies - fairness is the
    core invariant of this class.
    """

    def __init__(self, initial_amount: float = 10000, seed: int = 42):
        self.initial_amount = initial_amount
        self.seed = seed

    # ------------------------------------------------------------------
    # Environment helpers (lazy finrl import)
    # ------------------------------------------------------------------
    def _make_env(self, data: pd.DataFrame):
        """Build a trading env over ``data``. Imported lazily to keep the rest of
        this module usable without finrl / stable-baselines3 installed."""
        from src.envs.SingleStockTradingEnv import SingleStockTradingEnv

        return SingleStockTradingEnv(df=data, initial_amount=self.initial_amount)

    @staticmethod
    def _clean_obs(obs: np.ndarray) -> np.ndarray:
        """Replace NaN/inf in an observation, mirroring inference.py's guard."""
        if np.isnan(obs).any() or np.isinf(obs).any():
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def _run_env_loop(self, data: pd.DataFrame, action_fn) -> Dict[str, Any]:
        """Step an env to the end of ``data`` using ``action_fn(env, obs)`` to pick
        each action. Returns the env's ``asset_memory`` as the equity curve.

        Reuses the env-stepping pattern from ``inference.py`` (NaN cleaning + a
        ``max_steps`` safety bound to guard against a non-terminating env).
        """
        env = self._make_env(data)
        obs, _info = env.reset()
        obs = self._clean_obs(obs)

        actions: List[float] = []
        done = False
        truncated = False
        steps = 0
        max_steps = len(data) * 2  # safety bound

        while not (done or truncated) and steps < max_steps:
            action = action_fn(env, obs)
            obs, _reward, done, truncated, _info = env.step(action)
            obs = self._clean_obs(obs)
            actions.append(float(action[0] if hasattr(action, "__len__") else action))
            steps += 1

        curve = [float(v) for v in env.asset_memory]
        return {"portfolio_values": curve, "actions": actions}

    # ------------------------------------------------------------------
    # Strategy runners
    # ------------------------------------------------------------------
    def run_agent(self, model, data: pd.DataFrame) -> Dict[str, Any]:
        """Run the trained model's (deterministic) policy over ``data``."""

        def action_fn(env, obs):
            action, _states = model.predict(obs, deterministic=True)
            return action

        return self._run_env_loop(data, action_fn)

    def run_random(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run random actions under the fixed seed (reproducible)."""
        # The env's action space owns the RNG used by ``sample()``; seeding it makes
        # the random baseline deterministic across runs.
        def action_fn(env, obs):
            return env.action_space.sample()

        env = self._make_env(data)
        env.action_space.seed(self.seed)
        obs, _info = env.reset()
        obs = self._clean_obs(obs)

        actions: List[float] = []
        done = False
        truncated = False
        steps = 0
        max_steps = len(data) * 2

        while not (done or truncated) and steps < max_steps:
            action = env.action_space.sample()
            obs, _reward, done, truncated, _info = env.step(action)
            obs = self._clean_obs(obs)
            actions.append(float(action[0] if hasattr(action, "__len__") else action))
            steps += 1

        curve = [float(v) for v in env.asset_memory]
        return {"portfolio_values": curve, "actions": actions}

    def run_buy_and_hold(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Buy all affordable shares on day 1, pay the buy cost once, then hold.

        No further trades, so no further costs. ``curve[t] = cash_left + shares *
        price[t]``. This is the "do nothing smart" baseline: if the agent cannot
        beat it, it adds no value.
        """
        prices = np.asarray(data["close"].to_numpy(), dtype=float)
        if len(prices) == 0:
            return {"portfolio_values": [], "actions": []}

        price0 = prices[0]
        shares = float((self.initial_amount * (1 - BUY_COST_PCT)) // price0)
        cash_left = self.initial_amount - shares * price0 * (1 + BUY_COST_PCT)

        curve = [float(cash_left + shares * p) for p in prices]
        # One buy on day 0, hold thereafter.
        actions = [shares] + [0.0] * (len(prices) - 1)
        return {"portfolio_values": curve, "actions": actions}

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    def compare(
        self,
        model,
        data: pd.DataFrame,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
        ticker: Optional[str] = None,
        models_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Run all three strategies and score each on identical terms.

        Returns ``{"agent": {...}, "buy_and_hold": {...}, "random": {...},
        "curves": {name: [values]}}``.
        """
        # Credibility guardrails before any numbers are produced.
        self._check_out_of_sample(ticker, test_start, test_end, models_dir)
        if self._is_fabricated_data(data):
            # Refuse to "backtest" on the env's np.random.seed(42) dummy fallback -
            # those numbers are meaningless and must not pass silently (design §8).
            raise RuntimeError(
                "Backtest aborted: the input data looks fabricated (the env's "
                "dummy-data fallback was likely used - weekend dates and/or no "
                "benchmark column). This usually means the market-data download "
                "failed (e.g. yfinance rate-limiting). Retry once real data is "
                "available rather than reporting results on simulated prices."
            )

        agent = self.run_agent(model, data)
        buy_and_hold = self.run_buy_and_hold(data)
        random_run = self.run_random(data)

        bnh_curve = np.asarray(buy_and_hold["portfolio_values"], dtype=float)

        results = {
            "agent": calculate_trading_metrics(
                agent["portfolio_values"],
                benchmark_data=bnh_curve,
                initial_amount=self.initial_amount,
            ),
            "buy_and_hold": calculate_trading_metrics(
                buy_and_hold["portfolio_values"],
                initial_amount=self.initial_amount,
            ),
            "random": calculate_trading_metrics(
                random_run["portfolio_values"],
                benchmark_data=bnh_curve,
                initial_amount=self.initial_amount,
            ),
            "curves": {
                "agent": agent["portfolio_values"],
                "buy_and_hold": buy_and_hold["portfolio_values"],
                "random": random_run["portfolio_values"],
            },
        }
        return results

    # ------------------------------------------------------------------
    # Guardrails
    # ------------------------------------------------------------------
    @staticmethod
    def _load_metadata(ticker: Optional[str], models_dir: Optional[Path]) -> Optional[dict]:
        """Load ``models/<TICKER>.json`` training metadata if present."""
        if not ticker:
            return None
        base = Path(models_dir) if models_dir is not None else MODELS_DIR
        meta_path = base / f"{ticker}.json"
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None

    def _check_out_of_sample(
        self,
        ticker: Optional[str],
        test_start: Optional[str],
        test_end: Optional[str],
        models_dir: Optional[Path],
    ) -> None:
        """Assert the test window does not overlap training dates.

        Raises ``ValueError`` on a verified overlap; emits a visible warning when
        metadata or the test window is unavailable rather than silently proceeding.
        """
        meta = self._load_metadata(ticker, models_dir)
        if meta is None:
            warnings.warn(
                f"No training metadata found for '{ticker}'. Cannot verify the test "
                "window is out-of-sample - results may reflect memorized training "
                "data. (Train via the updated pipeline to write models/<TICKER>.json.)",
                stacklevel=2,
            )
            return

        if test_start is None or test_end is None:
            warnings.warn(
                "No explicit test window provided; skipping out-of-sample check.",
                stacklevel=2,
            )
            return

        train_start = pd.Timestamp(meta["train_start"])
        train_end = pd.Timestamp(meta["train_end"])
        ts = pd.Timestamp(test_start)
        te = pd.Timestamp(test_end)

        # Overlap unless the test window is entirely before or entirely after training.
        overlaps = not (ts > train_end or te < train_start)
        if overlaps:
            raise ValueError(
                f"Out-of-sample violation for '{ticker}': test window "
                f"[{test_start} .. {test_end}] overlaps training window "
                f"[{meta['train_start']} .. {meta['train_end']}]. Choose a test "
                "window that does not intersect the training dates."
            )

    @staticmethod
    def _is_fabricated_data(data: pd.DataFrame) -> bool:
        """Heuristically detect the env's ``np.random.seed(42)`` dummy fallback.

        The fallback uses ``pd.date_range(freq='D')`` (so weekend dates appear) and
        never attaches a benchmark column, neither of which happens with real
        market data fetched for trading days.
        """
        if "close_benchmark" not in data.columns:
            return True
        if "date" in data.columns:
            dates = pd.to_datetime(data["date"], errors="coerce")
            if dates.dt.dayofweek.ge(5).any():
                return True
        return False


# ----------------------------------------------------------------------
# Data loading + CLI
# ----------------------------------------------------------------------
def load_backtest_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch and process price data for ``ticker`` over ``[start, end]``.

    Reuses ``SingleStockTradingEnv.get_data`` so the indicators, benchmark merge
    and (if the network fails) the dummy-data fallback all match what training and
    inference see. Imported lazily so this module loads without finrl.
    """
    from src.envs.SingleStockTradingEnv import SingleStockTradingEnv

    env = SingleStockTradingEnv(ticker=ticker, start_date=start, end_date=end)
    return env.df


def _sanitize_for_json(obj: Any) -> Any:
    """Make metrics JSON-serializable (numpy scalars, non-finite floats)."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    return obj


def run_cli(args=None) -> Dict[str, Any]:
    """Entry point for ``python -m src.inference.backtest``."""
    import argparse

    from stable_baselines3 import PPO

    parser = argparse.ArgumentParser(
        description="Out-of-sample backtest: RL agent vs. buy-and-hold vs. random."
    )
    parser.add_argument("--ticker", required=True, help="Stock ticker (matches models/<TICKER>.zip)")
    parser.add_argument("--start", required=True, help="Test window start (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Test window end (YYYY-MM-DD)")
    parser.add_argument("--initial-amount", type=float, default=10000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    ns = parser.parse_args(args)

    models_dir = Path(ns.models_dir)
    model_path = models_dir / f"{ns.ticker}.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}")

    print(f"Loading model {model_path} ...")
    model = PPO.load(str(model_path))

    print(f"Fetching data for {ns.ticker} [{ns.start} .. {ns.end}] ...")
    data = load_backtest_data(ns.ticker, ns.start, ns.end)

    backtester = Backtester(initial_amount=ns.initial_amount, seed=ns.seed)
    results = backtester.compare(
        model,
        data,
        test_start=ns.start,
        test_end=ns.end,
        ticker=ns.ticker,
        models_dir=models_dir,
    )

    labels = {"agent": "RL AGENT", "buy_and_hold": "BUY & HOLD", "random": "RANDOM"}
    for key, label in labels.items():
        print(f"\n##### {label} #####")
        print(create_performance_report(results[key]))

    # Persist metrics + curves.
    output_dir = Path(ns.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"backtest_{ns.ticker}.json"
    payload = {
        "ticker": ns.ticker,
        "test_start": ns.start,
        "test_end": ns.end,
        "initial_amount": ns.initial_amount,
        "seed": ns.seed,
        "metrics": {k: results[k] for k in ("agent", "buy_and_hold", "random")},
        "curves": results["curves"],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(_sanitize_for_json(payload), fh, indent=2)
    print(f"\nResults written to {out_path}")

    return results


if __name__ == "__main__":
    run_cli()
