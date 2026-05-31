"""Offline tests for the Backtester (no network / yfinance)."""

import json

import numpy as np
import pandas as pd
import pytest

from src.inference.backtest import BUY_COST_PCT, Backtester


def test_buy_and_hold_return_matches_price_ratio_minus_cost(rising_data):
    """Buy-and-hold return ≈ price[-1]/price[0]-1, reduced by the one-side buy cost."""
    bt = Backtester(initial_amount=10000, seed=42)
    result = bt.run_buy_and_hold(rising_data)
    curve = result["portfolio_values"]

    prices = rising_data["close"].to_numpy()
    shares = (10000 * (1 - BUY_COST_PCT)) // prices[0]
    cash_left = 10000 - shares * prices[0] * (1 + BUY_COST_PCT)
    expected_final = cash_left + shares * prices[-1]

    assert curve[-1] == pytest.approx(expected_final)
    assert len(curve) == len(rising_data)

    # The realized return is slightly below the raw price ratio (cost drag),
    # and within one share-rounding step of it.
    got_return = (curve[-1] - 10000) / 10000
    raw_price_return = prices[-1] / prices[0] - 1
    assert got_return < raw_price_return
    assert raw_price_return - got_return < 0.05


def test_run_random_is_deterministic_under_fixed_seed(rising_data):
    """Same seed -> identical random equity curve across runs."""
    curve_a = Backtester(initial_amount=10000, seed=7).run_random(rising_data)["portfolio_values"]
    curve_b = Backtester(initial_amount=10000, seed=7).run_random(rising_data)["portfolio_values"]
    assert curve_a == pytest.approx(curve_b)


def test_all_runners_return_finite_curves_of_expected_length(rising_data, stub_model):
    bt = Backtester(initial_amount=10000, seed=42)
    n = len(rising_data)
    for result in (
        bt.run_agent(stub_model, rising_data),
        bt.run_buy_and_hold(rising_data),
        bt.run_random(rising_data),
    ):
        curve = result["portfolio_values"]
        assert len(curve) == n
        assert np.all(np.isfinite(curve))
        assert "actions" in result


def test_compare_returns_all_strategies_and_curves(rising_data, stub_model):
    bt = Backtester(initial_amount=10000, seed=42)
    results = bt.compare(stub_model, rising_data)
    assert set(results) == {"agent", "buy_and_hold", "random", "curves"}
    assert set(results["curves"]) == {"agent", "buy_and_hold", "random"}
    # Agent / random scored against buy-and-hold benchmark -> have beta.
    assert "beta" in results["agent"]
    assert "beta" in results["random"]


def test_compare_aborts_on_fabricated_data(stub_model):
    """compare() must refuse fabricated (dummy-fallback) data, not crash silently."""
    fake = pd.DataFrame(
        {"date": pd.date_range("2023-01-01", periods=10, freq="D"), "close": range(1, 11)}
    )
    bt = Backtester()
    with pytest.raises(RuntimeError, match="fabricated"):
        bt.compare(stub_model, fake)


def test_out_of_sample_guard_raises_on_overlap(tmp_path, rising_data, stub_model):
    """With training metadata present, an overlapping test window must raise."""
    meta = {"ticker": "TST", "train_start": "2020-01-01", "train_end": "2023-12-31"}
    (tmp_path / "TST.json").write_text(json.dumps(meta))

    bt = Backtester(initial_amount=10000, seed=42)
    with pytest.raises(ValueError, match="Out-of-sample violation"):
        bt.compare(
            stub_model,
            rising_data,
            test_start="2023-06-01",   # inside the training window
            test_end="2023-09-01",
            ticker="TST",
            models_dir=tmp_path,
        )


def test_out_of_sample_guard_passes_on_disjoint_window(tmp_path):
    meta = {"ticker": "TST", "train_start": "2020-01-01", "train_end": "2022-12-31"}
    (tmp_path / "TST.json").write_text(json.dumps(meta))
    bt = Backtester()
    # Should not raise for a window entirely after training.
    bt._check_out_of_sample("TST", "2023-01-01", "2023-06-01", tmp_path)


def test_missing_metadata_warns_instead_of_raising(tmp_path):
    bt = Backtester()
    with pytest.warns(UserWarning, match="No training metadata"):
        bt._check_out_of_sample("TST", "2023-01-01", "2023-06-01", tmp_path)


def test_fabricated_data_detection():
    # Weekend dates + no benchmark column => fabricated (dummy fallback).
    fake = pd.DataFrame(
        {"date": pd.date_range("2023-01-01", periods=10, freq="D"), "close": range(10)}
    )
    assert Backtester._is_fabricated_data(fake) is True

    # Business days + benchmark column => not flagged.
    real = pd.DataFrame(
        {
            "date": pd.bdate_range("2023-01-02", periods=10),
            "close": range(10),
            "close_benchmark": range(10),
        }
    )
    assert Backtester._is_fabricated_data(real) is False
