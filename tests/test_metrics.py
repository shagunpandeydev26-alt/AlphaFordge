"""Known-input tests for the performance metrics module."""

import numpy as np
import pytest

from src.utils.metrics import (
    PerformanceMetrics,
    calculate_max_drawdown,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_trading_metrics,
)


def test_flat_curve_zero_return_and_sharpe():
    """A flat equity curve has zero return and an (undefined -> 0) Sharpe."""
    flat = [10000.0] * 30
    returns = calculate_returns(flat)
    assert np.allclose(returns, 0.0)
    assert calculate_sharpe_ratio(returns) == 0.0

    metrics = calculate_trading_metrics(flat, initial_amount=10000.0)
    assert metrics["total_return_pct"] == pytest.approx(0.0)
    assert metrics["sharpe_ratio"] == pytest.approx(0.0)
    assert metrics["max_drawdown_pct"] == pytest.approx(0.0)


def test_monotonic_rising_curve_has_zero_drawdown():
    """A curve that only ever rises never draws down."""
    rising = list(np.linspace(10000.0, 15000.0, 50))
    assert calculate_max_drawdown(rising) == pytest.approx(0.0)
    metrics = calculate_trading_metrics(rising, initial_amount=10000.0)
    assert metrics["max_drawdown_pct"] == pytest.approx(0.0)
    assert metrics["total_return_pct"] == pytest.approx(50.0)


def test_crash_series_has_expected_negative_drawdown():
    """A peak-to-trough crash produces the expected drawdown magnitude."""
    curve = [10000.0, 12000.0, 15000.0, 9000.0, 6000.0]
    # Peak 15000 -> trough 6000 => (6000-15000)/15000 = -0.6
    assert calculate_max_drawdown(curve) == pytest.approx(-0.6)
    metrics = calculate_trading_metrics(curve, initial_amount=10000.0)
    assert metrics["max_drawdown_pct"] == pytest.approx(-60.0)
    assert metrics["max_drawdown_pct"] < 0


def test_calculate_all_metrics_runs_without_benchmark():
    """Regression test for bug #3: calculate_all_metrics used to TypeError."""
    pm = PerformanceMetrics()
    for v in [10000.0, 10100.0, 10050.0, 10200.0, 10300.0]:
        pm.add_data_point(v)
    metrics = pm.calculate_all_metrics()
    assert metrics  # non-empty
    assert "sharpe_ratio" in metrics
    assert "total_return_pct" in metrics
    # No benchmark recorded -> no benchmark-comparison fields.
    assert "beta" not in metrics


def test_calculate_all_metrics_with_benchmark_populates_beta():
    """When a benchmark is recorded, comparison fields appear."""
    pm = PerformanceMetrics()
    for v in [10000.0, 10100.0, 10050.0, 10200.0, 10300.0]:
        pm.add_data_point(v, benchmark_value=v * 0.99)
    metrics = pm.calculate_all_metrics()
    assert "beta" in metrics
    assert "excess_return_pct" in metrics
    assert np.isfinite(metrics["beta"])


def test_empty_inputs_are_safe():
    assert calculate_max_drawdown([]) == 0.0
    assert calculate_sharpe_ratio(np.array([])) == 0.0
    assert calculate_trading_metrics([]) == {}
