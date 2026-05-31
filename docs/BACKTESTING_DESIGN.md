# Design Doc: Out-of-Sample Backtesting & Baseline Comparison

**Status:** Proposed
**Author:** (you)
**Date:** 2026-06-13

---

## 1. Problem

The project can train a PPO agent and produce a portfolio value at the end of a
run, but it **cannot honestly answer the only question that matters**:

> *Is the agent actually better than just buying the stock and holding it?*

Three concrete defects make the current numbers untrustworthy:

1. **The benchmark is computed wrong.** In `src/inference/inference.py:177-185`
   the agent's return is a clean fraction
   `(final - initial) / initial`, but `benchmark_return` is
   `data['close'].pct_change().cumsum().iloc[-1]` — a cumulative **sum** of daily
   percent changes. Stock value compounds **multiplicatively**
   (`(1 + r).cumprod()`), not additively. The two numbers are on different
   scales, so "the agent beat the benchmark" is meaningless as written.

2. **No real baselines.** There is no buy-and-hold or random strategy run through
   the *same* dollars, dates, and transaction costs as the agent. Without that,
   there is nothing legitimate to compare against.

3. **The rich metrics library is dead/unused.** `src/utils/metrics.py` (~500
   lines: Sharpe, Sortino, Calmar, beta, VaR, CVaR, etc.) is never wired into a
   real evaluation. Worse, `PerformanceMetrics.calculate_all_metrics()`
   (`metrics.py:450`) calls `calculate_trading_metrics(...)` with keyword
   arguments (`benchmark_values`, `actions`, `prices`) that **do not exist** in
   the function's signature (`metrics.py:272`), so it raises `TypeError` if ever
   called. It has never been exercised.

Additionally, nothing enforces an **out-of-sample** test window, so reported
results may simply be the model recalling data it trained on.

## 2. Goals / Non-Goals

**Goals**
- Run three strategies — PPO agent, buy-and-hold, random — over an identical
  out-of-sample window with identical starting cash and transaction costs.
- Score all three with the existing `calculate_trading_metrics()` and emit a
  comparison table + a saved JSON/markdown report.
- Produce an equity-curve plot (all three lines) plus a drawdown subplot.
- Surface the result in the Streamlit UI and as a reusable function/CLI.
- Fix the two metrics bugs (#1 and #3 above) as part of the work.

**Non-Goals (explicitly out of scope for this change)**
- Adding new RL algorithms (A2C/SAC/etc.).
- Multi-asset / portfolio-allocation environments.
- Live or paper trading.
- Removing the silent dummy-data fallback (tracked separately; see §8).

## 3. Background — concepts the implementation relies on

| Term | Definition | Where it lives |
|---|---|---|
| **Equity curve** | Portfolio dollar value over time; the core artifact. | env `asset_memory` |
| **Total return** | `(final - initial) / initial`. | `metrics.py:297` |
| **Annualized return** | `(final/initial) ** (252/n_days) - 1`; lets unequal-length tests be compared. | `metrics.py:298` |
| **Sharpe ratio** | Excess return ÷ total volatility, annualized. >1 decent, >2 good. | `metrics.py:18` (correct) |
| **Max drawdown** | Worst peak-to-trough drop. High-return + deep-drawdown = uninvestable. | `metrics.py:66` |
| **Out-of-sample** | Test dates must not overlap training dates. Violating this is the cardinal ML-finance sin. | enforced here |
| **Transaction cost** | Env charges `buy_cost_pct`/`sell_cost_pct` = 0.1% per side. Baselines must pay the same. | env defaults |

Environment facts this design depends on (from
`src/envs/SingleStockTradingEnv.py`): `initial_amount=10000`, `hmax=100`,
`buy_cost_pct=[0.001]`, `sell_cost_pct=[0.001]`, continuous action in `[-1, 1]`
scaled by `hmax`, and the env records its equity curve in `self.asset_memory`.

## 4. Design

### 4.1 New module: `src/inference/backtest.py`

A `Backtester` class that runs strategies over one fixed `(data, initial_amount)`
context and returns equity curves. Each runner returns a `dict` with at least
`portfolio_values: list[float]` and `actions: list[float]`.

```python
class Backtester:
    def __init__(self, initial_amount: float = 10000, seed: int = 42): ...

    def run_agent(self, model, data) -> dict:
        # Reuses the existing env-stepping loop from inference.py:142-174:
        # env = SingleStockTradingEnv(df=data, initial_amount=self.initial_amount)
        # step with action, _ = model.predict(obs); collect env.asset_memory[-1]

    def run_buy_and_hold(self, data) -> dict:
        # shares = (initial_amount * (1 - buy_cost_pct)) // price[0]
        # cash_left = initial_amount - shares*price[0]*(1 + buy_cost_pct)
        # curve[t] = cash_left + shares * price[t]   (no further trades)

    def run_random(self, data) -> dict:
        # Same env loop as run_agent but action = env.action_space.sample()
        # under a fixed RNG seed for reproducibility.

    def compare(self, model, data) -> dict:
        # Runs all three, scores each via calculate_trading_metrics(),
        # using buy-and-hold's curve as the `benchmark_data` arg so beta /
        # information ratio / excess return populate. Returns:
        # { "agent": {...metrics}, "buy_and_hold": {...}, "random": {...},
        #   "curves": { name: [values] } }
```

**Why buy-and-hold is the `benchmark_data`:** `calculate_trading_metrics()`
already has a benchmark branch (`metrics.py:335-360`) that computes beta,
Treynor, information ratio, and excess return when given a benchmark series.
Feeding buy-and-hold there makes those fields meaningful "vs. the obvious
alternative" numbers for free.

### 4.2 Enforcing out-of-sample

`compare()` (and the CLI) take explicit `test_start` / `test_end`. When training
date metadata is available (see §4.4), assert no overlap and **raise loudly** on
violation; when it is not available, emit a visible warning rather than silently
proceeding. This is the credibility guardrail.

### 4.3 Plot: `src/utils/plot.py :: plot_equity_curves(curves, title) -> Figure`

matplotlib (already a dependency). Top panel: three equity curves on shared
dates. Bottom panel: drawdown of each. Returns a `Figure` so it can be both saved
to disk and rendered by Streamlit via `st.pyplot`.

### 4.4 Training-date metadata (small enabler)

To verify out-of-sample automatically, training should write a sidecar
`models/<TICKER>.json` with `{train_start, train_end, ticker, hyperparams}` next
to the existing `models/<TICKER>.zip`. The backtester reads it when present. If
absent (true for the 7 shipped models), fall back to the warning path. *This is
the only change outside the new module + plot + bug fixes, and it is additive.*

### 4.5 Surfaces

- **Streamlit** (`streamlit_app.py`): a "📊 Backtest vs. Baselines" expander —
  pick ticker + test window → table + `plot_equity_curves` figure.
- **CLI**: `python -m src.inference.backtest --ticker NFLX --start 2023-01-01
  --end 2024-12-31` → prints `create_performance_report()` for each strategy and
  writes `results/backtest_<ticker>.json`.

## 5. Bug fixes folded into this change

1. **`inference.py` benchmark** — replace `pct_change().cumsum()` with a true
   buy-and-hold equity curve from the new `Backtester`, and compare on the same
   scale (both fractions, or both dollar curves).
2. **`metrics.py:450`** — fix `PerformanceMetrics.calculate_all_metrics()` to call
   `calculate_trading_metrics(self.portfolio_values, benchmark_data=...,
   initial_amount=..., risk_free_rate=...)` with real arguments, or remove the
   class if nothing uses it after the refactor.

Each ships as its own commit with a one-line rationale.

## 6. Test plan (`tests/` — new)

- `test_metrics.py`: known-input checks — a flat curve → 0 return, 0 Sharpe;
  a monotonically rising curve → 0 drawdown; a crash → expected negative
  drawdown. Covers the bug-fixed `calculate_all_metrics`.
- `test_backtest.py`:
  - On a **synthetic monotonically rising series**, buy-and-hold return ≈
    `(price[-1]/price[0] - 1)` minus the round-trip not taken (one-side cost).
  - `run_random` with a fixed seed is **deterministic** (same curve twice).
  - All three runners return curves of the expected length and finite values.
  - Out-of-sample guard raises when given an overlapping window + metadata.
- Use a tiny offline DataFrame fixture (no network) so tests run in CI.

## 7. Milestones

1. `Backtester` + buy-and-hold + random + `compare()` (pure functions, testable).
2. Wire `calculate_trading_metrics` in; fix the two bugs.
3. `plot_equity_curves`.
4. Tests + a minimal GitHub Actions job to run them.
5. Streamlit section + CLI + sidecar metadata on training.

Milestones 1–3 alone deliver the resume-worthy core.

## 8. Risks & open questions

- **Dummy-data fallback** (`SingleStockTradingEnv.get_data`, the
  `np.random.seed(42)` branch) can silently feed fabricated prices into a
  "backtest." The backtester should at minimum detect/flag when fallback data is
  used. Full removal is out of scope but noted.
- **Network in CI** — yfinance must not be hit during tests; rely on cached
  fixtures.
- **Action threshold for buy/sell/hold stats** is currently `±0.1`
  (`inference.py:221-224`); keep consistent if we report action distributions.

## 9. What this buys you (the pitch)

> "I built an out-of-sample backtesting harness that benchmarks the RL agent
> against buy-and-hold and random baselines under identical transaction costs,
> scoring all three on Sharpe, Sortino, Calmar and max-drawdown, with equity-curve
> and drawdown plots. It surfaced that the agent beats buy-and-hold on Sharpe in
> low-volatility regimes but underperforms in high-volatility ones — which I
> report honestly instead of cherry-picking."

Demonstrating you know *when the model fails* is the part that makes a reviewer
trust the rest.
