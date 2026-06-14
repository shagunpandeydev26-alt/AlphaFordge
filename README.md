# AlphaFordge — RL Trading Agent

A modular reinforcement-learning trading system built on
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) and
[FinRL](https://github.com/AI4Finance-Foundation/FinRL). It provides a complete
pipeline for **training**, **evaluating**, **backtesting**, and **serving** PPO
agents for single-stock trading — with an out-of-sample backtesting harness that
benchmarks the agent against buy-and-hold and random baselines under identical
transaction costs.

![tests](https://github.com/shagunpandeydev26-alt/AlphaFordge/actions/workflows/tests.yml/badge.svg)

---

## Highlights

- **Modular architecture** — clean separation of environments, agents, rewards,
  data, training, inference, and UI.
- **PPO agent** with configurable hyperparameters (Stable-Baselines3).
- **Pluggable reward functions** — differential return, Sortino, risk-adjusted,
  drawdown-penalty, and transaction-cost-aware.
- **Out-of-sample backtesting** — runs the agent, buy-and-hold, and random
  strategies over the same window with the same starting cash and the same
  0.1%/side transaction costs, then scores all three on Sharpe, Sortino, Calmar,
  max-drawdown and more.
- **Equity-curve + drawdown plots** and a comparison table.
- **Interfaces**: a Streamlit web app, a FastAPI service, and a CLI.
- **Tested**: an offline `pytest` suite runs in CI on every push.

---

## Installation

```bash
git clone https://github.com/shagunpandeydev26-alt/AlphaFordge.git
cd AlphaFordge

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Reproducible, pinned dependency set (recommended):
pip install -r requirements-ci.txt
```

> The pins in `requirements-ci.txt` encode a few non-obvious compatibility
> constraints (numpy 1.26.x for the scikit-learn ABI; yfinance 0.2.52 vs. the
> alpaca/websockets conflict pulled in transitively by FinRL). Use them to avoid
> dependency headaches.

---

## Quick start

### Backtest the agent vs. baselines (the headline feature)

```bash
python -m src.inference.backtest --ticker NFLX --start 2023-01-01 --end 2024-12-31
```

This prints a performance report for the **RL agent**, **buy-and-hold**, and
**random** strategies, and writes `results/backtest_NFLX.json` (metrics + equity
curves). Seven pre-trained models ship in `models/` (AVGO, COST, GOOGL, NET,
NFLX, RELIANCE.NS, TSLA).

### Train a model

```bash
python src/main.py train --ticker AAPL --start_date 2018-01-01 --end_date 2023-01-01 --config production
```

Training also writes a sidecar `models/<TICKER>.json` recording the training
window and hyperparameters, which the backtester uses to verify a test window is
genuinely out-of-sample.

### Evaluate / predict

```bash
python src/main.py evaluate --ticker AAPL --model_path models/AAPL.zip --start_date 2023-01-01 --end_date 2024-01-01
python src/main.py predict  --ticker AAPL --model_path models/AAPL.zip --portfolio_value 10000 --num_shares 0
```

### Web app

```bash
streamlit run streamlit_app.py
```

Includes a **"📊 Backtest vs. Baselines"** section: pick a ticker + test window
and see the comparison table and equity/drawdown plot.

### API

```bash
python -m uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
# POST /predict, /predict/batch, /evaluate/{ticker}, GET /models, /health
```

---

## How the backtester works

The core question a trading model must answer is: *is it actually better than
just buying and holding the stock?* The `Backtester`
([`src/inference/backtest.py`](src/inference/backtest.py)) answers it honestly:

| Strategy | What it does | Why it's the right comparison |
|---|---|---|
| **RL agent** | The trained model's actions | The thing being evaluated |
| **Buy & hold** | Buy on day 1, pay the buy cost once, hold | The "do nothing smart" baseline |
| **Random** | Random actions under a fixed seed | "Is it learning, or just lucky?" sanity check |

All three start with the same cash and pay the same transaction costs. Buy-and-hold
is fed as the benchmark series so beta, Treynor, information ratio, and excess
return populate against the obvious alternative. The harness also:

- **Enforces out-of-sample testing** — raises if the test window overlaps the
  model's recorded training dates (and warns if no metadata is available).
- **Refuses fabricated data** — aborts if the environment's dummy-data fallback
  was used (e.g. a failed download), so a "backtest" on simulated prices can't
  pass silently.

---

## Project structure

```
AlphaFordge/
├── src/
│   ├── agents/        # PPO agent wrapper
│   ├── data/          # data loading + feature engineering
│   ├── envs/          # SingleStockTradingEnv (FinRL-based)
│   ├── inference/     # inference engine, FastAPI, backtester
│   ├── rewards/       # pluggable reward functions
│   ├── train/         # training pipeline + config
│   ├── utils/         # metrics, plotting, logging
│   └── main.py        # CLI entry point
├── tests/             # offline pytest suite
├── models/            # trained models + sidecar metadata
├── streamlit_app.py   # web UI
├── requirements-ci.txt
└── .github/workflows/ # CI
```

---

## Testing

```bash
pytest
```

The suite is fully offline (in-memory fixtures, no network) and covers the
metrics, the buy-and-hold math, random-seed determinism, the out-of-sample guard,
and fabricated-data detection. It runs in GitHub Actions on every push and PR.

---

## Disclaimer

This software is for educational and research purposes only. Past performance
does not guarantee future results. Nothing here is financial advice — always do
your own research before making investment decisions.
