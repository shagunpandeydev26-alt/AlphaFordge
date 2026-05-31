# RL Trading Agent - Modular Reinforcement Learning Trading System

A comprehensive, modular reinforcement learning trading system built with stable-baselines3 and FinRL. This project provides a complete pipeline for training, evaluating, and deploying RL agents for algorithmic trading.

## 🚀 Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for environments, agents, training, inference, and utilities
- **Multiple RL Algorithms**: Support for PPO and other stable-baselines3 algorithms
- **Flexible Reward Functions**: Extensible reward system with profit-based, Sharpe ratio, and risk-adjusted rewards
- **Comprehensive Evaluation**: Built-in backtesting, performance metrics, and comparison tools
- **Web Interface**: Streamlit-based UI for easy interaction and visualization
- **RESTful API**: FastAPI-based service for programmatic access
- **Feature Engineering**: Advanced technical indicators and market regime detection
- **Multiple Environments**: Single-stock, multi-asset, and continuous trading environments

## 📁 Project Structure

```
RLTradingAgent/
├── src/                          # Main source code
│   ├── agents/                   # RL agent implementations
│   ├── data/                     # Data loading and feature engineering
│   ├── envs/                     # Trading environments
│   ├── inference/                # Model inference and API
│   ├── rewards/                  # Reward function implementations
│   ├── train/                    # Training and evaluation
│   ├── ui/                       # Streamlit web interface
│   ├── utils/                    # Utilities and metrics
│   └── main.py                   # CLI entry point
├── scripts/                      # Converted notebooks and utilities
├── notebooks/                    # Original Jupyter notebooks
├── models/                       # Trained model storage
├── data/                         # Data storage
├── logs/                         # Training and application logs
├── results/                      # Evaluation results
└── requirements.txt              # Python dependencies
```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd RLTradingAgent
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup project structure**
```bash
python scripts/setup.py
```

## 🚀 Quick Start

### Training a Model

```bash
# Train a PPO agent for Apple stock
python src/main.py train --ticker AAPL --start-date 2020-01-01 --end-date 2023-01-01

# Train with custom parameters
python src/main.py train --ticker AAPL --total-timesteps 100000 --learning-rate 0.0003
```

### Running Evaluation

```bash
# Evaluate a trained model
python src/main.py evaluate --ticker AAPL --start-date 2023-01-01 --end-date 2024-01-01

# Compare multiple models
python src/main.py compare --tickers AAPL GOOGL MSFT --start-date 2023-01-01
```

### Starting the Web Interface

```bash
streamlit run src/ui/app.py
```

### Starting the API Server

```bash
python src/inference/api.py
```

## 📊 Environment Details

### Action Space
Actions represent trading decisions in continuous space [-1, 1]:
- **-1**: Sell maximum allowed
- **0**: Hold position  
- **1**: Buy maximum allowed

Actions are scaled by `hmax` (maximum holdings):
```
hmax = initial_amount / max_price
```

### State Space
The state includes:
- Portfolio value (1 dimension)
- Stock holdings and prices (2 × stock_dim)  
- Technical indicators (len(indicators) × stock_dim)

**State Space Size**: `1 + 2 × stock_dim + len(indicators) × stock_dim`

### Technical Indicators
- **MACD**: Moving Average Convergence Divergence
- **RSI**: Relative Strength Index
- **Bollinger Bands**: Price volatility bands
- **Volume**: Trading volume indicators
- **SMA/EMA**: Simple and Exponential Moving Averages
- **ATR**: Average True Range
- **Stochastic Oscillator**: Momentum indicator

### Reward Functions
1. **Profit Reward**: Direct portfolio value changes
2. **Sharpe Ratio Reward**: Risk-adjusted returns
3. **Drawdown Penalty**: Penalizes large losses
4. **Transaction Cost Aware**: Considers trading costs
5. **Risk Adjusted**: Balances returns with risk metrics

## 🎯 Usage Examples

### Custom Training Configuration

```python
from src.train import TradingTrainer, TrainingConfig
from src.rewards import SharpeRatioReward

config = TrainingConfig(
    ticker="AAPL",
    start_date="2020-01-01",
    end_date="2023-01-01",
    total_timesteps=100000,
    learning_rate=0.0003,
    reward_function=SharpeRatioReward()
)

trainer = TradingTrainer(config)
model = trainer.train()
```

### Programmatic Inference

```python
from src.inference import TradingInferenceEngine
from stable_baselines3 import PPO

# Load model and create inference engine
model = PPO.load("models/AAPL.zip")
engine = TradingInferenceEngine()

# Get prediction
action = engine.predict_action(model, env, portfolio_value=10000, num_shares=10)
```

### Custom Reward Function

```python
from src.rewards import BaseRewardFunction

class CustomReward(BaseRewardFunction):
    def calculate_reward(self, data: dict) -> float:
        portfolio_value = data['portfolio_value']
        benchmark_return = data.get('benchmark_return', 0)
        
        # Custom reward logic
        return portfolio_value * 0.1 - abs(benchmark_return) * 0.05
```

## 📈 API Usage

### Start API Server
```bash
python src/inference/api.py
```

### Make Predictions
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "ticker": "AAPL",
       "portfolio_value": 10000,
       "num_shares": 10
     }'
```

### Batch Evaluation
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "ticker": "AAPL", 
       "start_date": "2023-01-01",
       "end_date": "2023-12-31",
       "initial_amount": 10000
     }'
```

## 🧪 Testing and Evaluation

### Run Backtests
```bash
# Single model backtest
python src/main.py backtest --ticker AAPL --model-path models/AAPL.zip

# Cross-validation
python src/main.py cv --ticker AAPL --folds 5
```

### Generate Reports
```bash
# Comprehensive evaluation report
python src/main.py report --ticker AAPL --output-dir results/
```

## 📊 Performance Metrics

The system tracks comprehensive performance metrics:

- **Returns**: Total, annualized, and risk-adjusted returns
- **Risk Metrics**: Sharpe ratio, Sortino ratio, maximum drawdown
- **Trading Metrics**: Win rate, profit factor, trading frequency
- **Benchmark Comparison**: Alpha, beta, tracking error
- **Portfolio Metrics**: Volatility, VaR, Calmar ratio

## 🛡️ Risk Management

- **Transaction Costs**: 0.1% per transaction (configurable)
- **Position Limits**: Maximum position sizing controls
- **Drawdown Limits**: Automatic position reduction on large losses
- **Volatility Filters**: Reduced trading during high volatility periods

## 🔧 Configuration

### Environment Variables
```bash
export MODELS_DIR="./models"
export DATA_DIR="./data" 
export LOG_LEVEL="INFO"
```

### Training Configuration
```yaml
# config.yaml
training:
  total_timesteps: 100000
  learning_rate: 0.0003
  batch_size: 64
  n_steps: 2048

environment:
  initial_amount: 10000
  transaction_cost_pct: 0.001
  turbulence_threshold: 140

features:
  technical_indicators: true
  time_features: true
  market_regime: true
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FinRL](https://github.com/AI4Finance-Foundation/FinRL) for the trading environment foundation
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for RL algorithms
- [yfinance](https://github.com/ranaroussi/yfinance) for market data
- [ta](https://github.com/bukosabino/ta) for technical indicators

## 📞 Support

For questions and support:
- Create an issue in the GitHub repository
- Check the [documentation](docs/) 
- Review existing [discussions](../../discussions)

---

**Disclaimer**: This software is for educational and research purposes only. Past performance does not guarantee future results. Always conduct your own research before making investment decisions.

A trend-following momentum indicator that shows the relationship between two moving averages. It helps identify potential buy or sell signals based on crossovers and divergence.

$M A C D = E M A _ { 1 2 } - E M A _ { 2 6 }$

• boll_ub (Bollinger Upper Band)

The upper boundary of Bollinger Bands, which represents a price level where an asset may be overbought. It is calculated using a moving average and standard deviation. 

Bollinger Upper Band = SMA + k × σ

• boll_lb (Bollinger Lower Band)

The lower boundary of Bollinger Bands, indicating a price level where an asset may be oversold.It helps traders identify potential buying opportunities. 

## Bollinger Lower Band = SMA −k × σ

• rsi_30 (Relative Strength Index - 30 period)

A momentum oscillator that measures the speed and change of price movements. It ranges from 0 to 100, with values below 30 indicating oversold conditions and potential reversals. 

• cci_30 (Commodity Channel Index - 30 period)

A momentum-based indicator that identifies price trends and overbought/oversold conditions.Positive values suggest bullish momentum, while negative values indicate bearish trends. 

$d x \underline { } 3 0$(Directional Movement Index - 30 period)

Measures trend strength by comparing upward and downward movement. Higher values indicate a stronger trend, while lower values suggest weak or no trend. 

• close_30_sma (30-period Simple Moving Average of Close Price)

The average closing price over 30 periods. It smooths price fluctuations to help identify trends and potential support/resistance levels. 

• close_60_sma (60-period Simple Moving Average of Close Price)

Similar to the 30-period SMA but over a longer period, providing a broader view of price trends and reducing short-term noise. 

## • turbulence 

A measure of market volatility and instability. Higher turbulence values indicate unpredictable price movements, which can signal potential risk or upcoming market shifts. 

# Evaluation Parameters

## • Sharpe Ratio

The Sharpe Ratio measures the risk-adjusted return of an investment by comparing its excess return over the risk-free rate to the standard deviation of its returns. A higher Sharpe Ratio indicates better risk-adjusted performance. 

Sharp$e R a t i o = \frac { A v e r a g e R e t u r n - R i s k - F r e e e R t e } { s \tan d a r d D e v i a t i o n o f R e t u r n }$

## • Sortino Ratio

The Sortino Ratio is similar to the Sharpe Ratio but only considers downside risk (volatility of negative returns) rather than total volatility. It is a more refined measure of risk-adjusted return as it focuses on the harmful part of risk. 

$S o r t i n o R a t i o = \frac { A v e r a g e R e t u r n - R i s k \cdot F r e e R a t e } { D o w n s i d e D e v i a t i o n }$

# Reward Functions

## PnL (Profit & Loss)

Simple reward function that measures the total profit/loss or return obtained on day. Leads to issues such as early convergence of model.

Reward = Current Portfolio Value – Previous Portfolio Value


| day: 2515, episode: 0  |
| -- |
| begin_total_asset: 10000.00  |
| end_total_asset: 67467.96  |
| total_reward: 57467.96  |
| total_cost: 907.57  |
| total_trades: 2515  |
| Sharpe: 0.811  |



![](https://web-api.textin.com/ocr_image/external/f07f35cad64f1308.jpg)

# Moving Average of Return 

Smooths the return over a period to reduce volatility.

$R e w a r d = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } F$Return

## Custom Reward Function

Balances return with risk using hyperparameters α and β

Reward = α×return_moving_average − β×downside_return

where α,β are hyperparameters that control the weight of return and risk.

## Differential Return 

https://www.researchgate.net/publication/356127405_Portfolio_Performance_and Risk_Penalty_Measurement_with_Differential_Return 

Algorithm Used:

Proximal Policy Optimization (PPO)



