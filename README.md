# RL Trading Agent

## Environment 

The notebook uses a custom stock trading environment which extends StockTradingEnv of the open source library FinRL which is built on top of OpenAI Gym. It simulates a market where an agent interacts by making trading decisions. 

Stable Baselines3: A collection of RL algorithms.

yfinance: Fetches stock data from Yahoo Finance.

# Action

Actions involve placing trades such as buying, selling, or holding stocks. More formally, we define a singular Action as: 

A∈[-1,1]

which is scaled on basis of hmax i.e. no. of stocks the agent can purchase or sell at a time. 

hmax=1initial_amount / max_price ⌋

$s t o c k \underline { } \div \min = n o .$ of different stocks the agent can trade in

Action Space:

$A = s t o c k \underline { d i m ^ { * } } c A$

Transaction Costs: Buy/Sell costs are 0.1% per transaction to penalize the model for transactions similar to real-life scenarios. 

Turbulence Threshold: Helps prevent excessive trading in high-volatility situations. 

## State Space

state_space= 1 + 2 * stock_dim + len(indicators) * stock_dim

where: 

• 1 represents the portfolio value.

• 2 * stock_dim represents stock holdings and their prices.

• len(indicators) * stock_dim represents the number of technical indicators per stock. 

Technical Indicators Used:

## • volume 

Represents the number of shares or contracts traded in a given period. High volume often indicates strong interest and potential price movement. 

• macd (Moving Average Convergence Divergence)

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



