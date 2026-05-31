from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.config import INDICATORS
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

import numpy as np
import pandas as pd

class SingleStockTradingEnv(StockTradingEnv):
    def __init__(self, **kwargs):
        """
        Arguments:
        - ticker: str (optional) - Stock ticker symbol.
        - df: pd.DataFrame (optional) - Directly pass a DataFrame.
        - start_date: str (optional) - Start date for data fetching.
        - end_date: str (optional) - End date for data fetching.
        - hmax: int (default=100) - Max number of shares per trade.
        - initial_amount: int (default=10000) - Initial cash amount.
        - num_stock_shares: list (default=[0]) - Initial stock holdings.
        - Additional arguments are passed to the superclass.
        """

        ticker = kwargs.get("ticker", None)
        df = kwargs.get("df", None)
        start_date = kwargs.get("start_date", None)
        end_date = kwargs.get("end_date", None)

        # Fetch data if ticker is provided
        if ticker is not None:
            df = self.get_data(ticker, start_date, end_date)

        if df is None:
            raise ValueError("Either 'ticker' or 'df' must be provided.")

        indicators = [
            "volume", "macd", "boll_ub", "boll_lb", "rsi_30",
            "cci_30", "dx_30", "close_30_sma", "close_60_sma", "turbulence"
        ]

        # Default values for optional params
        default_params = {
            "stock_dim": 1,
            "hmax": 100,
            "initial_amount": 10000,
            "num_stock_shares": [0],
            "print_verbosity": 1,
            "buy_cost_pct": [0.001],
            "sell_cost_pct": [0.001],
            "turbulence_threshold": 100,
            "reward_scaling": 1e-4,
            "tech_indicator_list": indicators,
            "state_space": 3 + len(indicators),
            "action_space": 1,
        }
        default_params.update(kwargs)
        default_params.pop("ticker", None)
        default_params.pop("start_date", None)
        default_params.pop("end_date", None)
        default_params.pop("df", None)

        # Initialize parent class
        super().__init__(df=df, **default_params)

        self.reward_weights = [0.1, 0.01, 0.01, 1.0]


    def get_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        # Get data from Yahoo Finance
        benchmark_ticker = "^GSPC" # S&P 500

        df_s = YahooDownloader(start_date=start_date, end_date=end_date, ticker_list=[ticker]).fetch_data()
        df_benchmark = YahooDownloader(start_date=start_date, end_date=end_date, ticker_list=[benchmark_ticker]).fetch_data()

        df = pd.merge(df_s, df_benchmark[['date', 'close']], on='date', suffixes=('', '_benchmark'))

        # Add technical indicators and turbulence
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=INDICATORS,
            use_turbulence=True
        )
        return fe.preprocess_data(df)

    def step(self, actions):
        next_state, reward, terminal, truncated, info = super().step(actions)

        reward = self.reward_function(self, actions, next_state, reward, terminal, truncated, info)
        
        return next_state, reward, terminal, truncated, info

    def reward_function(self, actions, next_state, base_reward, terminal, truncated, info, *, reward_logging=False):
        df_total_value = pd.DataFrame(self.asset_memory, columns=["account_value"])
        df_total_value["date"] = self.date_memory
        df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)
        
        df_total_value["benchmark_value"] = self.df["close_benchmark"].iloc[:len(df_total_value)].reset_index(drop=True)
        df_total_value["benchmark_daily_return"] = df_total_value["benchmark_value"].pct_change(1)

        remove_nan = lambda x: 0 if np.isnan(x) else x
        
        # Compute mean daily returns and benchmark returns
        mean_returns = df_total_value["daily_return"].mean()
        std_returns = df_total_value["daily_return"].std()
        bench_returns = df_total_value["benchmark_daily_return"].mean()
        bench_std = df_total_value["benchmark_daily_return"].std()
        
        # Compute Beta
        beta = 1.0  
        if bench_std and not np.isnan(bench_std) and bench_std != 0:
            portfolio_returns = df_total_value["daily_return"].fillna(0)
            benchmark_returns = df_total_value["benchmark_daily_return"].fillna(0)
            if len(portfolio_returns) == len(benchmark_returns):
                covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
                beta = covariance / (bench_std ** 2)
        
        # Compute Sharpe Ratio
        sharpe = 0
        if std_returns and not np.isnan(std_returns):
            sharpe = (252**0.5) * mean_returns / std_returns
        
        # Compute Sortino Ratio
        downside_returns = df_total_value["daily_return"][df_total_value["daily_return"] < 0]
        downside_std = downside_returns.std(ddof=1)
        
        sortino = 0
        if downside_std and not np.isnan(downside_std):
            sortino = (252**0.5) * mean_returns / downside_std
        
        # Compute Treynor Ratio
        treynor = 0
        if beta and not np.isnan(beta) and beta != 0:
            treynor = (252**0.5) * mean_returns / beta
        
        # Compute Differential Return
        diff_return = 0
        if beta and not np.isnan(beta) and beta != 0:
            diff_return = (mean_returns - bench_returns) / beta

        # Weights obtained by grid search
        w1, w2, w3, w4 = self.reward_weights
        total_reward = (
            w1 * remove_nan(mean_returns) 
            - w2 * remove_nan(abs(downside_std)) 
            + w3 * remove_nan(treynor) 
            + w4 * remove_nan(diff_return)
        )
        
        if reward_logging:
            print(f"Mean Daily Returns: {mean_returns}")
            print(f"Benchmark Returns: {bench_returns}")
            print(f"Daily Return Standard Deviation: {std_returns}")
            print(f"Downside Only Standard Deviation: {downside_std}")
            print(f"Beta: {beta}")
            print(f"Sharpe Ratio: {sharpe}")
            print(f"Sortino Ratio: {sortino}")
            print(f"Treynor Ratio: {treynor}")
            print(f"Differential Return: {diff_return}")
            print(f"Reward: {total_reward}")
            print("-----------------------------")
        
        return total_reward