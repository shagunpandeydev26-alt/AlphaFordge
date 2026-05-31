from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.config import INDICATORS
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from src.rewards.reward_function import get_reward_function

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
        - reward_type: str (default="differential") - Type of reward function to use.
        - reward_weights: list (optional) - Custom weights for reward function.
        - Additional arguments are passed to the superclass.
        """

        ticker = kwargs.get("ticker", None)
        df = kwargs.get("df", None)
        start_date = kwargs.get("start_date", None)
        end_date = kwargs.get("end_date", None)
        reward_type = kwargs.pop("reward_type", "differential")
        reward_weights = kwargs.pop("reward_weights", None)

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

        # Set up reward function
        self.reward_calculator = get_reward_function(reward_type, reward_weights)
        self.reward_weights = reward_weights or [0.1, 0.01, 0.01, 1.0]


    def get_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        # Get data from Yahoo Finance
        benchmark_ticker = "^GSPC" # S&P 500

        try:
            # Try using YahooDownloader
            df_s = YahooDownloader(start_date=start_date, end_date=end_date, ticker_list=[ticker]).fetch_data()
            df_benchmark = YahooDownloader(start_date=start_date, end_date=end_date, ticker_list=[benchmark_ticker]).fetch_data()
        except Exception as e:
            print(f"YahooDownloader failed: {e}. Using fallback yfinance...")
            # Fallback to direct yfinance with better error handling
            import yfinance as yf
            
            try:
                # Get stock data
                stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if stock_data.empty:
                    raise ValueError(f"No data found for ticker {ticker}")
                
                # Format for FinRL compatibility
                df_s = stock_data.reset_index()
                df_s.columns = [col.lower() if isinstance(col, str) else col[0].lower() for col in df_s.columns]
                df_s['tic'] = ticker
                
                # Get benchmark data
                benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date, progress=False)
                df_benchmark = benchmark_data.reset_index()
                df_benchmark.columns = [col.lower() if isinstance(col, str) else col[0].lower() for col in df_benchmark.columns]
                df_benchmark = df_benchmark[['date', 'close']].rename(columns={'close': 'close_benchmark'})
                
                # Merge with benchmark
                df = pd.merge(df_s, df_benchmark, on='date', how='left')
                
            except Exception as fallback_error:
                print(f"Fallback yfinance also failed: {fallback_error}")
                # Create realistic dummy dataset if all else fails
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                n_days = len(dates)
                
                # Generate realistic price movements
                np.random.seed(42)  # For reproducibility
                base_price = 100.0
                price_changes = np.random.normal(0, 0.02, n_days)  # 2% daily volatility
                prices = [base_price]
                
                for change in price_changes[1:]:
                    new_price = prices[-1] * (1 + change)
                    prices.append(max(new_price, 1.0))  # Ensure positive prices
                
                # Create OHLCV data with realistic patterns
                opens = prices[:-1] + [prices[-1]]
                closes = prices
                
                # Highs and lows with some randomness
                highs = [max(o, c) * (1 + np.random.uniform(0, 0.01)) for o, c in zip(opens, closes)]
                lows = [min(o, c) * (1 - np.random.uniform(0, 0.01)) for o, c in zip(opens, closes)]
                
                # Volumes with some variation
                volumes = [1000000 * (1 + np.random.uniform(-0.3, 0.3)) for _ in range(n_days)]
                
                df = pd.DataFrame({
                    'date': dates,
                    'open': opens,
                    'high': highs,
                    'low': lows,
                    'close': closes,
                    'volume': volumes,
                    'tic': ticker
                })
                print(f"Using realistic dummy data for {ticker} with {n_days} days")
        
        if 'df' not in locals():
            # If YahooDownloader succeeded, merge the data
            df = pd.merge(df_s, df_benchmark[['date', 'close']], on='date', suffixes=('', '_benchmark'))

        # Ensure required columns exist
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic']
        for col in required_cols:
            if col not in df.columns:
                if col in ['open', 'high', 'low', 'close']:
                    df[col] = 100.0  # Default price
                elif col == 'volume':
                    df[col] = 1000000  # Default volume
                elif col == 'tic':
                    df[col] = ticker

        # Add technical indicators and turbulence with error handling
        try:
            fe = FeatureEngineer(
                use_technical_indicator=True,
                tech_indicator_list=INDICATORS,
                use_turbulence=True
            )
            processed_df = fe.preprocess_data(df)
            
            # Handle NaN values that might occur in technical indicators
            if processed_df.isnull().any().any():
                print("Warning: NaN values detected in processed data. Filling with forward/backward fill...")
                processed_df = processed_df.ffill().bfill()
                
                # If still NaN, fill with reasonable defaults
                if processed_df.isnull().any().any():
                    print("Warning: Still NaN values after forward/backward fill. Using default values...")
                    # Fill remaining NaNs with column means or zeros
                    for col in processed_df.columns:
                        if processed_df[col].isnull().any():
                            if col in ['macd', 'rsi_30', 'cci_30', 'dx_30']:
                                processed_df[col] = processed_df[col].fillna(0)  # Technical indicators default to 0
                            elif col in ['boll_ub', 'boll_lb', 'close_30_sma', 'close_60_sma']:
                                processed_df[col] = processed_df[col].fillna(processed_df['close'].mean())  # Price-based indicators use mean close
                            elif col == 'turbulence':
                                processed_df[col] = processed_df[col].fillna(0)  # Turbulence defaults to 0
                            else:
                                processed_df[col] = processed_df[col].fillna(0)  # Everything else defaults to 0
            
            return processed_df
            
        except Exception as fe_error:
            print(f"FeatureEngineer failed: {fe_error}. Using basic data...")
            # Return basic processed data without technical indicators but with required structure
            df = df.sort_values(['date', 'tic']).reset_index(drop=True)
            
            # Add missing technical indicator columns with default values
            default_indicators = {
                'macd': 0,
                'boll_ub': df['close'],
                'boll_lb': df['close'],
                'rsi_30': 50,  # RSI neutral value
                'cci_30': 0,
                'dx_30': 0,
                'close_30_sma': df['close'],
                'close_60_sma': df['close'],
                'turbulence': 0
            }
            
            for indicator, default_value in default_indicators.items():
                if indicator not in df.columns:
                    df[indicator] = default_value
            
            return df

    def step(self, actions):
        next_state, reward, terminal, truncated, info = super().step(actions)

        reward = self.reward_calculator.calculate(self, actions, next_state, reward, terminal, truncated, info)
        
        return next_state, reward, terminal, truncated, info

    def reward_function(self, actions, next_state, base_reward, terminal, truncated, info, *, reward_logging=False):
        """Legacy reward function - now delegates to reward calculator"""
        return self.reward_calculator.calculate(self, actions, next_state, base_reward, terminal, truncated, info, reward_logging)