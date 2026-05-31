"""
Experimental Single Stock Trading Environment Training Script
Converted from FinRLSingleStockTradingEnviroment.ipynb
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.config import INDICATORS
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import Logger, KVWriter, CSVOutputFormat
from gymnasium import spaces
import torch

warnings.filterwarnings("ignore")

class CustomStockTradingEnv(StockTradingEnv):
    """Custom Stock Trading Environment with advanced reward functions"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = Logger('results', [CSVOutputFormat])
        self.reward_function = None

        # action space for equal distribution of BUY/SELL and HOLD
        action_dim = 1 + kwargs["action_space"]
        print("Action Space =", action_dim)
        self.action_space = spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=float)

    def set_reward_function(self, reward_function):
        self.reward_function = reward_function

    def step(self, actions):
        hold = actions[0]
        purchase = actions[1:]
        if hold < 0:
            purchase[:] = 0

        # step
        next_state, reward, terminal, truncated, info = super().step(purchase)

        # custom reward function
        if self.reward_function is not None:
            reward = self.reward_function(self, actions, next_state, reward, terminal, truncated, info)
        
        return next_state, reward, terminal, truncated, info


def sample_custom_reward(self, actions, next_state, base_reward, terminal, truncated, info, reward_logging=False):
    """Custom reward function for RL trading"""
    
    # Convert memory to DataFrame
    df_total_value = pd.DataFrame(self.asset_memory, columns=["account_value"])
    df_total_value["date"] = self.date_memory
    df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)
    
    # Get benchmark values from the 'close_benchmark' column
    df_total_value["benchmark_value"] = self.df["close_benchmark"].iloc[:len(df_total_value)].reset_index(drop=True)
    df_total_value["benchmark_daily_return"] = df_total_value["benchmark_value"].pct_change(1)
    
    # Remove NaN values
    remove_nan = lambda x: 0 if np.isnan(x) else x
    
    # Calculate portfolio returns and risks
    mean_returns = df_total_value["daily_return"].mean()
    std_returns = df_total_value["daily_return"].std()
    
    # Calculate benchmark returns and risks
    bench_returns = df_total_value["benchmark_daily_return"].mean()
    bench_std = df_total_value["benchmark_daily_return"].std()
    
    # Calculate beta
    beta = 1.0  # default value
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
    
    # Compute final reward with all components
    total_reward = remove_nan(sortino)
    if total_reward == 0: 
        total_reward = -100  # drive towards transactions
    
    if reward_logging:
        print(f"Reward: {total_reward}")
        print("-----------------------------")
    
    return total_reward


def load_and_prepare_data(ticker="GOOGL", start_date="2015-01-01", end_date="2025-01-01"):
    """Load and prepare stock data with technical indicators"""
    
    benchmark_ticker = "^GSPC"  # S&P 500
    
    # Download data
    df_stock = YahooDownloader(start_date=start_date, end_date=end_date, ticker_list=[ticker]).fetch_data()
    df_benchmark = YahooDownloader(start_date=start_date, end_date=end_date, ticker_list=[benchmark_ticker]).fetch_data()
    
    # Merge with benchmark
    df = pd.merge(df_stock, df_benchmark[['date', 'close']], on='date', suffixes=('', '_benchmark'))
    
    # Add technical indicators
    fe = FeatureEngineer(
        use_technical_indicator=True, 
        tech_indicator_list=INDICATORS,
        use_turbulence=True
    )
    df = fe.preprocess_data(df)
    
    return df


def plot_stock_data(df, ticker):
    """Plot stock and benchmark data"""
    
    # Plot stock closing price over time
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=pd.to_datetime(df['date']), y=df["close"], label="Closing Price", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Stock Price (USD)")
    plt.title(f"{ticker} Closing Stock Over Time")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.show()

    # Plot benchmark data over time
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=pd.to_datetime(df['date']), y=df["close_benchmark"], label="Closing Price Benchmark", color="green")
    plt.xlabel("Date")
    plt.ylabel("Stock Price (USD)")
    plt.title("S&P 500 Over Time")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.show()


def create_environment(df):
    """Create and configure the trading environment"""
    
    indicators = ["volume", "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma", "turbulence"]
    turbulence_threshold = 100

    stock_dim = len(df["tic"].unique())
    max_price = df['close'].max()
    initial_amount = 10000
    hmax = int(initial_amount / max_price)

    env = CustomStockTradingEnv(
        df=df, 
        stock_dim=stock_dim, 
        hmax=hmax,
        initial_amount=initial_amount, 
        num_stock_shares=[0],
        make_plots=True,
        print_verbosity=1,
        buy_cost_pct=[0.1],
        sell_cost_pct=[0.1],
        turbulence_threshold=turbulence_threshold,
        reward_scaling=1e-4,
        tech_indicator_list=indicators,
        state_space=1 + 2 * stock_dim + len(indicators) * stock_dim,
        action_space=stock_dim,
    )

    env.set_reward_function(sample_custom_reward)
    os.makedirs("results", exist_ok=True)
    
    return env


def train_model(env, total_timesteps=100000):
    """Train the PPO model"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env.reset()
    vec_env = make_vec_env(lambda: Monitor(env, '/'), n_envs=1)

    model = PPO("MlpPolicy", vec_env, verbose=1, device=device)
    model.learn(total_timesteps=total_timesteps)
    
    return model, vec_env


def evaluate_model(model, vec_env):
    """Evaluate the trained model"""
    
    obs = vec_env.reset()
    vec_env.envs[0].unwrapped.episode = 0

    portfolio_values = []
    timesteps = []

    i = 0
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = vec_env.step(action)

        if done[0]:
            break

        portfolio_value = vec_env.envs[0].unwrapped.asset_memory[-1]
        portfolio_values.append(portfolio_value)
        timesteps.append(i)
        i += 1
    
    return portfolio_values, timesteps


def plot_results(portfolio_values, timesteps):
    """Plot training and evaluation results"""
    
    # Plot portfolio performance
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, portfolio_values, label="Portfolio Value", color='blue')
    plt.xlabel("Time Steps")
    plt.ylabel("Portfolio Value ($)")
    plt.title("RL Model Performance")
    plt.legend()
    plt.grid()
    plt.show()
    
    # Plot training logs if available
    try:
        logs = pd.read_csv("/monitor.csv", skiprows=1)
        logs.columns = ["reward", "episode_length", "timesteps"]
        logs["reward_smooth"] = logs["reward"].rolling(window=10).mean()

        plt.figure(figsize=(12, 6))
        plt.plot(logs["timesteps"], logs["reward_smooth"], label="Smoothed Reward")
        plt.xlabel("Timesteps")
        plt.ylabel("Reward")
        plt.title("Smoothed Episode Reward Over Time")
        plt.legend()
        plt.show()
    except:
        print("Training logs not available for plotting")


def main():
    """Main execution function"""
    
    # Configuration
    ticker = "GOOGL"
    start_date = "2015-01-01"
    end_date = "2025-01-01"
    total_timesteps = 100000
    
    print(f"Training RL model for {ticker}")
    
    # Load and prepare data
    df = load_and_prepare_data(ticker, start_date, end_date)
    print(f"Loaded {len(df)} data points")
    
    # Plot data (optional)
    # plot_stock_data(df, ticker)
    
    # Create environment
    env = create_environment(df)
    print("Environment created successfully")
    
    # Train model
    model, vec_env = train_model(env, total_timesteps)
    print("Model training completed")
    
    # Evaluate model
    portfolio_values, timesteps = evaluate_model(model, vec_env)
    print("Model evaluation completed")
    
    # Plot results
    plot_results(portfolio_values, timesteps)
    
    # Save model
    model_name = f"{ticker}_experimental_model"
    model.save(model_name)
    print(f"Model saved as {model_name}")


if __name__ == "__main__":
    main()
