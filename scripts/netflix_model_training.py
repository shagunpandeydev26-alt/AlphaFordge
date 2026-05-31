"""
Netflix (NFLX) Trading Model Training Script
Converted from Netflix.ipynb
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
from src.envs import SingleStockTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import torch

warnings.filterwarnings("ignore")

def load_netflix_data(start_date="2015-01-01", end_date="2025-01-01"):
    """Load and prepare Netflix data with technical indicators"""
    
    ticker = "NFLX"
    benchmark_ticker = "^GSPC"  # S&P 500
    
    print(f"Loading data for {ticker} from {start_date} to {end_date}")
    
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

def plot_netflix_data(df):
    """Plot Netflix stock and benchmark data"""
    
    ticker = "NFLX"
    benchmark_ticker = "^GSPC"
    
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
    plt.title(f"{benchmark_ticker} Over Time")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.show()

def create_netflix_environment(df):
    """Create and configure the Netflix trading environment"""
    
    # Use the optimized weights found through grid search
    env = SingleStockTradingEnv(
        df=df,
        hmax=100,
        initial_amount=10000,
        num_stock_shares=[0],
        print_verbosity=1,
        buy_cost_pct=[0.001],
        sell_cost_pct=[0.001],
        turbulence_threshold=100,
        reward_scaling=1e-4
    )
    
    return env

def train_netflix_model(env, total_timesteps=100000):
    """Train the PPO model for Netflix"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env.reset()
    vec_env = make_vec_env(lambda: Monitor(env, './'), n_envs=1)

    model = PPO("MlpPolicy", vec_env, verbose=1, device=device)
    
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    return model, vec_env

def evaluate_netflix_model(model, vec_env):
    """Evaluate the trained Netflix model"""
    
    print("Evaluating model performance...")
    
    obs = vec_env.reset()
    vec_env.envs[0].unwrapped.episode = 0

    portfolio_values = []
    actions_taken = []
    timesteps = []

    i = 0
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = vec_env.step(action)

        if done[0]:
            break

        portfolio_value = vec_env.envs[0].unwrapped.asset_memory[-1]
        portfolio_values.append(portfolio_value)
        actions_taken.append(action[0])
        timesteps.append(i)
        i += 1
    
    return portfolio_values, actions_taken, timesteps

def analyze_netflix_performance(portfolio_values, actions_taken, timesteps, df):
    """Analyze the performance of the Netflix trading model"""
    
    # Calculate performance metrics
    initial_value = 10000
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_value) / initial_value * 100
    
    # Calculate benchmark performance
    benchmark_initial = df['close_benchmark'].iloc[0]
    benchmark_final = df['close_benchmark'].iloc[-1]
    benchmark_return = (benchmark_final - benchmark_initial) / benchmark_initial * 100
    
    print(f"Performance Analysis for Netflix (NFLX):")
    print(f"Initial Portfolio Value: ${initial_value:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Benchmark (S&P 500) Return: {benchmark_return:.2f}%")
    print(f"Excess Return: {total_return - benchmark_return:.2f}%")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Portfolio performance
    axes[0, 0].plot(timesteps, portfolio_values, label="Portfolio Value", color='blue')
    axes[0, 0].set_xlabel("Time Steps")
    axes[0, 0].set_ylabel("Portfolio Value ($)")
    axes[0, 0].set_title("Netflix RL Model Portfolio Performance")
    axes[0, 0].legend()
    axes[0, 0].grid()
    
    # Actions taken
    axes[0, 1].plot(timesteps, actions_taken, label="Actions", color='red', alpha=0.7)
    axes[0, 1].set_xlabel("Time Steps")
    axes[0, 1].set_ylabel("Action Value")
    axes[0, 1].set_title("Trading Actions Over Time")
    axes[0, 1].legend()
    axes[0, 1].grid()
    
    # Portfolio vs Benchmark comparison
    # Normalize both to start at 100 for comparison
    portfolio_normalized = [(pv/portfolio_values[0]) * 100 for pv in portfolio_values]
    benchmark_subset = df['close_benchmark'].iloc[:len(portfolio_values)]
    benchmark_normalized = [(bv/benchmark_subset.iloc[0]) * 100 for bv in benchmark_subset]
    
    axes[1, 0].plot(timesteps, portfolio_normalized, label="RL Portfolio", color='blue')
    axes[1, 0].plot(timesteps, benchmark_normalized, label="S&P 500", color='green')
    axes[1, 0].set_xlabel("Time Steps")
    axes[1, 0].set_ylabel("Normalized Value")
    axes[1, 0].set_title("Portfolio vs Benchmark (Normalized)")
    axes[1, 0].legend()
    axes[1, 0].grid()
    
    # Daily returns distribution
    portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
    axes[1, 1].hist(portfolio_returns, bins=30, alpha=0.7, color='blue', density=True)
    axes[1, 1].set_xlabel("Daily Returns")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("Distribution of Portfolio Daily Returns")
    axes[1, 1].grid()
    
    plt.tight_layout()
    plt.show()
    
    # Additional metrics
    portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
    volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized volatility
    sharpe_ratio = (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))
    
    print(f"\nAdditional Metrics:")
    print(f"Annualized Volatility: {volatility:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {((pd.Series(portfolio_values).cummax() - pd.Series(portfolio_values)) / pd.Series(portfolio_values).cummax()).max():.2%}")

def save_netflix_model(model, filename="NFLX"):
    """Save the trained Netflix model"""
    
    # Create models directory if it doesn't exist
    models_dir = Path("../models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / f"{filename}.zip"
    model.save(str(model_path))
    print(f"Model saved to {model_path}")

def main():
    """Main execution function for Netflix model training"""
    
    print("Netflix (NFLX) RL Trading Model Training")
    print("=" * 50)
    
    # Configuration
    start_date = "2015-01-01"
    end_date = "2025-01-01"
    total_timesteps = 100000
    
    # Load and prepare data
    df = load_netflix_data(start_date, end_date)
    print(f"Loaded {len(df)} data points for Netflix")
    
    # Optional: Plot data for visual inspection
    plot_netflix_data(df)
    
    # Create environment
    env = create_netflix_environment(df)
    print("Netflix trading environment created successfully")
    
    # Train model
    model, vec_env = train_netflix_model(env, total_timesteps)
    print("Netflix model training completed")
    
    # Evaluate model
    portfolio_values, actions_taken, timesteps = evaluate_netflix_model(model, vec_env)
    print("Netflix model evaluation completed")
    
    # Analyze performance
    analyze_netflix_performance(portfolio_values, actions_taken, timesteps, df)
    
    # Save model
    save_netflix_model(model, "NFLX")
    
    print("\nNetflix model training and evaluation completed successfully!")

if __name__ == "__main__":
    main()
