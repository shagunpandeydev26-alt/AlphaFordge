"""
Grid Search Hyperparameter Optimization for RL Trading
Converted from GridSearch.ipynb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import warnings
from pathlib import Path
import sys
import os
import pickle
from datetime import datetime

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
import torch

warnings.filterwarnings("ignore")

class CustomStockTradingEnv(StockTradingEnv):
    """Custom Stock Trading Environment for hyperparameter optimization"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_function = None

    def set_reward_function(self, reward_function):
        self.reward_function = reward_function

    def step(self, actions):
        next_state, reward, terminal, truncated, info = super().step(actions)

        if self.reward_function is not None:
            reward = self.reward_function(self, actions, next_state, reward, terminal, truncated, info)
        
        return next_state, reward, terminal, truncated, info


def differential_return_reward(self, actions, next_state, base_reward, terminal, truncated, info, weights=[0.1, 0.01, 0.01, 1.0]):
    """Differential return reward function with configurable weights"""
    
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
    
    # Compute Sortino Ratio
    downside_returns = df_total_value["daily_return"][df_total_value["daily_return"] < 0]
    downside_std = downside_returns.std(ddof=1)
    
    # Compute Treynor Ratio
    treynor = 0
    if beta and not np.isnan(beta) and beta != 0:
        treynor = (252**0.5) * mean_returns / beta
    
    # Compute Differential Return
    diff_return = 0
    if beta and not np.isnan(beta) and beta != 0:
        diff_return = (mean_returns - bench_returns) / beta

    # Weighted combination
    w1, w2, w3, w4 = weights
    total_reward = (
        w1 * remove_nan(mean_returns) 
        - w2 * remove_nan(abs(downside_std)) 
        + w3 * remove_nan(treynor) 
        + w4 * remove_nan(diff_return)
    )
    
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


def create_environment(df, reward_weights=[0.1, 0.01, 0.01, 1.0]):
    """Create and configure the trading environment"""
    
    indicators = ["volume", "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma", "turbulence"]
    
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
        print_verbosity=0,  # Reduced verbosity for grid search
        buy_cost_pct=[0.001],
        sell_cost_pct=[0.001],
        turbulence_threshold=100,
        reward_scaling=1e-4,
        tech_indicator_list=indicators,
        state_space=1 + 2 * stock_dim + len(indicators) * stock_dim,
        action_space=stock_dim,
    )

    # Set reward function with specific weights
    reward_fn = lambda *args, **kwargs: differential_return_reward(*args, weights=reward_weights, **kwargs)
    env.set_reward_function(reward_fn)
    
    return env


def train_and_evaluate(df, reward_weights, total_timesteps=50000, evaluation_episodes=10):
    """Train model with given hyperparameters and evaluate performance"""
    
    try:
        # Create environment
        env = create_environment(df, reward_weights)
        env.reset()
        
        # Create vectorized environment
        vec_env = make_vec_env(lambda: Monitor(env, None), n_envs=1)
        
        # Train model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PPO("MlpPolicy", vec_env, verbose=0, device=device)
        model.learn(total_timesteps=total_timesteps)
        
        # Evaluate model
        total_rewards = []
        final_values = []
        
        for _ in range(evaluation_episodes):
            obs = vec_env.reset()
            episode_reward = 0
            
            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, done, info = vec_env.step(action)
                episode_reward += rewards[0]
                
                if done[0]:
                    final_value = vec_env.envs[0].unwrapped.asset_memory[-1]
                    final_values.append(final_value)
                    break
            
            total_rewards.append(episode_reward)
        
        # Calculate metrics
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        mean_final_value = np.mean(final_values)
        std_final_value = np.std(final_values)
        
        return {
            'weights': reward_weights,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_final_value': mean_final_value,
            'std_final_value': std_final_value,
            'sharpe_ratio': mean_reward / std_reward if std_reward > 0 else 0
        }
        
    except Exception as e:
        print(f"Error with weights {reward_weights}: {e}")
        return None


def grid_search(df, param_grid, total_timesteps=50000, evaluation_episodes=5):
    """Perform grid search over hyperparameter space"""
    
    results = []
    total_combinations = len(list(itertools.product(*param_grid.values())))
    
    print(f"Starting grid search with {total_combinations} combinations...")
    
    for i, combination in enumerate(itertools.product(*param_grid.values())):
        weights = list(combination)
        print(f"Progress: {i+1}/{total_combinations} - Testing weights: {weights}")
        
        result = train_and_evaluate(df, weights, total_timesteps, evaluation_episodes)
        
        if result is not None:
            results.append(result)
            print(f"Mean reward: {result['mean_reward']:.4f}, Final value: {result['mean_final_value']:.2f}")
        
        print("-" * 50)
    
    return results


def analyze_results(results):
    """Analyze and visualize grid search results"""
    
    if not results:
        print("No results to analyze!")
        return
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Sort by different metrics
    print("Top 5 configurations by mean reward:")
    top_reward = df_results.nlargest(5, 'mean_reward')[['weights', 'mean_reward', 'mean_final_value']]
    print(top_reward)
    print()
    
    print("Top 5 configurations by final portfolio value:")
    top_value = df_results.nlargest(5, 'mean_final_value')[['weights', 'mean_reward', 'mean_final_value']]
    print(top_value)
    print()
    
    print("Top 5 configurations by Sharpe ratio:")
    top_sharpe = df_results.nlargest(5, 'sharpe_ratio')[['weights', 'sharpe_ratio', 'mean_reward']]
    print(top_sharpe)
    print()
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Mean reward distribution
    axes[0, 0].hist(df_results['mean_reward'], bins=20, alpha=0.7)
    axes[0, 0].set_title('Distribution of Mean Rewards')
    axes[0, 0].set_xlabel('Mean Reward')
    axes[0, 0].set_ylabel('Frequency')
    
    # Final value distribution
    axes[0, 1].hist(df_results['mean_final_value'], bins=20, alpha=0.7, color='green')
    axes[0, 1].set_title('Distribution of Final Portfolio Values')
    axes[0, 1].set_xlabel('Final Portfolio Value')
    axes[0, 1].set_ylabel('Frequency')
    
    # Reward vs Final Value scatter
    axes[1, 0].scatter(df_results['mean_reward'], df_results['mean_final_value'], alpha=0.6)
    axes[1, 0].set_xlabel('Mean Reward')
    axes[1, 0].set_ylabel('Final Portfolio Value')
    axes[1, 0].set_title('Reward vs Portfolio Value')
    
    # Sharpe ratio distribution
    axes[1, 1].hist(df_results['sharpe_ratio'], bins=20, alpha=0.7, color='orange')
    axes[1, 1].set_title('Distribution of Sharpe Ratios')
    axes[1, 1].set_xlabel('Sharpe Ratio')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    return df_results


def save_results(results, filename=None):
    """Save grid search results to file"""
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"grid_search_results_{timestamp}.pkl"
    
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {filename}")


def load_results(filename):
    """Load grid search results from file"""
    
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    
    return results


def main():
    """Main execution function"""
    
    # Configuration
    ticker = "GOOGL"
    start_date = "2015-01-01"
    end_date = "2025-01-01"
    total_timesteps = 30000  # Reduced for faster grid search
    evaluation_episodes = 3  # Reduced for faster evaluation
    
    print(f"Starting grid search for {ticker}")
    
    # Load and prepare data
    df = load_and_prepare_data(ticker, start_date, end_date)
    print(f"Loaded {len(df)} data points")
    
    # Define parameter grid - reward function weights
    param_grid = {
        'w1': [0.05, 0.1, 0.2],      # Mean returns weight
        'w2': [0.005, 0.01, 0.02],   # Downside risk weight
        'w3': [0.005, 0.01, 0.02],   # Treynor ratio weight
        'w4': [0.5, 1.0, 2.0]        # Differential return weight
    }
    
    print(f"Parameter grid: {param_grid}")
    total_combinations = len(list(itertools.product(*param_grid.values())))
    print(f"Total combinations to test: {total_combinations}")
    
    # Perform grid search
    results = grid_search(df, param_grid, total_timesteps, evaluation_episodes)
    
    # Analyze results
    df_results = analyze_results(results)
    
    # Save results
    save_results(results)
    
    # Get best configuration
    if not df_results.empty:
        best_config = df_results.loc[df_results['mean_final_value'].idxmax()]
        print(f"\nBest configuration (by final value):")
        print(f"Weights: {best_config['weights']}")
        print(f"Mean reward: {best_config['mean_reward']:.4f}")
        print(f"Final portfolio value: {best_config['mean_final_value']:.2f}")
        print(f"Sharpe ratio: {best_config['sharpe_ratio']:.4f}")


if __name__ == "__main__":
    main()
