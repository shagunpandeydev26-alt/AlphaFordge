"""
Training module for RL trading agents
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.agents.PPOAgent import TradingPPOAgent, TradingCallback, create_ppo_agent
from src.envs.SingleStockTradingEnv import SingleStockTradingEnv
from src.data.data_loader import StockDataLoader
from src.utils.logger import setup_logger
from src.utils.metrics import calculate_trading_metrics
from src.train.config import TrainingConfig

warnings.filterwarnings("ignore")


class TradingTrainer:
    """Main training class for RL trading agents"""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer with configuration
        
        Args:
            config: Training configuration object
        """
        self.config = config
        self.logger = setup_logger("training", self.config.log_dir)
        self.data_loader = StockDataLoader(use_finrl=True)
        
        # Create directories
        os.makedirs(self.config.model_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        os.makedirs(self.config.results_dir, exist_ok=True)
        
        self.logger.info(f"Training configuration: {self.config}")
    
    def prepare_data(self) -> pd.DataFrame:
        """
        Load and prepare training data
        
        Returns:
            Preprocessed DataFrame with stock data
        """
        
        self.logger.info(f"Loading data for {self.config.ticker}")
        
        df = self.data_loader.load_single_stock_data(
            ticker=self.config.ticker,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            include_benchmark=True
        )
        
        # Validate data
        is_valid, issues = self.data_loader.validate_data(df)
        if not is_valid:
            self.logger.error(f"Data validation failed: {issues}")
            raise ValueError(f"Invalid data: {issues}")
        
        # Log data summary
        summary = self.data_loader.get_data_summary(df)
        self.logger.info(f"Data summary: {summary}")
        
        return df
    
    def create_environment(self, df: pd.DataFrame) -> SingleStockTradingEnv:
        """
        Create trading environment
        
        Args:
            df: Stock data DataFrame
            
        Returns:
            Configured trading environment
        """
        
        env_params = {
            'df': df,
            'hmax': self.config.hmax,
            'initial_amount': self.config.initial_amount,
            'num_stock_shares': [0],
            'buy_cost_pct': [self.config.transaction_cost],
            'sell_cost_pct': [self.config.transaction_cost],
            'turbulence_threshold': self.config.turbulence_threshold,
            'reward_scaling': self.config.reward_scaling,
            'print_verbosity': 1 if self.config.verbose else 0,
            'reward_type': self.config.reward_type,
            'reward_weights': self.config.reward_weights
        }
        
        env = SingleStockTradingEnv(**env_params)
        self.logger.info(f"Created environment with params: {env_params}")
        
        return env
    
    def create_agent(self, env: SingleStockTradingEnv) -> TradingPPOAgent:
        """
        Create PPO agent
        
        Args:
            env: Trading environment
            
        Returns:
            Configured PPO agent
        """
        
        agent_config = {
            'learning_rate': self.config.learning_rate,
            'n_steps': self.config.n_steps,
            'batch_size': self.config.batch_size,
            'n_epochs': self.config.n_epochs,
            'gamma': self.config.gamma,
            'gae_lambda': self.config.gae_lambda,
            'clip_range': self.config.clip_range,
            'ent_coef': self.config.ent_coef,
            'vf_coef': self.config.vf_coef,
            'max_grad_norm': self.config.max_grad_norm,
            'verbose': 1 if self.config.verbose else 0
        }
        
        agent = create_ppo_agent(env, agent_config)
        self.logger.info(f"Created PPO agent with config: {agent_config}")
        
        return agent
    
    def train(self) -> Dict[str, Any]:
        """
        Execute the full training pipeline
        
        Returns:
            Dictionary with training results
        """
        
        start_time = datetime.now()
        self.logger.info(f"Starting training at {start_time}")
        
        try:
            # Prepare data
            df = self.prepare_data()
            
            # Create environment
            env = self.create_environment(df)
            
            # Create agent
            agent = self.create_agent(env)
            
            # Create callback for monitoring
            callback = TradingCallback(
                eval_freq=self.config.eval_freq,
                n_eval_episodes=self.config.n_eval_episodes,
                verbose=1 if self.config.verbose else 0
            )
            
            # Train the agent
            self.logger.info(f"Starting training for {self.config.total_timesteps} timesteps")
            agent.train(
                total_timesteps=self.config.total_timesteps,
                callback=callback
            )
            
            # Evaluate the trained agent
            self.logger.info("Evaluating trained agent")
            eval_results = agent.evaluate(
                n_episodes=self.config.n_eval_episodes,
                deterministic=True
            )
            
            # Save the model
            model_path = Path(self.config.model_dir) / f"{self.config.ticker}.zip"
            agent.save(model_path)
            self.logger.info(f"Model saved to {model_path}")
            
            # Calculate additional metrics
            portfolio_values = eval_results['portfolio_values']
            trading_metrics = calculate_trading_metrics(
                portfolio_values=portfolio_values,
                benchmark_data=df['close_benchmark'].values,
                initial_amount=self.config.initial_amount
            )
            
            # Compile results
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            results = {
                'ticker': self.config.ticker,
                'training_time': training_time,
                'total_timesteps': self.config.total_timesteps,
                'evaluation_results': eval_results,
                'trading_metrics': trading_metrics,
                'model_path': str(model_path),
                'config': self.config.to_dict(),
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }
            
            # Save results
            results_path = Path(self.config.results_dir) / f"{self.config.ticker}_training_results.json"
            self._save_results(results, results_path)
            
            self.logger.info(f"Training completed successfully in {training_time:.2f} seconds")
            self.logger.info(f"Final portfolio value: ${eval_results['mean_portfolio_value']:.2f}")
            self.logger.info(f"Results saved to {results_path}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
    
    def _save_results(self, results: Dict[str, Any], path: Path) -> None:
        """Save training results to JSON file"""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_json = convert_numpy(results)
        
        with open(path, 'w') as f:
            json.dump(results_json, f, indent=2)


def train_single_stock(ticker: str,
                      start_date: str = "2015-01-01",
                      end_date: str = "2025-01-01",
                      total_timesteps: int = 100000,
                      **kwargs) -> Dict[str, Any]:
    """
    Convenience function to train a single stock model
    
    Args:
        ticker: Stock ticker symbol
        start_date: Training start date
        end_date: Training end date
        total_timesteps: Total training timesteps
        **kwargs: Additional configuration parameters
        
    Returns:
        Training results dictionary
    """
    
    config = TrainingConfig(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        total_timesteps=total_timesteps,
        **kwargs
    )
    
    trainer = TradingTrainer(config)
    return trainer.train()


def train_multiple_stocks(tickers: List[str],
                         start_date: str = "2015-01-01",
                         end_date: str = "2025-01-01",
                         total_timesteps: int = 100000,
                         **kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Train models for multiple stocks
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Training start date
        end_date: Training end date
        total_timesteps: Total training timesteps
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary mapping ticker to training results
    """
    
    results = {}
    
    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"Training model for {ticker}")
        print(f"{'='*50}")
        
        try:
            result = train_single_stock(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                total_timesteps=total_timesteps,
                **kwargs
            )
            results[ticker] = result
            print(f"✅ Successfully trained {ticker}")
            
        except Exception as e:
            print(f"❌ Failed to train {ticker}: {str(e)}")
            results[ticker] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RL trading agents")
    parser.add_argument("--ticker", type=str, default="GOOGL", help="Stock ticker")
    parser.add_argument("--start_date", type=str, default="2015-01-01", help="Start date")
    parser.add_argument("--end_date", type=str, default="2025-01-01", help="End date")
    parser.add_argument("--timesteps", type=int, default=100000, help="Training timesteps")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    results = train_single_stock(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        total_timesteps=args.timesteps,
        verbose=args.verbose
    )
    
    print(f"\nTraining completed for {args.ticker}")
    print(f"Final portfolio value: ${results['evaluation_results']['mean_portfolio_value']:.2f}")
    print(f"Model saved to: {results['model_path']}")
