# Loads the trained model and runs inference on new stock market data.

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import streamlit as st

from stable_baselines3 import PPO
from ..envs.SingleStockTradingEnv import SingleStockTradingEnv
from ..utils.metrics import PerformanceMetrics
from ..data.data_loader import DataLoader


class TradingInferenceEngine:
    """
    Handles inference operations for trained RL trading models.
    Provides methods for single predictions, batch inference, and model evaluation.
    """
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.metrics = PerformanceMetrics()
    
    def predict_action(self, model: PPO, env: SingleStockTradingEnv, 
                      portfolio_value: float, num_stock_shares: int) -> int:
        """
        Predict next action using the trained model.
        
        Args:
            model: Trained PPO model
            env: Trading environment
            portfolio_value: Current portfolio value
            num_stock_shares: Current number of shares held
            
        Returns:
            Recommended action (positive for buy, negative for sell, 0 for hold)
        """
        print(f"Predicting next action with {num_stock_shares} shares and {portfolio_value} portfolio value")
        
        # Get latest market data
        row_df = env.df.tail(1).reset_index(drop=True)
        if hasattr(st, 'write'):  # Only show in Streamlit context
            st.write(row_df)
        
        # Create prediction environment
        predict_env = SingleStockTradingEnv(
            df=row_df, 
            hmax=portfolio_value // max(env.df['close']) if max(env.df['close']) > 0 else 1,
            initial_amount=portfolio_value, 
            num_stock_shares=[num_stock_shares],
        )
        
        # Get model prediction
        obs, info = predict_env.reset()
        
        # Debug: Check for NaN values in observations
        if np.isnan(obs).any():
            print("Warning: NaN values detected in observations!")
            print(f"Observation shape: {obs.shape}")
            print(f"NaN positions: {np.where(np.isnan(obs))}")
            print(f"Observation values: {obs}")
            
            # Replace NaN values with zeros
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"Cleaned observation values: {obs}")
        
        try:
            action, _states = model.predict(obs)
            print(f"Predicted raw action: {action}")
        except Exception as pred_error:
            print(f"Model prediction failed: {pred_error}")
            # Fallback to a safe default action (hold)
            action = np.array([0.0])
            print("Using fallback action: hold")

        # Convert to actual number of shares
        action = int(action[0] * predict_env.hmax)

        # Validate action constraints
        action = self._validate_action(action, num_stock_shares, portfolio_value, 
                                     predict_env.df["close"].iloc[0])
        
        return action
    
    def _validate_action(self, action: int, num_stock_shares: int, 
                        portfolio_value: float, current_price: float) -> int:
        """Validate and constrain the predicted action based on available resources."""
        if action < 0:  # Selling
            tosell = -action
            if tosell > num_stock_shares:
                action = -num_stock_shares
        
        elif action > 0:  # Buying
            tobuy = action
            max_affordable = portfolio_value // current_price
            if tobuy > max_affordable:
                action = int(max_affordable)
        
        return action
    
    def batch_predict(self, model: PPO, ticker: str, start_date: str, 
                     end_date: str, initial_amount: float = 10000) -> Dict[str, Any]:
        """
        Run batch prediction over a date range.
        
        Args:
            model: Trained PPO model
            ticker: Stock ticker
            start_date: Start date for prediction
            end_date: End date for prediction
            initial_amount: Initial portfolio value
            
        Returns:
            Dictionary containing predictions and performance metrics
        """
        # Load data
        data = self.data_loader.load_stock_data(ticker, start_date, end_date)
        
        # Create environment
        env = SingleStockTradingEnv(
            df=data,
            initial_amount=initial_amount
        )
        
        # Run inference
        obs, info = env.reset()
        
        # Validate initial observation
        if np.isnan(obs).any():
            print("Warning: NaN values in initial batch observation, replacing with zeros")
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        actions = []
        rewards = []
        portfolio_values = []
        
        done = False
        step_count = 0
        max_steps = len(data) * 2  # Safety limit to prevent infinite loops
        
        while not done and step_count < max_steps:
            try:
                # Validate observation before prediction
                if np.isnan(obs).any():
                    print(f"Warning: NaN in observation at step {step_count}, cleaning...")
                    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
                
                action, _states = model.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                
                # Validate outputs
                if np.isnan(reward):
                    print(f"Warning: NaN reward at step {step_count}, setting to 0")
                    reward = 0.0
                
                actions.append(action[0] if hasattr(action, '__len__') else action)
                rewards.append(reward)
                portfolio_values.append(env.asset_memory[-1] if hasattr(env, 'asset_memory') and env.asset_memory else initial_amount)
                
                step_count += 1
                
                if done or truncated:
                    break
                    
            except Exception as step_error:
                print(f"Error during batch prediction step {step_count}: {step_error}")
                # Add safe default values and continue
                actions.append(0.0)  # Hold action
                rewards.append(0.0)
                portfolio_values.append(portfolio_values[-1] if portfolio_values else initial_amount)
                step_count += 1
                if step_count >= max_steps:
                    break
        
        # Calculate performance metrics
        returns = np.array(portfolio_values)
        benchmark_returns = data['close'].pct_change().fillna(0).cumsum()
        
        metrics = {
            'total_return': (portfolio_values[-1] - initial_amount) / initial_amount,
            'sharpe_ratio': self.metrics.calculate_sharpe_ratio(returns),
            'max_drawdown': self.metrics.calculate_max_drawdown(returns),
            'volatility': np.std(returns),
            'benchmark_return': benchmark_returns.iloc[-1]
        }
        
        return {
            'actions': actions,
            'rewards': rewards,
            'portfolio_values': portfolio_values,
            'metrics': metrics,
            'final_value': portfolio_values[-1]
        }
    
    def evaluate_model(self, model: PPO, ticker: str, test_start: str, 
                      test_end: str) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            model: Trained PPO model
            ticker: Stock ticker
            test_start: Test period start date
            test_end: Test period end date
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = self.batch_predict(model, ticker, test_start, test_end)
        
        # Additional evaluation metrics
        actions = np.array(results['actions'])
        portfolio_values = np.array(results['portfolio_values'])
        
        metrics = results['metrics'].copy()
        metrics.update({
            'win_rate': np.mean(np.array(results['rewards']) > 0),
            'avg_reward': np.mean(results['rewards']),
            'action_distribution': {
                'buy_pct': np.mean(actions > 0.1),
                'sell_pct': np.mean(actions < -0.1),
                'hold_pct': np.mean(np.abs(actions) <= 0.1)
            }
        })
        
        return metrics


class ModelInferenceAPI:
    """
    Higher-level API for model inference operations.
    Combines model loading and inference functionality.
    """
    
    def __init__(self, models_dir: Union[str, Path]):
        self.models_dir = Path(models_dir)
        self.inference_engine = TradingInferenceEngine()
        self._loaded_models = {}
    
    def load_model(self, ticker: str) -> Optional[PPO]:
        """Load model for a specific ticker."""
        if ticker in self._loaded_models:
            return self._loaded_models[ticker]
        
        model_path = self.models_dir / f"{ticker}.zip"
        if not model_path.exists():
            return None
        
        try:
            model = PPO.load(model_path)
            self._loaded_models[ticker] = model
            return model
        except Exception as e:
            print(f"Error loading model for {ticker}: {e}")
            return None
    
    def predict_for_ticker(self, ticker: str, portfolio_value: float, 
                          num_shares: int) -> Optional[int]:
        """Get prediction for a specific ticker."""
        model = self.load_model(ticker)
        if not model:
            return None
        
        # Create environment with latest data
        data = self.inference_engine.data_loader.load_stock_data(
            ticker, "2019-01-01", pd.Timestamp.now().strftime("%Y-%m-%d")
        )
        env = SingleStockTradingEnv(df=data)
        
        return self.inference_engine.predict_action(
            model, env, portfolio_value, num_shares
        )
