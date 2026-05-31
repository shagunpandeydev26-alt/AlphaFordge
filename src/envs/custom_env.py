# Custom Gym environment for RL training, defining state, action space, and loading custom reward functions.

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, Union
from abc import ABC, abstractmethod

from ..rewards.reward_function import BaseRewardFunction, ProfitReward
from ..data.feature_engineer import FeatureEngineer


class BaseCustomEnv(gym.Env, ABC):
    """
    Abstract base class for custom trading environments.
    Provides common functionality and interface for trading environments.
    """
    
    def __init__(self, 
                 df: pd.DataFrame,
                 initial_amount: float = 10000,
                 transaction_cost_pct: float = 0.001,
                 reward_function: Optional[BaseRewardFunction] = None,
                 feature_engineer: Optional[FeatureEngineer] = None):
        """
        Initialize the base custom environment.
        
        Args:
            df: Market data DataFrame
            initial_amount: Initial cash amount
            transaction_cost_pct: Transaction cost as percentage
            reward_function: Custom reward function
            feature_engineer: Feature engineering module
        """
        super().__init__()
        
        self.df = df.copy()
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_function = reward_function or ProfitReward()
        self.feature_engineer = feature_engineer or FeatureEngineer()
        
        # State variables
        self.current_step = 0
        self.cash = initial_amount
        self.positions = {}
        self.portfolio_value = initial_amount
        self.transaction_history = []
        self.state_memory = []
        
        # Setup action and observation spaces
        self._setup_spaces()
        
        # Process features if feature engineer provided
        if self.feature_engineer:
            self.df = self.feature_engineer.engineer_features(self.df)
        
    @abstractmethod
    def _setup_spaces(self):
        """Setup action and observation spaces. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        """Get current observation. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _execute_action(self, action: Union[int, float, np.ndarray]) -> Dict[str, Any]:
        """Execute the given action. Must be implemented by subclasses."""
        pass
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.cash = self.initial_amount
        self.positions = {}
        self.portfolio_value = self.initial_amount
        self.transaction_history = []
        self.state_memory = []
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: Union[int, float, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Execute action
        action_info = self._execute_action(action)
        
        # Calculate reward
        reward_data = {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'positions': self.positions,
            'action_info': action_info,
            'current_step': self.current_step,
            'market_data': self.df.iloc[self.current_step]
        }
        reward = self.reward_function.calculate_reward(reward_data)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= len(self.df) - 1
        truncated = False  # Can be used for time limits
        
        # Get new observation
        observation = self._get_observation()
        info = self._get_info()
        
        # Store state for analysis
        self.state_memory.append({
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'positions': self.positions.copy(),
            'action': action,
            'reward': reward
        })
        
        return observation, reward, terminated, truncated, info
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary for current state."""
        return {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'positions': self.positions,
            'current_step': self.current_step,
            'total_steps': len(self.df),
            'transaction_count': len(self.transaction_history)
        }
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        if self.current_step >= len(self.df):
            return self.cash
        
        current_prices = self.df.iloc[self.current_step]
        position_value = sum(
            qty * current_prices.get(asset, 0) 
            for asset, qty in self.positions.items()
        )
        return self.cash + position_value
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the current state of the environment."""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:.2f}")
            print(f"Cash: ${self.cash:.2f}")
            print(f"Positions: {self.positions}")
            print("-" * 40)
        return None


class MultiAssetTradingEnv(BaseCustomEnv):
    """
    Custom environment for multi-asset trading.
    Supports trading multiple assets with continuous actions.
    """
    
    def __init__(self, 
                 df: pd.DataFrame,
                 assets: list,
                 initial_amount: float = 10000,
                 max_position_pct: float = 0.3,
                 **kwargs):
        """
        Initialize multi-asset trading environment.
        
        Args:
            df: Market data DataFrame with columns for each asset
            assets: List of asset names to trade
            initial_amount: Initial cash amount
            max_position_pct: Maximum position size as percentage of portfolio
        """
        self.assets = assets
        self.max_position_pct = max_position_pct
        self.n_assets = len(assets)
        
        super().__init__(df=df, initial_amount=initial_amount, **kwargs)
        
        # Initialize positions for each asset
        self.positions = {asset: 0 for asset in assets}
    
    def _setup_spaces(self):
        """Setup action and observation spaces for multi-asset trading."""
        # Action space: continuous actions for each asset [-1, 1]
        # -1 = sell all, 0 = hold, 1 = buy max allowed
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.n_assets,), 
            dtype=np.float32
        )
        
        # Observation space: market data + portfolio state
        market_features = len(self.df.columns)
        portfolio_features = self.n_assets + 2  # positions + cash + portfolio_value
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(market_features + portfolio_features,),
            dtype=np.float32
        )
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation including market data and portfolio state."""
        if self.current_step >= len(self.df):
            # Return zeros if beyond data range
            return np.zeros(self.observation_space.shape)
        
        # Market data
        market_data = self.df.iloc[self.current_step].values
        
        # Portfolio state
        positions_array = np.array([self.positions[asset] for asset in self.assets])
        portfolio_state = np.concatenate([
            positions_array,
            [self.cash / self.initial_amount],  # Normalized cash
            [self.portfolio_value / self.initial_amount]  # Normalized portfolio value
        ])
        
        # Combine market and portfolio data
        observation = np.concatenate([market_data, portfolio_state])
        
        return observation.astype(np.float32)
    
    def _execute_action(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Execute trading action for multiple assets.
        
        Args:
            action: Array of actions for each asset [-1, 1]
            
        Returns:
            Dictionary with execution details
        """
        if self.current_step >= len(self.df):
            return {'executed_trades': []}
        
        current_prices = self.df.iloc[self.current_step]
        executed_trades = []
        
        for i, asset in enumerate(self.assets):
            if asset not in current_prices or pd.isna(current_prices[asset]):
                continue
                
            current_price = current_prices[asset]
            action_val = action[i]
            
            if abs(action_val) < 0.01:  # Threshold for no action
                continue
            
            # Calculate target position
            max_position_value = self.portfolio_value * self.max_position_pct
            max_shares = max_position_value / current_price
            target_position = action_val * max_shares
            
            # Calculate trade quantity
            current_position = self.positions[asset]
            trade_quantity = target_position - current_position
            
            if abs(trade_quantity) < 0.01:  # Minimum trade size
                continue
            
            # Execute trade
            trade_value = abs(trade_quantity) * current_price
            transaction_cost = trade_value * self.transaction_cost_pct
            
            if trade_quantity > 0:  # Buying
                total_cost = trade_value + transaction_cost
                if total_cost <= self.cash:
                    self.positions[asset] += trade_quantity
                    self.cash -= total_cost
                    executed_trades.append({
                        'asset': asset,
                        'action': 'BUY',
                        'quantity': trade_quantity,
                        'price': current_price,
                        'cost': total_cost
                    })
            else:  # Selling
                if abs(trade_quantity) <= current_position:
                    self.positions[asset] += trade_quantity  # trade_quantity is negative
                    self.cash += trade_value - transaction_cost
                    executed_trades.append({
                        'asset': asset,
                        'action': 'SELL',
                        'quantity': abs(trade_quantity),
                        'price': current_price,
                        'revenue': trade_value - transaction_cost
                    })
        
        # Update portfolio value
        self.portfolio_value = self._calculate_portfolio_value()
        
        # Record transaction
        if executed_trades:
            self.transaction_history.append({
                'step': self.current_step,
                'trades': executed_trades,
                'portfolio_value_after': self.portfolio_value
            })
        
        return {'executed_trades': executed_trades}


class ContinuousTradingEnv(BaseCustomEnv):
    """
    Simplified continuous trading environment for single asset.
    Actions represent portfolio allocation percentage.
    """
    
    def __init__(self, 
                 df: pd.DataFrame,
                 asset_column: str = 'close',
                 **kwargs):
        """
        Initialize continuous trading environment.
        
        Args:
            df: Market data DataFrame
            asset_column: Column name for asset price
        """
        self.asset_column = asset_column
        super().__init__(df=df, **kwargs)
        
        # Single position tracking
        self.shares_held = 0
    
    def _setup_spaces(self):
        """Setup action and observation spaces for continuous trading."""
        # Action space: portfolio allocation [0, 1]
        # 0 = all cash, 1 = all invested
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Observation space: market features + portfolio state
        market_features = len(self.df.columns)
        portfolio_features = 3  # shares_held, cash_ratio, portfolio_value_ratio
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(market_features + portfolio_features,),
            dtype=np.float32
        )
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape)
        
        # Market data
        market_data = self.df.iloc[self.current_step].values
        
        # Portfolio state (normalized)
        portfolio_state = np.array([
            self.shares_held / 1000,  # Normalize shares
            self.cash / self.initial_amount,
            self.portfolio_value / self.initial_amount
        ])
        
        observation = np.concatenate([market_data, portfolio_state])
        return observation.astype(np.float32)
    
    def _execute_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Execute continuous allocation action."""
        if self.current_step >= len(self.df):
            return {'action_taken': 'none'}
        
        allocation = action[0]  # Target allocation [0, 1]
        current_price = self.df.iloc[self.current_step][self.asset_column]
        
        if pd.isna(current_price):
            return {'action_taken': 'none'}
        
        # Calculate target position value
        target_investment = self.portfolio_value * allocation
        target_shares = target_investment / current_price
        
        # Calculate trade
        shares_to_trade = target_shares - self.shares_held
        
        if abs(shares_to_trade) < 0.01:
            return {'action_taken': 'hold'}
        
        # Execute trade
        trade_value = abs(shares_to_trade) * current_price
        transaction_cost = trade_value * self.transaction_cost_pct
        
        if shares_to_trade > 0:  # Buying
            total_cost = trade_value + transaction_cost
            if total_cost <= self.cash:
                self.shares_held += shares_to_trade
                self.cash -= total_cost
                action_taken = 'buy'
            else:
                action_taken = 'insufficient_cash'
        else:  # Selling
            if abs(shares_to_trade) <= self.shares_held:
                self.shares_held += shares_to_trade  # shares_to_trade is negative
                self.cash += trade_value - transaction_cost
                action_taken = 'sell'
            else:
                action_taken = 'insufficient_shares'
        
        # Update portfolio value
        self.portfolio_value = self.cash + self.shares_held * current_price
        
        return {
            'action_taken': action_taken,
            'shares_traded': shares_to_trade,
            'trade_value': trade_value,
            'allocation': allocation
        }
