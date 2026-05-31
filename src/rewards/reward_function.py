"""
Advanced reward functions for RL trading environments
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod


class BaseRewardFunction(ABC):
    """Abstract base class for reward functions"""
    
    @abstractmethod
    def calculate_reward(self, data: Dict[str, Any]) -> float:
        """Calculate reward based on environment data"""
        pass


class RewardCalculator(BaseRewardFunction):
    """Base class for reward calculations"""
    
    def __init__(self, weights: Optional[List[float]] = None):
        self.weights = weights or [0.1, 0.01, 0.01, 1.0]
    
    def calculate_reward(self, data: Dict[str, Any]) -> float:
        """Default implementation - delegates to calculate method"""
        # Extract parameters for backward compatibility
        env = data.get('env')
        actions = data.get('actions')
        next_state = data.get('next_state')
        base_reward = data.get('base_reward', 0)
        terminal = data.get('terminal', False)
        truncated = data.get('truncated', False)
        info = data.get('info', {})
        
        if env and hasattr(self, 'calculate'):
            return self.calculate(env, actions, next_state, base_reward, terminal, truncated, info)
        return 0.0
    
    def calculate(self, env, actions, next_state, base_reward, terminal, truncated, info) -> float:
        """Calculate reward based on environment state"""
        raise NotImplementedError


class ProfitReward(BaseRewardFunction):
    """Simple profit-based reward function"""
    
    def __init__(self, scaling_factor: float = 1e-4):
        self.scaling_factor = scaling_factor
    
    def calculate_reward(self, data: Dict[str, Any]) -> float:
        """Calculate reward based on portfolio value change"""
        portfolio_value = data.get('portfolio_value', 0)
        previous_value = data.get('previous_portfolio_value', portfolio_value)
        
        profit = portfolio_value - previous_value
        return profit * self.scaling_factor


class DifferentialReturnReward(RewardCalculator):
    """
    Differential return reward function based on:
    - Mean daily returns
    - Downside risk (downside standard deviation)
    - Treynor ratio
    - Differential return compared to benchmark
    """
    
    def __init__(self, weights: Optional[List[float]] = None):
        super().__init__(weights)
        self.w1, self.w2, self.w3, self.w4 = self.weights
    
    def calculate(self, env, actions, next_state, base_reward, terminal, truncated, info, reward_logging=False) -> float:
        """Calculate differential return reward"""
        
        df_total_value = pd.DataFrame(env.asset_memory, columns=["account_value"])
        df_total_value["date"] = env.date_memory
        df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)
        
        df_total_value["benchmark_value"] = env.df["close_benchmark"].iloc[:len(df_total_value)].reset_index(drop=True)
        df_total_value["benchmark_daily_return"] = df_total_value["benchmark_value"].pct_change(1)

        remove_nan = lambda x: 0 if np.isnan(x) else x
        
        # Compute mean daily returns and benchmark returns
        mean_returns = df_total_value["daily_return"].mean()
        std_returns = df_total_value["daily_return"].std()
        bench_returns = df_total_value["benchmark_daily_return"].mean()
        bench_std = df_total_value["benchmark_daily_return"].std()
        
        # Compute Beta
        beta = self._calculate_beta(df_total_value, bench_std)
        
        # Compute financial ratios
        sharpe = self._calculate_sharpe(mean_returns, std_returns)
        sortino = self._calculate_sortino(df_total_value, mean_returns)
        treynor = self._calculate_treynor(mean_returns, beta)
        diff_return = self._calculate_differential_return(mean_returns, bench_returns, beta)
        
        # Downside standard deviation
        downside_returns = df_total_value["daily_return"][df_total_value["daily_return"] < 0]
        downside_std = downside_returns.std(ddof=1)
        
        if reward_logging:
            self._log_metrics(mean_returns, bench_returns, std_returns, downside_std, 
                            beta, sharpe, sortino, treynor, diff_return)
        
        # Weighted combination
        total_reward = (
            self.w1 * remove_nan(mean_returns) 
            - self.w2 * remove_nan(abs(downside_std)) 
            + self.w3 * remove_nan(treynor) 
            + self.w4 * remove_nan(diff_return)
        )
        
        return total_reward
    
    def _calculate_beta(self, df_total_value: pd.DataFrame, bench_std: float) -> float:
        """Calculate portfolio beta"""
        beta = 1.0
        if bench_std and not np.isnan(bench_std) and bench_std != 0:
            portfolio_returns = df_total_value["daily_return"].fillna(0)
            benchmark_returns = df_total_value["benchmark_daily_return"].fillna(0)
            if len(portfolio_returns) == len(benchmark_returns):
                covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
                beta = covariance / (bench_std ** 2)
        return beta
    
    def _calculate_sharpe(self, mean_returns: float, std_returns: float) -> float:
        """Calculate Sharpe ratio"""
        sharpe = 0
        if std_returns and not np.isnan(std_returns) and std_returns != 0:
            sharpe = (252**0.5) * mean_returns / std_returns
        return sharpe
    
    def _calculate_sortino(self, df_total_value: pd.DataFrame, mean_returns: float) -> float:
        """Calculate Sortino ratio"""
        downside_returns = df_total_value["daily_return"][df_total_value["daily_return"] < 0]
        downside_std = downside_returns.std(ddof=1)
        
        sortino = 0
        if downside_std and not np.isnan(downside_std) and downside_std != 0:
            sortino = (252**0.5) * mean_returns / downside_std
        return sortino
    
    def _calculate_treynor(self, mean_returns: float, beta: float) -> float:
        """Calculate Treynor ratio"""
        treynor = 0
        if beta and not np.isnan(beta) and beta != 0:
            treynor = (252**0.5) * mean_returns / beta
        return treynor
    
    def _calculate_differential_return(self, mean_returns: float, bench_returns: float, beta: float) -> float:
        """Calculate differential return"""
        diff_return = 0
        if beta and not np.isnan(beta) and beta != 0:
            diff_return = (mean_returns - bench_returns) / beta
        return diff_return
    
    def _log_metrics(self, mean_returns, bench_returns, std_returns, downside_std, 
                    beta, sharpe, sortino, treynor, diff_return):
        """Log performance metrics"""
        print(f"Mean Daily Returns: {mean_returns}")
        print(f"Benchmark Returns: {bench_returns}")
        print(f"Daily Return Standard Deviation: {std_returns}")
        print(f"Downside Only Standard Deviation: {downside_std}")
        print(f"Beta: {beta}")
        print(f"Sharpe Ratio: {sharpe}")
        print(f"Sortino Ratio: {sortino}")
        print(f"Treynor Ratio: {treynor}")
        print(f"Differential Return: {diff_return}")
        print("-----------------------------")


class SortinoReward(RewardCalculator):
    """Reward function based on Sortino ratio"""
    
    def calculate(self, env, actions, next_state, base_reward, terminal, truncated, info, reward_logging=False) -> float:
        """Calculate Sortino-based reward"""
        
        df_total_value = pd.DataFrame(env.asset_memory, columns=["account_value"])
        df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)
        
        mean_returns = df_total_value["daily_return"].mean()
        downside_returns = df_total_value["daily_return"][df_total_value["daily_return"] < 0]
        downside_std = downside_returns.std(ddof=1)
        
        sortino = 0
        if downside_std and not np.isnan(downside_std) and downside_std != 0:
            sortino = (252**0.5) * mean_returns / downside_std
        
        total_reward = sortino if not np.isnan(sortino) else -100
        
        if reward_logging:
            print(f"Sortino Ratio: {sortino}")
            print(f"Reward: {total_reward}")
        
        return total_reward


class RiskAdjustedReturnReward(RewardCalculator):
    """Risk-adjusted return reward with multiple components"""
    
    def __init__(self, weights: Optional[List[float]] = None):
        super().__init__(weights)
        # Weights for: [return, volatility, max_drawdown, benchmark_excess]
        self.w_return, self.w_vol, self.w_dd, self.w_excess = self.weights
    
    def calculate(self, env, actions, next_state, base_reward, terminal, truncated, info, reward_logging=False) -> float:
        """Calculate risk-adjusted reward"""
        
        df_total_value = pd.DataFrame(env.asset_memory, columns=["account_value"])
        df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)
        
        # Add benchmark data
        df_total_value["benchmark_value"] = env.df["close_benchmark"].iloc[:len(df_total_value)].reset_index(drop=True)
        df_total_value["benchmark_daily_return"] = df_total_value["benchmark_value"].pct_change(1)
        
        remove_nan = lambda x: 0 if np.isnan(x) else x
        
        # Calculate metrics
        mean_returns = df_total_value["daily_return"].mean()
        volatility = df_total_value["daily_return"].std()
        
        # Max drawdown
        cumulative_returns = (1 + df_total_value["daily_return"]).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Excess return over benchmark
        bench_returns = df_total_value["benchmark_daily_return"].mean()
        excess_return = mean_returns - bench_returns
        
        # Combine components
        total_reward = (
            self.w_return * remove_nan(mean_returns) +
            self.w_vol * remove_nan(-volatility) +  # Negative because we want lower volatility
            self.w_dd * remove_nan(-max_drawdown) +  # Negative because drawdown is negative
            self.w_excess * remove_nan(excess_return)
        )
        
        if reward_logging:
            print(f"Mean Returns: {mean_returns}")
            print(f"Volatility: {volatility}")
            print(f"Max Drawdown: {max_drawdown}")
            print(f"Excess Return: {excess_return}")
            print(f"Total Reward: {total_reward}")
        
        return total_reward


class SharpeRatioReward(BaseRewardFunction):
    """Sharpe ratio-based reward function"""
    
    def __init__(self, risk_free_rate: float = 0.02, window: int = 252):
        self.risk_free_rate = risk_free_rate
        self.window = window
    
    def calculate_reward(self, data: Dict[str, Any]) -> float:
        """Calculate reward based on Sharpe ratio"""
        portfolio_value = data.get('portfolio_value', 0)
        cash = data.get('cash', 0)
        
        # Simple implementation - can be enhanced with historical data
        if hasattr(data.get('env'), 'asset_memory') and len(data['env'].asset_memory) > 1:
            returns = pd.Series(data['env'].asset_memory).pct_change().dropna()
            if len(returns) > 0:
                excess_return = returns.mean() - self.risk_free_rate / 252
                volatility = returns.std()
                if volatility > 0:
                    return excess_return / volatility
        
        return 0.0


class DrawdownPenaltyReward(BaseRewardFunction):
    """Reward function that penalizes large drawdowns"""
    
    def __init__(self, max_drawdown_threshold: float = 0.2, penalty_factor: float = 2.0):
        self.max_drawdown_threshold = max_drawdown_threshold
        self.penalty_factor = penalty_factor
    
    def calculate_reward(self, data: Dict[str, Any]) -> float:
        """Calculate reward with drawdown penalty"""
        portfolio_value = data.get('portfolio_value', 0)
        
        if hasattr(data.get('env'), 'asset_memory') and len(data['env'].asset_memory) > 1:
            values = pd.Series(data['env'].asset_memory)
            running_max = values.expanding().max()
            drawdown = (values - running_max) / running_max
            current_drawdown = drawdown.iloc[-1]
            
            # Base reward is portfolio change
            base_reward = values.iloc[-1] - values.iloc[-2] if len(values) > 1 else 0
            
            # Apply penalty if drawdown exceeds threshold
            if abs(current_drawdown) > self.max_drawdown_threshold:
                penalty = abs(current_drawdown) * self.penalty_factor
                return base_reward - penalty
            
            return base_reward
        
        return 0.0


class TransactionCostAwareReward(BaseRewardFunction):
    """Reward function that accounts for transaction costs"""
    
    def __init__(self, transaction_cost_pct: float = 0.001, base_scaling: float = 1e-4):
        self.transaction_cost_pct = transaction_cost_pct
        self.base_scaling = base_scaling
    
    def calculate_reward(self, data: Dict[str, Any]) -> float:
        """Calculate reward accounting for transaction costs"""
        portfolio_value = data.get('portfolio_value', 0)
        action_info = data.get('action_info', {})
        
        # Base profit reward
        previous_value = data.get('previous_portfolio_value', portfolio_value)
        profit = portfolio_value - previous_value
        
        # Calculate transaction costs
        transaction_cost = 0
        if 'executed_trades' in action_info:
            for trade in action_info['executed_trades']:
                trade_value = trade.get('cost', 0) + trade.get('revenue', 0)
                transaction_cost += trade_value * self.transaction_cost_pct
        
        # Return profit minus transaction costs
        return (profit - transaction_cost) * self.base_scaling


class RiskAdjustedReward(BaseRewardFunction):
    """Comprehensive risk-adjusted reward function"""
    
    def __init__(self, return_weight: float = 1.0, risk_weight: float = 0.5, 
                 benchmark_weight: float = 0.3):
        self.return_weight = return_weight
        self.risk_weight = risk_weight
        self.benchmark_weight = benchmark_weight
    
    def calculate_reward(self, data: Dict[str, Any]) -> float:
        """Calculate comprehensive risk-adjusted reward"""
        portfolio_value = data.get('portfolio_value', 0)
        
        if hasattr(data.get('env'), 'asset_memory') and len(data['env'].asset_memory) > 10:
            # Portfolio returns
            values = pd.Series(data['env'].asset_memory)
            returns = values.pct_change().dropna()
            
            # Return component
            mean_return = returns.mean()
            
            # Risk component (volatility)
            volatility = returns.std()
            
            # Benchmark comparison (if available)
            benchmark_excess = 0
            if hasattr(data.get('env'), 'df') and 'close_benchmark' in data['env'].df.columns:
                benchmark_data = data['env'].df['close_benchmark'].iloc[:len(values)]
                benchmark_returns = benchmark_data.pct_change().dropna()
                if len(benchmark_returns) > 0:
                    benchmark_excess = mean_return - benchmark_returns.mean()
            
            # Combine components
            reward = (self.return_weight * mean_return - 
                     self.risk_weight * volatility + 
                     self.benchmark_weight * benchmark_excess)
            
            return reward
        
        # Fallback to simple profit reward
        previous_value = data.get('previous_portfolio_value', portfolio_value)
        return (portfolio_value - previous_value) * 1e-4
    

def get_reward_function(reward_type: str = "differential", weights: Optional[List[float]] = None) -> RewardCalculator:
    """Factory function to get reward calculator instances"""
    
    reward_functions = {
        "differential": DifferentialReturnReward,
        "sortino": SortinoReward,
        "risk_adjusted": RiskAdjustedReturnReward
    }
    
    if reward_type not in reward_functions:
        raise ValueError(f"Unknown reward type: {reward_type}. Available types: {list(reward_functions.keys())}")
    
    return reward_functions[reward_type](weights)


# Legacy function for backward compatibility
def differential_return_reward_function(env, actions, next_state, base_reward, terminal, truncated, info, *, reward_logging=False):
    """Legacy wrapper for differential return reward function"""
    calculator = DifferentialReturnReward()
    return calculator.calculate(env, actions, next_state, base_reward, terminal, truncated, info, reward_logging)