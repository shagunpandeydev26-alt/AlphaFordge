"""
Financial and performance metrics for RL trading evaluation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats


def calculate_returns(portfolio_values: List[float]) -> np.ndarray:
    """Calculate daily returns from portfolio values"""
    returns = np.array(portfolio_values)
    returns = np.diff(returns) / returns[:-1]
    return returns


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Array of daily returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    daily_rf_rate = risk_free_rate / 252
    excess_returns = returns - daily_rf_rate
    
    return np.sqrt(252) * np.mean(excess_returns) / np.std(returns)


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sortino ratio
    
    Args:
        returns: Array of daily returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    daily_rf_rate = risk_free_rate / 252
    excess_returns = returns - daily_rf_rate
    
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return float('inf')
    
    downside_std = np.std(downside_returns)
    if downside_std == 0:
        return 0.0
    
    return np.sqrt(252) * np.mean(excess_returns) / downside_std


def calculate_max_drawdown(portfolio_values: List[float]) -> float:
    """
    Calculate maximum drawdown
    
    Args:
        portfolio_values: List of portfolio values over time
        
    Returns:
        Maximum drawdown as a percentage
    """
    if len(portfolio_values) == 0:
        return 0.0
    
    values = np.array(portfolio_values)
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    
    return np.min(drawdown)


def calculate_calmar_ratio(returns: np.ndarray, portfolio_values: List[float]) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown)
    
    Args:
        returns: Array of daily returns
        portfolio_values: List of portfolio values
        
    Returns:
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0
    
    annualized_return = np.mean(returns) * 252
    max_dd = abs(calculate_max_drawdown(portfolio_values))
    
    if max_dd == 0:
        return float('inf') if annualized_return > 0 else 0.0
    
    return annualized_return / max_dd


def calculate_information_ratio(portfolio_returns: np.ndarray, 
                              benchmark_returns: np.ndarray) -> float:
    """
    Calculate information ratio
    
    Args:
        portfolio_returns: Array of portfolio returns
        benchmark_returns: Array of benchmark returns
        
    Returns:
        Information ratio
    """
    if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
        return 0.0
    
    # Ensure same length
    min_len = min(len(portfolio_returns), len(benchmark_returns))
    portfolio_returns = portfolio_returns[:min_len]
    benchmark_returns = benchmark_returns[:min_len]
    
    excess_returns = portfolio_returns - benchmark_returns
    tracking_error = np.std(excess_returns)
    
    if tracking_error == 0:
        return 0.0
    
    return np.mean(excess_returns) / tracking_error


def calculate_beta(portfolio_returns: np.ndarray, 
                  benchmark_returns: np.ndarray) -> float:
    """
    Calculate portfolio beta
    
    Args:
        portfolio_returns: Array of portfolio returns
        benchmark_returns: Array of benchmark returns
        
    Returns:
        Portfolio beta
    """
    if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
        return 1.0
    
    # Ensure same length
    min_len = min(len(portfolio_returns), len(benchmark_returns))
    portfolio_returns = portfolio_returns[:min_len]
    benchmark_returns = benchmark_returns[:min_len]
    
    if np.var(benchmark_returns) == 0:
        return 1.0
    
    covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    
    return covariance / benchmark_variance


def calculate_treynor_ratio(returns: np.ndarray, 
                           benchmark_returns: np.ndarray,
                           risk_free_rate: float = 0.02) -> float:
    """
    Calculate Treynor ratio
    
    Args:
        returns: Array of portfolio returns
        benchmark_returns: Array of benchmark returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Treynor ratio
    """
    daily_rf_rate = risk_free_rate / 252
    beta = calculate_beta(returns, benchmark_returns)
    
    if beta == 0:
        return 0.0
    
    excess_return = np.mean(returns) - daily_rf_rate
    return (excess_return * 252) / beta


def calculate_var(returns: np.ndarray, confidence_level: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR)
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
        
    Returns:
        VaR value
    """
    if len(returns) == 0:
        return 0.0
    
    return np.percentile(returns, confidence_level * 100)


def calculate_cvar(returns: np.ndarray, confidence_level: float = 0.05) -> float:
    """
    Calculate Conditional Value at Risk (CVaR)
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level
        
    Returns:
        CVaR value
    """
    if len(returns) == 0:
        return 0.0
    
    var = calculate_var(returns, confidence_level)
    tail_returns = returns[returns <= var]
    
    if len(tail_returns) == 0:
        return var
    
    return np.mean(tail_returns)


def calculate_win_rate(returns: np.ndarray) -> float:
    """
    Calculate win rate (percentage of positive returns)
    
    Args:
        returns: Array of returns
        
    Returns:
        Win rate as percentage
    """
    if len(returns) == 0:
        return 0.0
    
    return np.mean(returns > 0) * 100


def calculate_profit_factor(returns: np.ndarray) -> float:
    """
    Calculate profit factor (total gains / total losses)
    
    Args:
        returns: Array of returns
        
    Returns:
        Profit factor
    """
    if len(returns) == 0:
        return 0.0
    
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    
    total_gains = np.sum(gains) if len(gains) > 0 else 0
    total_losses = abs(np.sum(losses)) if len(losses) > 0 else 0
    
    if total_losses == 0:
        return float('inf') if total_gains > 0 else 0.0
    
    return total_gains / total_losses


def calculate_trading_metrics(portfolio_values: List[float],
                            benchmark_data: Optional[np.ndarray] = None,
                            initial_amount: float = 10000,
                            risk_free_rate: float = 0.02) -> Dict[str, Any]:
    """
    Calculate comprehensive trading performance metrics
    
    Args:
        portfolio_values: List of portfolio values over time
        benchmark_data: Optional benchmark values for comparison
        initial_amount: Initial portfolio value
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Dictionary with all calculated metrics
    """
    
    if len(portfolio_values) == 0:
        return {}
    
    # Calculate returns
    returns = calculate_returns(portfolio_values)
    
    # Basic metrics
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_amount) / initial_amount * 100
    annualized_return = (final_value / initial_amount) ** (252 / len(portfolio_values)) - 1
    annualized_return *= 100
    
    # Risk metrics
    volatility = np.std(returns) * np.sqrt(252) * 100
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate)
    sortino = calculate_sortino_ratio(returns, risk_free_rate)
    max_dd = calculate_max_drawdown(portfolio_values) * 100
    calmar = calculate_calmar_ratio(returns, portfolio_values)
    
    # VaR metrics
    var_95 = calculate_var(returns, 0.05) * 100
    cvar_95 = calculate_cvar(returns, 0.05) * 100
    
    # Trading metrics
    win_rate = calculate_win_rate(returns)
    profit_factor = calculate_profit_factor(returns)
    
    metrics = {
        'final_value': final_value,
        'initial_value': initial_amount,
        'total_return_pct': total_return,
        'annualized_return_pct': annualized_return,
        'volatility_pct': volatility,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown_pct': max_dd,
        'calmar_ratio': calmar,
        'var_95_pct': var_95,
        'cvar_95_pct': cvar_95,
        'win_rate_pct': win_rate,
        'profit_factor': profit_factor,
        'num_trades': len(returns),
        'avg_daily_return_pct': np.mean(returns) * 100 if len(returns) > 0 else 0,
    }
    
    # Benchmark comparison metrics
    if benchmark_data is not None and len(benchmark_data) > 0:
        # Align lengths
        min_len = min(len(portfolio_values), len(benchmark_data))
        aligned_portfolio = portfolio_values[:min_len]
        aligned_benchmark = benchmark_data[:min_len]
        
        # Calculate benchmark returns
        benchmark_returns = calculate_returns(aligned_benchmark)
        portfolio_returns_aligned = calculate_returns(aligned_portfolio)
        
        # Benchmark metrics
        benchmark_total_return = (aligned_benchmark[-1] - aligned_benchmark[0]) / aligned_benchmark[0] * 100
        
        beta = calculate_beta(portfolio_returns_aligned, benchmark_returns)
        treynor = calculate_treynor_ratio(portfolio_returns_aligned, benchmark_returns, risk_free_rate)
        info_ratio = calculate_information_ratio(portfolio_returns_aligned, benchmark_returns)
        
        excess_return = total_return - benchmark_total_return
        
        metrics.update({
            'benchmark_total_return_pct': benchmark_total_return,
            'excess_return_pct': excess_return,
            'beta': beta,
            'treynor_ratio': treynor,
            'information_ratio': info_ratio,
        })
    
    return metrics


def create_performance_report(metrics: Dict[str, Any]) -> str:
    """
    Create a formatted performance report
    
    Args:
        metrics: Dictionary of calculated metrics
        
    Returns:
        Formatted string report
    """
    
    report = []
    report.append("="*60)
    report.append("TRADING PERFORMANCE REPORT")
    report.append("="*60)
    
    # Portfolio Performance
    report.append("\n📊 PORTFOLIO PERFORMANCE:")
    report.append(f"  Initial Value:         ${metrics.get('initial_value', 0):,.2f}")
    report.append(f"  Final Value:           ${metrics.get('final_value', 0):,.2f}")
    report.append(f"  Total Return:          {metrics.get('total_return_pct', 0):.2f}%")
    report.append(f"  Annualized Return:     {metrics.get('annualized_return_pct', 0):.2f}%")
    
    # Risk Metrics
    report.append("\n⚠️  RISK METRICS:")
    report.append(f"  Volatility:            {metrics.get('volatility_pct', 0):.2f}%")
    report.append(f"  Maximum Drawdown:      {metrics.get('max_drawdown_pct', 0):.2f}%")
    report.append(f"  VaR (95%):             {metrics.get('var_95_pct', 0):.2f}%")
    report.append(f"  CVaR (95%):            {metrics.get('cvar_95_pct', 0):.2f}%")
    
    # Risk-Adjusted Returns
    report.append("\n📈 RISK-ADJUSTED RETURNS:")
    report.append(f"  Sharpe Ratio:          {metrics.get('sharpe_ratio', 0):.4f}")
    report.append(f"  Sortino Ratio:         {metrics.get('sortino_ratio', 0):.4f}")
    report.append(f"  Calmar Ratio:          {metrics.get('calmar_ratio', 0):.4f}")
    
    # Trading Statistics
    report.append("\n🎯 TRADING STATISTICS:")
    report.append(f"  Win Rate:              {metrics.get('win_rate_pct', 0):.1f}%")
    report.append(f"  Profit Factor:         {metrics.get('profit_factor', 0):.2f}")
    report.append(f"  Number of Trades:      {metrics.get('num_trades', 0)}")
    report.append(f"  Avg Daily Return:      {metrics.get('avg_daily_return_pct', 0):.4f}%")
    
    # Benchmark Comparison (if available)
    if 'benchmark_total_return_pct' in metrics:
        report.append("\n🔍 BENCHMARK COMPARISON:")
        report.append(f"  Benchmark Return:      {metrics.get('benchmark_total_return_pct', 0):.2f}%")
        report.append(f"  Excess Return:         {metrics.get('excess_return_pct', 0):.2f}%")
        report.append(f"  Beta:                  {metrics.get('beta', 0):.4f}")
        report.append(f"  Treynor Ratio:         {metrics.get('treynor_ratio', 0):.4f}")
        report.append(f"  Information Ratio:     {metrics.get('information_ratio', 0):.4f}")
    
    report.append("\n" + "="*60)
    
    return "\n".join(report)


class PerformanceMetrics:
    """
    A comprehensive performance metrics calculator for trading strategies
    """
    
    def __init__(self):
        self.portfolio_values = []
        self.benchmark_values = []
        self.actions = []
        self.prices = []
        
    def add_data_point(self, portfolio_value: float, benchmark_value: float = None, 
                      action: int = None, price: float = None):
        """Add a new data point to the metrics calculator"""
        self.portfolio_values.append(portfolio_value)
        if benchmark_value is not None:
            self.benchmark_values.append(benchmark_value)
        if action is not None:
            self.actions.append(action)
        if price is not None:
            self.prices.append(price)
    
    def calculate_all_metrics(self) -> Dict[str, float]:
        """Calculate all available performance metrics"""
        if len(self.portfolio_values) < 2:
            return {}
            
        # Use the existing calculate_trading_metrics function
        metrics = calculate_trading_metrics(
            portfolio_values=self.portfolio_values,
            benchmark_values=self.benchmark_values,
            actions=self.actions,
            prices=self.prices
        )
        
        return metrics
    
    def get_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Get Sharpe ratio"""
        if len(self.portfolio_values) < 2:
            return 0.0
        returns = calculate_returns(self.portfolio_values)
        return calculate_sharpe_ratio(returns, risk_free_rate)
    
    def get_total_return(self) -> float:
        """Get total return percentage"""
        if len(self.portfolio_values) < 2:
            return 0.0
        return ((self.portfolio_values[-1] - self.portfolio_values[0]) / self.portfolio_values[0]) * 100
    
    def get_max_drawdown(self) -> float:
        """Get maximum drawdown percentage"""
        if len(self.portfolio_values) < 2:
            return 0.0
        return calculate_max_drawdown(self.portfolio_values) * 100
    
    def get_volatility(self) -> float:
        """Get annualized volatility"""
        if len(self.portfolio_values) < 2:
            return 0.0
        returns = calculate_returns(self.portfolio_values)
        return np.sqrt(252) * np.std(returns) * 100  # Annualized volatility as percentage
    
    def reset(self):
        """Reset all data"""
        self.portfolio_values = []
        self.benchmark_values = []
        self.actions = []
        self.prices = []
    
    def generate_report(self) -> str:
        """Generate a comprehensive performance report"""
        metrics = self.calculate_all_metrics()
        return create_performance_report(metrics)
