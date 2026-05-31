# Evaluates trained models using different performance metrics.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
from datetime import datetime

from stable_baselines3 import PPO
from ..envs.SingleStockTradingEnv import SingleStockTradingEnv
from ..utils.metrics import PerformanceMetrics
from ..utils.logger import setup_logger
from ..data.data_loader import DataLoader
from ..inference.inference import TradingInferenceEngine

logger = setup_logger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation class for RL trading agents.
    Provides various evaluation methods and visualization tools.
    """
    
    def __init__(self, models_dir: Union[str, Path] = "models"):
        """
        Initialize the model evaluator.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.data_loader = DataLoader()
        self.metrics = PerformanceMetrics()
        self.inference_engine = TradingInferenceEngine()
        
    def evaluate_single_model(self, 
                            model_path: Union[str, Path],
                            ticker: str,
                            start_date: str,
                            end_date: str,
                            initial_amount: float = 10000) -> Dict[str, Any]:
        """
        Evaluate a single model on specified data.
        
        Args:
            model_path: Path to the trained model
            ticker: Stock ticker symbol
            start_date: Evaluation start date
            end_date: Evaluation end date
            initial_amount: Initial portfolio amount
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Evaluating model {model_path} on {ticker} from {start_date} to {end_date}")
        
        try:
            # Load model and data
            model = PPO.load(model_path)
            data = self.data_loader.load_stock_data(ticker, start_date, end_date)
            
            if data.empty:
                raise ValueError(f"No data available for {ticker} in specified date range")
            
            # Run inference
            results = self.inference_engine.batch_predict(
                model, ticker, start_date, end_date, initial_amount
            )
            
            # Calculate additional metrics
            portfolio_values = np.array(results['portfolio_values'])
            daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            benchmark_returns = data['close'].pct_change().dropna()
            
            # Comprehensive evaluation metrics
            evaluation = {
                'model_path': str(model_path),
                'ticker': ticker,
                'period': f"{start_date} to {end_date}",
                'initial_amount': initial_amount,
                'final_value': results['final_value'],
                'total_return': results['metrics']['total_return'],
                'total_return_pct': results['metrics']['total_return'] * 100,
                'annualized_return': self._annualize_return(results['metrics']['total_return'], 
                                                          len(portfolio_values)),
                'sharpe_ratio': results['metrics']['sharpe_ratio'],
                'max_drawdown': results['metrics']['max_drawdown'],
                'max_drawdown_pct': results['metrics']['max_drawdown'] * 100,
                'volatility': results['metrics']['volatility'],
                'benchmark_return': results['metrics']['benchmark_return'],
                'alpha': results['metrics']['total_return'] - results['metrics']['benchmark_return'],
                'win_rate': np.mean(np.array(results['rewards']) > 0),
                'avg_reward': np.mean(results['rewards']),
                'profit_factor': self._calculate_profit_factor(results['rewards']),
                'calmar_ratio': self._calculate_calmar_ratio(
                    results['metrics']['total_return'], 
                    results['metrics']['max_drawdown']
                ),
                'sortino_ratio': self._calculate_sortino_ratio(daily_returns),
                'var_95': np.percentile(daily_returns, 5),
                'var_99': np.percentile(daily_returns, 1),
                'action_distribution': self._analyze_actions(results['actions']),
                'trading_frequency': len([a for a in results['actions'] if abs(a) > 0.1]) / len(results['actions']),
                'raw_results': results
            }
            
            logger.info(f"Evaluation completed. Total return: {evaluation['total_return_pct']:.2f}%")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def compare_models(self, 
                      model_configs: List[Dict],
                      ticker: str,
                      start_date: str,
                      end_date: str) -> pd.DataFrame:
        """
        Compare multiple models on the same dataset.
        
        Args:
            model_configs: List of model config dictionaries with 'path' and 'name'
            ticker: Stock ticker symbol
            start_date: Evaluation start date
            end_date: Evaluation end date
            
        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing {len(model_configs)} models on {ticker}")
        
        results = []
        for config in model_configs:
            try:
                evaluation = self.evaluate_single_model(
                    config['path'], ticker, start_date, end_date
                )
                evaluation['model_name'] = config.get('name', Path(config['path']).stem)
                results.append(evaluation)
            except Exception as e:
                logger.error(f"Failed to evaluate model {config.get('name', 'unknown')}: {e}")
                continue
        
        if not results:
            raise ValueError("No models could be evaluated successfully")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame([
            {
                'Model': r['model_name'],
                'Total Return (%)': r['total_return_pct'],
                'Annualized Return (%)': r['annualized_return'] * 100,
                'Sharpe Ratio': r['sharpe_ratio'],
                'Max Drawdown (%)': r['max_drawdown_pct'],
                'Volatility': r['volatility'],
                'Win Rate': r['win_rate'],
                'Calmar Ratio': r['calmar_ratio'],
                'Sortino Ratio': r['sortino_ratio'],
                'Trading Frequency': r['trading_frequency'],
                'Final Value': r['final_value']
            } for r in results
        ])
        
        # Sort by Sharpe ratio
        comparison_df = comparison_df.sort_values('Sharpe Ratio', ascending=False)
        
        return comparison_df
    
    def backtest_model(self, 
                      model_path: Union[str, Path],
                      ticker: str,
                      train_end: str,
                      test_end: str,
                      lookback_days: int = 252) -> Dict[str, Any]:
        """
        Perform walk-forward backtest of a model.
        
        Args:
            model_path: Path to trained model
            ticker: Stock ticker symbol
            train_end: End date of training period
            test_end: End date of test period
            lookback_days: Number of days for each test window
            
        Returns:
            Backtest results dictionary
        """
        logger.info(f"Starting backtest for {ticker} from {train_end} to {test_end}")
        
        model = PPO.load(model_path)
        
        # Generate test periods
        train_end_dt = pd.to_datetime(train_end)
        test_end_dt = pd.to_datetime(test_end)
        
        test_periods = []
        current_date = train_end_dt
        
        while current_date < test_end_dt:
            period_end = min(current_date + pd.Timedelta(days=lookback_days), test_end_dt)
            test_periods.append({
                'start': current_date.strftime('%Y-%m-%d'),
                'end': period_end.strftime('%Y-%m-%d')
            })
            current_date += pd.Timedelta(days=lookback_days)
        
        # Run backtest for each period
        period_results = []
        cumulative_return = 0
        
        for i, period in enumerate(test_periods):
            try:
                result = self.evaluate_single_model(
                    model_path, ticker, period['start'], period['end']
                )
                result['period_number'] = i + 1
                result['period_start'] = period['start']
                result['period_end'] = period['end']
                period_results.append(result)
                cumulative_return += result['total_return']
                
                logger.info(f"Period {i+1}: {result['total_return_pct']:.2f}% return")
                
            except Exception as e:
                logger.warning(f"Failed to evaluate period {period}: {e}")
                continue
        
        # Aggregate results
        if not period_results:
            raise ValueError("No periods could be evaluated successfully")
        
        returns = [r['total_return'] for r in period_results]
        sharpe_ratios = [r['sharpe_ratio'] for r in period_results]
        max_drawdowns = [r['max_drawdown'] for r in period_results]
        
        backtest_summary = {
            'model_path': str(model_path),
            'ticker': ticker,
            'backtest_period': f"{train_end} to {test_end}",
            'num_periods': len(period_results),
            'avg_period_return': np.mean(returns),
            'avg_period_return_pct': np.mean(returns) * 100,
            'cumulative_return': cumulative_return,
            'cumulative_return_pct': cumulative_return * 100,
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'win_rate': np.mean([r > 0 for r in returns]),
            'best_period_return': max(returns),
            'worst_period_return': min(returns),
            'volatility_of_returns': np.std(returns),
            'period_results': period_results
        }
        
        logger.info(f"Backtest completed. Cumulative return: {backtest_summary['cumulative_return_pct']:.2f}%")
        return backtest_summary
    
    def generate_evaluation_report(self, 
                                 evaluation_results: Dict[str, Any],
                                 output_dir: Union[str, Path] = "results") -> Path:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            evaluation_results: Results from model evaluation
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"evaluation_report_{timestamp}.json"
        
        # Prepare report data (remove non-serializable objects)
        report_data = evaluation_results.copy()
        if 'raw_results' in report_data:
            raw_results = report_data.pop('raw_results')
            report_data['summary_stats'] = {
                'num_actions': len(raw_results['actions']),
                'num_rewards': len(raw_results['rewards']),
                'num_portfolio_values': len(raw_results['portfolio_values'])
            }
        
        # Add metadata
        report_data['evaluation_metadata'] = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'evaluator_version': '1.0.0'
        }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {report_path}")
        return report_path
    
    def plot_performance(self, 
                        evaluation_results: Dict[str, Any],
                        save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Create performance visualization plots.
        
        Args:
            evaluation_results: Results from model evaluation
            save_path: Optional path to save the plot
        """
        if 'raw_results' not in evaluation_results:
            raise ValueError("Raw results not available for plotting")
        
        raw_results = evaluation_results['raw_results']
        portfolio_values = raw_results['portfolio_values']
        actions = raw_results['actions']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Performance Analysis - {evaluation_results['ticker']}", fontsize=16)
        
        # Portfolio value over time
        axes[0, 0].plot(portfolio_values, label='Portfolio Value')
        axes[0, 0].axhline(y=evaluation_results['initial_amount'], color='r', linestyle='--', label='Initial Value')
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Trading Days')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Actions over time
        axes[0, 1].plot(actions, alpha=0.7)
        axes[0, 1].set_title('Trading Actions Over Time')
        axes[0, 1].set_xlabel('Trading Days')
        axes[0, 1].set_ylabel('Action Value')
        axes[0, 1].grid(True)
        
        # Action distribution
        axes[1, 0].hist(actions, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Action Distribution')
        axes[1, 0].set_xlabel('Action Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True)
        
        # Drawdown plot
        portfolio_values_np = np.array(portfolio_values)
        running_max = np.maximum.accumulate(portfolio_values_np)
        drawdown = (portfolio_values_np - running_max) / running_max
        axes[1, 1].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.7, color='red')
        axes[1, 1].set_title(f'Drawdown (Max: {evaluation_results["max_drawdown_pct"]:.2f}%)')
        axes[1, 1].set_xlabel('Trading Days')
        axes[1, 1].set_ylabel('Drawdown (%)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plot saved to {save_path}")
        
        plt.show()
    
    # Helper methods
    def _annualize_return(self, total_return: float, num_days: int) -> float:
        """Annualize the total return."""
        if num_days <= 0:
            return 0
        return (1 + total_return) ** (252 / num_days) - 1
    
    def _calculate_profit_factor(self, rewards: List[float]) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        profits = [r for r in rewards if r > 0]
        losses = [abs(r) for r in rewards if r < 0]
        
        if not losses:
            return float('inf')
        if not profits:
            return 0
        
        return sum(profits) / sum(losses)
    
    def _calculate_calmar_ratio(self, total_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        if max_drawdown == 0:
            return float('inf')
        return total_return / abs(max_drawdown)
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio (return / downside deviation)."""
        if len(returns) == 0:
            return 0
        
        mean_return = np.mean(returns)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return float('inf')
        
        return mean_return / downside_deviation
    
    def _analyze_actions(self, actions: List[float]) -> Dict[str, float]:
        """Analyze the distribution of actions."""
        actions_np = np.array(actions)
        
        return {
            'buy_percentage': np.mean(actions_np > 0.1) * 100,
            'sell_percentage': np.mean(actions_np < -0.1) * 100,
            'hold_percentage': np.mean(np.abs(actions_np) <= 0.1) * 100,
            'avg_buy_size': np.mean(actions_np[actions_np > 0.1]) if np.any(actions_np > 0.1) else 0,
            'avg_sell_size': np.mean(np.abs(actions_np[actions_np < -0.1])) if np.any(actions_np < -0.1) else 0
        }
