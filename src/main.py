"""
Main entry point for the RL Trading Agent project
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.train.train import train_single_stock, train_multiple_stocks
from src.train.config import TrainingConfig, FAST_TRAINING_CONFIG, PRODUCTION_CONFIG, DEBUG_CONFIG
from src.data.data_loader import load_stock_data
from src.envs.SingleStockTradingEnv import SingleStockTradingEnv
from src.agents.PPOAgent import TradingPPOAgent, create_ppo_agent
from src.utils.metrics import calculate_trading_metrics, create_performance_report
from src.utils.logger import setup_logger


def train_model(args):
    """Training mode"""
    print(f"🚀 Training RL model for {args.ticker}")
    
    # Select configuration
    if args.config == "fast":
        base_config = FAST_TRAINING_CONFIG
    elif args.config == "production":
        base_config = PRODUCTION_CONFIG
    elif args.config == "debug":
        base_config = DEBUG_CONFIG
    else:
        base_config = TrainingConfig()
    
    # Override with command line arguments
    config_dict = base_config.to_dict()
    config_dict.update({
        'ticker': args.ticker,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'total_timesteps': args.timesteps,
        'verbose': args.verbose
    })
    
    config = TrainingConfig.from_dict(config_dict)
    
    # Train the model
    results = train_single_stock(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        total_timesteps=args.timesteps,
        verbose=args.verbose
    )
    
    print("✅ Training completed successfully!")
    print(f"📈 Final portfolio value: ${results['evaluation_results']['mean_portfolio_value']:.2f}")
    print(f"💾 Model saved to: {results['model_path']}")


def evaluate_model(args):
    """Evaluation mode"""
    print(f"📊 Evaluating model for {args.ticker}")
    
    # Load data
    df = load_stock_data(args.ticker, args.start_date, args.end_date)
    
    # Create environment
    env = SingleStockTradingEnv(df=df)
    
    # Load trained model
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    agent = create_ppo_agent(env)
    agent.load(model_path)
    
    # Evaluate
    results = agent.evaluate(n_episodes=args.n_episodes, deterministic=True)
    
    # Calculate detailed metrics
    portfolio_values = results['portfolio_values']
    benchmark_data = df['close_benchmark'].values
    metrics = calculate_trading_metrics(portfolio_values, benchmark_data)
    
    # Print report
    report = create_performance_report(metrics)
    print(report)


def inference_mode(args):
    """Inference mode for making predictions"""
    print(f"🔮 Running inference for {args.ticker}")
    
    # Load data
    df = load_stock_data(args.ticker, args.start_date, args.end_date)
    
    # Create environment
    env = SingleStockTradingEnv(
        df=df,
        initial_amount=args.portfolio_value,
        num_stock_shares=[args.num_shares]
    )
    
    # Load trained model
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    agent = create_ppo_agent(env)
    agent.load(model_path)
    
    # Make prediction
    obs, _ = env.reset()
    action, _ = agent.predict(obs, deterministic=True)
    
    # Interpret action
    action_value = int(action[0] * env.hmax)
    
    if action_value > 0:
        recommendation = "BUY"
        quantity = action_value
    elif action_value < 0:
        recommendation = "SELL" 
        quantity = abs(action_value)
    else:
        recommendation = "HOLD"
        quantity = 0
    
    print(f"🎯 Recommendation: {recommendation}")
    print(f"📊 Quantity: {quantity} shares")
    print(f"💰 Current portfolio value: ${args.portfolio_value:.2f}")
    print(f"🏷️  Current holdings: {args.num_shares} shares")


def batch_train(args):
    """Batch training mode for multiple stocks"""
    tickers = args.tickers.split(',')
    print(f"🔄 Batch training for tickers: {tickers}")
    
    results = train_multiple_stocks(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        total_timesteps=args.timesteps,
        verbose=args.verbose
    )
    
    # Summary report
    print("\n" + "="*60)
    print("BATCH TRAINING SUMMARY")
    print("="*60)
    
    for ticker, result in results.items():
        if 'error' in result:
            print(f"❌ {ticker}: FAILED - {result['error']}")
        else:
            final_value = result['evaluation_results']['mean_portfolio_value']
            print(f"✅ {ticker}: SUCCESS - Final value: ${final_value:.2f}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="RL Trading Agent")
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Training mode
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--ticker', type=str, default='GOOGL', help='Stock ticker')
    train_parser.add_argument('--start_date', type=str, default='2015-01-01', help='Start date')
    train_parser.add_argument('--end_date', type=str, default='2025-01-01', help='End date')
    train_parser.add_argument('--timesteps', type=int, default=100000, help='Training timesteps')
    train_parser.add_argument('--config', type=str, choices=['fast', 'production', 'debug'], 
                             default='production', help='Training configuration')
    train_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Evaluation mode
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--ticker', type=str, required=True, help='Stock ticker')
    eval_parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    eval_parser.add_argument('--start_date', type=str, default='2020-01-01', help='Start date')
    eval_parser.add_argument('--end_date', type=str, default='2025-01-01', help='End date')
    eval_parser.add_argument('--n_episodes', type=int, default=10, help='Number of evaluation episodes')
    
    # Inference mode
    infer_parser = subparsers.add_parser('predict', help='Make trading predictions')
    infer_parser.add_argument('--ticker', type=str, required=True, help='Stock ticker')
    infer_parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    infer_parser.add_argument('--portfolio_value', type=float, default=10000, help='Current portfolio value')
    infer_parser.add_argument('--num_shares', type=int, default=0, help='Current number of shares')
    infer_parser.add_argument('--start_date', type=str, default='2020-01-01', help='Start date')
    infer_parser.add_argument('--end_date', type=str, default='2025-01-01', help='End date')
    
    # Batch training mode
    batch_parser = subparsers.add_parser('batch_train', help='Train models for multiple stocks')
    batch_parser.add_argument('--tickers', type=str, required=True, 
                             help='Comma-separated list of tickers (e.g., GOOGL,AAPL,MSFT)')
    batch_parser.add_argument('--start_date', type=str, default='2015-01-01', help='Start date')
    batch_parser.add_argument('--end_date', type=str, default='2025-01-01', help='End date')
    batch_parser.add_argument('--timesteps', type=int, default=100000, help='Training timesteps')
    batch_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'evaluate':
        evaluate_model(args)
    elif args.mode == 'predict':
        inference_mode(args)
    elif args.mode == 'batch_train':
        batch_train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()