"""
RL Trading Agent - A modular reinforcement learning trading system.

This package provides a complete RL trading solution with modular components
for data loading, environment management, agent training, inference, and evaluation.
"""

__version__ = "1.0.0"
__author__ = "RL Trading Team"

# Import main components for easy access
from .envs import SingleStockTradingEnv
from .agents import TradingPPOAgent
from .train import TradingTrainer, TrainingConfig
from .data import DataLoader
from .utils import setup_logger, PerformanceMetrics
from .inference import TradingInferenceEngine

__all__ = [
    'SingleStockTradingEnv',
    'TradingPPOAgent', 
    'TradingTrainer',
    'TrainingConfig',
    'DataLoader',
    'setup_logger',
    'PerformanceMetrics',
    'TradingInferenceEngine'
]