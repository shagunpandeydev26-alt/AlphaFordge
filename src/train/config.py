"""
Training configuration classes and utilities
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration class for training RL trading agents"""
    
    # Data parameters
    ticker: str = "GOOGL"
    start_date: str = "2015-01-01"
    end_date: str = "2025-01-01"
    
    # Environment parameters
    initial_amount: int = 10000
    hmax: int = 100
    transaction_cost: float = 0.001
    turbulence_threshold: int = 100
    reward_scaling: float = 1e-4
    reward_type: str = "differential"
    reward_weights: Optional[List[float]] = field(default_factory=lambda: [0.1, 0.01, 0.01, 1.0])
    
    # Training parameters
    total_timesteps: int = 100000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Evaluation parameters
    eval_freq: int = 10000
    n_eval_episodes: int = 5
    
    # Output parameters
    model_dir: str = "models"
    log_dir: str = "logs"
    results_dir: str = "results"
    
    # General parameters
    verbose: bool = True
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        
        # Ensure directories are Path objects
        self.model_dir = Path(self.model_dir)
        self.log_dir = Path(self.log_dir)
        self.results_dir = Path(self.results_dir)
        
        # Validate parameters
        if self.initial_amount <= 0:
            raise ValueError("initial_amount must be positive")
        
        if self.hmax <= 0:
            raise ValueError("hmax must be positive")
        
        if not 0 <= self.transaction_cost <= 1:
            raise ValueError("transaction_cost must be between 0 and 1")
        
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be between 0 and 1")
        
        if self.reward_weights and len(self.reward_weights) != 4:
            raise ValueError("reward_weights must have exactly 4 elements")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create configuration from dictionary"""
        
        return cls(**config_dict)
    
    def save(self, path: Path) -> None:
        """Save configuration to JSON file"""
        import json
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'TrainingConfig':
        """Load configuration from JSON file"""
        import json
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)


@dataclass
class GridSearchConfig:
    """Configuration for hyperparameter grid search"""
    
    # Base configuration
    base_config: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Parameter grids
    learning_rates: List[float] = field(default_factory=lambda: [1e-4, 3e-4, 1e-3])
    batch_sizes: List[int] = field(default_factory=lambda: [32, 64, 128])
    n_epochs: List[int] = field(default_factory=lambda: [5, 10, 20])
    reward_weights: List[List[float]] = field(default_factory=lambda: [
        [0.1, 0.01, 0.01, 1.0],
        [0.2, 0.02, 0.02, 2.0],
        [0.05, 0.005, 0.005, 0.5]
    ])
    
    # Grid search parameters
    n_trials: int = 20  # Number of random trials if using random search
    cv_folds: int = 3   # Number of cross-validation folds
    metric: str = "mean_portfolio_value"  # Metric to optimize
    
    def get_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Get all parameter combinations for grid search"""
        import itertools
        
        combinations = []
        
        for lr, bs, ne, rw in itertools.product(
            self.learning_rates,
            self.batch_sizes, 
            self.n_epochs,
            self.reward_weights
        ):
            combinations.append({
                'learning_rate': lr,
                'batch_size': bs,
                'n_epochs': ne,
                'reward_weights': rw
            })
        
        return combinations


# Predefined configurations for different use cases
FAST_TRAINING_CONFIG = TrainingConfig(
    total_timesteps=50000,
    n_steps=1024,
    batch_size=32,
    n_epochs=5,
    eval_freq=5000,
    verbose=False
)

PRODUCTION_CONFIG = TrainingConfig(
    total_timesteps=200000,
    n_steps=4096,
    batch_size=128,
    n_epochs=15,
    eval_freq=20000,
    learning_rate=1e-4,
    verbose=True
)

DEBUG_CONFIG = TrainingConfig(
    total_timesteps=10000,
    n_steps=512,
    batch_size=16,
    n_epochs=3,
    eval_freq=1000,
    verbose=True
)

# Stock-specific configurations
TECH_STOCK_CONFIG = TrainingConfig(
    hmax=50,  # Lower max holdings for volatile tech stocks
    turbulence_threshold=150,
    reward_weights=[0.15, 0.02, 0.02, 1.5]
)

STABLE_STOCK_CONFIG = TrainingConfig(
    hmax=200,  # Higher max holdings for stable stocks
    turbulence_threshold=80,
    reward_weights=[0.05, 0.005, 0.005, 0.8]
)
