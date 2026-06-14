"""
AlphaFordge - A modular reinforcement learning trading system.

This package provides a complete RL trading solution with modular components
for data loading, environment management, agent training, inference, and evaluation.
"""

__version__ = "1.0.0"
__author__ = "AlphaFordge Team"

# Lazily-imported submodules — the heavy finrl/stable-baselines3/streamlit
# stack is loaded only when actually accessed, so that importing the top-level
# ``src`` package (e.g. during test collection) does not force the whole RL
# stack + wrds + seaborn into memory.
__lazy_map = {
    "SingleStockTradingEnv": ".envs",
    "TradingPPOAgent": ".agents",
    "TradingTrainer": ".train",
    "TrainingConfig": ".train",
    "DataLoader": ".data",
    "setup_logger": ".utils",
    "PerformanceMetrics": ".utils",
    "TradingInferenceEngine": ".inference",
}

__all__ = list(__lazy_map)


def __getattr__(name):
    if name in __lazy_map:
        import importlib

        mod = importlib.import_module(__lazy_map[name], __name__)
        attr = getattr(mod, name)
        setattr(__import__(__name__), name, attr)
        return attr
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__():
    return sorted(set(super().__dir__()) | set(__lazy_map))