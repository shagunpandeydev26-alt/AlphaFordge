"""
Agents module for the RL Trading Agent.

This module provides RL agent implementations for trading,
including PPO and other algorithm variants.
"""

from .PPOAgent import TradingPPOAgent, create_ppo_agent

__all__ = ['TradingPPOAgent', 'create_ppo_agent']