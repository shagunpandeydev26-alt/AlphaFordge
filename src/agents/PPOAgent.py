"""
PPO Agent implementation using Stable Baselines3 for RL trading
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Union
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger
import pandas as pd
from pathlib import Path


class TradingPPOAgent:
    """PPO Agent specifically configured for trading environments"""
    
    def __init__(self, 
                 env,
                 policy: str = "MlpPolicy",
                 learning_rate: float = 3e-4,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 ent_coef: float = 0.0,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 device: str = "auto",
                 verbose: int = 1,
                 **kwargs):
        """
        Initialize PPO agent with trading-specific configurations
        
        Args:
            env: Trading environment
            policy: Policy network type
            learning_rate: Learning rate for optimization
            n_steps: Number of steps to run for each update
            batch_size: Batch size for training
            n_epochs: Number of epochs for policy updates
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clipping parameter
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm
            device: Device to use for training
            verbose: Verbosity level
        """
        
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        
        # Create vectorized environment
        self.vec_env = make_vec_env(lambda: Monitor(env, None), n_envs=1)
        
        # Initialize PPO model
        self.model = PPO(
            policy=policy,
            env=self.vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            device=self.device,
            verbose=verbose,
            **kwargs
        )
        
        self.training_history = []
        self.evaluation_history = []
    
    def train(self, 
              total_timesteps: int,
              callback=None,
              log_interval: int = 1,
              tb_log_name: str = "PPO",
              reset_num_timesteps: bool = True) -> None:
        """
        Train the PPO agent
        
        Args:
            total_timesteps: Total timesteps for training
            callback: Callback function for training
            log_interval: Logging interval
            tb_log_name: Tensorboard log name
            reset_num_timesteps: Whether to reset timesteps
        """
        
        print(f"Starting PPO training for {total_timesteps} timesteps on {self.device}")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps
        )
        
        print("Training completed successfully")
    
    def predict(self, 
                observation,
                deterministic: bool = True,
                return_state: bool = False):
        """
        Predict action for given observation
        
        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic policy
            return_state: Whether to return internal state
            
        Returns:
            Predicted action and optionally internal state
        """
        
        return self.model.predict(observation, deterministic=deterministic)
    
    def evaluate(self, 
                 n_episodes: int = 10,
                 deterministic: bool = True,
                 render: bool = False) -> Dict[str, Any]:
        """
        Evaluate the trained agent
        
        Args:
            n_episodes: Number of episodes for evaluation
            deterministic: Whether to use deterministic policy
            render: Whether to render environment
            
        Returns:
            Dictionary with evaluation results
        """
        
        print(f"Evaluating agent for {n_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        portfolio_values = []
        
        for episode in range(n_episodes):
            obs = self.vec_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, done, info = self.vec_env.step(action)
                
                episode_reward += reward[0]
                episode_length += 1
                
                if done[0]:
                    final_value = self.vec_env.envs[0].unwrapped.asset_memory[-1]
                    portfolio_values.append(final_value)
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # Calculate evaluation metrics
        eval_results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'mean_portfolio_value': np.mean(portfolio_values),
            'std_portfolio_value': np.std(portfolio_values),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'portfolio_values': portfolio_values
        }
        
        # Calculate additional financial metrics
        if portfolio_values:
            initial_value = 10000  # Assuming standard initial amount
            returns = [(pv - initial_value) / initial_value * 100 for pv in portfolio_values]
            eval_results['mean_return'] = np.mean(returns)
            eval_results['std_return'] = np.std(returns)
            eval_results['sharpe_ratio'] = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        self.evaluation_history.append(eval_results)
        
        print(f"Evaluation completed:")
        print(f"  Mean reward: {eval_results['mean_reward']:.4f} ± {eval_results['std_reward']:.4f}")
        print(f"  Mean portfolio value: ${eval_results['mean_portfolio_value']:.2f} ± ${eval_results['std_portfolio_value']:.2f}")
        if 'mean_return' in eval_results:
            print(f"  Mean return: {eval_results['mean_return']:.2f}% ± {eval_results['std_return']:.2f}%")
            print(f"  Sharpe ratio: {eval_results['sharpe_ratio']:.4f}")
        
        return eval_results
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the trained model
        
        Args:
            path: Path to save the model
        """
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if not str(path).endswith('.zip'):
            path = path.with_suffix('.zip')
        
        self.model.save(str(path))
        print(f"Model saved to {path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """
        Load a trained model
        
        Args:
            path: Path to load the model from
        """
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self.model = PPO.load(str(path), env=self.vec_env)
        print(f"Model loaded from {path}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training statistics
        
        Returns:
            Dictionary with training statistics
        """
        
        return {
            'total_timesteps': self.model.num_timesteps,
            'device': str(self.device),
            'policy': self.model.policy.__class__.__name__,
            'learning_rate': self.model.learning_rate,
            'training_history': self.training_history,
            'evaluation_history': self.evaluation_history
        }
    
    def update_hyperparameters(self, **kwargs) -> None:
        """
        Update model hyperparameters
        
        Args:
            **kwargs: Hyperparameters to update
        """
        
        for param, value in kwargs.items():
            if hasattr(self.model, param):
                setattr(self.model, param, value)
                print(f"Updated {param} to {value}")
            else:
                print(f"Warning: {param} is not a valid hyperparameter")


class TradingCallback(BaseCallback):
    """Custom callback for training monitoring"""
    
    def __init__(self, 
                 eval_freq: int = 1000,
                 n_eval_episodes: int = 5,
                 verbose: int = 0):
        """
        Initialize callback
        
        Args:
            eval_freq: Frequency of evaluation
            n_eval_episodes: Number of episodes for evaluation
            verbose: Verbosity level
        """
        
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.evaluations = []
    
    def _on_step(self) -> bool:
        """
        Called at each training step
        
        Returns:
            Whether to continue training
        """
        
        if self.n_calls % self.eval_freq == 0:
            # Perform evaluation
            obs = self.training_env.reset()
            episode_rewards = []
            
            for _ in range(self.n_eval_episodes):
                episode_reward = 0
                done = False
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.training_env.step(action)
                    episode_reward += reward[0]
                    
                    if done[0]:
                        break
                
                episode_rewards.append(episode_reward)
                obs = self.training_env.reset()
            
            mean_reward = np.mean(episode_rewards)
            self.evaluations.append(mean_reward)
            
            if self.verbose > 0:
                print(f"Eval at step {self.n_calls}: Mean reward = {mean_reward:.4f}")
        
        return True


def create_ppo_agent(env, 
                    config: Optional[Dict[str, Any]] = None) -> TradingPPOAgent:
    """
    Factory function to create PPO agent with default trading configurations
    
    Args:
        env: Trading environment
        config: Optional configuration dictionary
        
    Returns:
        Configured PPO agent
    """
    
    default_config = {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.0,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'verbose': 1
    }
    
    if config:
        default_config.update(config)
    
    return TradingPPOAgent(env, **default_config)