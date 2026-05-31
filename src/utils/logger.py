"""
Logging utilities for RL trading project
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(name: str, 
                log_dir: Optional[str] = None,
                level: int = logging.INFO,
                console_output: bool = True,
                file_output: bool = True) -> logging.Logger:
    """
    Set up a logger with console and file handlers
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console_output: Whether to output to console
        file_output: Whether to output to file
        
    Returns:
        Configured logger
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_output and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Log file created: {log_file}")
    
    return logger


class TrainingLogger:
    """Custom logger for training progress"""
    
    def __init__(self, name: str, log_dir: Optional[str] = None):
        self.logger = setup_logger(name, log_dir)
        self.training_metrics = []
    
    def log_training_start(self, config):
        """Log training start"""
        self.logger.info("="*50)
        self.logger.info("TRAINING STARTED")
        self.logger.info("="*50)
        self.logger.info(f"Configuration: {config}")
    
    def log_epoch(self, epoch: int, metrics: dict):
        """Log epoch metrics"""
        self.logger.info(f"Epoch {epoch}: {metrics}")
        self.training_metrics.append({'epoch': epoch, **metrics})
    
    def log_evaluation(self, eval_metrics: dict):
        """Log evaluation results"""
        self.logger.info("EVALUATION RESULTS:")
        for metric, value in eval_metrics.items():
            self.logger.info(f"  {metric}: {value}")
    
    def log_training_end(self, final_metrics: dict):
        """Log training completion"""
        self.logger.info("="*50)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("="*50)
        self.logger.info(f"Final metrics: {final_metrics}")
    
    def save_metrics(self, filepath: Path):
        """Save training metrics to file"""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        
        self.logger.info(f"Training metrics saved to {filepath}")


def log_memory_usage(logger: logging.Logger):
    """Log current memory usage"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    except ImportError:
        logger.warning("psutil not available, cannot log memory usage")


def log_gpu_usage(logger: logging.Logger):
    """Log GPU usage if available"""
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_cached = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(f"GPU {i}: {memory_allocated:.2f}GB allocated, {memory_cached:.2f}GB cached")
        else:
            logger.info("No GPU available")
    except ImportError:
        logger.warning("PyTorch not available, cannot log GPU usage")
