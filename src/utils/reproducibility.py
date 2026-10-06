"""Reproducibility utilities for deterministic experiments."""
import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds across Python, NumPy, and PyTorch (if available) for reproducibility.
    
    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
