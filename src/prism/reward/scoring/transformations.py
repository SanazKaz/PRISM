"""
Simple score transformations for molecular optimization.
"""

import math
import numpy as np
import torch


def sigmoid(x: float, k: float = 1.0, center: float = 0.0) -> float:
    """
    Standard sigmoid function: σ(x) = 1 / (1 + e^(-k*(x - center)))
    
    Args:
        x: Input value
        k: Steepness parameter (higher = steeper curve)
        center: Center point where sigmoid = 0.5
        
    Returns:
        float: Value between 0 and 1
    """
    try:
        return 1.0 / (1.0 + math.exp(-k * (x - center)))
    except OverflowError:
        # Handle extreme values
        return 0.0 if x < center else 1.0


def reverse_sigmoid(x: float, k: float = 1.0, center: float = 0.0) -> float:
    """
    Reverse sigmoid function: σ(x) = 1 / (1 + e^(k*(x - center)))
    Goes from 1 to 0 as x increases.
    """
    try:
        return 1.0 / (1.0 + math.exp(k * (x - center)))
    except OverflowError:
        return 1.0 if x < center else 0.0


def double_sigmoid(x: float, 
                  low: float, 
                  high: float, 
                  coef_si: float = 20.0, 
                  coef_se: float = 20.0,
                  coef_div: float = 1.0) -> float:
    """
    Double sigmoid transformation creating a plateau between low and high values.
    Same as REINVENT's double_sigmoid.
    
    Args:
        x: Input value to transform
        low: Lower bound of target range
        high: Upper bound of target range
        coef_si: Steepness of left sigmoid (entry slope)
        coef_se: Steepness of right sigmoid (exit slope)
        coef_div: Normalization factor for input
        
    Returns:
        float: Transformed score between 0 and 1
    """
    # Normalize input
    x_norm = x / coef_div
    low_norm = low / coef_div  
    high_norm = high / coef_div
    
    # Left sigmoid (increases towards low)
    left_sigmoid = sigmoid(x_norm, k=coef_si, center=low_norm)
    
    # Right sigmoid (decreases after high)
    right_sigmoid = reverse_sigmoid(x_norm, k=coef_se, center=high_norm)
    
    # Combine both sigmoids
    return left_sigmoid * right_sigmoid

# =============================================================================
# PART 2: BATCH TRANSFORMATIONS
# Use this in scorer.py AFTER collecting the whole batch of scores.
# =============================================================================

def reshape_batch_rewards(rewards: torch.Tensor, valid_mask: torch.Tensor, config: dict) -> torch.Tensor:
    """
    Applies statistical transformations to the whole batch tensor.
    """
    shaped_rewards = rewards.clone()
    
    # 1. Centering (Subtract Batch Mean)
    if config.get('center_mean', False) and valid_mask.any():
        valid_scores = shaped_rewards[valid_mask]
        mean = valid_scores.mean()
        shaped_rewards[valid_mask] = valid_scores - mean

    # 2. Normalization (Divide by Std)
    if config.get('normalize_std', False) and valid_mask.any():
        valid_scores = shaped_rewards[valid_mask]
        std = valid_scores.std()
        if std > 1e-8:
            shaped_rewards[valid_mask] = valid_scores / std

    return shaped_rewards