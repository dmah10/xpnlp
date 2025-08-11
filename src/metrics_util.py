"""
Utility functions for computing sensitivity metrics in transformer model explanations.

This module provides helper functions for:
1. Identifying important tokens based on explanation scores
2. Converting soft explanations to discrete rationales
3. Parsing evaluation arguments
"""

import numpy as np
from typing import List, Optional, Union, Dict, Any


def _get_id_tokens_greater_th(
    soft_score_explanation: np.ndarray,
    th: float,
    only_pos: Optional[bool] = None
) -> np.ndarray:
    """Get indices of tokens with scores greater than threshold.
    
    Args:
        soft_score_explanation: Array of token importance scores
        th: Threshold value
        only_pos: Whether to consider only positive scores (unused)
        
    Returns:
        Array of token indices exceeding threshold
    """
    return np.where(soft_score_explanation > th)[0]


def _get_id_tokens_top_k(
    soft_score_explanation: np.ndarray,
    k: int,
    only_pos: bool = True
) -> Optional[np.ndarray]:
    """Get indices of top-k scoring tokens.
    
    Args:
        soft_score_explanation: Array of token importance scores
        k: Number of top tokens to select
        only_pos: Whether to consider only positive scores
        
    Returns:
        Array of top-k token indices, or None if no tokens selected
    """
    if only_pos:
        id_top_k = [
            i 
            for i in np.array(soft_score_explanation).argsort()[-k:][::-1]
            if soft_score_explanation[i] > 0
        ]
    else:
        soft_score_explanation = np.abs(np.array(soft_score_explanation))
        id_top_k = np.array(soft_score_explanation).argsort()[-k:][::-1]

    return np.array(id_top_k) if len(id_top_k) > 0 else None


def _get_id_tokens_percentage(
    soft_score_explanation: np.ndarray,
    percentage: float,
    only_pos: bool = True
) -> Optional[np.ndarray]:
    """Get indices of top percentage of tokens.
    
    Args:
        soft_score_explanation: Array of token importance scores
        percentage: Fraction of tokens to select (0-1)
        only_pos: Whether to consider only positive scores
        
    Returns:
        Array of token indices, or None if no tokens selected
    """
    v = int(percentage * len(soft_score_explanation))
    if v > 0 and v <= len(soft_score_explanation):
        return _get_id_tokens_top_k(soft_score_explanation, v, only_pos=only_pos)
    return None


def get_discrete_explanation_topK(
    score_explanation: np.ndarray,
    topK: int,
    only_pos: bool = False
) -> Optional[List[int]]:
    """Convert soft scores to discrete rationale using top-K tokens.
    
    Args:
        score_explanation: Array of token importance scores
        topK: Number of top tokens to include in rationale
        only_pos: Whether to consider only positive scores
        
    Returns:
        Binary mask (1 for selected tokens, 0 otherwise), or None if no tokens selected
    """
    # Get top-k token indices
    topk_indices = _get_id_tokens_top_k(score_explanation, topK, only_pos=only_pos)
    if topk_indices is None:
        return None
        
    # Create binary mask
    return [1 if i in topk_indices else 0 for i in range(len(score_explanation))]


def _check_and_define_get_id_discrete_rationale_function(based_on: str):
    """Get appropriate function for discrete rationale generation.
    
    Args:
        based_on: Method to use ('th', 'k', or 'perc')
        
    Returns:
        Function to generate discrete rationale
        
    Raises:
        ValueError: If based_on is not supported
    """
    if based_on == "th":
        return _get_id_tokens_greater_th
    elif based_on == "k":
        return _get_id_tokens_top_k
    elif based_on == "perc":
        return _get_id_tokens_percentage
    raise ValueError(f"{based_on} type not supported. Specify th, k or perc.")


def parse_evaluator_args(evaluator_args: Dict[str, Any]) -> tuple:
    """Parse arguments for evaluator configuration.
    
    Args:
        evaluator_args: Dictionary of evaluation parameters
        
    Returns:
        Tuple of:
        - remove_first_last: Whether to remove [CLS] and [SEP] tokens
        - only_pos: Whether to consider only positive scores
        - removal_args: Configuration for token removal
        - top_k_hard_rationale: Number of tokens for hard rationale
    """
    # Default: omit [CLS] and [SEP] tokens
    remove_first_last = evaluator_args.get("remove_first_last", True)
    
    # Default: consider only positive influence
    only_pos = evaluator_args.get("only_pos", True)
    
    # Default removal configuration (10% to 100% of tokens)
    removal_args = {
        "remove_tokens": True,
        "based_on": "perc",
        "thresholds": np.arange(0.1, 1.1, 0.1),
    }
    if removal_args_input := evaluator_args.get("removal_args"):
        removal_args.update(removal_args_input)
    
    # Default: 5 tokens for hard rationale
    top_k_hard_rationale = evaluator_args.get("top_k_rationale", 5)
    
    return remove_first_last, only_pos, removal_args, top_k_hard_rationale


def parse_evaluator_args_model(evaluator_args: Dict[str, Any]) -> tuple:
    """Parse arguments for model-based evaluator configuration.
    
    Extended version of parse_evaluator_args that includes model name.
    
    Args:
        evaluator_args: Dictionary of evaluation parameters
        
    Returns:
        Tuple of:
        - remove_first_last: Whether to remove [CLS] and [SEP] tokens
        - only_pos: Whether to consider only positive scores
        - removal_args: Configuration for token removal
        - top_k_hard_rationale: Number of tokens for hard rationale
        - name_model: Name of the model to use
    """
    # Get base parameters
    remove_first_last = evaluator_args.get("remove_first_last", True)
    only_pos = evaluator_args.get("only_pos", True)
    name_model = evaluator_args.get("name", None)
    
    # Default removal configuration
    removal_args = {
        "remove_tokens": True,
        "based_on": "perc",
        "thresholds": np.arange(0.1, 1.1, 0.1),
    }
    if removal_args_input := evaluator_args.get("removal_args"):
        removal_args.update(removal_args_input)
    
    # Default: 5 tokens for hard rationale
    top_k_hard_rationale = evaluator_args.get("top_k_rationale", 5)
    
    return remove_first_last, only_pos, removal_args, top_k_hard_rationale, name_model
