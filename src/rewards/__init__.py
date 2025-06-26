"""
RLT (Reasoning Like Teaching) Dense Reward System

This module implements the reward computation system for training language models
to generate reasoning that maximizes student understanding while maintaining naturalness.

Key Components:
- RLTRewardFunction: Main reward computation combining rSS and rKL
- StudentEvaluator: Computes solution scores measuring student understanding
- KLDivergenceCalculator: Computes KL divergence from baseline distribution
- RewardUtils: Helper functions for normalization, debugging, and visualization

Example Usage:
    from rewards import RLTRewardFunction, LocalStudentEvaluator, TransformerKLCalculator, RewardConfig
    
    # Initialize components
    student_eval = LocalStudentEvaluator(model_name="microsoft/phi-2")
    kl_calc = TransformerKLCalculator(model_name="gpt2-medium")
    config = RewardConfig(lambda_kl=0.5, normalize=True)
    
    # Create reward function
    reward_fn = RLTRewardFunction(student_eval, kl_calc, config)
    
    # Compute rewards
    rewards = reward_fn.compute_reward(
        reasoning=["First, we identify...", "Let's solve step by step..."],
        problem=["What is 2+2?", "Solve x^2 - 4 = 0"]
    )
"""

# Core reward function
from .reward_function import (
    RLTRewardFunction,
    AdaptiveRewardFunction,
    RewardConfig
)

# Student evaluators
from .student_evaluator import (
    BaseStudentEvaluator,
    LocalStudentEvaluator,
    APIStudentEvaluator,
    EnsembleStudentEvaluator
)

# KL divergence calculators
from .kl_divergence import (
    BaseKLCalculator,
    TransformerKLCalculator,
    SequenceKLCalculator,
    CachedKLCalculator,
    ApproximateKLCalculator
)

# Utilities
from .reward_utils import (
    normalize_rewards,
    compute_baseline_rewards,
    compute_advantage_estimates,
    RewardLogger,
    RewardDebugger
)

__all__ = [
    # Reward functions
    "RLTRewardFunction",
    "AdaptiveRewardFunction",
    "RewardConfig",
    
    # Student evaluators
    "BaseStudentEvaluator",
    "LocalStudentEvaluator",
    "APIStudentEvaluator",
    "EnsembleStudentEvaluator",
    
    # KL calculators
    "BaseKLCalculator",
    "TransformerKLCalculator",
    "SequenceKLCalculator",
    "CachedKLCalculator",
    "ApproximateKLCalculator",
    
    # Utilities
    "normalize_rewards",
    "compute_baseline_rewards",
    "compute_advantage_estimates",
    "RewardLogger",
    "RewardDebugger"
]

# Version info
__version__ = "0.1.0"
__author__ = "RLT Implementation based on Sakana AI methodology"