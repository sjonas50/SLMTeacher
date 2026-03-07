"""
Dense Reward Function for Reasoning Like Teaching (RLT)

This module implements the main reward computation following Sakana AI's RLT methodology.
The reward function combines Solution Score (rSS) and KL Divergence Score (rKL) to guide
the teacher model to produce reasoning that maximizes student understanding while
maintaining naturalness.

Mathematical formulation:
    r(y, x) = rSS(y, x) - λ * rKL(y, x)

where:
    - rSS: Solution Score measuring student understanding
    - rKL: KL Divergence from baseline distribution
    - λ: Trade-off parameter (lambda)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import torch
import logging

from .student_evaluator import StudentEvaluator
from .kl_divergence import KLDivergenceCalculator
from .reward_utils import normalize_rewards, compute_baseline_rewards


@dataclass
class RewardConfig:
    """Configuration for RLT reward computation"""
    lambda_kl: float = 0.5  # Trade-off parameter for KL divergence
    normalize: bool = True  # Whether to normalize rewards
    temperature: float = 1.0  # Temperature for probability distributions
    clip_rewards: bool = True  # Whether to clip extreme reward values
    clip_range: Tuple[float, float] = (-10.0, 10.0)  # Clipping range
    use_baseline: bool = True  # Whether to subtract baseline rewards
    debug: bool = False  # Enable debug logging


class RLTRewardFunction:
    """
    Main reward function for Reasoning Like Teaching (RLT).
    
    This class orchestrates the computation of dense rewards by combining
    solution scores from student models and KL divergence penalties.
    """
    
    def __init__(
        self,
        student_evaluator: StudentEvaluator,
        kl_calculator: KLDivergenceCalculator,
        config: Optional[RewardConfig] = None
    ):
        """
        Initialize RLT reward function.
        
        Args:
            student_evaluator: Evaluator for computing solution scores
            kl_calculator: Calculator for KL divergence scores
            config: Configuration for reward computation
        """
        self.student_evaluator = student_evaluator
        self.kl_calculator = kl_calculator
        self.config = config or RewardConfig()
        self.logger = logging.getLogger(__name__)
        
        if self.config.debug:
            logging.basicConfig(level=logging.DEBUG)
    
    def compute_reward(
        self,
        reasoning: Union[str, List[str]],
        problem: Union[str, List[str]],
        reference_reasoning: Optional[Union[str, List[str]]] = None,
        return_components: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute RLT reward for given reasoning.
        
        Args:
            reasoning: Generated reasoning text(s)
            problem: Problem statement(s)
            reference_reasoning: Optional reference reasoning for KL computation
            return_components: Whether to return individual reward components
            
        Returns:
            Reward values or dictionary with reward components
        """
        # Ensure inputs are lists
        if isinstance(reasoning, str):
            reasoning = [reasoning]
        if isinstance(problem, str):
            problem = [problem]
        if reference_reasoning and isinstance(reference_reasoning, str):
            reference_reasoning = [reference_reasoning]
        
        batch_size = len(reasoning)
        assert len(problem) == batch_size, "Reasoning and problem batch sizes must match"
        
        # Compute solution scores (rSS)
        self.logger.debug("Computing solution scores...")
        solution_scores = self.student_evaluator.evaluate_batch(
            reasoning=reasoning,
            problems=problem
        )
        
        # Compute KL divergence scores (rKL)
        self.logger.debug("Computing KL divergence scores...")
        kl_scores = self.kl_calculator.compute_batch(
            generated_reasoning=reasoning,
            reference_reasoning=reference_reasoning or reasoning,
            problems=problem
        )
        
        # Combine rewards: r = rSS - λ * rKL
        raw_rewards = solution_scores - self.config.lambda_kl * kl_scores
        
        if self.config.debug:
            self.logger.debug(f"Solution scores: {solution_scores}")
            self.logger.debug(f"KL scores: {kl_scores}")
            self.logger.debug(f"Raw rewards: {raw_rewards}")
        
        # Apply baseline if configured
        if self.config.use_baseline:
            baseline_rewards = compute_baseline_rewards(
                problems=problem,
                student_evaluator=self.student_evaluator
            )
            rewards = raw_rewards - baseline_rewards
        else:
            rewards = raw_rewards
        
        # Normalize rewards if configured
        if self.config.normalize:
            rewards = normalize_rewards(
                rewards,
                method='standardize',
                temperature=self.config.temperature
            )
        
        # Clip rewards if configured
        if self.config.clip_rewards:
            rewards = np.clip(rewards, *self.config.clip_range)
        
        if return_components:
            return {
                'total_reward': rewards,
                'solution_score': solution_scores,
                'kl_score': kl_scores,
                'raw_reward': raw_rewards,
                'lambda_kl': self.config.lambda_kl
            }
        
        return rewards
    
    def compute_advantage(
        self,
        reasoning_batch: List[str],
        problem_batch: List[str],
        baseline_reasoning: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Compute advantage values for policy gradient training.
        
        Args:
            reasoning_batch: Batch of generated reasoning
            problem_batch: Batch of problems
            baseline_reasoning: Optional baseline reasoning for comparison
            
        Returns:
            Advantage values
        """
        # Compute rewards for generated reasoning
        rewards = self.compute_reward(reasoning_batch, problem_batch)
        
        # Compute baseline rewards
        if baseline_reasoning:
            baseline_rewards = self.compute_reward(baseline_reasoning, problem_batch)
        else:
            baseline_rewards = compute_baseline_rewards(
                problems=problem_batch,
                student_evaluator=self.student_evaluator
            )
        
        # Advantage = rewards - baseline
        advantages = rewards - baseline_rewards
        
        return advantages
    
    def update_lambda(self, new_lambda: float):
        """Update the lambda parameter for KL trade-off."""
        self.config.lambda_kl = new_lambda
        self.logger.info(f"Updated lambda to: {new_lambda}")


class RewardFunction:
    """
    Simplified reward function matching the training script API.

    Wraps student evaluation into a compute_reward method that returns
    {'rewards': [...]} as expected by GRPOTrainer.
    """

    def __init__(
        self,
        student_evaluator=None,
        kl_weight: float = 0.1,
        correctness_weight: float = 0.5,
        confidence_weight: float = 0.3,
        length_penalty_weight: float = 0.1,
    ):
        self.student_evaluator = student_evaluator
        self.kl_weight = kl_weight
        self.correctness_weight = correctness_weight
        self.confidence_weight = confidence_weight
        self.length_penalty_weight = length_penalty_weight
        self.logger = logging.getLogger(__name__)

    def compute_reward(
        self,
        explanations: List[str],
        questions: List[str],
    ) -> Dict[str, List[float]]:
        """
        Compute rewards for a batch of explanations.

        Returns:
            Dict with 'rewards' key containing list of float scores.
        """
        if self.student_evaluator is not None:
            try:
                scores = self.student_evaluator.evaluate_batch(
                    reasoning=explanations,
                    problems=questions,
                )
                length_penalties = np.array([
                    -self.length_penalty_weight * max(0, len(e) / 1000 - 1)
                    for e in explanations
                ])
                rewards = self.correctness_weight * scores + length_penalties
                return {'rewards': rewards.tolist()}
            except Exception as e:
                self.logger.warning(f"Evaluator failed, using heuristic rewards: {e}")

        # Heuristic fallback when no evaluator or evaluator fails
        rewards = []
        for explanation in explanations:
            score = 0.0
            steps = explanation.count("Step") + explanation.count("step")
            score += min(steps * 0.1, 0.5)
            word_count = len(explanation.split())
            if 50 <= word_count <= 500:
                score += 0.3
            if word_count < 20 or word_count > 1000:
                score -= 0.2
            rewards.append(score)
        return {'rewards': rewards}
    
    def get_reward_statistics(
        self,
        reasoning_batch: List[str],
        problem_batch: List[str]
    ) -> Dict[str, float]:
        """
        Compute statistics for reward components.
        
        Returns dictionary with mean, std, min, max for each component.
        """
        components = self.compute_reward(
            reasoning_batch,
            problem_batch,
            return_components=True
        )
        
        stats = {}
        for key, values in components.items():
            if isinstance(values, np.ndarray):
                stats[f"{key}_mean"] = float(np.mean(values))
                stats[f"{key}_std"] = float(np.std(values))
                stats[f"{key}_min"] = float(np.min(values))
                stats[f"{key}_max"] = float(np.max(values))
        
        return stats


class AdaptiveRewardFunction(RLTRewardFunction):
    """
    Extension of RLT reward function with adaptive lambda scheduling.
    
    This class automatically adjusts the KL trade-off parameter based on
    training progress and reward statistics.
    """
    
    def __init__(
        self,
        student_evaluator: StudentEvaluator,
        kl_calculator: KLDivergenceCalculator,
        config: Optional[RewardConfig] = None,
        lambda_scheduler: Optional[str] = 'linear'
    ):
        super().__init__(student_evaluator, kl_calculator, config)
        self.lambda_scheduler = lambda_scheduler
        self.initial_lambda = self.config.lambda_kl
        self.step_count = 0
        self.reward_history = []
    
    def compute_reward(
        self,
        reasoning: Union[str, List[str]],
        problem: Union[str, List[str]],
        reference_reasoning: Optional[Union[str, List[str]]] = None,
        return_components: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Compute reward with adaptive lambda scheduling."""
        # Update lambda based on schedule
        self._update_lambda_schedule()
        
        # Compute rewards
        result = super().compute_reward(
            reasoning, problem, reference_reasoning, return_components
        )
        
        # Track reward history
        if isinstance(result, np.ndarray):
            self.reward_history.append(np.mean(result))
        else:
            self.reward_history.append(np.mean(result['total_reward']))
        
        self.step_count += 1
        
        return result
    
    def _update_lambda_schedule(self):
        """Update lambda based on the configured schedule."""
        if self.lambda_scheduler == 'linear':
            # Linear decay from initial_lambda to 0.1 * initial_lambda
            decay_rate = 0.9 * self.initial_lambda / 10000  # Decay over 10k steps
            new_lambda = max(
                0.1 * self.initial_lambda,
                self.initial_lambda - decay_rate * self.step_count
            )
        elif self.lambda_scheduler == 'exponential':
            # Exponential decay
            decay_rate = 0.9999
            new_lambda = self.initial_lambda * (decay_rate ** self.step_count)
        elif self.lambda_scheduler == 'adaptive':
            # Adaptive based on reward variance
            if len(self.reward_history) > 100:
                recent_variance = np.var(self.reward_history[-100:])
                if recent_variance > 1.0:
                    new_lambda = min(self.config.lambda_kl * 1.1, 2.0)
                else:
                    new_lambda = max(self.config.lambda_kl * 0.9, 0.1)
            else:
                new_lambda = self.config.lambda_kl
        else:
            # No scheduling
            new_lambda = self.config.lambda_kl
        
        self.config.lambda_kl = new_lambda