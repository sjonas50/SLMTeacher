"""
Utility Functions for RLT Reward System

This module provides helper functions for reward computation, normalization,
debugging, and visualization of the reward system.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def normalize_rewards(
    rewards: np.ndarray,
    method: str = 'standardize',
    temperature: float = 1.0,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Normalize reward values using various methods.
    
    Args:
        rewards: Array of reward values
        method: Normalization method ('standardize', 'minmax', 'softmax', 'robust')
        temperature: Temperature for softmax normalization
        epsilon: Small value for numerical stability
        
    Returns:
        Normalized rewards
    """
    if method == 'standardize':
        # Z-score normalization
        mean = np.mean(rewards)
        std = np.std(rewards) + epsilon
        return (rewards - mean) / std
    
    elif method == 'minmax':
        # Min-max normalization to [0, 1]
        min_val = np.min(rewards)
        max_val = np.max(rewards)
        range_val = max_val - min_val + epsilon
        return (rewards - min_val) / range_val
    
    elif method == 'softmax':
        # Softmax normalization with numerical stability (log-sum-exp trick)
        scaled = rewards / temperature
        scaled = scaled - np.max(scaled)  # shift for stability
        exp_rewards = np.exp(scaled)
        return exp_rewards / (np.sum(exp_rewards) + epsilon)
    
    elif method == 'robust':
        # Robust normalization using median and MAD
        median = np.median(rewards)
        mad = np.median(np.abs(rewards - median)) + epsilon
        return (rewards - median) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_baseline_rewards(
    problems: List[str],
    student_evaluator: Any,
    method: str = 'zero_shot'
) -> np.ndarray:
    """
    Compute baseline rewards for variance reduction.
    
    Args:
        problems: List of problem statements
        student_evaluator: Student evaluator instance
        method: Baseline method ('zero_shot', 'constant', 'learned')
        
    Returns:
        Baseline reward values
    """
    if method == 'zero_shot':
        # Use student's zero-shot performance as baseline
        empty_reasoning = [""] * len(problems)
        baseline_scores = student_evaluator.evaluate_batch(
            reasoning=empty_reasoning,
            problems=problems
        )
        return baseline_scores
    
    elif method == 'constant':
        # Use a constant baseline (e.g., average historical reward)
        return np.zeros(len(problems)) - 5.0  # Default low baseline
    
    elif method == 'learned':
        # Placeholder for learned baseline (e.g., value network)
        # In practice, this would use a separate neural network
        return np.zeros(len(problems))
    
    else:
        raise ValueError(f"Unknown baseline method: {method}")


def compute_advantage_estimates(
    rewards: np.ndarray,
    values: Optional[np.ndarray] = None,
    gamma: float = 0.99,
    lambda_gae: float = 0.95
) -> np.ndarray:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: Array of rewards
        values: Optional value estimates
        gamma: Discount factor
        lambda_gae: GAE lambda parameter
        
    Returns:
        Advantage estimates
    """
    if values is None:
        # Simple advantage: rewards - mean(rewards)
        return rewards - np.mean(rewards)
    
    # Compute TD residuals
    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
    
    # Compute GAE
    advantages = np.zeros_like(rewards)
    gae = 0
    
    for t in reversed(range(len(deltas))):
        gae = deltas[t] + gamma * lambda_gae * gae
        advantages[t] = gae
    
    return advantages


class RewardLogger:
    """Logger for tracking and analyzing rewards during training."""
    
    def __init__(
        self,
        log_dir: str = "./logs/rewards",
        experiment_name: Optional[str] = None
    ):
        """
        Initialize reward logger.
        
        Args:
            log_dir: Directory for saving logs
            experiment_name: Optional experiment identifier
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.log_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize tracking dictionaries
        self.reward_history = []
        self.component_history = []
        self.metadata = {
            "experiment_name": self.experiment_name,
            "start_time": datetime.now().isoformat()
        }
    
    def log_rewards(
        self,
        step: int,
        rewards: np.ndarray,
        components: Optional[Dict[str, np.ndarray]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log reward values and components."""
        entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "rewards": {
                "mean": float(np.mean(rewards)),
                "std": float(np.std(rewards)),
                "min": float(np.min(rewards)),
                "max": float(np.max(rewards)),
                "median": float(np.median(rewards))
            }
        }
        
        if metadata:
            entry["metadata"] = metadata
        
        self.reward_history.append(entry)
        
        # Log components if provided
        if components:
            component_entry = {
                "step": step,
                "timestamp": datetime.now().isoformat()
            }
            
            for name, values in components.items():
                if isinstance(values, np.ndarray):
                    component_entry[name] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values))
                    }
                else:
                    component_entry[name] = values
            
            self.component_history.append(component_entry)
        
        # Save periodically
        if step % 100 == 0:
            self.save_logs()
    
    def save_logs(self):
        """Save logs to disk."""
        # Save reward history
        with open(self.experiment_dir / "reward_history.json", "w") as f:
            json.dump(self.reward_history, f, indent=2)
        
        # Save component history
        if self.component_history:
            with open(self.experiment_dir / "component_history.json", "w") as f:
                json.dump(self.component_history, f, indent=2)
        
        # Save metadata
        self.metadata["last_update"] = datetime.now().isoformat()
        with open(self.experiment_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        self.logger.info(f"Saved logs to {self.experiment_dir}")
    
    def plot_reward_history(self, save_path: Optional[str] = None):
        """Plot reward history over training."""
        import matplotlib.pyplot as plt

        if not self.reward_history:
            self.logger.warning("No reward history to plot")
            return

        import pandas as pd

        # Convert to DataFrame
        df = pd.DataFrame([
            {
                "step": entry["step"],
                "mean": entry["rewards"]["mean"],
                "std": entry["rewards"]["std"],
                "min": entry["rewards"]["min"],
                "max": entry["rewards"]["max"]
            }
            for entry in self.reward_history
        ])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot mean with confidence interval
        ax.plot(df["step"], df["mean"], label="Mean Reward", linewidth=2)
        ax.fill_between(
            df["step"],
            df["mean"] - df["std"],
            df["mean"] + df["std"],
            alpha=0.3,
            label="±1 std"
        )
        
        # Plot min/max range
        ax.fill_between(
            df["step"],
            df["min"],
            df["max"],
            alpha=0.1,
            label="Min/Max Range"
        )
        
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Reward")
        ax.set_title(f"Reward History - {self.experiment_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.experiment_dir / "reward_history.png", dpi=300, bbox_inches="tight")
        
        plt.close()
    
    def plot_component_analysis(self, save_path: Optional[str] = None):
        """Plot analysis of reward components."""
        import matplotlib.pyplot as plt

        if not self.component_history:
            self.logger.warning("No component history to plot")
            return
        
        # Extract component names
        component_names = set()
        for entry in self.component_history:
            for key in entry.keys():
                if key not in ["step", "timestamp"]:
                    component_names.add(key)
        
        # Create subplots
        n_components = len(component_names)
        fig, axes = plt.subplots(n_components, 1, figsize=(10, 4 * n_components))
        if n_components == 1:
            axes = [axes]
        
        for idx, component in enumerate(sorted(component_names)):
            # Extract data for this component
            steps = []
            means = []
            stds = []
            
            for entry in self.component_history:
                if component in entry and isinstance(entry[component], dict):
                    steps.append(entry["step"])
                    means.append(entry[component]["mean"])
                    stds.append(entry[component]["std"])
            
            if not steps:
                continue
            
            # Plot
            ax = axes[idx]
            ax.plot(steps, means, label=f"{component} (mean)", linewidth=2)
            ax.fill_between(
                steps,
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                alpha=0.3
            )
            
            ax.set_xlabel("Training Step")
            ax.set_ylabel(component)
            ax.set_title(f"{component} over Training")
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.experiment_dir / "component_analysis.png", dpi=300, bbox_inches="tight")
        
        plt.close()


class RewardDebugger:
    """Debugging utilities for reward computation."""
    
    @staticmethod
    def analyze_reward_distribution(
        rewards: np.ndarray,
        components: Optional[Dict[str, np.ndarray]] = None,
        save_path: Optional[str] = None
    ):
        """Analyze and visualize reward distribution."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Reward histogram
        ax = axes[0, 0]
        ax.hist(rewards, bins=50, density=True, alpha=0.7, color='blue')
        ax.set_xlabel("Reward Value")
        ax.set_ylabel("Density")
        ax.set_title("Reward Distribution")
        ax.axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.3f}')
        ax.axvline(np.median(rewards), color='green', linestyle='--', label=f'Median: {np.median(rewards):.3f}')
        ax.legend()
        
        # Q-Q plot
        ax = axes[0, 1]
        from scipy import stats
        stats.probplot(rewards, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot (Normal Distribution)")
        
        # Box plot of components
        if components:
            ax = axes[1, 0]
            component_data = []
            component_labels = []
            
            for name, values in components.items():
                if isinstance(values, np.ndarray):
                    component_data.append(values)
                    component_labels.append(name)
            
            if component_data:
                ax.boxplot(component_data, labels=component_labels)
                ax.set_ylabel("Value")
                ax.set_title("Component Distributions")
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Correlation heatmap
        if components and len(components) > 1:
            ax = axes[1, 1]
            
            # Create correlation matrix
            component_arrays = []
            component_names = []
            
            for name, values in components.items():
                if isinstance(values, np.ndarray):
                    component_arrays.append(values)
                    component_names.append(name)
            
            if len(component_arrays) > 1:
                corr_matrix = np.corrcoef(component_arrays)
                
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    fmt='.3f',
                    xticklabels=component_names,
                    yticklabels=component_names,
                    ax=ax,
                    cmap='coolwarm',
                    center=0
                )
                ax.set_title("Component Correlations")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def check_reward_stability(
        reward_history: List[float],
        window_size: int = 100
    ) -> Dict[str, Any]:
        """Check reward stability over time."""
        if len(reward_history) < window_size:
            return {"stable": True, "message": "Not enough data"}
        
        # Compute rolling statistics
        rewards_array = np.array(reward_history)
        rolling_mean = np.convolve(
            rewards_array,
            np.ones(window_size) / window_size,
            mode='valid'
        )
        import pandas as pd
        rolling_std = pd.Series(rewards_array).rolling(window_size).std().dropna().values
        
        # Check for stability
        mean_change = np.abs(np.diff(rolling_mean)).max()
        std_change = np.abs(np.diff(rolling_std)).max()
        
        stability_info = {
            "stable": mean_change < 0.1 and std_change < 0.05,
            "mean_change": float(mean_change),
            "std_change": float(std_change),
            "current_mean": float(rolling_mean[-1]),
            "current_std": float(rolling_std[-1])
        }
        
        return stability_info
    
    @staticmethod
    def validate_reward_computation(
        reward_function: Any,
        test_cases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate reward computation with test cases."""
        results = []
        
        for i, test_case in enumerate(test_cases):
            try:
                # Compute reward
                reward = reward_function.compute_reward(
                    reasoning=test_case["reasoning"],
                    problem=test_case["problem"],
                    return_components=True
                )
                
                # Check validity
                issues = []
                
                # Check for NaN or Inf
                if np.any(np.isnan(reward["total_reward"])):
                    issues.append("NaN values in rewards")
                if np.any(np.isinf(reward["total_reward"])):
                    issues.append("Inf values in rewards")
                
                # Check component consistency
                expected_total = (
                    reward["solution_score"] - 
                    reward["lambda_kl"] * reward["kl_score"]
                )
                if not np.allclose(expected_total, reward["raw_reward"], rtol=1e-5):
                    issues.append("Component calculation mismatch")
                
                results.append({
                    "test_case": i,
                    "valid": len(issues) == 0,
                    "issues": issues,
                    "reward_stats": {
                        "mean": float(np.mean(reward["total_reward"])),
                        "std": float(np.std(reward["total_reward"]))
                    }
                })
                
            except Exception as e:
                results.append({
                    "test_case": i,
                    "valid": False,
                    "issues": [f"Exception: {str(e)}"],
                    "reward_stats": None
                })
        
        return results