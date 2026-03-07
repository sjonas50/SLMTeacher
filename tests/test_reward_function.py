"""Tests for RewardFunction (training script API)."""
import pytest
from src.rewards.reward_function import RewardFunction, RLTRewardFunction, RewardConfig


class TestRewardFunction:
    """Tests for the simplified RewardFunction wrapper."""

    def test_init_defaults(self):
        rf = RewardFunction()
        assert rf.kl_weight == 0.1
        assert rf.correctness_weight == 0.5
        assert rf.confidence_weight == 0.3
        assert rf.length_penalty_weight == 0.1

    def test_init_custom_weights(self):
        rf = RewardFunction(kl_weight=0.2, correctness_weight=0.8)
        assert rf.kl_weight == 0.2
        assert rf.correctness_weight == 0.8

    def test_heuristic_rewards_structure(self):
        rf = RewardFunction()
        result = rf.compute_reward(
            explanations=["Step 1: Understand the problem. Step 2: Solve it."],
            questions=["What is 2+2?"],
        )
        assert 'rewards' in result
        assert isinstance(result['rewards'], list)
        assert len(result['rewards']) == 1

    def test_heuristic_rewards_step_bonus(self):
        rf = RewardFunction()
        # More steps should score higher
        few_steps = rf.compute_reward(
            explanations=["The answer is 4."],
            questions=["What is 2+2?"],
        )
        many_steps = rf.compute_reward(
            explanations=[
                "Step 1: Identify the operation. Step 2: Add the numbers. "
                "Step 3: Verify. The answer is 4. " * 5
            ],
            questions=["What is 2+2?"],
        )
        assert many_steps['rewards'][0] >= few_steps['rewards'][0]

    def test_batch_rewards(self):
        rf = RewardFunction()
        result = rf.compute_reward(
            explanations=[
                "Step 1: Add them. Answer: 4",
                "Step 1: Multiply. Answer: 12",
                "Short.",
            ],
            questions=[
                "What is 2+2?",
                "What is 3*4?",
                "Hi?",
            ],
        )
        assert len(result['rewards']) == 3

    def test_length_penalty(self):
        rf = RewardFunction(length_penalty_weight=0.5)
        very_long = "word " * 2000
        result = rf.compute_reward(
            explanations=[very_long],
            questions=["Q?"],
        )
        # Very long text should get penalized
        assert result['rewards'][0] < 0.5


class TestRewardConfig:
    def test_defaults(self):
        config = RewardConfig()
        assert config.lambda_kl == 0.5
        assert config.normalize is True
        assert config.clip_rewards is True
        assert config.clip_range == (-10.0, 10.0)
