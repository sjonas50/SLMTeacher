"""Tests for GRPO configuration and trainer setup."""
import pytest
from src.training.grpo_trainer import GRPOConfig


class TestGRPOConfig:
    def test_defaults(self):
        config = GRPOConfig()
        assert config.learning_rate == 2e-5
        assert config.batch_size == 4
        assert config.gradient_accumulation_steps == 4
        assert config.num_epochs == 3
        assert config.group_size == 4
        assert config.clip_epsilon == 0.2
        assert config.normalize_rewards is True

    def test_custom_config(self):
        config = GRPOConfig(
            learning_rate=1e-4,
            batch_size=8,
            num_epochs=5,
        )
        assert config.learning_rate == 1e-4
        assert config.batch_size == 8
        assert config.num_epochs == 5
