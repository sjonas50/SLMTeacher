"""Tests for GRPO configuration and trainer setup."""
import pytest
from src.training.grpo_trainer import GRPOConfig, ExperimentTracker


class TestGRPOConfig:
    def test_defaults(self):
        config = GRPOConfig()
        assert config.learning_rate == 2e-5
        assert config.batch_size == 4
        assert config.gradient_accumulation_steps == 4
        assert config.num_epochs == 3
        assert config.group_size == 6
        assert config.lr_scheduler_type == "cosine"
        assert config.warmup_ratio == 0.1
        assert config.warmup_steps == 0
        assert config.clip_epsilon == 0.2
        assert config.kl_coef == 0.1
        assert config.entropy_coef == 0.01
        assert config.ref_update_freq == 1
        assert config.normalize_rewards is True
        assert config.tracker is None
        assert config.tracker_project == "slmteacher"
        assert config.tracker_run_name is None

    def test_custom_config(self):
        config = GRPOConfig(
            learning_rate=1e-4,
            batch_size=8,
            num_epochs=5,
        )
        assert config.learning_rate == 1e-4
        assert config.batch_size == 8
        assert config.num_epochs == 5

    def test_tracker_config(self):
        config = GRPOConfig(
            tracker="wandb",
            tracker_project="my-project",
            tracker_run_name="run-1",
        )
        assert config.tracker == "wandb"
        assert config.tracker_project == "my-project"
        assert config.tracker_run_name == "run-1"


class TestExperimentTracker:
    def test_no_tracker(self):
        config = GRPOConfig(tracker=None)
        tracker = ExperimentTracker(config)
        assert tracker.backend is None
        # Should be a no-op
        tracker.log({"loss": 0.5}, step=1)
        tracker.finish()

    def test_invalid_tracker_fallback(self):
        config = GRPOConfig(tracker="nonexistent")
        tracker = ExperimentTracker(config)
        # Falls through without matching any backend
        assert tracker.backend == "nonexistent" or tracker.backend is None
