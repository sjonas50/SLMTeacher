"""End-to-end integration tests for the training pipeline.

Verifies the full SFT warmup → GRPO training loop using lightweight
mock components (no real model loading or API calls).
"""
import math
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.training.grpo_trainer import GRPOConfig, GRPOTrainer


# ---------------------------------------------------------------------------
# Lightweight fakes
# ---------------------------------------------------------------------------

_FAKE_VOCAB_SIZE = 32


class _FakeInnerModel(torch.nn.Module):
    """Tiny nn.Module so deepcopy / parameters() work for reference policy."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, **kwargs):
        seq_len = kwargs.get("input_ids", torch.zeros(1, 5)).shape[1]
        logits = torch.randn(1, seq_len, _FAKE_VOCAB_SIZE)
        return type("Out", (), {"logits": logits})()


class FakeStudentModel:
    """Minimal student model that returns controllable log probs / losses."""

    def __init__(self):
        # A real nn.Module so deepcopy works for reference policy
        self.model = _FakeInnerModel()
        self._param = list(self.model.parameters())[0]
        self.device = torch.device("cpu")
        self.call_count = 0
        # Fake tokenizer for _compute_ref_log_probs
        self.tokenizer = _FakeTokenizer()

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def compute_loss(self, input_text: str, target_text: str) -> torch.Tensor:
        self.call_count += 1
        # Loss that depends on the parameter so gradients flow
        return (self._param.sum() ** 2) + 0.5

    def compute_log_probs(self, input_text: str, target_text: str) -> torch.Tensor:
        self.call_count += 1
        # Log prob depends on parameter
        return -((self._param.sum() - 1.0) ** 2) - 1.0

    def save_model(self, path: str):
        pass


class _FakeTokenizer:
    """Minimal tokenizer that returns consistent fake encodings."""

    model_max_length = 512
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "</s>"
    eos_token_id = 1

    def __call__(self, text, return_tensors="pt", **kwargs):
        # Return 1 token per word (rough approximation)
        tokens = text.split() or ["<empty>"]
        ids = torch.tensor([[hash(t) % _FAKE_VOCAB_SIZE for t in tokens]])
        return type("Enc", (), {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "to": lambda self, device: self,
        })()


class FakeTeacher:
    """Returns canned explanations."""

    def generate_explanation(self, question, answer, temperature=0.7, **kw):
        return f"To solve '{question}', note that the answer is {answer}."


class FakeRewardFunction:
    """Returns rewards in a predictable range for group-size testing."""

    def __init__(self):
        self._call = 0

    def compute_reward(self, explanations, questions, answers=None):
        self._call += 1
        # Vary rewards across calls to produce non-zero advantages
        reward = 0.3 + 0.1 * (self._call % 5)
        return {"rewards": [reward]}


class FakeEvaluator:
    def evaluate_batch(self, reasoning, problems, reference_answers=None):
        return np.array([0.5] * len(reasoning))

    def get_log_probabilities(self, inputs, targets):
        return np.array([-1.0] * len(inputs))


@dataclass
class FakeDataPoint:
    question: str
    solution: str
    subject: str = "math"
    difficulty: str = "easy"


def _collate(batch):
    return {
        "questions": [dp.question for dp in batch],
        "answers": [dp.solution for dp in batch],
        "subjects": [dp.subject for dp in batch],
        "difficulties": [dp.difficulty for dp in batch],
    }


def _make_data(n=8):
    return [FakeDataPoint(f"What is {i}+{i}?", str(2 * i)) for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGroupSize:
    def test_default_group_size_is_6(self):
        config = GRPOConfig()
        assert config.group_size == 6

    def test_group_size_explanations(self):
        """Each question should get exactly group_size explanations."""
        config = GRPOConfig(group_size=6)
        student = FakeStudentModel()
        trainer = GRPOTrainer(
            teacher=FakeTeacher(),
            student_model=student,
            student_evaluator=FakeEvaluator(),
            reward_function=FakeRewardFunction(),
            config=config,
        )
        batch = _collate(_make_data(2))
        result = trainer.generate_and_score_explanations(batch)
        for exps in result["explanations"]:
            assert len(exps) == 6
        for rews in result["rewards"]:
            assert len(rews) == 6


class TestScheduler:
    def test_cosine_scheduler_default(self):
        config = GRPOConfig(lr_scheduler_type="cosine", warmup_ratio=0.1)
        student = FakeStudentModel()
        trainer = GRPOTrainer(
            teacher=FakeTeacher(),
            student_model=student,
            student_evaluator=FakeEvaluator(),
            reward_function=FakeRewardFunction(),
            config=config,
        )
        total_steps = 100
        scheduler = trainer._create_scheduler(total_steps)
        # Warmup should be 10 steps (10% of 100)
        # After warmup, LR should start decaying with cosine
        lrs = []
        for _ in range(total_steps):
            lrs.append(scheduler.get_last_lr()[0])
            trainer.optimizer.step()
            scheduler.step()

        # LR should increase during warmup
        assert lrs[9] > lrs[0]
        # LR should decrease after warmup
        assert lrs[50] < lrs[10]
        # LR should approach 0 at the end
        assert lrs[-1] < lrs[50]

    def test_linear_scheduler(self):
        config = GRPOConfig(lr_scheduler_type="linear", warmup_ratio=0.1)
        student = FakeStudentModel()
        trainer = GRPOTrainer(
            teacher=FakeTeacher(),
            student_model=student,
            student_evaluator=FakeEvaluator(),
            reward_function=FakeRewardFunction(),
            config=config,
        )
        total_steps = 100
        scheduler = trainer._create_scheduler(total_steps)
        lrs = []
        for _ in range(total_steps):
            lrs.append(scheduler.get_last_lr()[0])
            trainer.optimizer.step()
            scheduler.step()

        # Linear decay: LR decreases roughly linearly after warmup
        assert lrs[50] < lrs[10]
        assert lrs[-1] < lrs[50]

    def test_warmup_steps_override(self):
        config = GRPOConfig(warmup_steps=20, warmup_ratio=0.5)
        student = FakeStudentModel()
        trainer = GRPOTrainer(
            teacher=FakeTeacher(),
            student_model=student,
            student_evaluator=FakeEvaluator(),
            reward_function=FakeRewardFunction(),
            config=config,
        )
        scheduler = trainer._create_scheduler(100)
        # warmup_steps=20 should override warmup_ratio=0.5
        # LR at step 19 should still be in warmup (below max)
        lrs = []
        for _ in range(25):
            lrs.append(scheduler.get_last_lr()[0])
            trainer.optimizer.step()
            scheduler.step()
        # LR should peak around step 20
        assert lrs[19] > lrs[0]


class TestSFTWarmup:
    def test_sft_warmup_runs(self):
        """SFT warmup should call compute_loss and run optimizer steps."""
        config = GRPOConfig(
            batch_size=2,
            gradient_accumulation_steps=1,
            group_size=2,  # small for speed
        )
        student = FakeStudentModel()
        trainer = GRPOTrainer(
            teacher=FakeTeacher(),
            student_model=student,
            student_evaluator=FakeEvaluator(),
            reward_function=FakeRewardFunction(),
            config=config,
        )
        loader = DataLoader(_make_data(4), batch_size=2, collate_fn=_collate)
        trainer.sft_warmup(loader, num_epochs=1)
        # Should have called compute_loss at least once per item
        assert student.call_count > 0


class TestGRPOTraining:
    def test_train_epoch_runs(self):
        """Full GRPO training epoch should run without errors."""
        config = GRPOConfig(
            batch_size=2,
            gradient_accumulation_steps=1,
            group_size=3,
            num_epochs=1,
        )
        student = FakeStudentModel()
        trainer = GRPOTrainer(
            teacher=FakeTeacher(),
            student_model=student,
            student_evaluator=FakeEvaluator(),
            reward_function=FakeRewardFunction(),
            config=config,
        )
        loader = DataLoader(_make_data(4), batch_size=2, collate_fn=_collate)
        trainer.train(loader, num_epochs=1)
        assert student.call_count > 0

    def test_advantages_nonzero(self):
        """With varied rewards, advantages should be non-zero for learning."""
        config = GRPOConfig(group_size=4, normalize_rewards=True)
        student = FakeStudentModel()
        reward_fn = FakeRewardFunction()
        trainer = GRPOTrainer(
            teacher=FakeTeacher(),
            student_model=student,
            student_evaluator=FakeEvaluator(),
            reward_function=reward_fn,
            config=config,
        )
        batch = _collate(_make_data(2))
        batch_results = trainer.generate_and_score_explanations(batch)

        # Verify rewards vary within at least one group
        has_variance = False
        for rews in batch_results["rewards"]:
            if max(rews) - min(rews) > 1e-6:
                has_variance = True
        assert has_variance, "Rewards should vary across explanations for non-zero advantages"

        _, metrics = trainer.train_student_on_explanations(batch_results)
        # At least some advantages should be non-zero
        assert any(abs(a) > 1e-6 for a in metrics["advantages"])

    def test_reference_policy_snapshot(self):
        """Reference policy should be created and used for PPO ratios."""
        config = GRPOConfig(group_size=2)
        student = FakeStudentModel()
        trainer = GRPOTrainer(
            teacher=FakeTeacher(),
            student_model=student,
            student_evaluator=FakeEvaluator(),
            reward_function=FakeRewardFunction(),
            config=config,
        )
        trainer._snapshot_reference_policy()
        assert hasattr(trainer, "_ref_model") or hasattr(trainer, "_ref_adapter_state")

    def test_eval_runs(self):
        """Evaluation should produce metrics."""
        config = GRPOConfig(group_size=2)
        student = FakeStudentModel()
        trainer = GRPOTrainer(
            teacher=FakeTeacher(),
            student_model=student,
            student_evaluator=FakeEvaluator(),
            reward_function=FakeRewardFunction(),
            config=config,
        )
        loader = DataLoader(_make_data(4), batch_size=2, collate_fn=_collate)
        metrics = trainer.evaluate(loader)
        assert "avg_reward" in metrics
        assert "std_reward" in metrics
        assert metrics["num_samples"] > 0


class TestFullPipeline:
    def test_sft_then_grpo(self):
        """SFT warmup followed by GRPO training — the full pipeline."""
        config = GRPOConfig(
            batch_size=2,
            gradient_accumulation_steps=1,
            group_size=3,
            num_epochs=1,
            lr_scheduler_type="cosine",
            warmup_ratio=0.2,
        )
        student = FakeStudentModel()
        trainer = GRPOTrainer(
            teacher=FakeTeacher(),
            student_model=student,
            student_evaluator=FakeEvaluator(),
            reward_function=FakeRewardFunction(),
            config=config,
        )
        data = _make_data(6)
        loader = DataLoader(data, batch_size=2, collate_fn=_collate)

        # Phase 1: SFT warmup
        trainer.sft_warmup(loader, num_epochs=1)
        sft_calls = student.call_count

        # Phase 2: GRPO
        student.call_count = 0
        trainer.train(loader, num_epochs=1)
        grpo_calls = student.call_count

        assert sft_calls > 0
        assert grpo_calls > 0

    def test_checkpoint_save(self, tmp_path):
        """Checkpointing should save without errors."""
        config = GRPOConfig(
            batch_size=2,
            gradient_accumulation_steps=1,
            group_size=2,
            num_epochs=1,
            checkpoint_dir=str(tmp_path),
        )
        student = FakeStudentModel()
        trainer = GRPOTrainer(
            teacher=FakeTeacher(),
            student_model=student,
            student_evaluator=FakeEvaluator(),
            reward_function=FakeRewardFunction(),
            config=config,
        )
        loader = DataLoader(_make_data(4), batch_size=2, collate_fn=_collate)
        trainer.train(loader, num_epochs=1)
        trainer.save_checkpoint(is_best=True)

        assert (tmp_path / "best_model.pt").exists()
        assert (tmp_path / "metrics.json").exists()
