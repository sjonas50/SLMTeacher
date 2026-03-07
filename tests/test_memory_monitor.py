"""Tests for MemoryMonitor."""
import pytest

peft = pytest.importorskip("peft", reason="peft not installed")
from src.models.optimized_model import MemoryMonitor, OptimizedModelConfig


class TestMemoryMonitor:
    def test_get_memory_stats(self):
        stats = MemoryMonitor.get_memory_stats()
        assert 'cpu_percent' in stats
        assert 'cpu_memory_percent' in stats
        assert 'cpu_available_gb' in stats
        assert isinstance(stats['cpu_available_gb'], float)

    def test_optimize_memory_runs(self):
        # Should not raise
        MemoryMonitor.optimize_memory()

    def test_estimate_3b_4bit(self):
        mem = MemoryMonitor.estimate_model_memory("Llama-3.2-3B-Instruct", use_4bit=True)
        assert 0.5 < mem < 3.0  # ~1.4 GB for 3B 4-bit

    def test_estimate_7b_4bit(self):
        mem = MemoryMonitor.estimate_model_memory("Mistral-7B-Instruct", use_4bit=True)
        assert 2.0 < mem < 5.0  # ~3.3 GB for 7B 4-bit

    def test_estimate_7b_fp16(self):
        mem = MemoryMonitor.estimate_model_memory("Mistral-7B-Instruct", use_4bit=False)
        assert 10.0 < mem < 16.0  # ~13 GB for 7B FP16

    def test_estimate_unknown_model(self):
        # Falls back to 7B default
        mem = MemoryMonitor.estimate_model_memory("some-unknown-model", use_4bit=True)
        assert mem > 0


class TestOptimizedModelConfig:
    def test_defaults(self):
        config = OptimizedModelConfig()
        assert config.use_4bit is True
        assert config.use_8bit is False
        assert config.gradient_checkpointing is True
        assert config.peft_method == "adalora"
        assert config.mixed_precision == "bf16"

    def test_custom_config(self):
        config = OptimizedModelConfig(
            model_name="test/model",
            use_4bit=False,
            peft_method="lora",
        )
        assert config.model_name == "test/model"
        assert config.use_4bit is False
        assert config.peft_method == "lora"
