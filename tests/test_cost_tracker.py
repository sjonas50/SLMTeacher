"""Tests for CostTracker."""
import pytest
from src.utils.cost_tracker import CostTracker, MODEL_PRICING, DEFAULT_PRICING


class TestCostTracker:
    def test_initial_state(self):
        tracker = CostTracker(budget_limit=10.0)
        summary = tracker.get_summary()
        assert summary['total_cost'] == 0.0
        assert summary['budget_remaining'] == 10.0
        assert summary['api_calls'] == 0
        assert summary['total_tokens'] == 0

    def test_add_usage(self):
        tracker = CostTracker(budget_limit=100.0)
        tracker.add_usage(input_tokens=1000, output_tokens=500)
        summary = tracker.get_summary()
        assert summary['api_calls'] == 1
        assert summary['total_tokens'] == 1500
        assert summary['total_cost'] > 0

    def test_cost_calculation(self):
        tracker = CostTracker(budget_limit=100.0)
        # 1M input tokens at $3/M = $3.00
        tracker.add_usage(input_tokens=1_000_000, output_tokens=0)
        summary = tracker.get_summary()
        assert summary['total_cost'] == pytest.approx(3.0, abs=0.01)

    def test_output_token_cost(self):
        tracker = CostTracker(budget_limit=100.0)
        # 1M output tokens at $15/M = $15.00
        tracker.add_usage(input_tokens=0, output_tokens=1_000_000)
        summary = tracker.get_summary()
        assert summary['total_cost'] == pytest.approx(15.0, abs=0.01)

    def test_budget_exceeded_raises(self):
        tracker = CostTracker(budget_limit=0.001)
        with pytest.raises(ValueError, match="Budget limit exceeded"):
            tracker.add_usage(input_tokens=1_000_000, output_tokens=1_000_000)

    def test_summary_returns_numeric_types(self):
        tracker = CostTracker(budget_limit=50.0)
        tracker.add_usage(input_tokens=100, output_tokens=50)
        summary = tracker.get_summary()

        assert isinstance(summary['total_cost'], float)
        assert isinstance(summary['budget_remaining'], float)
        assert isinstance(summary['average_cost_per_call'], float)
        assert isinstance(summary['runtime_minutes'], float)
        assert isinstance(summary['calls_per_minute'], float)
        assert isinstance(summary['api_calls'], int)
        assert isinstance(summary['total_tokens'], int)

    def test_multiple_calls_accumulate(self):
        tracker = CostTracker(budget_limit=100.0)
        for _ in range(5):
            tracker.add_usage(input_tokens=100, output_tokens=50)
        summary = tracker.get_summary()
        assert summary['api_calls'] == 5
        assert summary['total_tokens'] == 750

    def test_metadata_stored(self):
        tracker = CostTracker(budget_limit=100.0)
        tracker.add_usage(
            input_tokens=100,
            output_tokens=50,
            metadata={'model': 'test-model'}
        )
        assert tracker.costs['history'][0]['metadata']['model'] == 'test-model'

    def test_per_model_pricing_opus(self):
        tracker = CostTracker(budget_limit=100.0)
        # Opus: $15/M input, $75/M output
        tracker.add_usage(
            input_tokens=1_000_000,
            output_tokens=0,
            metadata={'model': 'claude-3-opus-20240229'}
        )
        assert tracker.get_summary()['total_cost'] == pytest.approx(15.0, abs=0.01)

    def test_per_model_pricing_haiku(self):
        tracker = CostTracker(budget_limit=100.0)
        # Haiku 3: $0.25/M input
        tracker.add_usage(
            input_tokens=1_000_000,
            output_tokens=0,
            metadata={'model': 'claude-3-haiku-20240307'}
        )
        assert tracker.get_summary()['total_cost'] == pytest.approx(0.25, abs=0.01)

    def test_unknown_model_uses_default_pricing(self):
        tracker = CostTracker(budget_limit=100.0)
        tracker.add_usage(
            input_tokens=1_000_000,
            output_tokens=0,
            metadata={'model': 'some-future-model'}
        )
        expected = DEFAULT_PRICING['input']
        assert tracker.get_summary()['total_cost'] == pytest.approx(expected, abs=0.01)

    def test_no_model_uses_default_pricing(self):
        tracker = CostTracker(budget_limit=100.0)
        tracker.add_usage(input_tokens=1_000_000, output_tokens=0)
        expected = DEFAULT_PRICING['input']
        assert tracker.get_summary()['total_cost'] == pytest.approx(expected, abs=0.01)

    def test_model_pricing_table_has_expected_models(self):
        assert 'claude-sonnet-4-6' in MODEL_PRICING
        assert 'claude-3-opus-20240229' in MODEL_PRICING
        assert 'claude-3-haiku-20240307' in MODEL_PRICING
