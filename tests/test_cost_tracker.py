"""Tests for CostTracker."""
import pytest
from src.utils.cost_tracker import CostTracker


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
