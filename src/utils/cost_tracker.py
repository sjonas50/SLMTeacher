"""
Cost tracking for API usage and budget management
"""
from typing import Dict, Optional
import json
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# Per-model pricing (USD per million tokens) — update when pricing changes
MODEL_PRICING = {
    # Claude 4.x / Sonnet 4
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4-5-20250514": {"input": 3.0, "output": 15.0},
    # Claude 3.5
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.0},
    # Claude 3
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
}

# Fallback pricing when model is unknown
DEFAULT_PRICING = {"input": 3.0, "output": 15.0}


class CostTracker:
    """Track API usage and costs throughout experiments."""

    def __init__(self, budget_limit: float = 10.0, save_path: Optional[str] = None):
        self.budget_limit = budget_limit
        self.save_path = save_path or "api_usage_log.json"
        self.costs = {
            'input_tokens': 0,
            'output_tokens': 0,
            'total_cost': 0.0,
            'api_calls': 0,
            'start_time': time.time(),
            'history': []
        }

    def _get_pricing(self, model: Optional[str] = None) -> Dict[str, float]:
        """Get pricing for a model, falling back to defaults."""
        if model and model in MODEL_PRICING:
            return MODEL_PRICING[model]
        if model:
            # Try prefix match for dated model variants
            for key, pricing in MODEL_PRICING.items():
                if model.startswith(key.rsplit("-", 1)[0]):
                    return pricing
            logger.warning("Unknown model '%s', using default pricing", model)
        return DEFAULT_PRICING

    def add_usage(self, input_tokens: int, output_tokens: int, metadata: Optional[Dict] = None):
        """Add token usage and calculate cost."""
        self.costs['input_tokens'] += input_tokens
        self.costs['output_tokens'] += output_tokens
        self.costs['api_calls'] += 1

        # Look up per-model pricing
        model = (metadata or {}).get('model')
        pricing = self._get_pricing(model)

        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']
        call_cost = input_cost + output_cost

        self.costs['total_cost'] += call_cost

        self.costs['history'].append({
            'timestamp': datetime.now().isoformat(),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost': call_cost,
            'cumulative_cost': self.costs['total_cost'],
            'metadata': metadata or {}
        })

        # Check budget
        if self.costs['total_cost'] > self.budget_limit:
            self.save_log()
            raise ValueError(f"Budget limit exceeded: ${self.costs['total_cost']:.2f} > ${self.budget_limit}")

        # Auto-save every 10 calls
        if self.costs['api_calls'] % 10 == 0:
            self.save_log()

    def get_summary(self) -> Dict:
        """Get cost summary with numeric values."""
        runtime = time.time() - self.costs['start_time']
        return {
            'total_cost': round(self.costs['total_cost'], 4),
            'budget_remaining': round(self.budget_limit - self.costs['total_cost'], 4),
            'api_calls': self.costs['api_calls'],
            'total_tokens': self.costs['input_tokens'] + self.costs['output_tokens'],
            'average_cost_per_call': round(self.costs['total_cost'] / max(1, self.costs['api_calls']), 4),
            'runtime_minutes': round(runtime / 60, 1),
            'calls_per_minute': round(self.costs['api_calls'] / max(1, runtime / 60), 2)
        }

    def save_log(self):
        """Save usage log to file."""
        with open(self.save_path, 'w') as f:
            json.dump(self.costs, f, indent=2)

    def load_log(self):
        """Load previous usage log."""
        try:
            with open(self.save_path, 'r') as f:
                self.costs = json.load(f)
        except FileNotFoundError:
            pass
