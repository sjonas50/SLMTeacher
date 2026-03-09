"""Performance Tracker — persists per-round metrics and detects convergence.

Writes to a JSON file that accumulates across rounds, supporting resume.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Tracks accuracy and metrics across continuous training rounds."""

    def __init__(self, metrics_path: str = "continuous_metrics.json"):
        self.metrics_path = metrics_path
        self.rounds: List[dict] = []

        if os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 0:
            with open(metrics_path) as f:
                data = json.load(f)
            self.rounds = data.get('rounds', [])
            logger.info("Loaded %d existing rounds from %s", len(self.rounds), metrics_path)

    def record_round(
        self,
        round_number: int,
        pre_accuracy: float,
        post_accuracy: float,
        category_accuracies: Dict[str, float],
        training_stats: Optional[Dict] = None,
        api_cost: float = 0.0,
    ):
        """Record metrics for a completed round."""
        record = {
            'round_number': round_number,
            'pre_accuracy': pre_accuracy,
            'post_accuracy': post_accuracy,
            'category_accuracies': category_accuracies,
            'training_stats': training_stats or {},
            'api_cost': api_cost,
            'cumulative_cost': sum(r.get('api_cost', 0) for r in self.rounds) + api_cost,
        }
        self.rounds.append(record)
        self.save()

    def is_converged(self, patience: int = 3, min_delta: float = 0.01) -> bool:
        """True if best accuracy hasn't improved by min_delta in last *patience* rounds."""
        if len(self.rounds) < patience + 1:
            return False

        accuracies = self.get_accuracy_curve()
        recent = accuracies[-patience:]
        earlier = accuracies[:-patience]

        if not earlier:
            return False

        best_earlier = max(earlier)
        best_recent = max(recent)

        return best_recent - best_earlier < min_delta

    def get_accuracy_curve(self) -> List[float]:
        """Return post_accuracy for each round."""
        return [r['post_accuracy'] for r in self.rounds]

    def get_best_round(self) -> Tuple[int, float]:
        """Return (round_number, accuracy) of the best round."""
        if not self.rounds:
            return (0, 0.0)
        best = max(self.rounds, key=lambda r: r['post_accuracy'])
        return (best['round_number'], best['post_accuracy'])

    def generate_summary(self) -> Dict:
        """Generate a summary of the entire training run."""
        if not self.rounds:
            return {'total_rounds': 0}

        accuracies = self.get_accuracy_curve()
        best_round, best_acc = self.get_best_round()

        return {
            'total_rounds': len(self.rounds),
            'best_round': best_round,
            'best_accuracy': best_acc,
            'final_accuracy': accuracies[-1] if accuracies else 0.0,
            'initial_accuracy': self.rounds[0]['pre_accuracy'],
            'total_improvement': accuracies[-1] - self.rounds[0]['pre_accuracy'] if accuracies else 0.0,
            'total_cost': self.rounds[-1].get('cumulative_cost', 0.0),
            'accuracy_curve': accuracies,
        }

    def save(self):
        """Persist metrics to disk."""
        os.makedirs(os.path.dirname(self.metrics_path) or '.', exist_ok=True)
        with open(self.metrics_path, 'w') as f:
            json.dump({'rounds': self.rounds}, f, indent=2)
