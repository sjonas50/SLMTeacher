"""Tests for PerformanceTracker."""
import os
import tempfile

from src.training.performance_tracker import PerformanceTracker


class TestPerformanceTracker:
    def test_record_round(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            tracker = PerformanceTracker(path)
            tracker.record_round(0, pre_accuracy=0.1, post_accuracy=0.2,
                                 category_accuracies={"math:easy": 0.3})
            assert len(tracker.rounds) == 1
            assert tracker.rounds[0]['post_accuracy'] == 0.2
        finally:
            os.unlink(path)

    def test_accuracy_curve(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            tracker = PerformanceTracker(path)
            for i in range(3):
                tracker.record_round(i, pre_accuracy=0.1 * i, post_accuracy=0.1 * (i + 1),
                                     category_accuracies={})
            curve = tracker.get_accuracy_curve()
            assert len(curve) == 3
            assert abs(curve[0] - 0.1) < 1e-9
            assert abs(curve[1] - 0.2) < 1e-9
            assert abs(curve[2] - 0.3) < 1e-9
        finally:
            os.unlink(path)

    def test_best_round(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            tracker = PerformanceTracker(path)
            tracker.record_round(0, 0.0, 0.5, {})
            tracker.record_round(1, 0.5, 0.8, {})
            tracker.record_round(2, 0.8, 0.6, {})

            best_round, best_acc = tracker.get_best_round()
            assert best_round == 1
            assert best_acc == 0.8
        finally:
            os.unlink(path)

    def test_not_converged_when_improving(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            tracker = PerformanceTracker(path)
            for i in range(5):
                tracker.record_round(i, 0.0, 0.1 * (i + 1), {})

            assert not tracker.is_converged(patience=3, min_delta=0.01)
        finally:
            os.unlink(path)

    def test_converged_when_plateaued(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            tracker = PerformanceTracker(path)
            # Rising then flat
            tracker.record_round(0, 0.0, 0.5, {})
            tracker.record_round(1, 0.5, 0.7, {})
            tracker.record_round(2, 0.7, 0.7, {})
            tracker.record_round(3, 0.7, 0.7, {})
            tracker.record_round(4, 0.7, 0.7, {})

            assert tracker.is_converged(patience=3, min_delta=0.01)
        finally:
            os.unlink(path)

    def test_save_load_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            tracker1 = PerformanceTracker(path)
            tracker1.record_round(0, 0.1, 0.2, {"math:easy": 0.3})
            tracker1.record_round(1, 0.2, 0.4, {"math:easy": 0.5})

            tracker2 = PerformanceTracker(path)
            assert len(tracker2.rounds) == 2
            assert tracker2.rounds[1]['post_accuracy'] == 0.4
        finally:
            os.unlink(path)

    def test_generate_summary(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            tracker = PerformanceTracker(path)
            tracker.record_round(0, 0.1, 0.3, {})
            tracker.record_round(1, 0.3, 0.5, {})

            summary = tracker.generate_summary()
            assert summary['total_rounds'] == 2
            assert summary['best_accuracy'] == 0.5
            assert summary['initial_accuracy'] == 0.1
            assert summary['total_improvement'] == 0.5 - 0.1
        finally:
            os.unlink(path)

    def test_empty_tracker(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            os.unlink(path)  # Ensure doesn't exist
            tracker = PerformanceTracker(path)
            assert tracker.get_accuracy_curve() == []
            assert tracker.get_best_round() == (0, 0.0)
            assert not tracker.is_converged()
            assert tracker.generate_summary()['total_rounds'] == 0
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_cumulative_cost(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            tracker = PerformanceTracker(path)
            tracker.record_round(0, 0.1, 0.2, {}, api_cost=1.50)
            tracker.record_round(1, 0.2, 0.3, {}, api_cost=2.00)

            assert tracker.rounds[1]['cumulative_cost'] == 3.50
        finally:
            os.unlink(path)
