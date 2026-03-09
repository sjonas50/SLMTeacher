"""Tests for StudentAssessor."""
from dataclasses import dataclass
from unittest.mock import MagicMock

from src.training.student_assessor import StudentAssessor, AssessmentResult


@dataclass
class FakeDataPoint:
    question: str
    solution: str
    subject: str
    difficulty: str


def _make_model(answers: dict):
    """Create a mock student model that returns fixed answers."""
    model = MagicMock()
    model.model = MagicMock()
    model.model.eval = MagicMock()

    def gen(prompt, **kwargs):
        for q, a in answers.items():
            if q in prompt:
                return {'generated_texts': [a]}
        return {'generated_texts': ['']}

    model.generate_optimized = MagicMock(side_effect=gen)
    return model


class TestStudentAssessor:
    def _benchmark(self):
        return [
            FakeDataPoint("What is 2+2?", "4", "math", "easy"),
            FakeDataPoint("What is 3*3?", "9", "math", "medium"),
            FakeDataPoint("What is 10/5?", "2", "math", "hard"),
        ]

    def test_assess_returns_result(self):
        data = self._benchmark()
        model = _make_model({"2+2": "4", "3*3": "9", "10/5": "2"})
        assessor = StudentAssessor(model, data)
        result = assessor.assess(0)

        assert isinstance(result, AssessmentResult)
        assert result.round_number == 0
        assert result.overall_accuracy == 1.0

    def test_category_accuracy(self):
        data = self._benchmark()
        model = _make_model({"2+2": "4", "3*3": "wrong", "10/5": "wrong"})
        assessor = StudentAssessor(model, data)
        result = assessor.assess(0)

        assert result.category_accuracies["math:easy"] == 1.0
        assert result.category_accuracies["math:medium"] == 0.0
        assert result.category_accuracies["math:hard"] == 0.0

    def test_weak_categories(self):
        data = self._benchmark()
        model = _make_model({"2+2": "4", "3*3": "wrong", "10/5": "wrong"})
        assessor = StudentAssessor(model, data, weak_threshold=0.5)
        result = assessor.assess(0)

        assert "math:medium" in result.weak_categories
        assert "math:hard" in result.weak_categories
        assert "math:easy" not in result.weak_categories

    def test_problem_history_tracked(self):
        data = self._benchmark()
        model = _make_model({"2+2": "4", "3*3": "9", "10/5": "2"})
        assessor = StudentAssessor(model, data)
        assessor.assess(0)
        assessor.assess(1)

        for qhash in assessor.problem_history:
            assert len(assessor.problem_history[qhash]) == 2

    def test_persistently_failing(self):
        data = self._benchmark()
        model = _make_model({"2+2": "4", "3*3": "wrong", "10/5": "wrong"})
        assessor = StudentAssessor(model, data, regen_after_failures=2)

        assessor.assess(0)
        result = assessor.assess(1)

        # 3*3 and 10/5 should be persistently failing after 2 rounds
        assert len(result.persistently_failing) == 2

    def test_serialization_roundtrip(self):
        data = self._benchmark()
        model = _make_model({"2+2": "4", "3*3": "9", "10/5": "2"})
        assessor = StudentAssessor(model, data)
        assessor.assess(0)

        state = assessor.to_dict()
        restored = StudentAssessor.from_dict(state, model, data)

        assert len(restored.problem_history) == len(assessor.problem_history)
