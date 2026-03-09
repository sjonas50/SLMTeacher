"""Tests for AdaptiveDataSelector."""
from dataclasses import dataclass
from unittest.mock import MagicMock

from src.data.explanation_generator import ExplanationDataset, _question_key
from src.training.adaptive_data_selector import AdaptiveDataSelector, SelectionResult
from src.training.student_assessor import AssessmentResult


@dataclass
class FakeDataPoint:
    question: str
    solution: str
    subject: str
    difficulty: str


def _make_train_data():
    data = []
    for i in range(20):
        data.append(FakeDataPoint(f"easy q{i}", str(i), "math", "easy"))
    for i in range(20):
        data.append(FakeDataPoint(f"hard q{i}", str(i), "math", "hard"))
    return data


def _make_assessment(easy_acc=0.9, hard_acc=0.1, failing=None):
    return AssessmentResult(
        overall_accuracy=(easy_acc + hard_acc) / 2,
        category_accuracies={"math:easy": easy_acc, "math:hard": hard_acc},
        per_problem_scores={},
        weak_categories=["math:hard"] if hard_acc < 0.6 else [],
        persistently_failing=failing or [],
        round_number=0,
    )


class TestAdaptiveDataSelector:
    def test_select_returns_result(self):
        data = _make_train_data()
        selector = AdaptiveDataSelector(data, min_problems=5, max_problems=30)
        assessment = _make_assessment()
        result = selector.select(assessment)

        assert isinstance(result, SelectionResult)
        assert len(result.selected_data) >= 5

    def test_weak_category_gets_higher_weight(self):
        data = _make_train_data()
        selector = AdaptiveDataSelector(data, min_problems=5, max_problems=100)
        assessment = _make_assessment(easy_acc=0.9, hard_acc=0.1)
        result = selector.select(assessment)

        assert result.category_weights["math:hard"] > result.category_weights["math:easy"]

    def test_persistently_failing_included(self):
        data = _make_train_data()
        failing_hash = _question_key("hard q0")
        selector = AdaptiveDataSelector(data, min_problems=5, max_problems=30)
        assessment = _make_assessment(failing=[failing_hash])
        result = selector.select(assessment)

        selected_hashes = {_question_key(dp.question) for dp in result.selected_data}
        assert failing_hash in selected_hashes

    def test_regen_candidates_found(self):
        data = _make_train_data()
        ds = ExplanationDataset()
        ds.add("hard q0", "0", "some explanation", 0.7)

        failing_hash = _question_key("hard q0")
        selector = AdaptiveDataSelector(data, min_problems=5, max_problems=30)
        assessment = _make_assessment(failing=[failing_hash])
        result = selector.select(assessment, explanation_dataset=ds)

        assert len(result.regen_candidates) == 1
        assert len(result.regen_candidates[0].temperatures) > 0

    def test_no_regen_without_dataset(self):
        data = _make_train_data()
        failing_hash = _question_key("hard q0")
        selector = AdaptiveDataSelector(data, min_problems=5, max_problems=30)
        assessment = _make_assessment(failing=[failing_hash])
        result = selector.select(assessment, explanation_dataset=None)

        assert len(result.regen_candidates) == 0

    def test_selection_size_bounded(self):
        data = _make_train_data()
        selector = AdaptiveDataSelector(data, min_problems=10, max_problems=15)
        assessment = _make_assessment()
        result = selector.select(assessment)

        assert 10 <= len(result.selected_data) <= 15
