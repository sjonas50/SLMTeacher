"""Tests for LocalStudentEvaluator._compare_answers."""
from src.rewards.student_evaluator import LocalStudentEvaluator


class TestCompareAnswers:
    compare = staticmethod(LocalStudentEvaluator._compare_answers)

    def test_exact_match(self):
        assert self.compare("42", "42") == 1.0

    def test_exact_match_case_insensitive(self):
        assert self.compare("Answer is X", "answer is x") == 1.0

    def test_exact_match_whitespace(self):
        assert self.compare("  42  ", "42") == 1.0

    def test_numerical_match_integer(self):
        assert self.compare("The answer is 42", "42") == 1.0

    def test_numerical_match_float(self):
        assert self.compare("Result: 3.14", "pi is approximately 3.14") == 1.0

    def test_numerical_within_1_percent(self):
        # 101 is 1% off from 100
        assert self.compare("101", "100") == 0.9

    def test_numerical_within_5_percent(self):
        # 104 is 4% off from 100
        assert self.compare("104", "100") == 0.7

    def test_numerical_within_5_percent_boundary(self):
        # 105 is exactly 5% off from 100
        assert self.compare("105", "100") == 0.7

    def test_numerical_within_10_percent(self):
        # 108 is 8% off from 100
        assert self.compare("108", "100") == 0.5

    def test_numerical_within_25_percent(self):
        # 120 is 20% off from 100
        assert self.compare("120", "100") == 0.3

    def test_numerical_too_far(self):
        # 130 is 30% off from 100
        assert self.compare("130", "100") == 0.0

    def test_wrong_answer(self):
        assert self.compare("The answer is 7", "42") == 0.0

    def test_no_numbers_no_overlap(self):
        assert self.compare("yes", "no") == 0.0

    def test_token_overlap_partial(self):
        # "the answer" overlaps with "the answer is 42" (2/4 tokens)
        score = self.compare("the answer", "the answer is 42")
        assert 0.0 < score <= 0.5

    def test_token_overlap_full(self):
        # All ref tokens present
        score = self.compare("the quick brown fox", "quick brown fox")
        assert score == 0.5  # capped at 0.5

    def test_negative_numbers(self):
        assert self.compare("-5", "The answer is -5") == 1.0

    def test_last_number_used(self):
        # Should use last number from each string
        assert self.compare("Step 1: 10, Step 2: 42", "42") == 1.0

    def test_empty_strings(self):
        assert self.compare("", "") == 1.0

    def test_zero_reference(self):
        # Division by zero guard: ref_val == 0, exact match only
        assert self.compare("0", "0") == 1.0
        assert self.compare("0.01", "0") == 0.0
