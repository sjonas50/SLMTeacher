"""Student Assessor — evaluates the student model and tracks per-problem history.

Used by ContinuousTrainer to identify weak areas across training rounds.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.data.explanation_generator import _question_key
from src.rewards.student_evaluator import LocalStudentEvaluator

logger = logging.getLogger(__name__)


@dataclass
class AssessmentResult:
    """Result of assessing the student on a benchmark set."""
    overall_accuracy: float
    category_accuracies: Dict[str, float]  # "math:medium" → 0.65
    per_problem_scores: Dict[str, float]   # question_hash → score
    weak_categories: List[str]
    persistently_failing: List[str]        # question hashes
    round_number: int


class StudentAssessor:
    """Evaluates a student model on a fixed benchmark and tracks history."""

    def __init__(
        self,
        student_model,
        benchmark_data: list,
        weak_threshold: float = 0.6,
        regen_after_failures: int = 2,
    ):
        self.student_model = student_model
        self.benchmark_data = benchmark_data
        self.weak_threshold = weak_threshold
        self.regen_after_failures = regen_after_failures
        # question_hash → [(round, score), ...]
        self.problem_history: Dict[str, List[Tuple[int, float]]] = {}

    def assess(self, round_number: int) -> AssessmentResult:
        """Run the student on all benchmark problems and return results."""
        if hasattr(self.student_model, 'model'):
            self.student_model.model.eval()

        per_problem_scores: Dict[str, float] = {}
        category_scores: Dict[str, List[float]] = {}

        with torch.no_grad():
            for dp in self.benchmark_data:
                qhash = _question_key(dp.question)
                prompt = f"Question: {dp.question}\nAnswer:"

                if hasattr(self.student_model, 'generate_optimized'):
                    result = self.student_model.generate_optimized(
                        prompt, max_new_tokens=50, do_sample=False,
                    )
                    generated = result['generated_texts'][0]
                else:
                    generated = ""

                score = LocalStudentEvaluator._compare_answers(generated, dp.solution)

                per_problem_scores[qhash] = score
                self.problem_history.setdefault(qhash, []).append((round_number, score))

                cat = f"{dp.subject}:{dp.difficulty}"
                category_scores.setdefault(cat, []).append(score)

        # Compute category accuracies
        category_accuracies = {
            cat: float(np.mean(scores)) for cat, scores in category_scores.items()
        }

        overall = float(np.mean(list(per_problem_scores.values()))) if per_problem_scores else 0.0
        weak = [cat for cat, acc in category_accuracies.items() if acc < self.weak_threshold]
        failing = self.get_persistently_failing(self.regen_after_failures)

        result = AssessmentResult(
            overall_accuracy=overall,
            category_accuracies=category_accuracies,
            per_problem_scores=per_problem_scores,
            weak_categories=weak,
            persistently_failing=failing,
            round_number=round_number,
        )

        logger.info(
            "Round %d assessment: accuracy=%.1f%%, weak_categories=%d, persistent_failures=%d",
            round_number, overall * 100, len(weak), len(failing),
        )
        return result

    def get_persistently_failing(self, min_rounds: int = 2) -> List[str]:
        """Return question hashes that scored < 1.0 for the last *min_rounds* consecutive rounds."""
        failing = []
        for qhash, history in self.problem_history.items():
            if len(history) < min_rounds:
                continue
            recent = history[-min_rounds:]
            if all(score < 1.0 for _, score in recent):
                failing.append(qhash)
        return failing

    def to_dict(self) -> dict:
        return {
            'problem_history': {
                k: [(r, s) for r, s in v] for k, v in self.problem_history.items()
            },
            'weak_threshold': self.weak_threshold,
            'regen_after_failures': self.regen_after_failures,
        }

    @classmethod
    def from_dict(cls, data: dict, student_model, benchmark_data) -> 'StudentAssessor':
        assessor = cls(
            student_model=student_model,
            benchmark_data=benchmark_data,
            weak_threshold=data.get('weak_threshold', 0.6),
            regen_after_failures=data.get('regen_after_failures', 2),
        )
        assessor.problem_history = {
            k: [(r, s) for r, s in v]
            for k, v in data.get('problem_history', {}).items()
        }
        return assessor
