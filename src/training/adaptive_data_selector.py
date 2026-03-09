"""Adaptive Data Selector — weighted sampling that targets student weaknesses.

Given an AssessmentResult, selects training problems that oversample weak
categories and identifies problems needing fresh teacher explanations.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.data.explanation_generator import ExplanationDataset, _question_key
from src.training.student_assessor import AssessmentResult

logger = logging.getLogger(__name__)


@dataclass
class RegenCandidate:
    """A problem that needs fresh teacher explanations."""
    data_point: object  # RLTDataPoint
    temperatures: List[float]
    reason: str


@dataclass
class SelectionResult:
    """Result of adaptive data selection."""
    selected_data: list              # List[RLTDataPoint]
    category_weights: Dict[str, float]
    regen_candidates: List[RegenCandidate]


class AdaptiveDataSelector:
    """Selects training data weighted toward the student's weak areas."""

    def __init__(
        self,
        train_data: list,
        min_problems: int = 50,
        max_problems: int = 500,
    ):
        self.train_data = train_data
        self.min_problems = min_problems
        self.max_problems = max_problems

        # Index by category
        self._by_category: Dict[str, list] = {}
        self._by_hash: Dict[str, object] = {}
        for dp in train_data:
            cat = f"{dp.subject}:{dp.difficulty}"
            self._by_category.setdefault(cat, []).append(dp)
            self._by_hash[_question_key(dp.question)] = dp

    def select(
        self,
        assessment: AssessmentResult,
        explanation_dataset: Optional[ExplanationDataset] = None,
    ) -> SelectionResult:
        """Select training data based on assessment results."""

        # 1. Compute category weights
        weights = self._compute_weights(assessment.category_accuracies)

        # 2. Compute per-category sample counts
        total_weight = sum(weights.values()) or 1.0
        raw_total = max(self.min_problems, min(self.max_problems, len(self.train_data)))
        counts: Dict[str, int] = {}
        for cat, w in weights.items():
            counts[cat] = max(1, int(raw_total * w / total_weight))

        # 3. Sample from each category
        selected = []
        for cat, n in counts.items():
            pool = self._by_category.get(cat, [])
            if not pool:
                continue
            sampled = random.choices(pool, k=min(n, len(pool) * 3))
            # Deduplicate while preserving order
            seen = set()
            for dp in sampled:
                h = _question_key(dp.question)
                if h not in seen:
                    seen.add(h)
                    selected.append(dp)

        # 4. Always include persistently failing problems (from train set)
        for qhash in assessment.persistently_failing:
            if qhash in self._by_hash:
                dp = self._by_hash[qhash]
                if not any(_question_key(s.question) == qhash for s in selected):
                    selected.append(dp)

        # 5. Clamp total
        if len(selected) > self.max_problems:
            selected = random.sample(selected, self.max_problems)
        elif len(selected) < self.min_problems:
            # Pad with random samples from train_data
            remaining = [dp for dp in self.train_data if dp not in selected]
            pad = random.sample(remaining, min(self.min_problems - len(selected), len(remaining)))
            selected.extend(pad)

        # 6. Identify regen candidates
        regen = self._find_regen_candidates(assessment, explanation_dataset)

        logger.info(
            "Selected %d training problems (%d categories, %d regen candidates)",
            len(selected), len(counts), len(regen),
        )
        return SelectionResult(
            selected_data=selected,
            category_weights=weights,
            regen_candidates=regen,
        )

    def _compute_weights(self, category_accuracies: Dict[str, float]) -> Dict[str, float]:
        """Higher weight for weaker categories."""
        weights = {}
        for cat in self._by_category:
            acc = category_accuracies.get(cat, 0.0)
            w = 3.0 / (acc + 0.1)
            # Reduce weight for strong categories
            if acc > 0.8:
                w *= 0.5
            weights[cat] = w
        return weights

    def _find_regen_candidates(
        self,
        assessment: AssessmentResult,
        explanation_dataset: Optional[ExplanationDataset],
    ) -> List[RegenCandidate]:
        """Find problems that need fresh explanations."""
        if explanation_dataset is None:
            return []

        candidates = []
        for qhash in assessment.persistently_failing:
            dp = self._by_hash.get(qhash)
            if dp is None:
                continue
            if not explanation_dataset.has_question(dp.question):
                continue

            # Find temperatures already used
            existing = explanation_dataset.get_explanations(dp.question, group_size=100)
            used_temps = {e.get('temperature', 0.7) for e in existing}

            # Suggest new temperatures
            new_temps = []
            for base in [0.3, 0.5, 0.7, 0.9, 1.0]:
                if base not in used_temps and len(new_temps) < 3:
                    new_temps.append(base)

            if not new_temps:
                # All standard temps used; try shifted values
                new_temps = [min(1.0, max(0.1, t + 0.15)) for t in sorted(used_temps)[:3]]

            candidates.append(RegenCandidate(
                data_point=dp,
                temperatures=new_temps,
                reason=f"failing {len(assessment.persistently_failing)} consecutive rounds",
            ))

        return candidates

    def add_data(self, new_data: list) -> None:
        """Add new problems to the training pool."""
        for dp in new_data:
            cat = f"{dp.subject}:{dp.difficulty}"
            self._by_category.setdefault(cat, []).append(dp)
            self._by_hash[_question_key(dp.question)] = dp
        self.train_data.extend(new_data)
