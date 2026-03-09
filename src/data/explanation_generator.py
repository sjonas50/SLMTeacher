"""Pre-generation pipeline for teacher explanations.

Decouples API calls from the training loop by generating all explanations
upfront and storing them on disk as JSONL.  Supports incremental saves and
resume so a crashed run can pick up where it left off.
"""

import hashlib
import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _question_key(question: str) -> str:
    return hashlib.sha256(question.strip().encode()).hexdigest()


class ExplanationDataset:
    """In-memory store for pre-generated explanations, backed by JSONL."""

    def __init__(self):
        self._data: Dict[str, List[dict]] = {}

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------
    def add(
        self,
        question: str,
        answer: str,
        explanation: str,
        temperature: float,
    ) -> None:
        key = _question_key(question)
        entry = {
            "explanation": explanation,
            "temperature": temperature,
            "question": question,
            "answer": answer,
        }
        self._data.setdefault(key, []).append(entry)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_explanations(self, question: str, group_size: int, strategy: str = "first") -> List[dict]:
        """Return up to *group_size* explanations for *question*.

        Args:
            strategy: "first" returns the oldest entries (default),
                      "latest" returns the newest (useful after regeneration).
        """
        key = _question_key(question)
        entries = self._data.get(key, [])
        if strategy == "latest":
            return entries[-group_size:]
        return entries[:group_size]

    def has_question(self, question: str) -> bool:
        key = _question_key(question)
        return key in self._data

    def __len__(self) -> int:
        return sum(len(v) for v in self._data.values())

    @property
    def num_questions(self) -> int:
        return len(self._data)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Write all entries to *path* as JSONL (one JSON object per line)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            for entries in self._data.values():
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")
        logger.info("Saved %d explanations to %s", len(self), path)

    def save_incremental(self, path: str, entry: dict) -> None:
        """Append a single entry to *path* for crash resilience."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    @classmethod
    def load(cls, path: str) -> "ExplanationDataset":
        """Load from a JSONL file."""
        ds = cls()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                ds.add(
                    question=entry["question"],
                    answer=entry["answer"],
                    explanation=entry["explanation"],
                    temperature=entry.get("temperature", 0.7),
                )
        logger.info("Loaded %d explanations (%d questions) from %s",
                     len(ds), ds.num_questions, path)
        return ds


def pregenerate_explanations(
    teacher,
    data_points: list,
    group_size: int = 4,
    temperatures: Optional[List[float]] = None,
    output_path: str = "explanations.jsonl",
) -> ExplanationDataset:
    """Generate explanations for all *data_points* and save to disk.

    Args:
        teacher: Teacher object with ``generate_explanation(question, answer, temperature)``.
        data_points: Iterable of objects with ``.question`` and ``.solution`` attributes.
        group_size: Number of explanations per question.
        temperatures: Per-group temperatures.  Defaults to ``[0.6, 0.7, ..., 0.6 + 0.1*(G-1)]``.
        output_path: JSONL file to write.

    Returns:
        The populated ``ExplanationDataset``.
    """
    if temperatures is None:
        temperatures = [min(1.0, 0.6 + 0.1 * g) for g in range(group_size)]
    else:
        temperatures = (temperatures * group_size)[:group_size]

    # Resume support: load existing if present
    if os.path.exists(output_path):
        dataset = ExplanationDataset.load(output_path)
        logger.info("Resuming — %d questions already generated", dataset.num_questions)
    else:
        dataset = ExplanationDataset()

    total = len(data_points)
    for idx, dp in enumerate(data_points):
        question = dp.question
        answer = dp.solution

        if dataset.has_question(question):
            continue

        for temp in temperatures:
            try:
                explanation = teacher.generate_explanation(
                    question=question,
                    answer=answer,
                    temperature=temp,
                )
                if isinstance(explanation, dict):
                    explanation = explanation.get("explanation", str(explanation))
            except Exception as e:
                logger.warning("Failed to generate explanation for q=%d: %s", idx, e)
                explanation = ""

            dataset.add(question, answer, explanation, temp)
            dataset.save_incremental(output_path, {
                "question": question,
                "answer": answer,
                "explanation": explanation,
                "temperature": temp,
            })

        if (idx + 1) % 50 == 0:
            logger.info("Pre-generated %d / %d questions", idx + 1, total)

    logger.info("Pre-generation complete: %d explanations for %d questions",
                len(dataset), dataset.num_questions)
    return dataset
