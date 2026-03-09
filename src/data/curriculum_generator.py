"""Teacher-generated curriculum from text descriptions.

Given a plain-text learning objective (e.g. "become an expert in calculus"),
this module uses Claude to decompose the objective into topics, generate
question-answer pairs across difficulty levels, and optionally self-verify
each problem for correctness.

Generated problems are persisted to a JSON cache for resume support.
"""

import json
import logging
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.data.data_processor import RLTDataPoint

logger = logging.getLogger(__name__)


@dataclass
class CurriculumConfig:
    total_problems: int = 200
    problems_per_topic: int = 10
    eval_fraction: float = 0.2
    verification_mode: str = "self"  # "self" or "none"
    cache_path: str = "curriculum_cache.json"


@dataclass
class TopicDecomposition:
    description: str
    subject: str
    topics: List[Dict[str, Any]]


class CurriculumGenerator:
    """Generates a full curriculum from a text description using Claude."""

    def __init__(self, teacher, config: CurriculumConfig):
        self.teacher = teacher
        self.config = config
        self._generated: List[RLTDataPoint] = []
        self._topic_decomposition: Optional[TopicDecomposition] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_curriculum(
        self, description: str
    ) -> Tuple[List[RLTDataPoint], List[RLTDataPoint]]:
        """Generate full curriculum from text description.

        Returns:
            (train_data, eval_data) split according to ``eval_fraction``.
        """
        # 1. Decompose description into topics
        self._topic_decomposition = self._decompose_topics(description)
        logger.info(
            "Decomposed into %d topics (subject=%s)",
            len(self._topic_decomposition.topics),
            self._topic_decomposition.subject,
        )

        # 2. Generate problems across topics and difficulties
        for topic in self._topic_decomposition.topics:
            for subtopic in topic["subtopics"]:
                for difficulty in ["easy", "medium", "hard"]:
                    if len(self._generated) >= self.config.total_problems:
                        break
                    problems = self._generate_problems(
                        topic=topic["name"],
                        subtopic=subtopic,
                        difficulty=difficulty,
                        count=self.config.problems_per_topic,
                    )
                    self._generated.extend(problems)
                    self._save_cache()
                    logger.info(
                        "Generated %d/%d problems (%s / %s / %s)",
                        len(self._generated),
                        self.config.total_problems,
                        topic["name"],
                        subtopic,
                        difficulty,
                    )
                if len(self._generated) >= self.config.total_problems:
                    break
            if len(self._generated) >= self.config.total_problems:
                break

        # 3. Split into train/eval
        random.shuffle(self._generated)
        split = int(len(self._generated) * (1 - self.config.eval_fraction))
        train_data = self._generated[:split]
        eval_data = self._generated[split:]
        logger.info(
            "Curriculum complete: %d train, %d eval problems",
            len(train_data),
            len(eval_data),
        )
        return train_data, eval_data

    def generate_targeted_problems(
        self, weak_categories: List[str], count: int = 20
    ) -> List[RLTDataPoint]:
        """Generate new problems targeting weak areas.

        Args:
            weak_categories: list of ``"subject:difficulty"`` strings from assessment.
            count: total number of new problems to generate.
        """
        if not weak_categories:
            return []

        new_problems: List[RLTDataPoint] = []
        per_category = max(1, count // len(weak_categories))

        for category in weak_categories:
            parts = category.rsplit(":", 1)
            if len(parts) != 2:
                continue
            cat_subject, difficulty = parts
            topic_name = self._find_topic_for_category(cat_subject)
            problems = self._generate_problems(
                topic=topic_name,
                subtopic=cat_subject,
                difficulty=difficulty,
                count=per_category,
            )
            new_problems.extend(problems)

        self._generated.extend(new_problems)
        self._save_cache()
        logger.info("Generated %d targeted problems for weak areas", len(new_problems))
        return new_problems

    # ------------------------------------------------------------------
    # Topic decomposition
    # ------------------------------------------------------------------

    def _decompose_topics(self, description: str) -> TopicDecomposition:
        """Ask Claude to break description into structured topics."""
        prompt = (
            "Break this learning objective into a structured curriculum.\n\n"
            f"Learning objective: {description}\n\n"
            "Respond in this exact JSON format (no other text):\n"
            "{\n"
            '  "subject": "math" or "science" or "general",\n'
            '  "topics": [\n'
            "    {\n"
            '      "name": "Topic Name",\n'
            '      "subtopics": ["Subtopic 1", "Subtopic 2", "Subtopic 3"]\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Create 3-6 topics with 2-4 subtopics each. "
            "Topics should progress from foundational to advanced."
        )

        response_text, _, _ = self.teacher._call_claude_api(prompt, temperature=0.3)
        data = _extract_json(response_text)

        return TopicDecomposition(
            description=description,
            subject=data.get("subject", "general"),
            topics=data["topics"],
        )

    # ------------------------------------------------------------------
    # Problem generation
    # ------------------------------------------------------------------

    def _generate_problems(
        self, topic: str, subtopic: str, difficulty: str, count: int
    ) -> List[RLTDataPoint]:
        """Generate a batch of problems for a specific topic/difficulty."""
        subject = (
            self._topic_decomposition.subject
            if self._topic_decomposition
            else "general"
        )

        prompt = (
            f'Generate exactly {count} {difficulty} practice problems about '
            f'"{subtopic}" (part of {topic}).\n\n'
            "CRITICAL RULES:\n"
            "- Each answer must be SHORT (a number, a single word, or at most one sentence)\n"
            "- Answers must be UNAMBIGUOUS — only one correct answer possible\n"
            "- Do NOT include explanations in the answer field\n\n"
            "Respond in this exact JSON format (no other text):\n"
            "{\n"
            '  "problems": [\n'
            '    {"question": "...", "answer": "..."},\n'
            '    {"question": "...", "answer": "..."}\n'
            "  ]\n"
            "}\n\n"
            f"Difficulty level: {difficulty}\n"
            "- easy: straightforward application of basics\n"
            "- medium: requires multi-step reasoning\n"
            "- hard: requires deep understanding and creative problem-solving"
        )

        try:
            response_text, _, _ = self.teacher._call_claude_api(
                prompt, temperature=0.7, max_tokens=2048
            )
            data = _extract_json(response_text)
        except Exception as e:
            logger.warning("Failed to generate problems for %s/%s: %s", subtopic, difficulty, e)
            return []

        problems: List[RLTDataPoint] = []
        for item in data.get("problems", []):
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()

            if not question or not answer or len(question) < 10:
                continue

            # Optional self-verification
            if self.config.verification_mode == "self":
                if not self._verify_problem(question, answer):
                    logger.debug("Verification failed for: %s", question[:60])
                    continue

            problems.append(
                RLTDataPoint(
                    question=question,
                    solution=answer,
                    subject=subject,
                    difficulty=difficulty,
                )
            )

        return problems

    def _verify_problem(self, question: str, answer: str) -> bool:
        """Ask Claude to independently solve the problem and check agreement."""
        prompt = (
            "Solve this problem independently. "
            "Give ONLY the final answer, nothing else.\n\n"
            f"Question: {question}"
        )

        try:
            response_text, _, _ = self.teacher._call_claude_api(
                prompt, temperature=0.0
            )
        except Exception as e:
            logger.warning("Verification API call failed: %s", e)
            return True  # on failure, keep the problem rather than silently discard

        from src.rewards.student_evaluator import LocalStudentEvaluator

        score = LocalStudentEvaluator._compare_answers(response_text.strip(), answer)
        return score >= 0.5

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_topic_for_category(self, category_subject: str) -> str:
        """Find the best matching topic name for a category string."""
        if not self._topic_decomposition:
            return category_subject
        for topic in self._topic_decomposition.topics:
            for subtopic in topic["subtopics"]:
                if category_subject.lower() in subtopic.lower():
                    return topic["name"]
        if self._topic_decomposition.topics:
            return self._topic_decomposition.topics[0]["name"]
        return category_subject

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_cache(self):
        """Save generated problems to disk for resume."""
        data: Dict[str, Any] = {
            "decomposition": (
                {
                    "description": self._topic_decomposition.description,
                    "subject": self._topic_decomposition.subject,
                    "topics": self._topic_decomposition.topics,
                }
                if self._topic_decomposition
                else None
            ),
            "problems": [dp.to_dict() for dp in self._generated],
        }
        os.makedirs(os.path.dirname(self.config.cache_path) or ".", exist_ok=True)
        with open(self.config.cache_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_cache(self) -> bool:
        """Load from cache. Returns True if cache was loaded."""
        if not os.path.exists(self.config.cache_path):
            return False
        try:
            with open(self.config.cache_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load curriculum cache: %s", e)
            return False

        decomp = data.get("decomposition")
        if decomp:
            self._topic_decomposition = TopicDecomposition(
                description=decomp["description"],
                subject=decomp["subject"],
                topics=decomp["topics"],
            )
        self._generated = [
            RLTDataPoint.from_dict(p) for p in data.get("problems", [])
        ]
        if self._generated:
            logger.info("Loaded %d problems from curriculum cache", len(self._generated))
        return len(self._generated) > 0


def _extract_json(text: str) -> dict:
    """Extract a JSON object from Claude's response text.

    Handles raw JSON, fenced code blocks, and JSON embedded in prose.
    """
    text = text.strip()

    # 1. Direct parse
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # 2. Extract from ```json ... ``` block (greedy to capture full nested JSON)
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Find first { ... } in text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # 4. Handle truncated JSON: close open brackets and retry
    if start != -1:
        fragment = text[start:]
        # Strip trailing code fence
        fragment = re.sub(r"\s*```\s*$", "", fragment)
        # Close open array brackets and braces
        open_brackets = fragment.count("[") - fragment.count("]")
        open_braces = fragment.count("{") - fragment.count("}")
        fragment = fragment.rstrip(", \n")
        fragment += "]" * max(0, open_brackets) + "}" * max(0, open_braces)
        try:
            return json.loads(fragment)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from response: {text[:200]}")
