"""Tests for CurriculumGenerator."""
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

from src.data.curriculum_generator import (
    CurriculumConfig,
    CurriculumGenerator,
    TopicDecomposition,
    _extract_json,
)
from src.data.data_processor import RLTDataPoint


# ── _extract_json ────────────────────────────────────────────────────

class TestExtractJson:
    def test_direct_json(self):
        result = _extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = _extract_json(text)
        assert result == {"key": "value"}

    def test_json_in_prose(self):
        text = 'Here is the result:\n{"key": "value"}\nDone.'
        result = _extract_json(text)
        assert result == {"key": "value"}

    def test_invalid_raises(self):
        try:
            _extract_json("no json here")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


# ── Helpers ──────────────────────────────────────────────────────────

DECOMPOSE_RESPONSE = json.dumps({
    "subject": "math",
    "topics": [
        {
            "name": "Basics",
            "subtopics": ["Addition", "Subtraction"],
        },
    ],
})

PROBLEMS_RESPONSE = json.dumps({
    "problems": [
        {"question": "What is 2 + 3?", "answer": "5"},
        {"question": "What is 10 - 4?", "answer": "6"},
        {"question": "too short", "answer": "x"},
    ],
})


def _make_teacher(responses=None):
    """Create a mock teacher that returns canned responses."""
    teacher = MagicMock()
    calls = iter(responses or [])

    def fake_api(prompt, temperature, max_tokens=None):
        try:
            return (next(calls), 100, 50)
        except StopIteration:
            return ('{"problems": []}', 10, 10)

    teacher._call_claude_api = MagicMock(side_effect=fake_api)
    return teacher


# ── CurriculumGenerator ─────────────────────────────────────────────

class TestCurriculumGenerator:
    def test_decompose_topics(self):
        teacher = _make_teacher([DECOMPOSE_RESPONSE])
        gen = CurriculumGenerator(teacher, CurriculumConfig())
        decomp = gen._decompose_topics("learn basic math")

        assert isinstance(decomp, TopicDecomposition)
        assert decomp.subject == "math"
        assert len(decomp.topics) == 1
        assert decomp.topics[0]["name"] == "Basics"

    def test_generate_problems_creates_datapoints(self):
        teacher = _make_teacher([PROBLEMS_RESPONSE])
        config = CurriculumConfig(verification_mode="none")
        gen = CurriculumGenerator(teacher, config)
        gen._topic_decomposition = TopicDecomposition(
            description="test", subject="math", topics=[],
        )

        problems = gen._generate_problems("Basics", "Addition", "easy", 3)
        # "too short" question (< 10 chars) should be filtered out
        assert len(problems) == 2
        assert all(isinstance(p, RLTDataPoint) for p in problems)
        assert problems[0].subject == "math"
        assert problems[0].difficulty == "easy"

    def test_generate_problems_filters_short_questions(self):
        response = json.dumps({
            "problems": [
                {"question": "short", "answer": "x"},
                {"question": "This is a valid question about math?", "answer": "42"},
            ],
        })
        teacher = _make_teacher([response])
        config = CurriculumConfig(verification_mode="none")
        gen = CurriculumGenerator(teacher, config)
        gen._topic_decomposition = TopicDecomposition(
            description="test", subject="math", topics=[],
        )

        problems = gen._generate_problems("Basics", "Addition", "easy", 2)
        assert len(problems) == 1
        assert "valid question" in problems[0].question

    def test_verification_filters_bad_problems(self):
        # Problem generation response
        gen_response = json.dumps({
            "problems": [
                {"question": "What is the square root of 144?", "answer": "12"},
                {"question": "What is the capital of France?", "answer": "Berlin"},
            ],
        })
        # Verification responses (one correct, one wrong)
        teacher = _make_teacher([gen_response, "12", "Paris"])
        config = CurriculumConfig(verification_mode="self")
        gen = CurriculumGenerator(teacher, config)
        gen._topic_decomposition = TopicDecomposition(
            description="test", subject="math", topics=[],
        )

        problems = gen._generate_problems("Basics", "Roots", "easy", 2)
        # First problem verified correctly (12 == 12), second fails (Paris != Berlin)
        assert len(problems) == 1
        assert "144" in problems[0].question

    def test_cache_save_load_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            cache_path = f.name
        try:
            config = CurriculumConfig(
                verification_mode="none", cache_path=cache_path,
            )
            teacher = _make_teacher([DECOMPOSE_RESPONSE, PROBLEMS_RESPONSE])
            gen1 = CurriculumGenerator(teacher, config)
            gen1._topic_decomposition = TopicDecomposition(
                description="learn math", subject="math",
                topics=[{"name": "Basics", "subtopics": ["Addition"]}],
            )
            gen1._generated = [
                RLTDataPoint("What is 2+3?", "5", "math", "easy"),
                RLTDataPoint("What is 10-4?", "6", "math", "easy"),
            ]
            gen1._save_cache()

            gen2 = CurriculumGenerator(MagicMock(), config)
            loaded = gen2._load_cache()

            assert loaded is True
            assert len(gen2._generated) == 2
            assert gen2._generated[0].question == "What is 2+3?"
            assert gen2._topic_decomposition is not None
            assert gen2._topic_decomposition.subject == "math"
        finally:
            os.unlink(cache_path)

    def test_generate_targeted_problems(self):
        response = json.dumps({
            "problems": [
                {"question": "Hard derivative: d/dx of x^3?", "answer": "3x^2"},
            ],
        })
        teacher = _make_teacher([response])
        config = CurriculumConfig(verification_mode="none")
        gen = CurriculumGenerator(teacher, config)
        gen._topic_decomposition = TopicDecomposition(
            description="learn calculus", subject="math",
            topics=[{"name": "Calculus", "subtopics": ["Derivatives"]}],
        )

        result = gen.generate_targeted_problems(["Derivatives:hard"], count=5)
        assert len(result) == 1
        assert result[0].difficulty == "hard"

    def test_full_curriculum_produces_train_eval_split(self):
        # Decompose response + enough problem responses
        responses = [DECOMPOSE_RESPONSE]
        # 2 subtopics × 3 difficulties = 6 calls for problems
        for _ in range(6):
            responses.append(PROBLEMS_RESPONSE)

        teacher = _make_teacher(responses)
        config = CurriculumConfig(
            total_problems=10,
            problems_per_topic=3,
            eval_fraction=0.2,
            verification_mode="none",
        )
        gen = CurriculumGenerator(teacher, config)
        train_data, eval_data = gen.generate_curriculum("learn basic math")

        total = len(train_data) + len(eval_data)
        assert total > 0
        # Eval should be roughly 20% of total
        assert len(eval_data) >= 1
        assert len(train_data) >= len(eval_data)

    def test_empty_targeted_returns_empty(self):
        gen = CurriculumGenerator(MagicMock(), CurriculumConfig())
        assert gen.generate_targeted_problems([]) == []

    def test_load_cache_missing_file(self):
        config = CurriculumConfig(cache_path="/nonexistent/path.json")
        gen = CurriculumGenerator(MagicMock(), config)
        assert gen._load_cache() is False
