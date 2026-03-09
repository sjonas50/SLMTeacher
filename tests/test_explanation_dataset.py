"""Tests for ExplanationDataset save/load/resume."""
import os
import tempfile

from src.data.explanation_generator import ExplanationDataset, _question_key


class TestQuestionKey:
    def test_deterministic(self):
        assert _question_key("hello") == _question_key("hello")

    def test_different_questions_differ(self):
        assert _question_key("a") != _question_key("b")

    def test_strips_whitespace(self):
        assert _question_key("  hello  ") == _question_key("hello")


class TestExplanationDataset:
    def test_add_and_get(self):
        ds = ExplanationDataset()
        ds.add("What is 2+2?", "4", "Two plus two equals four.", 0.7)
        results = ds.get_explanations("What is 2+2?", group_size=5)
        assert len(results) == 1
        assert results[0]["explanation"] == "Two plus two equals four."

    def test_group_size_limit(self):
        ds = ExplanationDataset()
        for i in range(5):
            ds.add("q", "a", f"expl_{i}", 0.7)
        assert len(ds.get_explanations("q", group_size=3)) == 3

    def test_missing_question(self):
        ds = ExplanationDataset()
        assert ds.get_explanations("nonexistent", group_size=4) == []

    def test_has_question(self):
        ds = ExplanationDataset()
        assert not ds.has_question("q")
        ds.add("q", "a", "e", 0.7)
        assert ds.has_question("q")

    def test_len(self):
        ds = ExplanationDataset()
        assert len(ds) == 0
        ds.add("q1", "a1", "e1", 0.7)
        ds.add("q1", "a1", "e2", 0.8)
        ds.add("q2", "a2", "e3", 0.7)
        assert len(ds) == 3

    def test_num_questions(self):
        ds = ExplanationDataset()
        ds.add("q1", "a1", "e1", 0.7)
        ds.add("q1", "a1", "e2", 0.8)
        ds.add("q2", "a2", "e3", 0.7)
        assert ds.num_questions == 2

    def test_save_load_roundtrip(self):
        ds = ExplanationDataset()
        ds.add("What is 2+2?", "4", "It's four.", 0.6)
        ds.add("What is 2+2?", "4", "Two plus two.", 0.8)
        ds.add("What is 3*3?", "9", "Nine.", 0.7)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            ds.save(path)
            loaded = ExplanationDataset.load(path)
            assert len(loaded) == 3
            assert loaded.num_questions == 2
            results = loaded.get_explanations("What is 2+2?", group_size=10)
            assert len(results) == 2
        finally:
            os.unlink(path)

    def test_incremental_save(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            ds = ExplanationDataset()
            entry = {"question": "q", "answer": "a", "explanation": "e", "temperature": 0.7}
            ds.save_incremental(path, entry)
            ds.save_incremental(path, entry)

            loaded = ExplanationDataset.load(path)
            assert len(loaded) == 2
        finally:
            os.unlink(path)

    def test_resume_deduplication(self):
        """Loading existing data and checking has_question prevents re-generation."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            ds = ExplanationDataset()
            ds.add("q1", "a1", "e1", 0.7)
            ds.save(path)

            loaded = ExplanationDataset.load(path)
            assert loaded.has_question("q1")
            assert not loaded.has_question("q2")
        finally:
            os.unlink(path)
