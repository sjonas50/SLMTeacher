"""Tests for ClaudeRLTTeacher (unit tests with mocked API)."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.teachers.claude_teacher import (
    ClaudeRLTTeacher,
    ClaudeConfig,
    RateLimitConfig,
    CacheConfig,
    FallbackStrategy,
    RateLimiter,
    ExplanationCache,
)
from src.utils.cost_tracker import CostTracker


class TestClaudeConfig:
    def test_defaults(self):
        config = ClaudeConfig()
        assert "claude-sonnet-4" in config.model
        assert config.temperature == 0.7
        assert config.max_tokens == 1024
        assert config.max_retries == 3

    def test_custom(self):
        config = ClaudeConfig(temperature=0.5, max_tokens=512)
        assert config.temperature == 0.5
        assert config.max_tokens == 512


class TestRateLimiter:
    def test_allows_first_request(self):
        limiter = RateLimiter(RateLimitConfig(requests_per_minute=10))
        can_proceed, wait = limiter.check_limits()
        assert can_proceed is True
        assert wait is None

    def test_records_usage(self):
        limiter = RateLimiter(RateLimitConfig(requests_per_minute=10))
        limiter.record_usage(100)
        assert len(limiter.request_times) == 1
        assert len(limiter.token_usage) == 1

    def test_blocks_after_limit(self):
        config = RateLimitConfig(requests_per_minute=2)
        limiter = RateLimiter(config)
        limiter.record_usage(100)
        limiter.record_usage(100)
        can_proceed, wait = limiter.check_limits()
        assert can_proceed is False
        assert wait is not None and wait >= 0


class TestExplanationCache:
    def test_cache_miss(self, tmp_path):
        config = CacheConfig(enabled=True, cache_dir=tmp_path / "cache")
        cache = ExplanationCache(config)
        result = cache.get("q", "a", 0.7)
        assert result is None

    def test_cache_set_and_get(self, tmp_path):
        config = CacheConfig(enabled=True, cache_dir=tmp_path / "cache")
        cache = ExplanationCache(config)
        cache.set("q", "a", 0.7, "explanation text")
        result = cache.get("q", "a", 0.7)
        assert result == "explanation text"

    def test_cache_disabled(self, tmp_path):
        config = CacheConfig(enabled=False, cache_dir=tmp_path / "cache")
        cache = ExplanationCache(config)
        cache.set("q", "a", 0.7, "text")
        assert cache.get("q", "a", 0.7) is None

    def test_max_size_eviction(self, tmp_path):
        config = CacheConfig(enabled=True, max_size=2, cache_dir=tmp_path / "cache")
        cache = ExplanationCache(config)
        cache.set("q1", "a1", 0.7, "e1")
        cache.set("q2", "a2", 0.7, "e2")
        cache.set("q3", "a3", 0.7, "e3")  # should evict q1
        assert cache.get("q1", "a1", 0.7) is None
        assert cache.get("q3", "a3", 0.7) == "e3"

    def test_clear(self, tmp_path):
        config = CacheConfig(enabled=True, cache_dir=tmp_path / "cache")
        cache = ExplanationCache(config)
        cache.set("q", "a", 0.7, "text")
        cache.clear()
        assert cache.get("q", "a", 0.7) is None


class TestClaudeRLTTeacher:
    def test_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                ClaudeRLTTeacher(api_key=None)

    def test_init_with_key(self):
        teacher = ClaudeRLTTeacher(api_key="test-key-123")
        assert teacher.api_key == "test-key-123"
        assert teacher.stats['total_requests'] == 0

    def test_empty_input_raises(self):
        teacher = ClaudeRLTTeacher(api_key="test-key")
        with pytest.raises(ValueError, match="non-empty"):
            teacher.generate_explanation("", "answer")

    def test_invalid_temperature_raises(self):
        teacher = ClaudeRLTTeacher(api_key="test-key")
        with pytest.raises(ValueError, match="Temperature"):
            teacher.generate_explanation("q", "a", temperature=2.0)

    def test_fallback_template(self):
        teacher = ClaudeRLTTeacher(
            api_key="test-key",
            fallback_strategy=FallbackStrategy.TEMPLATE,
        )
        # Mock the API to fail
        teacher._call_claude_api = Mock(side_effect=Exception("API down"))
        result = teacher.generate_explanation("What is 2+2?", "4")
        assert "2+2" in result
        assert "4" in result
        assert teacher.stats['fallback_uses'] == 1

    def test_stats_tracking(self):
        teacher = ClaudeRLTTeacher(api_key="test-key")
        stats = teacher.get_stats()
        assert stats['total_requests'] == 0
        assert stats['cache_hits'] == 0
        assert 'cache_hit_rate' in stats
