"""
Unit tests for ClaudeRLTTeacher class.

Run with: python -m pytest src/teachers/test_claude_teacher.py -v
"""
import os
import json
import time
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import pytest
import numpy as np

from claude_teacher import (
    ClaudeRLTTeacher,
    ClaudeConfig,
    RateLimitConfig,
    CacheConfig,
    FallbackStrategy,
    RateLimiter,
    ExplanationCache,
    create_teacher_from_env
)


class TestClaudeConfig:
    """Test ClaudeConfig dataclass."""
    
    def test_default_values(self):
        config = ClaudeConfig()
        assert config.model == "claude-3-5-sonnet-20241022"
        assert config.temperature == 0.7
        assert config.max_tokens == 1024
        assert config.api_timeout == 30
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.system_prompt is None
    
    def test_custom_values(self):
        config = ClaudeConfig(
            model="claude-3-opus-20240229",
            temperature=0.5,
            max_tokens=2048
        )
        assert config.model == "claude-3-opus-20240229"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048


class TestRateLimiter:
    """Test RateLimiter class."""
    
    def test_init(self):
        config = RateLimitConfig(requests_per_minute=10, tokens_per_minute=10000)
        limiter = RateLimiter(config)
        assert limiter.config == config
        assert limiter.request_times == []
        assert limiter.token_usage == []
    
    def test_check_limits_allows_first_request(self):
        config = RateLimitConfig(requests_per_minute=10)
        limiter = RateLimiter(config)
        can_proceed, wait_time = limiter.check_limits(1000)
        assert can_proceed is True
        assert wait_time is None
    
    def test_check_limits_enforces_request_limit(self):
        config = RateLimitConfig(requests_per_minute=2)
        limiter = RateLimiter(config)
        
        # Fill up the limit
        current_time = time.time()
        limiter.request_times = [current_time - 30, current_time - 20]
        
        can_proceed, wait_time = limiter.check_limits()
        assert can_proceed is False
        assert wait_time is not None
        assert wait_time > 0
    
    def test_check_limits_enforces_token_limit(self):
        config = RateLimitConfig(tokens_per_minute=5000)
        limiter = RateLimiter(config)
        
        # Fill up token limit
        current_time = time.time()
        limiter.token_usage = [(current_time - 30, 4000)]
        
        can_proceed, wait_time = limiter.check_limits(2000)
        assert can_proceed is False
        assert wait_time is not None
    
    def test_record_usage(self):
        config = RateLimitConfig()
        limiter = RateLimiter(config)
        
        limiter.record_usage(1500)
        assert len(limiter.request_times) == 1
        assert len(limiter.token_usage) == 1
        assert limiter.token_usage[0][1] == 1500


class TestExplanationCache:
    """Test ExplanationCache class."""
    
    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            cache = ExplanationCache(config)
            assert cache.config == config
            assert len(cache.cache) == 0
    
    def test_generate_key_deterministic(self):
        config = CacheConfig(enabled=True)
        cache = ExplanationCache(config)
        
        key1 = cache._generate_key("question", "answer", 0.7)
        key2 = cache._generate_key("question", "answer", 0.7)
        key3 = cache._generate_key("different", "answer", 0.7)
        
        assert key1 == key2
        assert key1 != key3
    
    def test_get_set_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(enabled=True, cache_dir=Path(tmpdir))
            cache = ExplanationCache(config)
            
            # Cache miss
            result = cache.get("question", "answer", 0.7)
            assert result is None
            
            # Set value
            cache.set("question", "answer", 0.7, "explanation")
            
            # Cache hit
            result = cache.get("question", "answer", 0.7)
            assert result == "explanation"
    
    def test_cache_disabled(self):
        config = CacheConfig(enabled=False)
        cache = ExplanationCache(config)
        
        cache.set("question", "answer", 0.7, "explanation")
        result = cache.get("question", "answer", 0.7)
        assert result is None
    
    def test_lru_eviction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(enabled=True, max_size=2, cache_dir=Path(tmpdir))
            cache = ExplanationCache(config)
            
            cache.set("q1", "a1", 0.7, "exp1")
            cache.set("q2", "a2", 0.7, "exp2")
            cache.set("q3", "a3", 0.7, "exp3")  # Should evict q1
            
            assert cache.get("q1", "a1", 0.7) is None
            assert cache.get("q2", "a2", 0.7) == "exp2"
            assert cache.get("q3", "a3", 0.7) == "exp3"
    
    def test_ttl_expiration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(enabled=True, ttl_hours=0, cache_dir=Path(tmpdir))
            cache = ExplanationCache(config)
            
            cache.set("question", "answer", 0.7, "explanation")
            # Should be expired immediately with ttl_hours=0
            time.sleep(0.1)
            result = cache.get("question", "answer", 0.7)
            assert result is None


class TestClaudeRLTTeacher:
    """Test ClaudeRLTTeacher class."""
    
    @pytest.fixture
    def mock_anthropic(self):
        """Mock Anthropic client."""
        with patch('claude_teacher.Anthropic') as mock:
            yield mock
    
    @pytest.fixture
    def teacher_with_mocked_api(self, mock_anthropic):
        """Create teacher with mocked API."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        # Mock API response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test explanation")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_client.messages.create.return_value = mock_response
        
        teacher = ClaudeRLTTeacher(
            api_key="test_key",
            fallback_strategy=FallbackStrategy.RAISE
        )
        return teacher, mock_client
    
    def test_init_requires_api_key(self):
        with pytest.raises(ValueError, match="Claude API key not provided"):
            ClaudeRLTTeacher(api_key=None)
    
    def test_init_with_env_var(self, monkeypatch, mock_anthropic):
        monkeypatch.setenv("CLAUDE_API_KEY", "test_key_from_env")
        teacher = ClaudeRLTTeacher()
        assert teacher.api_key == "test_key_from_env"
    
    def test_generate_explanation_basic(self, teacher_with_mocked_api):
        teacher, mock_client = teacher_with_mocked_api
        
        explanation = teacher.generate_explanation(
            question="What is 2+2?",
            answer="4"
        )
        
        assert explanation == "Test explanation"
        assert mock_client.messages.create.called
        assert teacher.stats['api_calls'] == 1
        assert teacher.stats['total_requests'] == 1
    
    def test_generate_explanation_validation(self, teacher_with_mocked_api):
        teacher, _ = teacher_with_mocked_api
        
        # Empty question
        with pytest.raises(ValueError, match="must be non-empty"):
            teacher.generate_explanation("", "answer")
        
        # Empty answer
        with pytest.raises(ValueError, match="must be non-empty"):
            teacher.generate_explanation("question", "")
        
        # Invalid temperature
        with pytest.raises(ValueError, match="Temperature must be between"):
            teacher.generate_explanation("question", "answer", temperature=1.5)
    
    def test_caching_behavior(self, teacher_with_mocked_api):
        teacher, mock_client = teacher_with_mocked_api
        
        # First call - should hit API
        exp1 = teacher.generate_explanation("What is 2+2?", "4", temperature=0.7)
        assert mock_client.messages.create.call_count == 1
        
        # Second call with same params - should hit cache
        exp2 = teacher.generate_explanation("What is 2+2?", "4", temperature=0.7)
        assert mock_client.messages.create.call_count == 1  # No additional call
        assert exp1 == exp2
        assert teacher.stats['cache_hits'] == 1
    
    def test_batch_processing(self, teacher_with_mocked_api):
        teacher, mock_client = teacher_with_mocked_api
        
        questions = ["Q1", "Q2", "Q3"]
        answers = ["A1", "A2", "A3"]
        
        results = teacher.batch_generate_explanations(
            questions=questions,
            answers=answers,
            max_workers=2
        )
        
        assert len(results) == 3
        assert all(r['success'] for r in results)
        assert mock_client.messages.create.call_count == 3
    
    def test_batch_processing_validation(self, teacher_with_mocked_api):
        teacher, _ = teacher_with_mocked_api
        
        # Mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            teacher.batch_generate_explanations(
                questions=["Q1", "Q2"],
                answers=["A1"]
            )
    
    def test_fallback_template_strategy(self):
        teacher = ClaudeRLTTeacher(
            api_key="test_key",
            fallback_strategy=FallbackStrategy.TEMPLATE
        )
        
        # Mock API to fail
        with patch.object(teacher, '_call_claude_api', side_effect=Exception("API Error")):
            explanation = teacher.generate_explanation("What is 2+2?", "4")
            
            assert "Let me explain" in explanation
            assert "What is 2+2?" in explanation
            assert "4" in explanation
            assert teacher.stats['fallback_uses'] == 1
    
    def test_fallback_raise_strategy(self):
        teacher = ClaudeRLTTeacher(
            api_key="test_key",
            fallback_strategy=FallbackStrategy.RAISE
        )
        
        # Mock API to fail
        with patch.object(teacher, '_call_claude_api', side_effect=Exception("API Error")):
            with pytest.raises(Exception, match="API Error"):
                teacher.generate_explanation("What is 2+2?", "4")
    
    def test_custom_fallback_callback(self):
        def custom_fallback(question, answer):
            return f"Custom: {question} -> {answer}"
        
        teacher = ClaudeRLTTeacher(
            api_key="test_key",
            fallback_strategy=FallbackStrategy.TEMPLATE,
            fallback_callback=custom_fallback
        )
        
        with patch.object(teacher, '_call_claude_api', side_effect=Exception("API Error")):
            explanation = teacher.generate_explanation("What is 2+2?", "4")
            assert explanation == "Custom: What is 2+2? -> 4"
    
    def test_rate_limiting_wait(self, teacher_with_mocked_api):
        teacher, _ = teacher_with_mocked_api
        
        # Fill rate limit
        teacher.rate_limiter.request_times = [time.time()] * 50
        
        with patch('time.sleep') as mock_sleep:
            teacher._wait_for_rate_limit()
            assert mock_sleep.called
    
    def test_get_stats(self, teacher_with_mocked_api):
        teacher, _ = teacher_with_mocked_api
        
        # Generate some activity
        teacher.generate_explanation("Q1", "A1")
        teacher.generate_explanation("Q1", "A1")  # Cache hit
        
        stats = teacher.get_stats()
        assert stats['total_requests'] == 2
        assert stats['api_calls'] == 1
        assert stats['cache_hits'] == 1
        assert 'cost_summary' in stats
        assert 'cache_hit_rate' in stats
    
    def test_retry_logic(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        # First two calls fail, third succeeds
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Success")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        
        mock_client.messages.create.side_effect = [
            Exception("Timeout"),
            Exception("Connection Error"),
            mock_response
        ]
        
        teacher = ClaudeRLTTeacher(
            api_key="test_key",
            claude_config=ClaudeConfig(max_retries=3, retry_delay=0.01)
        )
        
        explanation = teacher.generate_explanation("Q", "A")
        assert explanation == "Success"
        assert mock_client.messages.create.call_count == 3


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_teacher_from_env(self, monkeypatch, mock_anthropic):
        # Set environment variables
        monkeypatch.setenv("CLAUDE_API_KEY", "test_key")
        monkeypatch.setenv("CLAUDE_MODEL", "claude-3-opus-20240229")
        monkeypatch.setenv("CLAUDE_BUDGET_LIMIT", "50.0")
        monkeypatch.setenv("CLAUDE_CACHE_ENABLED", "false")
        
        with patch('claude_teacher.Anthropic'):
            teacher = create_teacher_from_env()
            
            assert teacher.api_key == "test_key"
            assert teacher.claude_config.model == "claude-3-opus-20240229"
            assert teacher.cost_tracker.budget_limit == 50.0
            assert teacher.cache_config.enabled is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])