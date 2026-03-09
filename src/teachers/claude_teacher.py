"""
Claude-based Reinforcement Learning Teacher (RLT) implementation.

This module provides a production-ready teacher class that uses Claude API
to generate high-quality explanations for the RLT training pipeline.
Based on Sakana AI's RLT methodology for training small language models.
"""
import os
import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from enum import Enum

from anthropic import Anthropic, APIError, APIConnectionError, APITimeoutError
import numpy as np

# Import the cost tracker
from ..utils.cost_tracker import CostTracker


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subject-specific system prompts and difficulty instructions
# ---------------------------------------------------------------------------
SUBJECT_PROMPTS: Dict[str, str] = {
    "math": (
        "You are a mathematics tutor. Show every algebraic step explicitly, "
        "verify intermediate computations, and check the final answer by "
        "substitution or estimation where possible."
    ),
    "science": (
        "You are a science tutor. State the relevant physical laws or concepts, "
        "include units throughout the calculation, and explain the reasoning "
        "behind each step."
    ),
    "default": (
        "You are a clear and patient tutor. Break the solution into logical "
        "steps, explain the reasoning at each stage, and verify the final answer."
    ),
}

DIFFICULTY_INSTRUCTIONS: Dict[str, str] = {
    "easy": "Use simple language and concrete examples. Keep the explanation short.",
    "medium": "Balance detail and clarity. Include key reasoning steps.",
    "hard": (
        "Show advanced reasoning. Address common misconceptions and explain "
        "why alternative approaches would fail."
    ),
}


class FallbackStrategy(Enum):
    """Fallback strategies when Claude API is unavailable."""
    TEMPLATE = "template"  # Use template-based explanations
    CACHE_ONLY = "cache_only"  # Only return cached results
    RAISE = "raise"  # Raise exception immediately
    RETRY = "retry"  # Retry with exponential backoff


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 50
    tokens_per_minute: int = 100000
    burst_size: int = 10
    wait_on_limit: bool = True


@dataclass
class CacheConfig:
    """Configuration for caching system."""
    enabled: bool = True
    max_size: int = 10000
    ttl_hours: int = 24
    cache_dir: Path = field(default_factory=lambda: Path(".claude_cache"))


@dataclass
class ClaudeConfig:
    """Configuration for Claude API and teacher behavior."""
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.7
    max_tokens: int = 1024
    api_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    system_prompt: Optional[str] = None


class RateLimiter:
    """Token bucket rate limiter implementation."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_times: List[float] = []
        self.token_usage: List[Tuple[float, int]] = []
        
    def check_limits(self, estimated_tokens: int = 1000) -> Tuple[bool, Optional[float]]:
        """
        Check if request can proceed within rate limits.
        
        Returns:
            Tuple of (can_proceed, wait_time_seconds)
        """
        current_time = time.time()
        
        # Clean old entries
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        self.token_usage = [(t, tokens) for t, tokens in self.token_usage if current_time - t < 60]
        
        # Check request limit
        if len(self.request_times) >= self.config.requests_per_minute:
            wait_time = 60 - (current_time - self.request_times[0])
            return False, wait_time if wait_time > 0 else 0
        
        # Check token limit
        total_tokens = sum(tokens for _, tokens in self.token_usage) + estimated_tokens
        if total_tokens > self.config.tokens_per_minute:
            wait_time = 60 - (current_time - self.token_usage[0][0])
            return False, wait_time if wait_time > 0 else 0
        
        return True, None
    
    def record_usage(self, tokens_used: int):
        """Record API usage for rate limiting."""
        current_time = time.time()
        self.request_times.append(current_time)
        self.token_usage.append((current_time, tokens_used))


class ExplanationCache:
    """LRU cache for storing generated explanations."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.cache_dir = config.cache_dir
        
        if config.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()
    
    def _generate_key(self, question: str, answer: str, temperature: float) -> str:
        """Generate cache key from inputs."""
        content = f"{question}|{answer}|{temperature:.2f}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, question: str, answer: str, temperature: float) -> Optional[str]:
        """Retrieve cached explanation if available and not expired."""
        if not self.config.enabled:
            return None
        
        key = self._generate_key(question, answer, temperature)
        
        if key in self.cache:
            entry = self.cache[key]
            # Check TTL
            created_at = datetime.fromisoformat(entry['created_at'])
            if datetime.now() - created_at < timedelta(hours=self.config.ttl_hours):
                # Move to end (LRU)
                self.cache.move_to_end(key)
                logger.debug(f"Cache hit for key: {key}")
                return entry['explanation']
            else:
                # Expired
                del self.cache[key]
        
        return None
    
    def set(self, question: str, answer: str, temperature: float, explanation: str):
        """Store explanation in cache."""
        if not self.config.enabled:
            return
        
        key = self._generate_key(question, answer, temperature)
        
        # Enforce max size
        if len(self.cache) >= self.config.max_size:
            self.cache.popitem(last=False)
        
        self.cache[key] = {
            'question': question,
            'answer': answer,
            'temperature': temperature,
            'explanation': explanation,
            'created_at': datetime.now().isoformat()
        }
        
        # Persist to disk
        self._save_cache()
    
    def _save_cache(self):
        """Persist cache to disk."""
        cache_file = self.cache_dir / "explanations.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(dict(self.cache), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _load_cache(self):
        """Load cache from disk."""
        cache_file = self.cache_dir / "explanations.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    self.cache = OrderedDict(data)
                logger.info(f"Loaded {len(self.cache)} cached explanations")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        if self.config.enabled:
            self._save_cache()


class ClaudeRLTTeacher:
    """
    Production-ready Claude-based teacher for RLT training.
    
    This teacher uses Claude API to generate high-quality explanations
    for question-answer pairs, following the RLT methodology where teachers
    are rewarded based on student understanding.
    
    Features:
        - Secure API key management via environment variables
        - Comprehensive error handling and retry logic
        - Built-in caching to avoid duplicate API calls
        - Rate limiting with token bucket algorithm
        - Batch processing capabilities
        - Cost tracking integration
        - Multiple fallback strategies
        
    Example:
        ```python
        teacher = ClaudeRLTTeacher(
            api_key=os.getenv("CLAUDE_API_KEY"),
            cost_tracker=CostTracker(budget_limit=50.0)
        )
        
        # Single explanation
        explanation = teacher.generate_explanation(
            question="What is 15 * 8?",
            answer="120"
        )
        
        # Batch processing
        results = teacher.batch_generate_explanations(
            questions=["What is 2+2?", "What is 3*4?"],
            answers=["4", "12"]
        )
        ```
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cost_tracker: Optional[CostTracker] = None,
        claude_config: Optional[ClaudeConfig] = None,
        rate_limit_config: Optional[RateLimitConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        fallback_strategy: FallbackStrategy = FallbackStrategy.TEMPLATE,
        fallback_callback: Optional[Callable] = None
    ):
        """
        Initialize the Claude RLT Teacher.
        
        Args:
            api_key: Claude API key. If None, reads from CLAUDE_API_KEY env var.
            cost_tracker: CostTracker instance for monitoring API costs.
            claude_config: Configuration for Claude API behavior.
            rate_limit_config: Configuration for rate limiting.
            cache_config: Configuration for caching system.
            fallback_strategy: Strategy to use when API is unavailable.
            fallback_callback: Custom callback for generating fallback explanations.
        
        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        # API key management
        self.api_key = api_key or os.getenv("CLAUDE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Claude API key not provided. Set CLAUDE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Initialize configurations
        self.claude_config = claude_config or ClaudeConfig()
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self.cache_config = cache_config or CacheConfig()
        
        # Initialize components
        self.client = Anthropic(api_key=self.api_key)
        self.cost_tracker = cost_tracker or CostTracker(budget_limit=100.0)
        self.rate_limiter = RateLimiter(self.rate_limit_config)
        self.cache = ExplanationCache(self.cache_config)
        
        # Fallback configuration
        self.fallback_strategy = fallback_strategy
        self.fallback_callback = fallback_callback
        
        # System prompt for RLT explanations
        self.system_prompt = self.claude_config.system_prompt or self._get_default_system_prompt()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'api_calls': 0,
            'fallback_uses': 0,
            'errors': 0
        }
        
        logger.info(f"ClaudeRLTTeacher initialized with model: {self.claude_config.model}")
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for RLT explanations."""
        return """You are an expert teacher following the Reinforcement Learning Teachers (RLT) methodology.
        
Your task is to generate clear, step-by-step explanations that help students understand the reasoning process.
Your explanations will be evaluated based on how well students can understand and apply them.

Guidelines:
1. Start by clearly understanding what the question asks
2. Break down the solution into logical, easy-to-follow steps
3. Use simple language and avoid unnecessary complexity
4. Show your reasoning at each step
5. Connect each step to the next logically
6. End with a clear conclusion that ties back to the original question

Remember: Your goal is maximum student understanding, not just providing the answer."""
    
    @contextmanager
    def _error_handling(self, operation: str):
        """Context manager for consistent error handling."""
        try:
            yield
        except APITimeoutError as e:
            logger.error(f"API timeout during {operation}: {e}")
            self.stats['errors'] += 1
            raise
        except APIConnectionError as e:
            logger.error(f"API connection error during {operation}: {e}")
            self.stats['errors'] += 1
            raise
        except APIError as e:
            logger.error(f"API error during {operation}: {e}")
            self.stats['errors'] += 1
            raise
        except Exception as e:
            logger.error(f"Unexpected error during {operation}: {e}")
            self.stats['errors'] += 1
            raise
    
    def _wait_for_rate_limit(self, estimated_tokens: int = 1000):
        """Wait if necessary to respect rate limits."""
        if not self.rate_limit_config.wait_on_limit:
            return
        
        can_proceed, wait_time = self.rate_limiter.check_limits(estimated_tokens)
        if not can_proceed and wait_time:
            logger.info(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
    
    def _call_claude_api(self, prompt: str, temperature: float, max_tokens: Optional[int] = None) -> Tuple[str, int, int]:
        """
        Make API call to Claude with retry logic.
        
        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        for attempt in range(self.claude_config.max_retries):
            try:
                with self._error_handling(f"API call (attempt {attempt + 1})"):
                    # Wait for rate limit if necessary
                    self._wait_for_rate_limit()
                    
                    # Make API call
                    response = self.client.messages.create(
                        model=self.claude_config.model,
                        messages=[{"role": "user", "content": prompt}],
                        system=self.system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens or self.claude_config.max_tokens,
                        timeout=self.claude_config.api_timeout
                    )
                    
                    # Extract response and token counts
                    text = response.content[0].text
                    input_tokens = response.usage.input_tokens
                    output_tokens = response.usage.output_tokens
                    
                    # Record usage
                    self.rate_limiter.record_usage(input_tokens + output_tokens)
                    self.cost_tracker.add_usage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        metadata={
                            'model': self.claude_config.model,
                            'temperature': temperature,
                            'operation': 'generate_explanation'
                        }
                    )
                    
                    return text, input_tokens, output_tokens
                    
            except (APITimeoutError, APIConnectionError) as e:
                if attempt < self.claude_config.max_retries - 1:
                    wait_time = self.claude_config.retry_delay * (2 ** attempt)
                    logger.warning(f"API error on attempt {attempt + 1}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
    
    def _generate_fallback_explanation(self, question: str, answer: str) -> str:
        """Generate fallback explanation when API is unavailable."""
        if self.fallback_callback:
            return self.fallback_callback(question, answer)
        
        # Default template-based fallback
        return f"""Let me explain how to solve this problem step by step.

**Question**: {question}

**Step 1: Understanding the Problem**
First, I need to understand what the question is asking. {question}

**Step 2: Working Through the Solution**
To solve this problem, I'll work through it systematically.

**Step 3: Final Answer**
After working through the problem, I arrive at the answer: {answer}

**Verification**
Let me verify this answer makes sense in the context of the original question.

The answer is {answer}."""
    
    def generate_explanation(
        self,
        question: str,
        answer: str,
        temperature: Optional[float] = None,
        use_cache: bool = True,
        subject: str = "default",
        difficulty: str = "medium",
    ) -> str:
        """
        Generate an explanation for a question-answer pair.

        Args:
            question: The question to explain.
            answer: The correct answer.
            temperature: Sampling temperature (0.0-1.0). If None, uses config default.
            use_cache: Whether to use caching.
            subject: Subject area for specialized prompts (math, science, default).
            difficulty: Difficulty level (easy, medium, hard).

        Returns:
            Generated explanation string.

        Raises:
            APIError: If API call fails after all retries.
            ValueError: If inputs are invalid.
        """
        # Validate inputs
        if not question or not answer:
            raise ValueError("Question and answer must be non-empty strings")

        temperature = temperature if temperature is not None else self.claude_config.temperature
        if not 0.0 <= temperature <= 1.0:
            raise ValueError(f"Temperature must be between 0.0 and 1.0, got {temperature}")

        self.stats['total_requests'] += 1

        # Check cache
        if use_cache:
            cached = self.cache.get(question, answer, temperature)
            if cached:
                self.stats['cache_hits'] += 1
                logger.debug(f"Returning cached explanation for: {question[:50]}...")
                return cached

        # Build structured prompt with subject + difficulty awareness
        system_prompt = SUBJECT_PROMPTS.get(subject, SUBJECT_PROMPTS["default"])
        diff_instruction = DIFFICULTY_INSTRUCTIONS.get(difficulty, DIFFICULTY_INSTRUCTIONS["medium"])

        prompt = f"""{system_prompt}
{diff_instruction}

Question: {question}
Correct Answer: {answer}

Explain step by step:
1. Understanding: What is the question asking?
2. Approach: What method or concept applies?
3. Steps: Show each calculation or reasoning step.
4. Answer: State the final answer clearly.
5. Check: Verify the answer is correct."""
        
        try:
            # Call API
            explanation, _, _ = self._call_claude_api(prompt, temperature)
            self.stats['api_calls'] += 1
            
            # Cache the result
            if use_cache:
                self.cache.set(question, answer, temperature, explanation)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            
            # Handle fallback
            if self.fallback_strategy == FallbackStrategy.RAISE:
                raise
            elif self.fallback_strategy == FallbackStrategy.CACHE_ONLY:
                # Try any temperature in cache
                for temp in [0.3, 0.5, 0.7, 0.9]:
                    cached = self.cache.get(question, answer, temp)
                    if cached:
                        self.stats['fallback_uses'] += 1
                        return cached
                raise ValueError("No cached explanation available")
            elif self.fallback_strategy == FallbackStrategy.TEMPLATE:
                self.stats['fallback_uses'] += 1
                return self._generate_fallback_explanation(question, answer)
            else:
                raise
    
    def batch_generate_explanations(
        self,
        questions: List[str],
        answers: List[str],
        temperatures: Optional[List[float]] = None,
        max_workers: int = 5,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for multiple question-answer pairs in parallel.
        
        Args:
            questions: List of questions.
            answers: List of corresponding answers.
            temperatures: Optional list of temperatures. If None, uses random sampling.
            max_workers: Maximum number of parallel workers.
            progress_callback: Optional callback for progress updates (completed, total).
        
        Returns:
            List of dictionaries containing explanations and metadata.
        
        Raises:
            ValueError: If input lists have different lengths.
        """
        # Validate inputs
        if len(questions) != len(answers):
            raise ValueError("Questions and answers lists must have the same length")
        
        n_items = len(questions)
        if temperatures and len(temperatures) != n_items:
            raise ValueError("Temperatures list must match questions/answers length")
        
        # Generate temperatures if not provided
        if not temperatures:
            temperatures = np.random.uniform(0.1, 0.9, n_items).tolist()
        
        results = []
        completed = 0
        
        # Process in parallel with thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {}
            for i, (q, a, t) in enumerate(zip(questions, answers, temperatures)):
                future = executor.submit(self._generate_with_metadata, q, a, t)
                future_to_idx[future] = i
            
            # Process completed tasks
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, n_items)
                        
                except Exception as e:
                    logger.error(f"Failed to generate explanation for item {idx}: {e}")
                    # Add error result
                    results.append((idx, {
                        'question': questions[idx],
                        'answer': answers[idx],
                        'temperature': temperatures[idx],
                        'explanation': None,
                        'error': str(e),
                        'success': False
                    }))
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, n_items)
        
        # Sort results by original index
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
    
    def _generate_with_metadata(self, question: str, answer: str, temperature: float) -> Dict[str, Any]:
        """Generate explanation with metadata."""
        # Check cache before generating to detect cache hits
        cached_value = self.cache.get(question, answer, temperature)
        was_cached = cached_value is not None

        explanation = self.generate_explanation(question, answer, temperature)
        return {
            'question': question,
            'answer': answer,
            'temperature': temperature,
            'explanation': explanation,
            'success': True,
            'cached': was_cached,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        cache_hit_rate = (
            self.stats['cache_hits'] / self.stats['total_requests'] 
            if self.stats['total_requests'] > 0 else 0.0
        )
        
        return {
            **self.stats,
            'cache_hit_rate': f"{cache_hit_rate:.2%}",
            'cost_summary': self.cost_tracker.get_summary(),
            'cache_size': len(self.cache.cache) if self.cache_config.enabled else 0
        }
    
    def clear_cache(self):
        """Clear the explanation cache."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def save_stats(self, filepath: str):
        """Save usage statistics to file."""
        stats = self.get_stats()
        stats['timestamp'] = datetime.now().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Stats saved to {filepath}")


# Utility functions for testing and demonstration
def create_teacher_from_env() -> ClaudeRLTTeacher:
    """
    Create a ClaudeRLTTeacher instance using environment variables.
    
    Environment variables:
        - CLAUDE_API_KEY: API key (required)
        - CLAUDE_MODEL: Model to use (default: claude-sonnet-4-6)
        - CLAUDE_BUDGET_LIMIT: Budget limit in USD (default: 10.0)
        - CLAUDE_CACHE_ENABLED: Enable caching (default: true)
    
    Returns:
        Configured ClaudeRLTTeacher instance.
    """
    # Read environment variables
    api_key = os.getenv("CLAUDE_API_KEY")
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    budget_limit = float(os.getenv("CLAUDE_BUDGET_LIMIT", "10.0"))
    cache_enabled = os.getenv("CLAUDE_CACHE_ENABLED", "true").lower() == "true"
    
    # Create configurations
    claude_config = ClaudeConfig(model=model)
    cache_config = CacheConfig(enabled=cache_enabled)
    cost_tracker = CostTracker(budget_limit=budget_limit)
    
    # Create and return teacher
    return ClaudeRLTTeacher(
        api_key=api_key,
        cost_tracker=cost_tracker,
        claude_config=claude_config,
        cache_config=cache_config
    )


if __name__ == "__main__":
    # Example usage and testing
    print("Claude RLT Teacher - Example Usage")
    print("=" * 50)
    
    # Create teacher instance
    try:
        teacher = create_teacher_from_env()
        print("✓ Teacher initialized successfully")
        
        # Test single explanation
        print("\nGenerating single explanation...")
        explanation = teacher.generate_explanation(
            question="A train travels at 60 mph for 3 hours. How far does it go?",
            answer="180 miles",
            temperature=0.7
        )
        print(f"Explanation:\n{explanation}\n")
        
        # Test batch processing
        print("Testing batch processing...")
        questions = [
            "What is 15 × 8?",
            "If a rectangle has length 10 and width 5, what is its area?"
        ]
        answers = ["120", "50 square units"]
        
        results = teacher.batch_generate_explanations(
            questions=questions,
            answers=answers,
            max_workers=2
        )
        
        print(f"\nGenerated {len(results)} explanations")
        
        # Print statistics
        stats = teacher.get_stats()
        print(f"\nUsage Statistics:")
        print(f"- Total requests: {stats['total_requests']}")
        print(f"- Cache hits: {stats['cache_hits']} ({stats['cache_hit_rate']})")
        print(f"- API calls: {stats['api_calls']}")
        print(f"- Cost summary: {stats['cost_summary']}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nMake sure CLAUDE_API_KEY is set in your environment variables.")