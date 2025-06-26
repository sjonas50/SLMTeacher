# Claude RLT Teacher

Production-ready implementation of a Claude-based teacher for the Reinforcement Learning Teachers (RLT) training pipeline. This teacher generates high-quality explanations that are used to train smaller student models following Sakana AI's RLT methodology.

## Features

- 🔐 **Secure API Key Management**: Uses environment variables for API keys
- 🚨 **Comprehensive Error Handling**: Handles all API exceptions gracefully
- 💾 **Built-in Caching**: LRU cache to avoid duplicate API calls
- 🚦 **Rate Limiting**: Token bucket algorithm for API rate limits
- 🚀 **Batch Processing**: Parallel processing for multiple explanations
- 💰 **Cost Tracking**: Integrated with CostTracker for budget management
- 🔄 **Fallback Mechanisms**: Multiple strategies when API is unavailable
- 📊 **Production Ready**: Type hints, docstrings, and comprehensive logging

## Installation

```bash
# Set your Claude API key
export CLAUDE_API_KEY="your-api-key-here"

# Install required dependencies
pip install anthropic numpy
```

## Quick Start

```python
from src.teachers import ClaudeRLTTeacher, create_teacher_from_env

# Option 1: Create from environment variables
teacher = create_teacher_from_env()

# Option 2: Manual configuration
from src.utils.cost_tracker import CostTracker

teacher = ClaudeRLTTeacher(
    api_key="your-api-key",
    cost_tracker=CostTracker(budget_limit=50.0)
)

# Generate an explanation
explanation = teacher.generate_explanation(
    question="A train travels at 60 mph for 3 hours. How far does it go?",
    answer="180 miles",
    temperature=0.7
)

print(explanation)
```

## Configuration Options

### Claude Configuration

```python
from src.teachers import ClaudeConfig

config = ClaudeConfig(
    model="claude-3-5-sonnet-20241022",  # Model to use
    temperature=0.7,                      # Sampling temperature
    max_tokens=1024,                      # Max tokens per response
    api_timeout=30,                       # API timeout in seconds
    max_retries=3,                        # Number of retries on failure
    retry_delay=1.0,                      # Base delay between retries
    system_prompt=None                    # Custom system prompt
)
```

### Rate Limiting Configuration

```python
from src.teachers import RateLimitConfig

rate_config = RateLimitConfig(
    requests_per_minute=50,      # Max requests per minute
    tokens_per_minute=100000,    # Max tokens per minute
    burst_size=10,               # Burst capacity
    wait_on_limit=True           # Wait when limit reached
)
```

### Cache Configuration

```python
from src.teachers import CacheConfig
from pathlib import Path

cache_config = CacheConfig(
    enabled=True,                        # Enable caching
    max_size=10000,                      # Max cached items
    ttl_hours=24,                        # Cache TTL in hours
    cache_dir=Path(".claude_cache")      # Cache directory
)
```

## Advanced Usage

### Batch Processing

Process multiple question-answer pairs efficiently:

```python
questions = [
    "What is 25% of 80?",
    "If x + 5 = 12, what is x?",
    "What is the area of a circle with radius 3?"
]
answers = ["20", "7", "28.26"]

# Process in parallel
results = teacher.batch_generate_explanations(
    questions=questions,
    answers=answers,
    max_workers=5,
    progress_callback=lambda done, total: print(f"{done}/{total}")
)

for result in results:
    if result['success']:
        print(f"Q: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"Explanation: {result['explanation'][:100]}...")
```

### Fallback Strategies

Handle API unavailability gracefully:

```python
from src.teachers import FallbackStrategy

# Option 1: Use template-based fallback
teacher = ClaudeRLTTeacher(
    api_key=api_key,
    fallback_strategy=FallbackStrategy.TEMPLATE
)

# Option 2: Use cached results only
teacher = ClaudeRLTTeacher(
    api_key=api_key,
    fallback_strategy=FallbackStrategy.CACHE_ONLY
)

# Option 3: Custom fallback function
def my_fallback(question: str, answer: str) -> str:
    return f"[Offline] The answer to '{question}' is {answer}."

teacher = ClaudeRLTTeacher(
    api_key=api_key,
    fallback_strategy=FallbackStrategy.TEMPLATE,
    fallback_callback=my_fallback
)
```

### RLT Training Integration

Generate diverse explanations for RLT training:

```python
# Generate explanations with temperature diversity
training_data = []

for temp in [0.3, 0.5, 0.7, 0.9]:
    explanation = teacher.generate_explanation(
        question="What is the perimeter of a square with side 5?",
        answer="20",
        temperature=temp
    )
    
    # In real RLT, student model calculates understanding score
    # Here we simulate it
    understanding_score = evaluate_with_student_model(explanation)
    
    training_data.append({
        'explanation': explanation,
        'temperature': temp,
        'understanding_score': understanding_score
    })

# Filter high-quality explanations (top 25%)
threshold = np.percentile(
    [d['understanding_score'] for d in training_data], 
    75
)
high_quality = [d for d in training_data if d['understanding_score'] >= threshold]
```

## Monitoring and Statistics

Track usage and performance:

```python
# Get comprehensive statistics
stats = teacher.get_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Cache hit rate: {stats['cache_hit_rate']}")
print(f"API calls: {stats['api_calls']}")
print(f"Fallback uses: {stats['fallback_uses']}")
print(f"Cost summary: {stats['cost_summary']}")

# Save statistics to file
teacher.save_stats("usage_stats.json")

# Clear cache if needed
teacher.clear_cache()
```

## Environment Variables

The teacher supports these environment variables:

- `CLAUDE_API_KEY`: Your Claude API key (required)
- `CLAUDE_MODEL`: Model to use (default: claude-3-5-sonnet-20241022)
- `CLAUDE_BUDGET_LIMIT`: Budget limit in USD (default: 10.0)
- `CLAUDE_CACHE_ENABLED`: Enable caching (default: true)

## Error Handling

The teacher handles various error scenarios:

```python
try:
    explanation = teacher.generate_explanation(question, answer)
except ValueError as e:
    # Invalid inputs (empty question/answer, invalid temperature)
    print(f"Invalid input: {e}")
except APIError as e:
    # Claude API errors
    print(f"API error: {e}")
except Exception as e:
    # Other errors
    print(f"Unexpected error: {e}")
```

## Testing

Run the test suite:

```bash
# Run unit tests
python -m pytest src/teachers/test_claude_teacher.py -v

# Run example demonstrations
python src/teachers/claude_teacher_example.py
```

## Best Practices

1. **API Key Security**: Always use environment variables for API keys
2. **Budget Management**: Set appropriate budget limits and monitor costs
3. **Caching**: Enable caching to reduce API calls and costs
4. **Rate Limiting**: Configure rate limits based on your API plan
5. **Error Handling**: Always handle potential API failures gracefully
6. **Batch Processing**: Use batch processing for multiple explanations
7. **Temperature Diversity**: Vary temperature for diverse explanations in RLT

## Integration with RLT Pipeline

The ClaudeRLTTeacher is designed to integrate seamlessly with the RLT training pipeline:

1. **Generate Explanations**: Teacher generates explanations for Q&A pairs
2. **Student Evaluation**: Student model evaluates understanding
3. **Dense Rewards**: Teacher receives rewards based on student understanding
4. **Quality Filtering**: Select high-quality explanations for training
5. **Fine-tuning**: Use filtered explanations to improve the teacher

## License

This implementation follows the RLT methodology from Sakana AI's research. Please refer to their paper for commercial usage guidelines.