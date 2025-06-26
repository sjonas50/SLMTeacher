# RLT Data Pipeline

A production-ready data pipeline for Reinforcement Learning from Teachers (RLT) training, supporting multiple datasets with automatic downloading, caching, and efficient batch generation.

## Features

- **Multi-Dataset Support**: GSM8K, MATH, ARC-C, and custom JSON formats
- **Automatic Downloading**: Datasets are downloaded automatically when first accessed
- **Intelligent Caching**: Reduces redundant downloads and API calls
- **Batch Generation**: Efficient batching with curriculum learning support
- **Data Augmentation**: Create variations of training examples
- **Format Flexibility**: Export to JSON, JSONL, or CSV formats
- **Production Ready**: Comprehensive error handling and logging

## Quick Start

```python
from src.data import quick_start

# Quick test of the pipeline
data, processor, cache_stats = quick_start('gsm8k', max_samples=100)
```

## Basic Usage

### Loading Datasets

```python
from src.data import DataLoader

# Initialize loader
loader = DataLoader()

# Load GSM8K dataset
gsm8k_data = loader.load_dataset('gsm8k', split='train', max_samples=1000)

# Load MATH dataset
math_data = loader.load_dataset('math', split='train', subset='algebra')

# Load ARC-Challenge dataset
arc_data = loader.load_dataset('arc-c', split='train')

# Load custom JSON dataset
custom_data = loader.load_dataset('path/to/custom_dataset.json')
```

### Data Processing

```python
from src.data import DataProcessor, RLTDataPoint

# Initialize processor
processor = DataProcessor()

# Create a data point
data_point = RLTDataPoint(
    question="What is 25% of 80?",
    solution="20",
    subject="math",
    difficulty="easy"
)

# Format for teacher model
teacher_input = processor.format_teacher_input(data_point)
print(teacher_input.prompt)

# Format for student model
student_input = processor.format_student_input(
    question=data_point.question,
    explanation="To find 25% of 80, multiply 80 by 0.25: 80 × 0.25 = 20"
)
print(student_input.prompt)
```

### Batch Generation

```python
from src.data import BatchGenerator

# Initialize batch generator
generator = BatchGenerator(loader, batch_size=32)

# Generate batches with curriculum learning
batch_iterator = generator.generate_batches(
    dataset_names=['gsm8k', 'math'],
    num_epochs=3,
    curriculum_learning=True
)

for batch, metadata in batch_iterator:
    # Process batch
    print(f"Batch size: {len(batch)}, Epoch: {metadata['epoch']}")
```

### Caching

```python
from src.data import CacheManager

# Initialize cache
cache = CacheManager()

# Cache an explanation
cache_key = cache.cache_explanation(
    question="What is the derivative of x^2?",
    solution="2x",
    explanation="Using the power rule, d/dx(x^n) = nx^(n-1), so d/dx(x^2) = 2x",
    model="claude-sonnet-4",
    temperature=0.7
)

# Retrieve from cache
cached = cache.get_explanation(
    question="What is the derivative of x^2?",
    solution="2x",
    model="claude-sonnet-4",
    temperature=0.7
)

# Get cache statistics
stats = cache.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
```

## Advanced Features

### Data Augmentation

```python
# Enable augmentation
processor = DataProcessor(enable_augmentation=True)

# Augment a data point
augmented = processor.augment_data_point(
    data_point,
    strategies=['paraphrase', 'difficulty']
)
```

### Dataset Merging

```python
# Merge multiple datasets with proportions
merged_data = loader.merge_datasets(
    dataset_names=['gsm8k', 'math', 'arc-c'],
    proportions=[0.5, 0.3, 0.2],  # 50% GSM8K, 30% MATH, 20% ARC
    total_samples=10000
)
```

### Custom Dataset Format

Custom JSON datasets should follow this format:

```json
[
  {
    "question": "What is the capital of France?",
    "solution": "Paris",
    "subject": "geography",
    "difficulty": "easy"
  },
  {
    "question": "Solve for x: 2x + 5 = 13",
    "solution": "x = 4",
    "subject": "math",
    "difficulty": "medium"
  }
]
```

### Export Datasets

```python
# Export to different formats
loader.export_dataset(data, 'output.json', format='json')
loader.export_dataset(data, 'output.jsonl', format='jsonl')
loader.export_dataset(data, 'output.csv', format='csv')
```

## Configuration

### Cache Configuration

```python
cache = CacheManager(
    cache_dir="~/.cache/rlt_data",      # Cache directory
    max_cache_size_gb=10.0,             # Maximum cache size
    enable_compression=True,             # Compress cached data
    cache_ttl_days=30                   # Cache time-to-live
)
```

### Data Loader Configuration

```python
loader = DataLoader(
    data_dir="~/.rlt_data",             # Dataset directory
    auto_download=True,                 # Auto-download datasets
    verify_checksums=True               # Verify dataset integrity
)
```

## Supported Datasets

| Dataset | Description | Size | Subjects |
|---------|-------------|------|----------|
| GSM8K | Grade School Math | 8.5K | Math |
| MATH | Competition Mathematics | 12.5K | Math (multiple topics) |
| ARC-C | AI2 Reasoning Challenge | 7.7K | Science |
| Custom | Your own JSON datasets | Any | Any |

## Custom Prompt Templates

```python
# Custom teacher prompt
teacher_template = """
As an expert educator, explain how to solve this problem:
Question: {question}
Correct Answer: {solution}
Provide a clear, step-by-step explanation:
"""

# Custom student prompt
student_template = """
Problem: {question}
Teacher's Explanation: {explanation}
Based on the explanation above, the answer is:
"""

# Use custom templates
processor = DataProcessor(
    teacher_prompt_template=teacher_template,
    student_prompt_template=student_template
)
```

## Error Handling

The pipeline includes comprehensive error handling:

```python
try:
    data = loader.load_dataset('unknown_dataset')
except ValueError as e:
    print(f"Dataset error: {e}")

# Validate data points
is_valid, issues = processor.validate_data_point(data_point)
if not is_valid:
    print(f"Validation issues: {issues}")
```

## Performance Tips

1. **Use Caching**: Enable caching to avoid redundant downloads
2. **Batch Size**: Adjust batch size based on available memory
3. **Curriculum Learning**: Start with easier examples
4. **Dynamic Batching**: Automatically adjust batch size based on sequence length
5. **Compression**: Enable compression for large datasets

## Examples

See `example_usage.py` for comprehensive examples covering all features.

## Dependencies

- Python 3.8+
- requests (for downloading)
- tqdm (for progress bars)
- pandas (optional, for CSV export)
- datasets (optional, for HuggingFace datasets)

## License

This module is part of the RLT training system. See the main project license for details.