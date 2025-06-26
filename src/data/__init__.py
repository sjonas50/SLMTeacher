"""
RLT Data Pipeline Module

This module provides comprehensive data loading, processing, and caching functionality
for the Reinforcement Learning from Teachers (RLT) training pipeline.
"""

# Import main classes
from .data_loader import (
    DataLoader,
    BatchGenerator,
    load_dataset,
    create_batch_iterator,
    DATASET_CONFIGS
)

from .data_processor import (
    RLTDataPoint,
    TeacherInput,
    StudentInput,
    DataProcessor,
    create_balanced_batches,
    merge_datasets
)

from .cache_manager import (
    CacheManager,
    get_default_cache,
    cache_dataset,
    get_cached_dataset,
    cache_explanation,
    get_cached_explanation
)

# Version
__version__ = "1.0.0"

# Module metadata
__all__ = [
    # Data structures
    'RLTDataPoint',
    'TeacherInput', 
    'StudentInput',
    
    # Main classes
    'DataLoader',
    'DataProcessor',
    'CacheManager',
    'BatchGenerator',
    
    # Convenience functions
    'load_dataset',
    'create_batch_iterator',
    'create_balanced_batches',
    'merge_datasets',
    'get_default_cache',
    'cache_dataset',
    'get_cached_dataset',
    'cache_explanation',
    'get_cached_explanation',
    
    # Constants
    'DATASET_CONFIGS'
]


def quick_start(dataset_name: str = 'gsm8k', max_samples: int = 100):
    """
    Quick start function to test the data pipeline.
    
    Args:
        dataset_name: Name of dataset to load
        max_samples: Maximum number of samples to load
        
    Returns:
        Tuple of (data_points, processor, cache_stats)
    """
    # Initialize components
    loader = DataLoader()
    processor = DataProcessor()
    cache = get_default_cache()
    
    # Load dataset
    print(f"Loading {dataset_name} dataset...")
    data = loader.load_dataset(dataset_name, max_samples=max_samples)
    
    # Process first sample
    if data:
        sample = data[0]
        teacher_input = processor.format_teacher_input(sample)
        
        print(f"\nSample data point:")
        print(f"Question: {sample.question[:100]}...")
        print(f"Solution: {sample.solution}")
        print(f"Subject: {sample.subject}")
        print(f"Difficulty: {sample.difficulty}")
        
        print(f"\nFormatted teacher input:")
        print(teacher_input.prompt[:200] + "...")
    
    # Get cache stats
    cache_stats = cache.get_cache_stats()
    print(f"\nCache statistics:")
    print(f"Total entries: {cache_stats['total_entries']}")
    print(f"Cache size: {cache_stats['total_size_mb']:.2f} MB")
    print(f"Hit rate: {cache_stats['hit_rate']:.2%}")
    
    return data, processor, cache_stats