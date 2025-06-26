#!/usr/bin/env python3
"""
Example usage of the RLT Data Pipeline

This script demonstrates how to use the data pipeline for loading, processing,
and caching datasets for RLT training.
"""

import json
from pathlib import Path
from src.data import (
    DataLoader, 
    DataProcessor, 
    CacheManager,
    BatchGenerator,
    RLTDataPoint,
    quick_start
)


def example_basic_usage():
    """Basic usage example."""
    print("=" * 50)
    print("Basic Usage Example")
    print("=" * 50)
    
    # Initialize data loader
    loader = DataLoader()
    
    # Load GSM8K dataset
    print("\nLoading GSM8K dataset...")
    gsm8k_data = loader.load_dataset('gsm8k', split='train', max_samples=10)
    
    print(f"Loaded {len(gsm8k_data)} samples")
    
    # Display first sample
    if gsm8k_data:
        sample = gsm8k_data[0]
        print(f"\nFirst sample:")
        print(f"Question: {sample.question}")
        print(f"Solution: {sample.solution}")
        print(f"Subject: {sample.subject}")
        print(f"Difficulty: {sample.difficulty}")


def example_data_processing():
    """Data processing example."""
    print("\n" + "=" * 50)
    print("Data Processing Example")
    print("=" * 50)
    
    # Initialize processor
    processor = DataProcessor()
    
    # Create sample data point
    sample = RLTDataPoint(
        question="What is 25% of 120?",
        solution="30",
        subject="math",
        difficulty="easy"
    )
    
    # Format for teacher model
    teacher_input = processor.format_teacher_input(sample)
    print("\nTeacher Input:")
    print(teacher_input.prompt)
    
    # Format for student model
    explanation = "To find 25% of 120, we multiply 120 by 0.25: 120 × 0.25 = 30"
    student_input = processor.format_student_input(sample.question, explanation)
    print("\nStudent Input:")
    print(student_input.prompt)


def example_batch_generation():
    """Batch generation example."""
    print("\n" + "=" * 50)
    print("Batch Generation Example")
    print("=" * 50)
    
    # Initialize components
    loader = DataLoader()
    generator = BatchGenerator(loader, batch_size=5)
    
    # Generate batches
    print("\nGenerating batches from GSM8K...")
    batch_iterator = generator.generate_batches(
        'gsm8k',
        num_epochs=1,
        max_samples=20
    )
    
    # Process first few batches
    for i, (batch, metadata) in enumerate(batch_iterator):
        if i >= 2:  # Only show first 2 batches
            break
        
        print(f"\nBatch {i + 1}:")
        print(f"  Size: {len(batch)}")
        print(f"  Metadata: {metadata}")
        print(f"  First question: {batch[0].question[:50]}...")


def example_caching():
    """Caching example."""
    print("\n" + "=" * 50)
    print("Caching Example")
    print("=" * 50)
    
    # Initialize cache manager
    cache = CacheManager()
    
    # Cache an explanation
    question = "What is the derivative of x^2?"
    solution = "2x"
    explanation = "The derivative of x^2 is 2x using the power rule."
    
    print("\nCaching explanation...")
    cache_key = cache.cache_explanation(
        question=question,
        solution=solution,
        explanation=explanation,
        model="claude-sonnet-4",
        temperature=0.7,
        additional_data={'tokens': 50}
    )
    
    print(f"Cached with key: {cache_key[:8]}...")
    
    # Retrieve from cache
    print("\nRetrieving from cache...")
    cached_data = cache.get_explanation(
        question=question,
        solution=solution,
        model="claude-sonnet-4",
        temperature=0.7
    )
    
    if cached_data:
        print(f"Retrieved explanation: {cached_data['explanation']}")
        print(f"Cache hit!")
    
    # Show cache statistics
    stats = cache.get_cache_stats()
    print(f"\nCache Statistics:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  API calls saved: {stats['api_calls_saved']}")


def example_custom_dataset():
    """Custom dataset loading example."""
    print("\n" + "=" * 50)
    print("Custom Dataset Example")
    print("=" * 50)
    
    # Create a sample custom dataset
    custom_data = [
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "subject": "geography",
            "difficulty": "easy"
        },
        {
            "question": "Solve: 2x + 5 = 13",
            "answer": "x = 4",
            "subject": "math",
            "difficulty": "medium"
        }
    ]
    
    # Save to file
    custom_path = Path("sample_custom_dataset.json")
    with open(custom_path, 'w') as f:
        json.dump(custom_data, f, indent=2)
    
    print(f"\nCreated custom dataset at: {custom_path}")
    
    # Load custom dataset
    loader = DataLoader()
    custom_dataset = loader.load_dataset(str(custom_path))
    
    print(f"Loaded {len(custom_dataset)} samples from custom dataset")
    
    for i, sample in enumerate(custom_dataset):
        print(f"\nSample {i + 1}:")
        print(f"  Question: {sample.question}")
        print(f"  Solution: {sample.solution}")
        print(f"  Subject: {sample.subject}")
    
    # Clean up
    custom_path.unlink()


def example_data_augmentation():
    """Data augmentation example."""
    print("\n" + "=" * 50)
    print("Data Augmentation Example")
    print("=" * 50)
    
    # Initialize processor with augmentation
    processor = DataProcessor(enable_augmentation=True)
    
    # Create sample data point
    sample = RLTDataPoint(
        question="Calculate the area of a rectangle with length 10 and width 5",
        solution="50",
        subject="math",
        difficulty="easy"
    )
    
    print(f"\nOriginal question: {sample.question}")
    
    # Apply augmentation
    augmented = processor.augment_data_point(sample)
    
    print(f"\nGenerated {len(augmented)} variations:")
    for i, aug_sample in enumerate(augmented):
        print(f"\nVariation {i + 1}:")
        print(f"  Question: {aug_sample.question}")
        print(f"  Solution: {aug_sample.solution}")


def example_dataset_merging():
    """Dataset merging example."""
    print("\n" + "=" * 50)
    print("Dataset Merging Example")
    print("=" * 50)
    
    # Initialize loader
    loader = DataLoader()
    
    # Create sample datasets
    dataset1 = [
        RLTDataPoint("Q1", "A1", "math", "easy"),
        RLTDataPoint("Q2", "A2", "math", "medium")
    ]
    
    dataset2 = [
        RLTDataPoint("Q3", "A3", "physics", "hard"),
        RLTDataPoint("Q4", "A4", "physics", "medium")
    ]
    
    # Use the processor's merge function
    from src.data import merge_datasets
    merged = merge_datasets([dataset1, dataset2])
    
    print(f"\nMerged {len(merged)} samples from 2 datasets")
    
    # Show distribution
    subjects = {}
    for sample in merged:
        subjects[sample.subject] = subjects.get(sample.subject, 0) + 1
    
    print("\nSubject distribution:")
    for subject, count in subjects.items():
        print(f"  {subject}: {count}")


def example_export_dataset():
    """Dataset export example."""
    print("\n" + "=" * 50)
    print("Dataset Export Example")
    print("=" * 50)
    
    # Create sample dataset
    data = [
        RLTDataPoint(
            question="What is 2 + 2?",
            solution="4",
            subject="math",
            difficulty="easy"
        ),
        RLTDataPoint(
            question="What is the speed of light?",
            solution="299,792,458 meters per second",
            subject="physics",
            difficulty="medium"
        )
    ]
    
    # Initialize loader
    loader = DataLoader()
    
    # Export in different formats
    formats = ['json', 'jsonl', 'csv']
    
    for fmt in formats:
        output_path = f"sample_export.{fmt}"
        loader.export_dataset(data, output_path, format=fmt)
        print(f"\nExported to {output_path}")
        
        # Clean up
        Path(output_path).unlink()


def main():
    """Run all examples."""
    examples = [
        example_basic_usage,
        example_data_processing,
        example_batch_generation,
        example_caching,
        example_custom_dataset,
        example_data_augmentation,
        example_dataset_merging,
        example_export_dataset
    ]
    
    print("RLT Data Pipeline Examples")
    print("=" * 50)
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    
    # Run quick start
    print("\n" + "=" * 50)
    print("Running Quick Start")
    print("=" * 50)
    
    try:
        data, processor, cache_stats = quick_start('gsm8k', max_samples=5)
        print(f"\nQuick start completed successfully!")
    except Exception as e:
        print(f"Quick start error: {e}")


if __name__ == "__main__":
    main()