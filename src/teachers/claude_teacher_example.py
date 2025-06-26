"""
Example usage of ClaudeRLTTeacher in the RLT training pipeline.

This script demonstrates how to use the Claude teacher for generating
high-quality explanations that can be used to train student models.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.teachers import ClaudeRLTTeacher, ClaudeConfig, CacheConfig, FallbackStrategy
from src.utils.cost_tracker import CostTracker


def demonstrate_basic_usage():
    """Demonstrate basic usage of ClaudeRLTTeacher."""
    print("=== Basic Usage Demo ===\n")
    
    # Initialize with minimal configuration
    teacher = ClaudeRLTTeacher(
        api_key=os.getenv("CLAUDE_API_KEY"),
        fallback_strategy=FallbackStrategy.TEMPLATE
    )
    
    # Generate a single explanation
    question = "A store sells 15 apples per day for 6 days. How many apples in total?"
    answer = "90 apples"
    
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print("\nGenerating explanation...")
    
    explanation = teacher.generate_explanation(question, answer, temperature=0.7)
    print(f"\nExplanation:\n{explanation}")
    
    # Show statistics
    stats = teacher.get_stats()
    print(f"\nStats: {json.dumps(stats, indent=2)}")


def demonstrate_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("\n\n=== Batch Processing Demo ===\n")
    
    # Initialize with custom configuration
    cost_tracker = CostTracker(budget_limit=25.0)
    claude_config = ClaudeConfig(
        temperature=0.6,
        max_tokens=800
    )
    
    teacher = ClaudeRLTTeacher(
        api_key=os.getenv("CLAUDE_API_KEY"),
        cost_tracker=cost_tracker,
        claude_config=claude_config
    )
    
    # Sample math problems for batch processing
    math_problems = [
        {"question": "What is 25% of 80?", "answer": "20"},
        {"question": "If x + 5 = 12, what is x?", "answer": "7"},
        {"question": "What is the area of a circle with radius 3? (use π = 3.14)", "answer": "28.26"},
        {"question": "Simplify: 2(x + 3) - x", "answer": "x + 6"},
        {"question": "What is the median of 3, 7, 9, 2, 5?", "answer": "5"}
    ]
    
    questions = [p["question"] for p in math_problems]
    answers = [p["answer"] for p in math_problems]
    
    # Progress callback
    def progress_update(completed, total):
        print(f"Progress: {completed}/{total} ({completed/total*100:.0f}%)")
    
    print(f"Processing {len(questions)} problems in batch...")
    results = teacher.batch_generate_explanations(
        questions=questions,
        answers=answers,
        max_workers=3,
        progress_callback=progress_update
    )
    
    # Display results
    print("\n--- Batch Results ---")
    for i, result in enumerate(results):
        print(f"\nProblem {i+1}:")
        print(f"Q: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Cached: {result.get('cached', False)}")
            print(f"Explanation preview: {result['explanation'][:100]}...")
    
    # Cost summary
    print(f"\nCost Summary: {json.dumps(cost_tracker.get_summary(), indent=2)}")


def demonstrate_rlt_training_integration():
    """Demonstrate integration with RLT training pipeline."""
    print("\n\n=== RLT Training Integration Demo ===\n")
    
    # Initialize teacher with caching for efficiency
    cache_config = CacheConfig(
        enabled=True,
        max_size=5000,
        ttl_hours=48
    )
    
    teacher = ClaudeRLTTeacher(
        api_key=os.getenv("CLAUDE_API_KEY"),
        cache_config=cache_config
    )
    
    # Simulate RLT training data generation
    training_problems = [
        {"question": "A rectangle has length 8m and width 5m. What is its perimeter?", "answer": "26m"},
        {"question": "If 3x = 15, what is x?", "answer": "5"},
        {"question": "What is 40% of 250?", "answer": "100"},
    ]
    
    # Generate explanations with temperature diversity (key RLT concept)
    print("Generating diverse explanations for RLT training...")
    training_data = []
    
    for problem in training_problems:
        # Generate multiple explanations with different temperatures
        for temp in [0.3, 0.7, 0.9]:
            explanation = teacher.generate_explanation(
                problem["question"], 
                problem["answer"],
                temperature=temp
            )
            
            # Simulate student understanding score (would come from actual student model)
            # In real RLT, this would be calculated by the student model
            understanding_score = np.random.uniform(0.6, 0.95)
            
            training_data.append({
                "question": problem["question"],
                "answer": problem["answer"],
                "explanation": explanation,
                "temperature": temp,
                "understanding_score": understanding_score
            })
            
            print(f"Generated explanation (temp={temp}), understanding score: {understanding_score:.3f}")
    
    # Filter high-quality explanations (RLT approach)
    threshold = np.percentile([d["understanding_score"] for d in training_data], 75)
    high_quality = [d for d in training_data if d["understanding_score"] >= threshold]
    
    print(f"\nFiltered to {len(high_quality)}/{len(training_data)} high-quality explanations")
    print(f"Quality threshold: {threshold:.3f}")
    
    # Save training data
    output_file = "rlt_training_data.json"
    with open(output_file, 'w') as f:
        json.dump(high_quality, f, indent=2)
    print(f"\nSaved high-quality training data to {output_file}")


def demonstrate_fallback_handling():
    """Demonstrate fallback mechanisms for API unavailability."""
    print("\n\n=== Fallback Handling Demo ===\n")
    
    # Custom fallback function
    def custom_fallback(question: str, answer: str) -> str:
        return f"""[Fallback Explanation]
        
Question: {question}

To solve this problem, let's think step by step:
1. First, understand what we're looking for
2. Apply the relevant mathematical concepts
3. Work through the calculation
4. Verify our answer: {answer}

The answer is {answer}."""
    
    # Initialize with custom fallback
    teacher = ClaudeRLTTeacher(
        api_key="invalid_key_for_demo",  # Invalid key to trigger fallback
        fallback_strategy=FallbackStrategy.TEMPLATE,
        fallback_callback=custom_fallback
    )
    
    # This will use fallback
    explanation = teacher.generate_explanation(
        "What is 7 × 8?",
        "56"
    )
    
    print("Fallback explanation generated:")
    print(explanation)
    
    stats = teacher.get_stats()
    print(f"\nFallback uses: {stats['fallback_uses']}")


def main():
    """Run all demonstrations."""
    # Check for API key
    if not os.getenv("CLAUDE_API_KEY"):
        print("Warning: CLAUDE_API_KEY not set. Some demos will use fallback mode.")
        print("Set your API key with: export CLAUDE_API_KEY='your-key-here'\n")
    
    try:
        # Run demonstrations
        demonstrate_basic_usage()
        demonstrate_batch_processing()
        demonstrate_rlt_training_integration()
        demonstrate_fallback_handling()
        
        print("\n\n=== All Demonstrations Complete ===")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()