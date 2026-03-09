"""
Integration script for using ClaudeRLTTeacher with the RLT notebook.

This script shows how to replace the notebook's basic teacher model
with the production-ready Claude teacher for better explanations.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.teachers import ClaudeRLTTeacher, create_teacher_from_env
from src.utils.cost_tracker import CostTracker


@dataclass
class RLTConfig:
    """Configuration matching the notebook's RLTConfig."""
    teacher_model_name: str = "claude-sonnet-4-6"
    student_model_name: str = "microsoft/DialoGPT-small"
    max_length: int = 512
    batch_size: int = 2
    learning_rate: float = 2e-4
    num_epochs: int = 1
    warmup_steps: int = 50
    temperature_range: Tuple[float, float] = (0.1, 0.9)
    reward_threshold_percentile: int = 75


class NotebookClaudeTeacher:
    """
    Wrapper to make ClaudeRLTTeacher compatible with the notebook's RLTTeacher interface.
    """
    
    def __init__(self, config: RLTConfig, budget_limit: float = 10.0):
        """Initialize the Claude teacher for notebook integration."""
        self.config = config
        
        # Initialize cost tracker
        self.cost_tracker = CostTracker(budget_limit=budget_limit)
        
        # Initialize Claude teacher
        self.claude_teacher = ClaudeRLTTeacher(
            api_key=os.getenv("CLAUDE_API_KEY"),
            cost_tracker=self.cost_tracker
        )
        
        print(f"✅ Claude teacher initialized with budget limit: ${budget_limit}")
    
    def generate_explanation(self, question: str, answer: str, temperature: float = 0.7) -> str:
        """
        Generate explanation matching the notebook's interface.
        
        This method matches the signature of the notebook's RLTTeacher.generate_explanation
        """
        return self.claude_teacher.generate_explanation(
            question=question,
            answer=answer,
            temperature=temperature
        )
    
    def batch_generate_for_dataset(self, dataset: List[Dict[str, str]], num_samples: int = None) -> List[Dict]:
        """
        Generate explanations for a dataset with temperature diversity.
        
        Args:
            dataset: List of dicts with 'question' and 'answer' keys
            num_samples: Number of samples to process (None = all)
        
        Returns:
            List of training data items with explanations and metadata
        """
        if num_samples:
            dataset = dataset[:num_samples]
        
        training_data = []
        
        for i, example in enumerate(dataset):
            question = example.get('question', '')
            answer = example.get('answer', '')
            
            if not question or not answer:
                continue
            
            # Generate with random temperature for diversity (RLT approach)
            temperature = np.random.uniform(*self.config.temperature_range)
            
            try:
                explanation = self.generate_explanation(question, answer, temperature)
                
                training_item = {
                    'question': question,
                    'answer': answer,
                    'explanation': explanation,
                    'temperature': temperature,
                    'success': True
                }
                
            except Exception as e:
                print(f"Failed to generate explanation for item {i}: {e}")
                training_item = {
                    'question': question,
                    'answer': answer,
                    'explanation': None,
                    'temperature': temperature,
                    'success': False,
                    'error': str(e)
                }
            
            training_data.append(training_item)
            
            # Progress update
            if (i + 1) % 5 == 0:
                print(f"Progress: {i+1}/{len(dataset)} samples processed")
        
        # Print cost summary
        print(f"\nCost Summary: {json.dumps(self.cost_tracker.get_summary(), indent=2)}")
        
        return training_data


def integrate_with_notebook_data():
    """
    Example of integrating Claude teacher with the notebook's sample data.
    """
    print("=== Notebook Integration Example ===\n")
    
    # Sample data from the notebook
    sample_problems = [
        {
            "question": "A store sells 15 apples per day and operates 6 days a week. How many apples does it sell in 4 weeks?",
            "answer": "360"
        },
        {
            "question": "If a rectangle has length 8 meters and width 5 meters, what is its area?", 
            "answer": "40 square meters"
        },
        {
            "question": "Sarah has 24 stickers. She gives away 1/3 of them to her friends. How many stickers does she have left?",
            "answer": "16"
        },
        {
            "question": "A train travels 60 miles per hour for 3 hours. How far does it travel?",
            "answer": "180 miles"
        },
        {
            "question": "If 5 notebooks cost $25, how much does one notebook cost?",
            "answer": "$5"
        }
    ]
    
    # Initialize config
    config = RLTConfig()
    
    # Create Claude teacher
    teacher = NotebookClaudeTeacher(config, budget_limit=5.0)
    
    # Generate training data
    print("Generating explanations for training data...\n")
    training_data = teacher.batch_generate_for_dataset(sample_problems)
    
    # Display results
    print("\n=== Generated Training Data ===")
    for i, item in enumerate(training_data):
        print(f"\nProblem {i+1}:")
        print(f"Q: {item['question']}")
        print(f"A: {item['answer']}")
        print(f"Temperature: {item['temperature']:.2f}")
        if item['success']:
            print(f"Explanation preview: {item['explanation'][:150]}...")
        else:
            print(f"Error: {item.get('error', 'Unknown error')}")
    
    # Simulate student scoring (in real RLT, this comes from student model)
    print("\n=== Simulating Student Understanding Scores ===")
    for item in training_data:
        if item['success']:
            # Simulate understanding score based on temperature
            # (In reality, this would come from the student model)
            base_score = 0.7
            temp_factor = 1 - abs(item['temperature'] - 0.5) * 0.5
            noise = np.random.uniform(-0.1, 0.1)
            item['understanding_score'] = max(0, min(1, base_score * temp_factor + noise))
            print(f"Temp {item['temperature']:.2f} -> Score: {item['understanding_score']:.3f}")
    
    # Filter high-quality explanations
    successful_items = [item for item in training_data if item['success']]
    scores = [item['understanding_score'] for item in successful_items]
    threshold = np.percentile(scores, config.reward_threshold_percentile)
    
    high_quality = [
        item for item in successful_items 
        if item['understanding_score'] >= threshold
    ]
    
    print(f"\n=== Quality Filtering ===")
    print(f"Total explanations: {len(successful_items)}")
    print(f"Quality threshold (top {100-config.reward_threshold_percentile}%): {threshold:.3f}")
    print(f"High-quality explanations: {len(high_quality)}")
    
    # Save for use in training
    output_file = "claude_rlt_training_data.json"
    with open(output_file, 'w') as f:
        json.dump(high_quality, f, indent=2)
    print(f"\nSaved high-quality training data to {output_file}")
    
    return high_quality


def create_notebook_compatible_teacher():
    """
    Create a teacher instance that can be directly used in the notebook.
    
    Usage in notebook:
        from src.teachers.notebook_integration import create_notebook_compatible_teacher
        teacher = create_notebook_compatible_teacher()
        
        # Use it just like the notebook's teacher
        explanation = teacher.generate_explanation(question, answer, temperature)
    """
    config = RLTConfig()
    return NotebookClaudeTeacher(config, budget_limit=10.0)


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("CLAUDE_API_KEY"):
        print("Error: CLAUDE_API_KEY environment variable not set!")
        print("Set it with: export CLAUDE_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Run integration example
    try:
        high_quality_data = integrate_with_notebook_data()
        
        print("\n=== Integration Complete ===")
        print(f"Generated {len(high_quality_data)} high-quality training examples")
        print("\nYou can now use these explanations to train your teacher model!")
        
    except Exception as e:
        print(f"\nError during integration: {e}")
        import traceback
        traceback.print_exc()