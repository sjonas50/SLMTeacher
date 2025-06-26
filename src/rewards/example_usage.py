"""
Example Usage of RLT Dense Reward System

This script demonstrates how to use the RLT reward system for training
language models with reasoning-focused rewards.
"""

import numpy as np
import torch
from typing import List, Dict

# Import RLT components
from rewards import (
    RLTRewardFunction,
    AdaptiveRewardFunction,
    RewardConfig,
    LocalStudentEvaluator,
    APIStudentEvaluator,
    EnsembleStudentEvaluator,
    TransformerKLCalculator,
    CachedKLCalculator,
    ApproximateKLCalculator,
    RewardLogger,
    RewardDebugger
)


def example_basic_usage():
    """Basic example of computing RLT rewards."""
    print("=== Basic RLT Reward Computation ===\n")
    
    # Initialize student evaluator (using a small local model)
    print("Initializing student evaluator...")
    student_eval = LocalStudentEvaluator(
        model_name="microsoft/phi-2",  # Small efficient model
        device="cuda" if torch.cuda.is_available() else "cpu",
        temperature=0.1
    )
    
    # Initialize KL divergence calculator
    print("Initializing KL calculator...")
    kl_calc = TransformerKLCalculator(
        model_name="gpt2-medium",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Configure reward computation
    config = RewardConfig(
        lambda_kl=0.5,  # Balance between solution score and KL
        normalize=True,
        temperature=1.0,
        clip_rewards=True,
        clip_range=(-10.0, 10.0),
        debug=True
    )
    
    # Create reward function
    reward_fn = RLTRewardFunction(student_eval, kl_calc, config)
    
    # Example data
    problems = [
        "What is 15 + 27?",
        "Solve for x: 2x + 5 = 13",
        "Find the area of a rectangle with length 6 and width 4"
    ]
    
    teacher_reasoning = [
        "To find 15 + 27, I'll add the tens place (10 + 20 = 30) and ones place (5 + 7 = 12), giving us 30 + 12 = 42.",
        "To solve 2x + 5 = 13, I'll subtract 5 from both sides: 2x = 8, then divide by 2: x = 4.",
        "The area of a rectangle is length × width. So, 6 × 4 = 24 square units."
    ]
    
    # Compute rewards
    print("\nComputing rewards...")
    rewards = reward_fn.compute_reward(
        reasoning=teacher_reasoning,
        problem=problems,
        return_components=True
    )
    
    # Display results
    print("\nResults:")
    print(f"Total rewards: {rewards['total_reward']}")
    print(f"Solution scores: {rewards['solution_score']}")
    print(f"KL scores: {rewards['kl_score']}")
    print(f"Lambda (KL weight): {rewards['lambda_kl']}")
    
    # Get statistics
    stats = reward_fn.get_reward_statistics(teacher_reasoning, problems)
    print("\nReward Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")


def example_adaptive_rewards():
    """Example using adaptive reward function with lambda scheduling."""
    print("\n=== Adaptive Reward Function ===\n")
    
    # Use approximate KL for faster computation
    student_eval = LocalStudentEvaluator(model_name="microsoft/phi-2")
    kl_calc = ApproximateKLCalculator()
    
    # Create adaptive reward function
    adaptive_reward_fn = AdaptiveRewardFunction(
        student_eval,
        kl_calc,
        config=RewardConfig(lambda_kl=1.0),
        lambda_scheduler='linear'
    )
    
    # Simulate training loop
    print("Simulating training with adaptive lambda...")
    for step in range(5):
        reasoning = [f"Step {step}: Let me solve this step by step..."]
        problem = ["What is 10 + 5?"]
        
        rewards = adaptive_reward_fn.compute_reward(reasoning, problem)
        
        print(f"Step {step}: λ={adaptive_reward_fn.config.lambda_kl:.3f}, "
              f"reward={float(rewards[0]):.3f}")


def example_ensemble_evaluation():
    """Example using ensemble of student evaluators."""
    print("\n=== Ensemble Student Evaluation ===\n")
    
    # Create multiple student evaluators
    evaluators = []
    
    # Local model evaluator
    if torch.cuda.is_available():
        evaluators.append(
            LocalStudentEvaluator(model_name="microsoft/phi-2")
        )
    
    # Approximate evaluator (always available)
    evaluators.append(
        LocalStudentEvaluator(model_name="gpt2")  # Smaller fallback
    )
    
    # Create ensemble
    ensemble_eval = EnsembleStudentEvaluator(
        evaluators=evaluators,
        weights=[0.7, 0.3],  # Weight the better model higher
        aggregation="weighted"
    )
    
    # Use with reward function
    kl_calc = ApproximateKLCalculator()
    reward_fn = RLTRewardFunction(ensemble_eval, kl_calc)
    
    # Test
    rewards = reward_fn.compute_reward(
        reasoning=["First, I'll break down the problem..."],
        problem=["Calculate 25% of 80"]
    )
    
    print(f"Ensemble reward: {float(rewards[0]):.3f}")


def example_caching_and_logging():
    """Example with caching and comprehensive logging."""
    print("\n=== Caching and Logging ===\n")
    
    # Setup components with caching
    student_eval = LocalStudentEvaluator(model_name="gpt2")
    base_kl_calc = ApproximateKLCalculator()
    cached_kl_calc = CachedKLCalculator(base_kl_calc, cache_size=1000)
    
    # Create reward function
    reward_fn = RLTRewardFunction(student_eval, cached_kl_calc)
    
    # Initialize logger
    logger = RewardLogger(experiment_name="rlt_example")
    
    # Simulate training
    print("Running training simulation with logging...")
    
    # Training data
    problems = [
        "What is 2 + 2?",
        "What is 3 × 4?",
        "What is 10 - 6?"
    ]
    
    reasoning_templates = [
        "Let me solve this: {problem}. The answer is obtained by {method}.",
        "To solve {problem}, I'll {method} to get the result.",
        "For {problem}, we need to {method}. This gives us the answer."
    ]
    
    for step in range(10):
        # Generate reasoning (simulated)
        reasoning = []
        for i, prob in enumerate(problems):
            template = reasoning_templates[i % len(reasoning_templates)]
            reason = template.format(
                problem=prob,
                method="performing the calculation"
            )
            reasoning.append(reason)
        
        # Compute rewards
        reward_components = reward_fn.compute_reward(
            reasoning=reasoning,
            problem=problems,
            return_components=True
        )
        
        # Log rewards
        logger.log_rewards(
            step=step,
            rewards=reward_components['total_reward'],
            components=reward_components,
            metadata={"batch_size": len(problems)}
        )
        
        if step % 3 == 0:
            print(f"Step {step}: Mean reward = {np.mean(reward_components['total_reward']):.3f}")
    
    # Save and visualize
    logger.save_logs()
    logger.plot_reward_history()
    logger.plot_component_analysis()
    print(f"\nLogs saved to: {logger.experiment_dir}")


def example_debugging_rewards():
    """Example of debugging reward computation."""
    print("\n=== Reward Debugging ===\n")
    
    # Setup
    student_eval = LocalStudentEvaluator(model_name="gpt2")
    kl_calc = ApproximateKLCalculator()
    reward_fn = RLTRewardFunction(student_eval, kl_calc)
    
    # Generate test data
    problems = ["What is " + str(i) + " + " + str(i+1) + "?" for i in range(50)]
    reasoning = [f"To add {i} + {i+1}, I get {2*i+1}." for i in range(50)]
    
    # Compute rewards with components
    reward_components = reward_fn.compute_reward(
        reasoning=reasoning,
        problem=problems,
        return_components=True
    )
    
    # Analyze distribution
    print("Analyzing reward distribution...")
    RewardDebugger.analyze_reward_distribution(
        rewards=reward_components['total_reward'],
        components=reward_components,
        save_path="reward_debug_analysis.png"
    )
    
    # Validate computation
    test_cases = [
        {
            "reasoning": "To solve 1 + 1, I add them to get 2.",
            "problem": "What is 1 + 1?"
        },
        {
            "reasoning": "",  # Empty reasoning
            "problem": "What is 5 × 5?"
        },
        {
            "reasoning": "This is completely unrelated text about cooking.",
            "problem": "Solve x^2 = 16"
        }
    ]
    
    print("\nValidating reward computation...")
    validation_results = RewardDebugger.validate_reward_computation(
        reward_function=reward_fn,
        test_cases=test_cases
    )
    
    for result in validation_results:
        print(f"\nTest case {result['test_case']}:")
        print(f"  Valid: {result['valid']}")
        if result['issues']:
            print(f"  Issues: {', '.join(result['issues'])}")
        if result['reward_stats']:
            print(f"  Reward mean: {result['reward_stats']['mean']:.3f}")


def example_api_based_evaluation():
    """Example using API-based student evaluation (requires API key)."""
    print("\n=== API-Based Evaluation ===\n")
    
    # Note: This requires an actual API key
    # For demonstration, we'll show the setup
    
    print("API-based evaluation setup (requires API key):")
    print("""
    # Initialize API evaluator
    api_eval = APIStudentEvaluator(
        api_type="openai",
        api_key="your-api-key-here",
        model_name="gpt-3.5-turbo",
        temperature=0.1
    )
    
    # Use with reward function
    kl_calc = TransformerKLCalculator()
    reward_fn = RLTRewardFunction(api_eval, kl_calc)
    
    # Compute rewards
    rewards = reward_fn.compute_reward(
        reasoning=["Step-by-step solution..."],
        problem=["Complex math problem..."]
    )
    """)


if __name__ == "__main__":
    print("RLT Dense Reward System - Example Usage\n")
    print("This demonstrates various features of the RLT reward implementation.\n")
    
    # Run examples
    try:
        example_basic_usage()
    except Exception as e:
        print(f"Basic usage example failed: {e}")
    
    try:
        example_adaptive_rewards()
    except Exception as e:
        print(f"Adaptive rewards example failed: {e}")
    
    try:
        example_ensemble_evaluation()
    except Exception as e:
        print(f"Ensemble evaluation example failed: {e}")
    
    try:
        example_caching_and_logging()
    except Exception as e:
        print(f"Caching/logging example failed: {e}")
    
    try:
        example_debugging_rewards()
    except Exception as e:
        print(f"Debugging example failed: {e}")
    
    example_api_based_evaluation()
    
    print("\n\nAll examples completed!")
    print("\nTo use this in your training loop:")
    print("""
    1. Initialize the reward function with your choice of student evaluator and KL calculator
    2. During training, compute rewards for generated reasoning
    3. Use rewards for policy gradient updates or other RL algorithms
    4. Monitor with RewardLogger and debug with RewardDebugger
    5. Adjust lambda_kl to balance solution quality vs. naturalness
    """)