# RLT Dense Reward System Implementation

This module implements the Reasoning Like Teaching (RLT) dense reward system based on Sakana AI's methodology. The reward function guides language models to generate reasoning that maximizes student understanding while maintaining naturalness.

## Mathematical Formulation

The RLT reward function combines two key components:

```
r(y, x) = rSS(y, x) - λ * rKL(y, x)
```

Where:
- `rSS`: Solution Score - measures how well a student model can solve problems using the generated reasoning
- `rKL`: KL Divergence Score - measures deviation from a baseline distribution to maintain naturalness
- `λ`: Trade-off parameter balancing the two objectives

## Components

### 1. **Reward Function** (`reward_function.py`)
- `RLTRewardFunction`: Main reward computation class
- `AdaptiveRewardFunction`: Extension with adaptive lambda scheduling
- `RewardConfig`: Configuration dataclass for reward parameters

### 2. **Student Evaluator** (`student_evaluator.py`)
- `LocalStudentEvaluator`: Uses local transformer models (e.g., Phi-2, GPT-2)
- `APIStudentEvaluator`: Uses API-based models (OpenAI, Anthropic)
- `EnsembleStudentEvaluator`: Combines multiple evaluators for robustness

### 3. **KL Divergence** (`kl_divergence.py`)
- `TransformerKLCalculator`: Token-level KL divergence using transformer models
- `SequenceKLCalculator`: Sequence-level KL divergence
- `CachedKLCalculator`: Adds caching for efficiency
- `ApproximateKLCalculator`: Fast approximation using embeddings

### 4. **Utilities** (`reward_utils.py`)
- Reward normalization functions
- Baseline computation
- `RewardLogger`: Comprehensive logging and tracking
- `RewardDebugger`: Debugging and visualization tools

## Installation

```bash
# Install required dependencies
pip install torch transformers numpy pandas matplotlib seaborn scipy

# Optional: For API-based evaluation
pip install aiohttp

# Optional: For approximate KL calculation
pip install sentence-transformers
```

## Quick Start

```python
from rewards import RLTRewardFunction, LocalStudentEvaluator, TransformerKLCalculator, RewardConfig

# Initialize components
student_eval = LocalStudentEvaluator(model_name="microsoft/phi-2")
kl_calc = TransformerKLCalculator(model_name="gpt2-medium")
config = RewardConfig(lambda_kl=0.5, normalize=True)

# Create reward function
reward_fn = RLTRewardFunction(student_eval, kl_calc, config)

# Compute rewards
rewards = reward_fn.compute_reward(
    reasoning=["First, let's break down the problem..."],
    problem=["What is 15 + 27?"]
)
```

## Advanced Usage

### Adaptive Lambda Scheduling

```python
from rewards import AdaptiveRewardFunction

# Create adaptive reward function
adaptive_fn = AdaptiveRewardFunction(
    student_eval, kl_calc,
    lambda_scheduler='linear'  # or 'exponential', 'adaptive'
)
```

### Ensemble Evaluation

```python
from rewards import EnsembleStudentEvaluator

# Combine multiple student models
evaluators = [
    LocalStudentEvaluator("microsoft/phi-2"),
    LocalStudentEvaluator("gpt2-medium")
]
ensemble = EnsembleStudentEvaluator(evaluators, weights=[0.6, 0.4])
```

### Logging and Monitoring

```python
from rewards import RewardLogger

# Initialize logger
logger = RewardLogger(experiment_name="rlt_training")

# Log rewards during training
for step in range(num_steps):
    rewards = reward_fn.compute_reward(reasoning, problems)
    logger.log_rewards(step, rewards)

# Visualize results
logger.plot_reward_history()
logger.plot_component_analysis()
```

### Debugging

```python
from rewards import RewardDebugger

# Analyze reward distribution
RewardDebugger.analyze_reward_distribution(
    rewards, components,
    save_path="analysis.png"
)

# Validate computation
results = RewardDebugger.validate_reward_computation(
    reward_fn, test_cases
)
```

## Configuration Options

### RewardConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_kl` | 0.5 | Trade-off between solution score and KL divergence |
| `normalize` | True | Whether to normalize rewards |
| `temperature` | 1.0 | Temperature for probability distributions |
| `clip_rewards` | True | Whether to clip extreme reward values |
| `clip_range` | (-10, 10) | Range for reward clipping |
| `use_baseline` | True | Whether to subtract baseline rewards |
| `debug` | False | Enable debug logging |

## Model Compatibility

### Student Models (Recommended)
- `microsoft/phi-2`: Efficient 2.7B parameter model
- `google/flan-t5-small`: Lightweight T5 variant
- `EleutherAI/pythia-1b`: Open-source alternative
- Any GPT-2 variant for testing

### Reference Models for KL
- `gpt2-medium`: Good balance of size and quality
- `EleutherAI/gpt-neo-1.3B`: Larger open-source option
- Domain-specific models for specialized tasks

## Performance Considerations

1. **Batching**: Always process multiple examples together for efficiency
2. **Caching**: Use `CachedKLCalculator` for repeated computations
3. **Approximation**: Use `ApproximateKLCalculator` for faster inference
4. **Model Selection**: Balance model quality with computational cost
5. **Device Usage**: Utilize GPU when available

## Integration with RL Training

```python
# Example integration with PPO training loop
for epoch in range(num_epochs):
    # Generate reasoning with current policy
    reasoning = policy_model.generate(problems)
    
    # Compute rewards
    rewards = reward_fn.compute_reward(reasoning, problems)
    
    # Compute advantages for policy gradient
    advantages = reward_fn.compute_advantage(
        reasoning, problems, baseline_reasoning
    )
    
    # Update policy with computed advantages
    policy_loss = compute_policy_loss(advantages)
    optimizer.step()
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use smaller models
2. **Slow Computation**: Enable caching, use approximations
3. **Unstable Rewards**: Adjust normalization, check for NaN values
4. **Poor Student Performance**: Try different student models or ensemble

### Debug Mode

Enable debug mode for detailed logging:
```python
config = RewardConfig(debug=True)
```

## Citation

This implementation is based on the Reasoning Like Teaching methodology from Sakana AI. If you use this code, please cite the original RLT paper.

## License

This implementation is provided as-is for research and educational purposes.