# Claude as Evaluator - High-Quality RLT Training

## Overview

While the standard RLT implementation uses a local model to evaluate student understanding, using Claude Sonnet 4 as the evaluator provides the highest quality assessment and training signals.

## Architecture with Claude Evaluation

```
1. Teacher (Claude API) → Generates explanations
2. Student (Local HF Model) → Learns from explanations (trainable)
3. Evaluator (Claude API) → Assesses student understanding
```

## Benefits of Claude Evaluation

### 1. **Superior Assessment Quality**
- Claude provides more accurate evaluation of student understanding
- Better distinguishes between surface-level mimicry and true comprehension
- More nuanced scoring of partial understanding

### 2. **Better Training Signals**
- Higher quality rewards lead to better student learning
- More reliable convergence during training
- Students learn deeper reasoning patterns

### 3. **Consistent Standards**
- Same high-quality model for both teaching and evaluation
- Ensures alignment between what's taught and what's assessed

## Cost Considerations

Using Claude for evaluation doubles API usage:
- 1 API call for teacher explanation generation
- 1 API call for student evaluation

### Cost Estimation Example
- 1000 training samples × 4 explanations each × 3 epochs = 12,000 teacher calls
- 12,000 evaluation calls (if using pure Claude mode)
- Total: 24,000 API calls (~$48 at typical rates)

## Usage Modes

### 1. **Pure Claude Mode** (Highest Quality)
```json
{
  "evaluation": {
    "mode": "claude",
    "model": "claude-3-5-sonnet-20241022",
    "temperature": 0.1
  }
}
```

### 2. **Hybrid Mode** (Balanced)
```json
{
  "evaluation": {
    "mode": "hybrid",
    "claude_frequency": 0.2,  // Use Claude 20% of the time
    "local_model": "meta-llama/Llama-3.2-3B-Instruct"
  }
}
```

### 3. **Local Mode** (Low Cost)
```json
{
  "evaluation": {
    "mode": "local",
    "model_name": "meta-llama/Llama-3.2-3B-Instruct"
  }
}
```

## Running Training with Claude Evaluation

```bash
# Create configuration
python train_optimized_rlt_claude_eval.py --create-config

# Edit configuration to set evaluation mode
# Then run training
python train_optimized_rlt_claude_eval.py

# Or override evaluation mode
python train_optimized_rlt_claude_eval.py --eval-mode claude
```

## Evaluation Process

### What Claude Evaluates

When assessing student understanding, Claude considers:

1. **Correctness**: Is the final answer right?
2. **Understanding**: Does the student grasp the teacher's approach?
3. **Application**: Are reasoning steps properly applied?
4. **Clarity**: Is the work logical and clear?

### Scoring Scale
- `1.0`: Perfect understanding and application
- `0.8-0.9`: Good understanding with minor issues
- `0.6-0.7`: Moderate understanding
- `0.4-0.5`: Poor understanding
- `0.0-0.3`: No understanding / incorrect

## Best Practices

### 1. **Start with Hybrid Mode**
- Begin with 10-20% Claude evaluation
- Monitor quality improvements
- Increase frequency if needed

### 2. **Use Claude for Critical Training**
- Final model training runs
- Difficult problem domains
- When quality is paramount

### 3. **Monitor Costs**
- Set budget limits in configuration
- Track API usage during training
- Use caching to avoid duplicate evaluations

### 4. **Optimize Batch Sizes**
- Smaller batches reduce memory but increase API calls
- Find optimal balance for your use case

## Performance Comparison

| Evaluation Mode | Quality | Cost | Speed |
|----------------|---------|------|-------|
| Claude (100%) | ⭐⭐⭐⭐⭐ | $$$ | Slower |
| Hybrid (20%) | ⭐⭐⭐⭐ | $$ | Medium |
| Local Only | ⭐⭐⭐ | $ | Fast |

## When to Use Claude Evaluation

### Recommended For:
- Production model training
- Research experiments
- Difficult reasoning tasks
- When budget allows

### Consider Alternatives For:
- Initial prototyping
- Large-scale experiments
- Budget-constrained projects
- Simple problem domains

## Technical Implementation

The Claude evaluator (`src/rewards/claude_evaluator.py`) provides:
- Structured evaluation prompts
- JSON response parsing
- Caching for efficiency
- Rate limiting
- Cost tracking
- Error handling

## Example Configuration

```json
{
  "budget_limit": 100.0,
  "teacher": {
    "model": "claude-3-5-sonnet-20241022",
    "temperature": 0.7
  },
  "evaluation": {
    "mode": "hybrid",
    "model": "claude-3-5-sonnet-20241022",
    "temperature": 0.1,
    "claude_frequency": 0.2,
    "local_model": "meta-llama/Llama-3.2-3B-Instruct"
  },
  "training": {
    "num_epochs": 3,
    "batch_size": 1,
    "gradient_accumulation_steps": 8
  }
}
```

## Conclusion

Using Claude as an evaluator provides the highest quality training signals for RLT, leading to better student models. While it increases costs, the quality improvements often justify the investment, especially for production models or challenging domains.