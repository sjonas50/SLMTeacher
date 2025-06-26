# Evaluator Options in RLT Training

## Overview

The evaluator component assesses how well the student model understands and can apply teacher explanations. This is crucial for computing high-quality reward signals during training.

## Available Options

### 1. **Local Model Evaluator** (Default)
- **Cost**: Low ($)
- **Quality**: Good (⭐⭐⭐)
- **Speed**: Fast
- **Use Case**: Budget-conscious training, prototyping

```python
evaluator = StudentEvaluator(
    model_name="meta-llama/Llama-3.2-3B-Instruct"
)
```

### 2. **Claude API Evaluator**
- **Cost**: High ($$$)
- **Quality**: Excellent (⭐⭐⭐⭐⭐)
- **Speed**: Slower (API calls)
- **Use Case**: Production models, research, high-stakes applications

```python
evaluator = ClaudeStudentEvaluator(
    model="claude-3-5-sonnet-20241022",
    temperature=0.1  # Low for consistent evaluation
)
```

### 3. **Hybrid Evaluator**
- **Cost**: Medium ($$)
- **Quality**: Very Good (⭐⭐⭐⭐)
- **Speed**: Medium
- **Use Case**: Balanced approach, most practical for many scenarios

```python
evaluator = HybridEvaluator(
    claude_evaluator=claude_eval,
    local_evaluator=local_eval,
    claude_eval_frequency=0.2  # Use Claude 20% of time
)
```

## Comparison Table

| Feature | Local Model | Claude API | Hybrid (20% Claude) |
|---------|------------|------------|-------------------|
| **Quality** | Good | Excellent | Very Good |
| **Cost per 1K samples** | ~$0 | ~$2 | ~$0.40 |
| **Speed** | 100 samples/min | 10 samples/min | 80 samples/min |
| **Consistency** | Variable | Very High | High |
| **Resource Usage** | GPU/CPU | Network only | GPU/CPU + Network |

## When to Use Each

### Use Local Model When:
- Budget is tight
- Running initial experiments
- Training on simple tasks
- Need fast iteration

### Use Claude API When:
- Training production models
- Quality is paramount
- Working on complex reasoning tasks
- Budget allows for best results

### Use Hybrid When:
- Want balance of quality and cost
- Training important but not critical models
- Need better than local but full Claude is too expensive
- Most real-world scenarios

## Configuration Examples

### Local Only (Lowest Cost)
```json
{
  "evaluation": {
    "mode": "local",
    "model_name": "meta-llama/Llama-3.2-3B-Instruct",
    "batch_size": 8
  }
}
```

### Claude Only (Highest Quality)
```json
{
  "evaluation": {
    "mode": "claude",
    "model": "claude-3-5-sonnet-20241022",
    "temperature": 0.1,
    "max_tokens": 512
  }
}
```

### Hybrid (Recommended)
```json
{
  "evaluation": {
    "mode": "hybrid",
    "claude_frequency": 0.2,
    "model": "claude-3-5-sonnet-20241022",
    "local_model": "meta-llama/Llama-3.2-3B-Instruct"
  }
}
```

## Impact on Training

### Quality Impact
Using Claude as evaluator typically results in:
- 15-25% better final model performance
- More stable training convergence
- Better generalization to unseen problems

### Cost Impact
For a typical training run (1000 samples, 3 epochs, 4 variants):
- Local: ~$0 (just compute)
- Hybrid (20%): ~$10-20
- Claude: ~$50-100

## Best Practices

1. **Start with Hybrid**: Begin with 10-20% Claude evaluation
2. **Monitor Quality**: Track improvement vs cost
3. **Use Caching**: Enable caching to avoid duplicate evaluations
4. **Budget Limits**: Always set budget limits when using Claude
5. **Critical Runs**: Use full Claude for final production training

## Technical Details

The evaluator's role in the reward computation:
```
reward = rSS - λ * rKL
         ↑
    Evaluator measures this
```

The evaluator determines rSS (solution score) by assessing how well the student can solve problems after seeing the teacher's explanation.

## Conclusion

Choice of evaluator is a key decision balancing:
- **Quality**: Better evaluation → better training signals → better final model
- **Cost**: API calls add up quickly in large training runs
- **Speed**: API calls are slower than local inference

For most users, the **hybrid approach with 20% Claude evaluation** provides the best balance of quality, cost, and speed.