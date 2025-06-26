# RLT Training Options Summary

## Available Training Scripts

### 1. **train_optimized_rlt_corrected.py** (Recommended Start)
- **Teacher**: Claude API
- **Student**: Local HF model (trainable)
- **Evaluator**: Local model
- **Cost**: Medium (teacher API calls only)
- **Use**: Standard high-quality training

```bash
python train_optimized_rlt_corrected.py --create-config
python train_optimized_rlt_corrected.py
```

### 2. **train_optimized_rlt_claude_eval.py** (Highest Quality)
- **Teacher**: Claude API
- **Student**: Local HF model (trainable)
- **Evaluator**: Claude API / Hybrid / Local
- **Cost**: High (teacher + evaluator API calls)
- **Use**: Production models, best results

```bash
python train_optimized_rlt_claude_eval.py --create-config
python train_optimized_rlt_claude_eval.py --eval-mode hybrid
```

### 3. **train_rlt_model.py** (Low Cost)
- **Teacher**: Local HF model
- **Student**: Same or different local model
- **Evaluator**: Local model
- **Cost**: Low (no API calls)
- **Use**: Experimentation, limited budget

```bash
python train_rlt_model.py --teacher-model meta-llama/Llama-3.2-3B-Instruct
```

## Configuration Comparison

| Feature | Corrected Script | Claude Eval Script | Standard Script |
|---------|-----------------|-------------------|-----------------|
| Teacher Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Evaluation Quality | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Training Cost | $$ | $$$ | $ |
| Speed | Medium | Slow | Fast |
| Best For | Most users | Production | Budget/Testing |

## Decision Tree

```
Do you need the absolute best quality?
├─ Yes → Use train_optimized_rlt_claude_eval.py with mode="claude"
└─ No → Is budget a major concern?
    ├─ Yes → Use train_rlt_model.py (all local)
    └─ No → Use train_optimized_rlt_corrected.py (recommended)
            or train_optimized_rlt_claude_eval.py with mode="hybrid"
```

## Cost Examples (1000 samples, 3 epochs)

### Corrected Script (Claude teacher, local eval)
- Teacher API calls: 12,000
- Evaluator API calls: 0
- **Estimated cost: $24**

### Claude Eval Script - Hybrid Mode (20% Claude)
- Teacher API calls: 12,000
- Evaluator API calls: 2,400
- **Estimated cost: $29**

### Claude Eval Script - Full Claude
- Teacher API calls: 12,000
- Evaluator API calls: 12,000
- **Estimated cost: $48**

### Standard Script (All Local)
- API calls: 0
- **Estimated cost: $0** (just compute)

## Optimization Features

All scripts support:
- ✅ Flash Attention 2
- ✅ QLoRA 4-bit quantization
- ✅ AdaLoRA adaptive ranks
- ✅ Gradient checkpointing
- ✅ Mixed precision (BF16)
- ✅ Memory monitoring
- ✅ Cost tracking

## Quick Start Recommendations

### For Most Users:
```bash
# Good balance of quality and cost
python train_optimized_rlt_corrected.py --create-config
python train_optimized_rlt_corrected.py
```

### For Production/Research:
```bash
# Highest quality with controlled costs
python train_optimized_rlt_claude_eval.py --create-config
# Edit config to set mode: "hybrid" with claude_frequency: 0.2
python train_optimized_rlt_claude_eval.py
```

### For Experimentation:
```bash
# Fast iteration, no API costs
python train_rlt_model.py \
    --teacher-model meta-llama/Llama-3.2-3B-Instruct \
    --student-model microsoft/phi-2 \
    --use-lora
```

## Key Points to Remember

1. **Teacher Role**: Generates explanations (Claude is best)
2. **Student Role**: Learns from explanations (this gets trained!)
3. **Evaluator Role**: Measures understanding (Claude gives best signals)
4. **Cost vs Quality**: Better evaluation = better final model
5. **Start Simple**: Begin with corrected script, upgrade if needed

## Architecture Reminder

```
Claude Teacher → Explanations → Student Model (trains) ← Rewards ← Evaluator
     (API)                         (Local)                    (Local/API)
```

The student model is always what gets trained, regardless of which script you use!