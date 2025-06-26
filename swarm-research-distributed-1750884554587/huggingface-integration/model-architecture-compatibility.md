# Model Architecture Compatibility Analysis

## Meta (Llama) Models

### Llama 3.2 Series (≤21B params)
- **Llama-3.2-1B/3B**: Ideal for student models, efficient inference
- **Llama-3.2-11B**: Good balance for teacher models
- **Architecture**: RMSNorm, SwiGLU activation, RoPE embeddings
- **Key Modules**: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

### Compatibility Considerations
1. **Tokenizer Alignment**:
   - Llama models use SentencePiece tokenizer
   - Special tokens: <|begin_of_text|>, <|end_of_text|>, <|eot_id|>
   - Vocabulary size: 128,256 tokens
   
2. **Architecture Features**:
   - Grouped Query Attention (GQA) in newer models
   - RoPE base frequency: 500,000
   - Layer normalization: RMSNorm (eps=1e-5)

## Mistral Models

### Mistral Series (≤21B params)
- **Mistral-7B-v0.3**: Excellent teacher model
- **Mistral-7B-Instruct**: Pre-trained for instruction following
- **Architecture**: Similar to Llama but with sliding window attention
- **Key Modules**: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

### Compatibility Considerations
1. **Sliding Window Attention**:
   - Window size: 4096 tokens
   - Reduces memory usage for long sequences
   - Compatible with Flash Attention 2

2. **Tokenizer**:
   - Uses SentencePiece like Llama
   - Vocabulary size: 32,000 tokens
   - Different special token handling

## Teacher-Student Compatibility Matrix

| Teacher Model | Student Model | Compatibility | Notes |
|--------------|---------------|---------------|-------|
| Llama-3.2-11B | Llama-3.2-3B | ⭐⭐⭐⭐⭐ | Same architecture family, optimal |
| Llama-3.2-11B | Llama-3.2-1B | ⭐⭐⭐⭐ | Good, may need distillation tuning |
| Mistral-7B | Llama-3.2-3B | ⭐⭐⭐ | Different attention, needs adaptation |
| Llama-3.2-11B | Mistral-7B | ⭐⭐⭐ | Tokenizer alignment required |
| Mistral-7B | Mistral-7B | ⭐⭐⭐⭐⭐ | Same model, different LoRA configs |

## Architecture-Specific Optimizations

### For Llama Models
```python
# Optimal configuration for Llama models
config = {
    "rope_theta": 500000,  # RoPE base frequency
    "use_cache": True,     # KV cache for generation
    "attention_dropout": 0.0,  # No dropout during inference
    "hidden_dropout": 0.0,
    "tie_word_embeddings": False,  # Llama doesn't tie embeddings
}
```

### For Mistral Models
```python
# Optimal configuration for Mistral models
config = {
    "sliding_window": 4096,
    "rope_theta": 10000,
    "use_sliding_window": True,
    "max_position_embeddings": 32768,
}
```

## Recommendations

1. **Same Family Training**: Use models from the same family (all Llama or all Mistral) for best compatibility
2. **Cross-Architecture**: When mixing architectures, implement tokenizer alignment layers
3. **Size Ratios**: Maintain 3:1 to 5:1 teacher-student parameter ratios for optimal distillation
4. **Attention Mechanisms**: Account for different attention implementations when computing KL divergence