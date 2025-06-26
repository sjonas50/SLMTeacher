# PEFT Methods for Meta/Mistral Architectures

## Current Implementation Analysis

The codebase uses basic LoRA with:
- Fixed rank (r=32)
- Standard target modules
- No architecture-specific optimizations

## Advanced PEFT Strategies

### 1. Architecture-Specific LoRA Configuration

#### Llama Models Optimal LoRA
```python
from peft import LoraConfig, TaskType

def get_llama_lora_config(model_size: str):
    """Get optimized LoRA config for Llama models."""
    
    # Size-specific configurations
    configs = {
        "1B": {"r": 16, "lora_alpha": 32},
        "3B": {"r": 32, "lora_alpha": 64},
        "7B": {"r": 64, "lora_alpha": 128},
        "11B": {"r": 128, "lora_alpha": 256},
    }
    
    base_config = configs.get(model_size, {"r": 64, "lora_alpha": 128})
    
    return LoraConfig(
        r=base_config["r"],
        lora_alpha=base_config["lora_alpha"],
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",      # MLP
            "lm_head"  # Output layer for better adaptation
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["embed_tokens", "lm_head"],  # Save embeddings
    )
```

#### Mistral Models Optimal LoRA
```python
def get_mistral_lora_config():
    """Get optimized LoRA config for Mistral models."""
    return LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # Mistral-specific: don't modify embeddings
        modules_to_save=None,
        # Enable for sliding window attention
        use_rslora=True,  # Rank-stabilized LoRA
    )
```

### 2. QLoRA with Advanced Quantization

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

def get_optimized_qlora_config():
    """4-bit quantization with NF4 for maximum efficiency."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Normal Float 4
        bnb_4bit_compute_dtype=torch.bfloat16,  # Better than fp16
        bnb_4bit_use_double_quant=True,  # Double quantization
        bnb_4bit_qlora_config={
            "use_qlora": True,
            "use_gradient_checkpointing": True,
        }
    )

# Apply QLoRA
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=get_optimized_qlora_config(),
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)
```

### 3. IA³ (Infused Adapter by Inhibiting and Amplifying)

```python
from peft import IA3Config, get_peft_model

def apply_ia3_to_model(model, model_type="llama"):
    """IA³ for even more parameter efficiency."""
    
    target_modules = {
        "llama": ["k_proj", "v_proj", "down_proj"],
        "mistral": ["k_proj", "v_proj", "down_proj"]
    }
    
    ia3_config = IA3Config(
        target_modules=target_modules.get(model_type),
        feedforward_modules=["down_proj"],
        modules_to_save=["lm_head"],
    )
    
    return get_peft_model(model, ia3_config)
```

### 4. AdaLoRA (Adaptive LoRA)

```python
from peft import AdaLoraConfig

def get_adalora_config(model_type="llama"):
    """Adaptive rank allocation for LoRA."""
    return AdaLoraConfig(
        r=64,
        target_modules=get_target_modules(model_type),
        lora_alpha=32,
        lora_dropout=0.1,
        # AdaLoRA specific
        target_r=8,  # Target average rank
        init_r=16,   # Initial rank
        tinit=0,     # Warmup steps
        tfinal=1000, # Final step for pruning
        deltaT=10,   # Pruning interval
        orth_reg_weight=0.5,
    )
```

### 5. Multi-Task PEFT

```python
from peft import MultitaskPromptTuningConfig, get_peft_model

class MultiTaskPEFT:
    """PEFT for teacher-student with multiple tasks."""
    
    def __init__(self, base_model, tasks):
        self.base_model = base_model
        self.tasks = tasks
        self.peft_configs = {}
        
    def add_task_adapter(self, task_name, peft_config):
        """Add task-specific adapter."""
        self.peft_configs[task_name] = peft_config
        
    def get_model_for_task(self, task_name):
        """Get model with task-specific adapter."""
        model = get_peft_model(
            self.base_model,
            self.peft_configs[task_name]
        )
        model.set_adapter(task_name)
        return model

# Usage
multi_peft = MultiTaskPEFT(base_model, ["reasoning", "math", "coding"])
multi_peft.add_task_adapter("reasoning", get_llama_lora_config("7B"))
multi_peft.add_task_adapter("math", get_adalora_config("llama"))
```

### 6. Hybrid PEFT Methods

```python
from peft import PeftModel, LoraConfig, PrefixTuningConfig

class HybridPEFT:
    """Combine multiple PEFT methods for maximum efficiency."""
    
    def __init__(self, base_model):
        self.base_model = base_model
        
    def apply_hybrid_peft(self):
        # First: Apply LoRA to attention layers
        lora_config = LoraConfig(
            r=32,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=64,
        )
        
        # Second: Apply prefix tuning for task adaptation
        prefix_config = PrefixTuningConfig(
            num_virtual_tokens=10,
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Combine methods
        model = get_peft_model(self.base_model, lora_config)
        # Additional PEFT layers can be stacked
        
        return model
```

## Memory and Performance Comparison

| Method | Memory Usage | Trainable Params | Performance | Best Use Case |
|--------|--------------|------------------|-------------|---------------|
| LoRA | Low | 0.1-1% | ⭐⭐⭐⭐ | General fine-tuning |
| QLoRA | Very Low | 0.1-1% | ⭐⭐⭐ | Memory-constrained |
| IA³ | Ultra Low | 0.01% | ⭐⭐⭐ | Extreme efficiency |
| AdaLoRA | Low | Dynamic | ⭐⭐⭐⭐⭐ | Automatic optimization |
| Prefix Tuning | Medium | <0.1% | ⭐⭐⭐ | Task adaptation |

## Teacher-Student Specific Optimizations

### 1. Asymmetric PEFT
```python
def setup_teacher_student_peft(teacher_model, student_model):
    """Different PEFT strategies for teacher and student."""
    
    # Teacher: Larger LoRA for better reasoning
    teacher_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
    )
    
    # Student: Smaller LoRA for efficiency
    student_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Fewer modules
    )
    
    teacher_peft = get_peft_model(teacher_model, teacher_config)
    student_peft = get_peft_model(student_model, student_config)
    
    return teacher_peft, student_peft
```

### 2. Progressive PEFT
```python
class ProgressivePEFT:
    """Gradually increase PEFT complexity during training."""
    
    def __init__(self, model):
        self.model = model
        self.current_rank = 8
        
    def update_rank(self, step, total_steps):
        """Increase LoRA rank during training."""
        progress = step / total_steps
        
        if progress > 0.3 and self.current_rank == 8:
            self.current_rank = 16
            self._reinitialize_lora()
        elif progress > 0.6 and self.current_rank == 16:
            self.current_rank = 32
            self._reinitialize_lora()
```

## Recommendations

1. **For Llama Models**: Use standard LoRA with all linear layers
2. **For Mistral Models**: Focus on attention layers due to sliding window
3. **Memory Constrained**: Use QLoRA with 4-bit quantization
4. **Multi-Task**: Implement task-specific adapters
5. **Production**: Use AdaLoRA for automatic rank optimization