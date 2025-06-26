# Model Loading and Memory Management Strategies

## Current Implementation Analysis

The codebase uses:
- Basic `device_map="auto"`
- 8-bit quantization via BitsAndBytes
- No explicit memory management

## Advanced Memory Management Techniques

### 1. Efficient Model Loading Pipeline

```python
import torch
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM

class EfficientModelLoader:
    def __init__(self, max_memory=None):
        self.max_memory = max_memory or self._get_available_memory()
        
    def _get_available_memory(self):
        """Calculate available memory per device."""
        if torch.cuda.is_available():
            gpu_memory = {}
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.get_device_properties(i).total_memory
                # Reserve 2GB for other operations
                gpu_memory[i] = f"{int(mem * 0.9 / 1e9)}GiB"
            return {**gpu_memory, "cpu": "100GiB"}
        return {"cpu": "100GiB"}
    
    def load_model_efficiently(self, model_name, dtype=torch.float16):
        """Load model with optimal memory allocation."""
        # Step 1: Load config
        config = AutoConfig.from_pretrained(model_name)
        
        # Step 2: Initialize empty model
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        
        # Step 3: Infer device map
        device_map = infer_auto_device_map(
            model,
            max_memory=self.max_memory,
            dtype=dtype,
            no_split_module_classes=["LlamaDecoderLayer", "MistralDecoderLayer"]
        )
        
        # Step 4: Load and dispatch
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=model_name,
            device_map=device_map,
            dtype=dtype,
            offload_folder="offload",
            offload_state_dict=True
        )
        
        return model, device_map
```

### 2. Dynamic Memory Management

```python
import gc
import torch.cuda as cuda

class MemoryManager:
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.initial_memory = {}
        self._record_initial_memory()
    
    def _record_initial_memory(self):
        """Record initial memory state."""
        if cuda.is_available():
            for i in range(cuda.device_count()):
                self.initial_memory[i] = cuda.memory_allocated(i)
    
    def optimize_memory(self):
        """Free up memory when approaching threshold."""
        if not cuda.is_available():
            return
        
        for device_id in range(cuda.device_count()):
            mem_allocated = cuda.memory_allocated(device_id)
            mem_total = cuda.get_device_properties(device_id).total_memory
            
            if mem_allocated / mem_total > self.threshold:
                # Clear cache
                cuda.empty_cache()
                gc.collect()
                
                # Log memory status
                print(f"GPU {device_id}: {mem_allocated/1e9:.2f}GB / {mem_total/1e9:.2f}GB")
    
    def get_memory_stats(self):
        """Get detailed memory statistics."""
        stats = {}
        if cuda.is_available():
            for i in range(cuda.device_count()):
                stats[f"gpu_{i}"] = {
                    "allocated": cuda.memory_allocated(i) / 1e9,
                    "reserved": cuda.memory_reserved(i) / 1e9,
                    "free": (cuda.get_device_properties(i).total_memory - 
                            cuda.memory_allocated(i)) / 1e9
                }
        return stats
```

### 3. Model Sharding for Multi-GPU

```python
from transformers import AutoModelForCausalLM
import torch.distributed as dist

class ModelShardingManager:
    def __init__(self, model_name, num_gpus=None):
        self.model_name = model_name
        self.num_gpus = num_gpus or torch.cuda.device_count()
        
    def create_device_map(self, model_config):
        """Create balanced device map for model sharding."""
        num_layers = model_config.num_hidden_layers
        layers_per_gpu = num_layers // self.num_gpus
        
        device_map = {"model.embed_tokens": 0, "model.norm": self.num_gpus - 1}
        
        # Distribute layers evenly
        for i in range(num_layers):
            device_id = min(i // layers_per_gpu, self.num_gpus - 1)
            device_map[f"model.layers.{i}"] = device_id
        
        # Output layer on last GPU
        device_map["lm_head"] = self.num_gpus - 1
        
        return device_map
    
    def load_sharded_model(self):
        """Load model with custom sharding."""
        config = AutoConfig.from_pretrained(self.model_name)
        device_map = self.create_device_map(config)
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,
            torch_dtype=torch.float16,
            offload_folder="offload_folder"
        )
        
        return model, device_map
```

### 4. Gradient Accumulation Memory Optimization

```python
class GradientAccumulationOptimizer:
    def __init__(self, model, target_batch_size=32, micro_batch_size=4):
        self.model = model
        self.target_batch_size = target_batch_size
        self.micro_batch_size = micro_batch_size
        self.accumulation_steps = target_batch_size // micro_batch_size
        
    def training_step(self, batch, optimizer, step):
        """Memory-efficient training step."""
        # Split batch into micro-batches
        micro_batches = self._split_batch(batch, self.micro_batch_size)
        
        total_loss = 0
        for i, micro_batch in enumerate(micro_batches):
            # Forward pass
            outputs = self.model(**micro_batch)
            loss = outputs.loss / self.accumulation_steps
            
            # Backward pass
            loss.backward()
            total_loss += loss.item()
            
            # Clear intermediate activations
            if i < len(micro_batches) - 1:
                self._clear_intermediate_memory()
        
        # Update weights
        if (step + 1) % self.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        return total_loss
    
    def _clear_intermediate_memory(self):
        """Clear intermediate tensors to save memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### 5. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()
        
    def train_step(self, batch):
        """Training step with automatic mixed precision."""
        self.optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast(dtype=torch.float16):
            outputs = self.model(**batch)
            loss = outputs.loss
        
        # Scaled backward pass
        self.scaler.scale(loss).backward()
        
        # Unscale and clip gradients
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Update weights
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
```

### 6. Teacher-Student Memory Optimization

```python
class TeacherStudentMemoryManager:
    def __init__(self, teacher_name, student_name):
        self.teacher_name = teacher_name
        self.student_name = student_name
        self.memory_manager = MemoryManager()
        
    def load_models_optimally(self):
        """Load teacher and student with memory optimization."""
        # Load teacher on primary GPU
        teacher = AutoModelForCausalLM.from_pretrained(
            self.teacher_name,
            device_map={"": 0},
            torch_dtype=torch.float16,
            load_in_8bit=True  # Quantize teacher
        )
        
        # Check memory before loading student
        self.memory_manager.optimize_memory()
        stats = self.memory_manager.get_memory_stats()
        
        # Load student on available device
        if stats.get("gpu_1", {}).get("free", 0) > 10:  # 10GB free
            student_device = {"": 1}
        else:
            student_device = "auto"  # Distribute across devices
        
        student = AutoModelForCausalLM.from_pretrained(
            self.student_name,
            device_map=student_device,
            torch_dtype=torch.float16,
            load_in_4bit=True  # More aggressive quantization
        )
        
        return teacher, student
```

## Memory Usage Patterns

| Model Size | FP32 Memory | FP16 Memory | INT8 Memory | INT4 Memory |
|------------|-------------|-------------|-------------|-------------|
| 1B params  | 4GB         | 2GB         | 1GB         | 0.5GB       |
| 3B params  | 12GB        | 6GB         | 3GB         | 1.5GB       |
| 7B params  | 28GB        | 14GB        | 7GB         | 3.5GB       |
| 11B params | 44GB        | 22GB        | 11GB        | 5.5GB       |
| 13B params | 52GB        | 26GB        | 13GB        | 6.5GB       |

## Best Practices

### 1. Model Loading Checklist
- [ ] Calculate available memory before loading
- [ ] Use appropriate dtype (fp16/bf16)
- [ ] Enable offloading for large models
- [ ] Split models across GPUs when possible
- [ ] Clear cache after loading

### 2. Training Memory Optimization
- [ ] Use gradient accumulation
- [ ] Enable gradient checkpointing
- [ ] Implement mixed precision training
- [ ] Monitor memory usage continuously
- [ ] Use PEFT methods for large models

### 3. Inference Optimization
- [ ] Use KV cache efficiently
- [ ] Implement batch processing
- [ ] Enable torch.compile() for PyTorch 2.0+
- [ ] Use streaming generation for long outputs
- [ ] Clear intermediate tensors

## Integration Example

```python
# Optimized teacher-student pipeline
memory_manager = MemoryManager()
loader = EfficientModelLoader()

# Load teacher efficiently
print("Loading teacher model...")
teacher_model, teacher_map = loader.load_model_efficiently(
    "meta-llama/Llama-3.2-11B-Instruct",
    dtype=torch.float16
)

# Check memory before loading student
memory_manager.optimize_memory()
stats = memory_manager.get_memory_stats()
print(f"Memory stats: {stats}")

# Load student with remaining memory
print("Loading student model...")
student_model, student_map = loader.load_model_efficiently(
    "meta-llama/Llama-3.2-3B-Instruct",
    dtype=torch.float16
)

# Setup training with memory optimization
trainer = TeacherStudentMemoryManager(teacher_model, student_model)
```