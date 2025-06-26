"""
Optimized HuggingFace Model with Flash Attention 2, QLoRA, and Advanced Memory Management
"""
import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    PreTrainedModel
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType,
    AdaLoraConfig
)
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import psutil
import GPUtil
from dataclasses import dataclass
import gc
import warnings


@dataclass
class OptimizedModelConfig:
    """Configuration for optimized model loading and training."""
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Quantization settings
    use_4bit: bool = True  # QLoRA with 4-bit
    use_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # Flash Attention settings
    use_flash_attention_2: bool = True
    
    # PEFT settings
    use_peft: bool = True
    peft_method: str = "adalora"  # "lora" or "adalora"
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    adalora_init_r: int = 12
    adalora_target_r: int = 32
    adalora_tinit: int = 50
    adalora_tfinal: int = 1000
    adalora_delta_t: int = 10
    
    # Memory optimization
    gradient_checkpointing: bool = True
    max_memory: Optional[Dict] = None
    device_map: str = "auto"
    
    # Mixed precision
    mixed_precision: str = "bf16"  # "bf16", "fp16", or "no"
    
    # Batch and sequence settings
    enable_gradient_accumulation: bool = True
    gradient_accumulation_steps: int = 4
    
    # KV cache settings
    use_cache: bool = True
    max_cache_length: int = 2048


class MemoryMonitor:
    """Monitor and manage GPU/CPU memory usage."""
    
    @staticmethod
    def get_memory_stats() -> Dict:
        """Get current memory statistics."""
        stats = {
            'cpu_percent': psutil.cpu_percent(),
            'cpu_memory_percent': psutil.virtual_memory().percent,
            'cpu_available_gb': psutil.virtual_memory().available / (1024**3)
        }
        
        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                stats[f'gpu_{i}_memory_percent'] = gpu.memoryUtil * 100
                stats[f'gpu_{i}_memory_free_gb'] = gpu.memoryFree / 1024
                stats[f'gpu_{i}_memory_used_gb'] = gpu.memoryUsed / 1024
        
        return stats
    
    @staticmethod
    def optimize_memory():
        """Run memory optimization."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def estimate_model_memory(model_name: str, use_4bit: bool = True) -> float:
        """Estimate memory requirements for a model."""
        # Rough estimates based on parameter count
        param_estimates = {
            "3B": 3e9,
            "7B": 7e9,
            "11B": 11e9,
            "13B": 13e9,
            "21B": 21e9
        }
        
        # Extract parameter count from model name
        params = 7e9  # default
        for key, value in param_estimates.items():
            if key in model_name:
                params = value
                break
        
        # Calculate memory based on quantization
        if use_4bit:
            bytes_per_param = 0.5  # 4-bit
        else:
            bytes_per_param = 2  # FP16
        
        return (params * bytes_per_param) / (1024**3)  # GB


class OptimizedHFModel:
    """Optimized HuggingFace model with advanced features."""
    
    def __init__(self, config: OptimizedModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check memory before loading
        self.memory_monitor = MemoryMonitor()
        initial_stats = self.memory_monitor.get_memory_stats()
        print("Initial memory stats:", initial_stats)
        
        # Estimate memory requirements
        estimated_memory = self.memory_monitor.estimate_model_memory(
            config.model_name, 
            config.use_4bit
        )
        print(f"Estimated model memory: {estimated_memory:.2f} GB")
        
        # Load model with optimizations
        self._load_model()
        
        # Setup PEFT if requested
        if config.use_peft:
            self._setup_peft()
        
        # Final memory stats
        final_stats = self.memory_monitor.get_memory_stats()
        print("Final memory stats:", final_stats)
    
    def _load_model(self):
        """Load model with all optimizations."""
        # Setup quantization config
        quantization_config = None
        if self.config.use_4bit and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant
            )
        elif self.config.use_8bit and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
        
        # Model loading kwargs
        model_kwargs = {
            "quantization_config": quantization_config,
            "device_map": self.config.device_map,
            "trust_remote_code": True,
            "use_cache": self.config.use_cache
        }
        
        # Add Flash Attention 2 if available
        if self.config.use_flash_attention_2:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("✓ Flash Attention 2 enabled")
            except Exception as e:
                warnings.warn(f"Flash Attention 2 not available: {e}")
                model_kwargs["attn_implementation"] = "eager"
        
        # Set max memory if specified
        if self.config.max_memory:
            model_kwargs["max_memory"] = self.config.max_memory
        
        # Load model
        print(f"Loading model: {self.config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def _setup_peft(self):
        """Setup PEFT (LoRA or AdaLoRA) for efficient training."""
        # Prepare model for k-bit training
        if self.config.use_4bit or self.config.use_8bit:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config.gradient_checkpointing
            )
        
        # Get target modules based on architecture
        target_modules = self._get_target_modules()
        
        # Create PEFT config
        if self.config.peft_method == "adalora":
            peft_config = AdaLoraConfig(
                init_r=self.config.adalora_init_r,
                target_r=self.config.adalora_target_r,
                tinit=self.config.adalora_tinit,
                tfinal=self.config.adalora_tfinal,
                deltaT=self.config.adalora_delta_t,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=target_modules,
                task_type=TaskType.CAUSAL_LM
            )
            print(f"✓ AdaLoRA configured (init_r={self.config.adalora_init_r}, target_r={self.config.adalora_target_r})")
        else:
            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            print(f"✓ LoRA configured (r={self.config.lora_r})")
        
        # Apply PEFT
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
    
    def _get_target_modules(self) -> List[str]:
        """Get target modules based on model architecture."""
        model_type = self.model.config.model_type.lower()
        
        # Architecture-specific module targeting
        if "llama" in model_type:
            # Target all linear layers for Llama models
            return ["q_proj", "v_proj", "k_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"]
        elif "mistral" in model_type:
            # Focus on attention layers for Mistral
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif "gpt" in model_type:
            return ["c_attn", "c_proj", "c_fc"]
        elif "phi" in model_type:
            return ["q_proj", "v_proj", "k_proj", "dense", "fc1", "fc2"]
        else:
            # Default to attention modules
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    def generate_optimized(
        self,
        input_text: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        batch_size: int = 1,
        **kwargs
    ) -> Dict:
        """Generate text with optimized settings."""
        # Tokenize input
        inputs = self.tokenizer(
            [input_text] * batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_cache_length
        ).to(self.model.device)
        
        # Configure generation
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "top_p": top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **kwargs
        }
        
        # Use mixed precision context if enabled
        if self.config.mixed_precision == "bf16":
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model.generate(**inputs, **gen_kwargs)
        elif self.config.mixed_precision == "fp16":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = self.model.generate(**inputs, **gen_kwargs)
        else:
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode outputs
        generated_texts = self.tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return {
            "generated_texts": generated_texts,
            "input_length": inputs.input_ids.shape[1],
            "output_length": outputs.shape[1]
        }
    
    def prepare_training_args(self) -> TrainingArguments:
        """Create optimized training arguments."""
        return TrainingArguments(
            output_dir="./checkpoints",
            per_device_train_batch_size=1,  # Small due to large models
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=self.config.gradient_checkpointing,
            bf16=self.config.mixed_precision == "bf16",
            fp16=self.config.mixed_precision == "fp16",
            optim="paged_adamw_8bit" if self.config.use_4bit else "adamw_torch",
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            eval_strategy="steps",
            eval_steps=50,
            max_grad_norm=1.0,
            warmup_steps=100,
            group_by_length=True,  # Dynamic batching
            ddp_find_unused_parameters=False,
            report_to=["tensorboard"]
        )
    
    def batch_generate_with_sorting(
        self,
        texts: List[str],
        max_new_tokens: int = 512,
        **kwargs
    ) -> List[Dict]:
        """Generate for batch with sequence length sorting for efficiency."""
        # Sort by length for efficient batching
        indexed_texts = [(i, text) for i, text in enumerate(texts)]
        sorted_texts = sorted(indexed_texts, key=lambda x: len(x[1]))
        
        results = []
        batch_size = 4  # Adjust based on available memory
        
        for i in range(0, len(sorted_texts), batch_size):
            batch = sorted_texts[i:i+batch_size]
            batch_texts = [text for _, text in batch]
            
            # Generate for batch
            batch_results = self.generate_optimized(
                batch_texts[0] if len(batch_texts) == 1 else batch_texts,
                max_new_tokens=max_new_tokens,
                batch_size=len(batch_texts),
                **kwargs
            )
            
            # Store results with original indices
            for j, (orig_idx, _) in enumerate(batch):
                results.append((orig_idx, batch_results["generated_texts"][j]))
        
        # Sort back to original order
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]
    
    def save_model(self, save_path: str):
        """Save model with optimizations preserved."""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save config
        import json
        config_path = f"{save_path}/optimization_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        print(f"✓ Model saved to {save_path}")
    
    def cleanup(self):
        """Clean up resources."""
        self.memory_monitor.optimize_memory()
        print("✓ Memory cleaned up")