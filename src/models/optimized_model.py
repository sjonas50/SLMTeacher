"""
Optimized HuggingFace Model with Flash Attention 2, QLoRA, and Advanced Memory Management
"""
import gc
import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    AdaLoraConfig,
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

logger = logging.getLogger(__name__)


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
            for i in range(torch.cuda.device_count()):
                mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                mem_total = torch.cuda.get_device_properties(i).total_mem / (1024**3)
                mem_free = mem_total - mem_allocated
                stats[f'gpu_{i}_memory_used_gb'] = round(mem_allocated, 2)
                stats[f'gpu_{i}_memory_reserved_gb'] = round(mem_reserved, 2)
                stats[f'gpu_{i}_memory_free_gb'] = round(mem_free, 2)
                stats[f'gpu_{i}_memory_percent'] = round((mem_allocated / mem_total) * 100, 1) if mem_total > 0 else 0
        elif torch.backends.mps.is_available():
            mem_allocated = torch.mps.current_allocated_memory() / (1024**3)
            stats['mps_memory_used_gb'] = round(mem_allocated, 2)

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
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Auto-disable CUDA-only features on non-CUDA devices
        if not torch.cuda.is_available():
            if config.use_4bit or config.use_8bit:
                logger.info("Quantization requires CUDA — disabling (using fp32)")
                config.use_4bit = False
                config.use_8bit = False
            if config.device_map == "auto":
                config.device_map = None  # explicit placement instead

        # Check memory before loading
        self.memory_monitor = MemoryMonitor()
        initial_stats = self.memory_monitor.get_memory_stats()
        logger.info("Initial memory stats: %s", initial_stats)

        # Estimate memory requirements
        estimated_memory = self.memory_monitor.estimate_model_memory(
            config.model_name,
            config.use_4bit
        )
        logger.info("Estimated model memory: %.2f GB", estimated_memory)

        # Load model with optimizations
        self._load_model()

        # Setup PEFT if requested
        if config.use_peft:
            self._setup_peft()

        # Final memory stats
        final_stats = self.memory_monitor.get_memory_stats()
        logger.info("Final memory stats: %s", final_stats)
    
    def _load_model(self):
        """Load model with all optimizations."""
        # Setup quantization config
        quantization_config = None
        if self.config.use_4bit and torch.cuda.is_available():
            valid_dtypes = {"float16", "bfloat16", "float32"}
            if self.config.bnb_4bit_compute_dtype not in valid_dtypes:
                raise ValueError(
                    f"bnb_4bit_compute_dtype must be one of {valid_dtypes}, "
                    f"got '{self.config.bnb_4bit_compute_dtype}'"
                )
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
        
        # use_cache must be False when gradient checkpointing is enabled
        use_cache = self.config.use_cache and not self.config.gradient_checkpointing

        # Model loading kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "use_cache": use_cache,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        if self.config.device_map is not None:
            model_kwargs["device_map"] = self.config.device_map
        
        # Add Flash Attention 2 if available (CUDA only)
        if self.config.use_flash_attention_2 and torch.cuda.is_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Flash Attention 2 enabled")
        elif self.config.use_flash_attention_2:
            warnings.warn("Flash Attention 2 requires CUDA; using sdpa attention")
            model_kwargs["attn_implementation"] = "sdpa"
        
        # Set max memory if specified
        if self.config.max_memory:
            model_kwargs["max_memory"] = self.config.max_memory
        
        # Load model
        logger.info("Loading model: %s", self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )

        # Move to device when not using device_map (MPS / CPU)
        if self.config.device_map is None:
            logger.info("Moving model to %s", self.device)
            self.model.to(self.device)

        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
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
            try:
                peft_config = AdaLoraConfig(
                    init_r=self.config.adalora_init_r,
                    target_r=self.config.adalora_target_r,
                    tinit=self.config.adalora_tinit,
                    tfinal=self.config.adalora_tfinal,
                    deltaT=self.config.adalora_delta_t,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=target_modules,
                    task_type=TaskType.CAUSAL_LM,
                    total_step=self.config.adalora_tfinal,
                )
                logger.info("AdaLoRA configured (init_r=%d, target_r=%d)",
                            self.config.adalora_init_r, self.config.adalora_target_r)
            except (TypeError, ValueError) as e:
                logger.warning("AdaLoRA init failed (%s), falling back to LoRA", e)
                self.config.peft_method = "lora"

        if self.config.peft_method != "adalora":
            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            logger.info("LoRA configured (r=%d)", self.config.lora_r)
        
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
        device_type = "cuda" if self.device.type == "cuda" else self.device.type
        if self.config.mixed_precision == "bf16" and device_type in ("cuda", "cpu"):
            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                outputs = self.model.generate(**inputs, **gen_kwargs)
        elif self.config.mixed_precision == "fp16" and device_type == "cuda":
            with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
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

    def _get_prefix_len(self, input_text: str, full_text: str) -> int:
        """Compute prefix token length using consistent tokenization.

        Tokenizes ``input_text`` with the same settings as the full sequence
        so that BPE merges are identical up to the boundary.
        """
        prefix_ids = self.tokenizer(
            input_text, return_tensors="pt", truncation=True,
            max_length=self.tokenizer.model_max_length,
            add_special_tokens=True,
        ).input_ids
        return prefix_ids.shape[1]

    def compute_loss(self, input_text: str, target_text: str) -> torch.Tensor:
        """Compute cross-entropy loss for target_text given input_text as prefix.

        Tokenizes the concatenation of input_text and target_text, masks the
        prefix tokens with -100 so that only the target portion contributes to
        the loss, then runs a forward pass and returns ``outputs.loss``.
        """
        full_text = input_text + target_text
        encoding = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            add_special_tokens=True,
        ).to(self.device)

        input_ids = encoding.input_ids
        labels = input_ids.clone()
        prefix_len = min(self._get_prefix_len(input_text, full_text), labels.shape[1])
        labels[:, :prefix_len] = -100

        outputs = self.model(input_ids=input_ids, labels=labels)
        return outputs.loss

    def compute_log_probs(self, input_text: str, target_text: str) -> torch.Tensor:
        """Compute sum of log probabilities of target tokens given input prefix.

        Returns a scalar tensor with gradients enabled, suitable for use in
        policy-gradient objectives (e.g. GRPO / PPO).  Returns the **sum**
        (not mean) so that PPO ratios are not biased by answer length.
        """
        full_text = input_text + target_text
        encoding = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            add_special_tokens=True,
        ).to(self.device)

        input_ids = encoding.input_ids  # (1, seq_len)
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

        # Shift: predict token t+1 from position t
        shift_logits = logits[:, :-1, :]  # (1, seq_len-1, vocab)
        shift_labels = input_ids[:, 1:]   # (1, seq_len-1)

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # (1, seq_len-1)

        # Target tokens: in the shifted sequence, the first target token to
        # predict is at position (prefix_len - 1) because shift moves left by 1
        prefix_len = self._get_prefix_len(input_text, full_text)
        target_start = max(0, min(prefix_len - 1, token_log_probs.shape[1]))
        target_log_probs = token_log_probs[:, target_start:]

        if target_log_probs.numel() == 0:
            return torch.tensor(-100.0, device=self.device, requires_grad=True)

        return target_log_probs.sum()

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
        
        logger.info("Model saved to %s", save_path)

    def cleanup(self):
        """Clean up resources."""
        self.memory_monitor.optimize_memory()
        logger.info("Memory cleaned up")