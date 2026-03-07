"""
Hugging Face Teacher Model for RLT Training
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

logger = logging.getLogger(__name__)


class HFTeacherModel:
    """Trainable Hugging Face model for RLT teacher role."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        use_lora: bool = True,
        use_8bit: bool = True,
        lora_config: Optional[Dict] = None,
        device_map: str = "auto"
    ):
        self.model_name = model_name
        self.use_lora = use_lora
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Setup quantization for memory efficiency
        bnb_config = None
        if use_8bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
        
        # Load model and tokenizer
        logger.info("Loading teacher model: %s", model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup LoRA if requested
        if use_lora:
            self.setup_lora(lora_config)
        
        # Teacher prompt template
        self.teacher_prompt = """You are an expert teacher. Given a question and its answer, provide a clear step-by-step explanation.

Question: {question}
Answer: {answer}

Explanation:"""
    
    def setup_lora(self, lora_config: Optional[Dict] = None):
        """Configure LoRA for efficient training."""
        if lora_config is None:
            lora_config = {
                "r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": TaskType.CAUSAL_LM
            }
        
        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Get target modules based on model architecture
        target_modules = self._get_target_modules()
        
        # Create LoRA config
        peft_config = LoraConfig(
            **lora_config,
            target_modules=target_modules
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
    
    def _get_target_modules(self) -> List[str]:
        """Get target modules based on model architecture."""
        model_type = self.model.config.model_type.lower()
        
        if "llama" in model_type:
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif "gpt" in model_type:
            return ["c_attn", "c_proj"]
        elif "phi" in model_type:
            return ["q_proj", "v_proj", "k_proj", "dense"]
        else:
            # Default to common attention modules
            return ["q_proj", "v_proj"]
    
    def generate_explanation(
        self,
        question: str,
        answer: str,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        return_logprobs: bool = True
    ) -> Dict:
        """Generate explanation with optional log probabilities."""
        # Format prompt
        prompt = self.teacher_prompt.format(
            question=question,
            answer=answer
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.model.device)
        
        # Generate
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=return_logprobs
            )
        
        # Extract explanation
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        explanation = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        result = {
            'explanation': explanation,
            'generated_ids': generated_ids
        }
        
        # Compute log probabilities if requested
        if return_logprobs and outputs.scores:
            logprobs = self._compute_logprobs(outputs.scores, generated_ids)
            result['logprobs'] = logprobs
        
        return result
    
    def _compute_logprobs(
        self, 
        scores: Tuple[torch.Tensor], 
        generated_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probabilities from generation scores."""
        logprobs = []
        
        for i, score in enumerate(scores):
            # Get log probabilities
            log_probs = torch.log_softmax(score, dim=-1)
            # Get probability of generated token
            token_id = generated_ids[i]
            token_logprob = log_probs[0, token_id].item()
            logprobs.append(token_logprob)
        
        return torch.tensor(logprobs)
    
    def generate_batch_explanations(
        self,
        questions: List[str],
        answers: List[str],
        temperatures: Optional[List[float]] = None,
        **kwargs
    ) -> List[Dict]:
        """Generate explanations for a batch of questions."""
        if temperatures is None:
            temperatures = [0.7] * len(questions)
        
        results = []
        for q, a, t in zip(questions, answers, temperatures):
            result = self.generate_explanation(q, a, t, **kwargs)
            results.append(result)
        
        return results
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute language modeling loss for training."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss
    
    def save_model(self, save_path: str):
        """Save model and tokenizer."""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    def load_model(self, load_path: str):
        """Load model from checkpoint."""
        self.model = AutoModelForCausalLM.from_pretrained(
            load_path,
            device_map=self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)