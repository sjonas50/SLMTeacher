"""
KL Divergence Calculator for RLT Reward System

This module implements the KL divergence computation (rKL) that measures
how much the generated reasoning deviates from a baseline distribution.
This helps maintain naturalness and prevents the model from gaming the reward.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from abc import ABC, abstractmethod


class BaseKLCalculator(ABC):
    """Abstract base class for KL divergence calculators."""
    
    @abstractmethod
    def compute_batch(
        self,
        generated_reasoning: List[str],
        reference_reasoning: List[str],
        problems: List[str]
    ) -> np.ndarray:
        """
        Compute KL divergence between generated and reference distributions.
        
        Args:
            generated_reasoning: List of generated reasoning texts
            reference_reasoning: List of reference reasoning texts
            problems: List of problem statements
            
        Returns:
            Array of KL divergence scores
        """
        pass


class TransformerKLCalculator(BaseKLCalculator):
    """
    KL divergence calculator using transformer language models.
    
    This calculator computes KL divergence between the distributions
    produced by a reference model and the generated reasoning.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2-medium",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512,
        use_fp16: bool = True,
        chunk_size: int = 100
    ):
        """
        Initialize KL calculator with transformer model.
        
        Args:
            model_name: HuggingFace model identifier for reference model
            device: Device to run model on
            max_length: Maximum sequence length
            use_fp16: Whether to use half precision
            chunk_size: Size of chunks for processing long sequences
        """
        self.device = device
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
        
        # Load model and tokenizer
        self.logger.info(f"Loading reference model for KL: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        if use_fp16 and device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(device)
        
        self.model.eval()
    
    def compute_batch(
        self,
        generated_reasoning: List[str],
        reference_reasoning: List[str],
        problems: List[str]
    ) -> np.ndarray:
        """
        Compute KL divergence for a batch of reasoning pairs.
        
        KL(P||Q) = sum(P(x) * log(P(x)/Q(x)))
        where P is generated distribution, Q is reference distribution
        """
        batch_size = len(generated_reasoning)
        kl_scores = []
        
        for i in range(batch_size):
            # Format context with problem
            context = f"Problem: {problems[i]}\n\nReasoning: "
            
            # Compute KL divergence for this pair
            kl_div = self._compute_kl_single(
                context=context,
                generated_text=generated_reasoning[i],
                reference_text=reference_reasoning[i]
            )
            
            kl_scores.append(kl_div)
        
        return np.array(kl_scores)
    
    def _compute_kl_single(
        self,
        context: str,
        generated_text: str,
        reference_text: str
    ) -> float:
        """
        Compute KL divergence for a single text pair.
        
        This computes token-level KL divergence between the distributions
        produced when generating the two texts.
        """
        # Tokenize inputs
        context_tokens = self.tokenizer(
            context,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length // 2
        ).to(self.device)
        
        generated_full = context + generated_text
        reference_full = context + reference_text
        
        # Get token distributions
        gen_logits = self._get_token_logits(generated_full, len(context_tokens.input_ids[0]))
        ref_logits = self._get_token_logits(reference_full, len(context_tokens.input_ids[0]))
        
        # Ensure same length (truncate to shorter)
        min_len = min(gen_logits.shape[0], ref_logits.shape[0])
        gen_logits = gen_logits[:min_len]
        ref_logits = ref_logits[:min_len]
        
        # Compute KL divergence
        # Convert logits to log probabilities
        gen_log_probs = F.log_softmax(gen_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        
        # Compute KL divergence: KL(P||Q) = sum(P * (log P - log Q))
        gen_probs = gen_log_probs.exp()
        kl_div = (gen_probs * (gen_log_probs - ref_log_probs)).sum(dim=-1)

        # Clamp to avoid negative KL from numerical errors
        kl_div = kl_div.clamp(min=0.0)

        # Average over tokens
        avg_kl = kl_div.mean().item()

        # Guard against NaN/Inf
        if not np.isfinite(avg_kl):
            self.logger.warning("Non-finite KL divergence detected, returning 0.0")
            return 0.0

        return avg_kl
    
    def _get_token_logits(self, text: str, context_length: int) -> torch.Tensor:
        """Get logits for tokens after context."""
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits[0]  # Remove batch dimension
            
            # Get logits for generated tokens (after context)
            if context_length > 0:
                logits = logits[context_length-1:-1]  # Shift for next token prediction
            
            return logits


class SequenceKLCalculator(BaseKLCalculator):
    """
    KL divergence calculator at the sequence level.
    
    This calculator treats entire sequences as samples and computes
    KL divergence between sequence distributions.
    """
    
    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        num_samples: int = 100,
        temperature: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize sequence-level KL calculator.
        
        Args:
            base_model: Base model for computing distributions
            tokenizer: Tokenizer for the model
            num_samples: Number of samples for Monte Carlo estimation
            temperature: Temperature for sampling
            device: Device to run computations on
        """
        self.model = base_model
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.temperature = temperature
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        self.model.to(device)
        self.model.eval()
    
    def compute_batch(
        self,
        generated_reasoning: List[str],
        reference_reasoning: List[str],
        problems: List[str]
    ) -> np.ndarray:
        """
        Compute sequence-level KL divergence.
        
        This uses importance sampling to estimate KL divergence between
        sequence distributions.
        """
        batch_size = len(generated_reasoning)
        kl_scores = []
        
        for i in range(batch_size):
            context = f"Problem: {problems[i]}\n\nReasoning: "
            
            # Compute sequence probabilities
            gen_log_prob = self._get_sequence_log_prob(
                context + generated_reasoning[i]
            )
            ref_log_prob = self._get_sequence_log_prob(
                context + reference_reasoning[i]
            )
            
            # Simple approximation: |log P(gen) - log P(ref)|
            # This is related to KL divergence but not exact
            kl_approx = abs(gen_log_prob - ref_log_prob)
            
            kl_scores.append(kl_approx)
        
        return np.array(kl_scores)
    
    def _get_sequence_log_prob(self, text: str) -> float:
        """Compute log probability of entire sequence."""
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = encoding.input_ids[..., 1:].contiguous()
            
            # Compute log probabilities
            log_probs = F.log_softmax(shift_logits, dim=-1)
            
            # Gather log probs of actual tokens
            token_log_probs = log_probs.gather(
                dim=-1,
                index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Sum log probabilities
            total_log_prob = token_log_probs.sum().item()
            
            return total_log_prob


class CachedKLCalculator(BaseKLCalculator):
    """
    KL calculator with caching for efficiency.
    
    This wrapper adds caching to avoid recomputing KL divergences
    for repeated text pairs.
    """
    
    def __init__(
        self,
        base_calculator: BaseKLCalculator,
        cache_size: int = 10000
    ):
        """
        Initialize cached KL calculator.
        
        Args:
            base_calculator: Underlying KL calculator
            cache_size: Maximum number of cached entries
        """
        self.base_calculator = base_calculator
        self.cache_size = cache_size
        self.cache: Dict[str, float] = {}
        self.logger = logging.getLogger(__name__)
    
    def compute_batch(
        self,
        generated_reasoning: List[str],
        reference_reasoning: List[str],
        problems: List[str]
    ) -> np.ndarray:
        """Compute KL divergence with caching."""
        kl_scores = []
        
        for gen, ref, prob in zip(generated_reasoning, reference_reasoning, problems):
            # Create cache key
            cache_key = f"{prob}||{gen}||{ref}"
            
            if cache_key in self.cache:
                kl_scores.append(self.cache[cache_key])
                self.logger.debug("Cache hit for KL computation")
            else:
                # Compute KL divergence
                kl_div = self.base_calculator.compute_batch(
                    [gen], [ref], [prob]
                )[0]
                
                # Update cache
                self.cache[cache_key] = kl_div
                kl_scores.append(kl_div)
                
                # Evict old entries if cache is full
                if len(self.cache) > self.cache_size:
                    # Remove oldest entry (simple FIFO)
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
        
        return np.array(kl_scores)
    
    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()
        self.logger.info("Cleared KL cache")


class ApproximateKLCalculator(BaseKLCalculator):
    """
    Fast approximate KL divergence calculator.
    
    This calculator uses various approximations to speed up KL computation
    while maintaining reasonable accuracy.
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize approximate KL calculator.
        
        Args:
            embedding_model: Model for computing embeddings
            device: Device to run computations on
        """
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Load sentence transformer for embeddings
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_model.to(device)
        except ImportError:
            self.logger.warning("sentence-transformers not installed, using simple approximation")
            self.embedding_model = None
    
    def compute_batch(
        self,
        generated_reasoning: List[str],
        reference_reasoning: List[str],
        problems: List[str]
    ) -> np.ndarray:
        """
        Compute approximate KL divergence using embeddings.
        
        This uses cosine distance in embedding space as a proxy for KL divergence.
        """
        if self.embedding_model is None:
            # Fallback to simple text distance
            return self._simple_text_distance(
                generated_reasoning,
                reference_reasoning
            )
        
        # Compute embeddings
        gen_embeddings = self.embedding_model.encode(
            generated_reasoning,
            convert_to_tensor=True,
            device=self.device
        )
        ref_embeddings = self.embedding_model.encode(
            reference_reasoning,
            convert_to_tensor=True,
            device=self.device
        )
        
        # Compute cosine similarity and map to [0, 1] range
        cos_sim = F.cosine_similarity(gen_embeddings, ref_embeddings)
        # Cosine similarity can be negative; shift to [0, 1] for log safety
        similarity = (cos_sim + 1.0) / 2.0

        # Convert to KL-like score (higher distance = higher KL)
        kl_approx = -torch.log(similarity.clamp(min=1e-6))

        return kl_approx.cpu().numpy()
    
    def _simple_text_distance(
        self,
        generated: List[str],
        reference: List[str]
    ) -> np.ndarray:
        """Simple text-based distance as KL approximation."""
        distances = []
        
        for gen, ref in zip(generated, reference):
            # Normalized edit distance
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, gen, ref).ratio()
            
            # Convert to KL-like score
            distance = -np.log(max(similarity, 1e-6))
            distances.append(distance)

        return np.array(distances)


# Compatibility alias used by reward_function.py
KLDivergenceCalculator = BaseKLCalculator