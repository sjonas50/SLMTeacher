"""
Student Evaluator for Computing Solution Scores (rSS)

This module implements the solution score computation that measures how well
a student model can understand and solve problems given the teacher's reasoning.
Supports both local models and API-based models (e.g., OpenAI, Anthropic).
"""

import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import asyncio
import aiohttp
import json
import logging
from concurrent.futures import ThreadPoolExecutor
import re


class BaseStudentEvaluator(ABC):
    """Abstract base class for student evaluators."""
    
    @abstractmethod
    def evaluate_batch(
        self,
        reasoning: List[str],
        problems: List[str]
    ) -> np.ndarray:
        """
        Evaluate student understanding given reasoning and problems.
        
        Args:
            reasoning: List of teacher reasoning texts
            problems: List of problem statements
            
        Returns:
            Array of solution scores
        """
        pass
    
    @abstractmethod
    def get_log_probabilities(
        self,
        inputs: List[str],
        targets: List[str]
    ) -> np.ndarray:
        """
        Get log probabilities of targets given inputs.
        
        Args:
            inputs: List of input texts
            targets: List of target texts
            
        Returns:
            Array of log probabilities
        """
        pass


class LocalStudentEvaluator(BaseStudentEvaluator):
    """
    Student evaluator using local transformer models.

    This evaluator uses a smaller model (student) to assess how well
    it can solve problems given the teacher's reasoning.
    """

    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        device: Optional[str] = None,
        max_length: int = 512,
        temperature: float = 0.1,
        use_fp16: bool = True,
        batch_size: int = 8,  # accepted for compatibility
    ):
        """
        Initialize local student evaluator.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on
            max_length: Maximum sequence length
            temperature: Temperature for probability computation
            use_fp16: Whether to use half precision
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)

        # Load model and tokenizer
        self.logger.info(f"Loading student model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with appropriate precision
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
    
    def evaluate_batch(
        self,
        reasoning: List[str],
        problems: List[str]
    ) -> np.ndarray:
        """
        Evaluate student understanding using log probability of correct solutions.
        
        The solution score measures how likely the student is to produce
        the correct answer given the teacher's reasoning.
        """
        batch_size = len(reasoning)
        scores = []
        
        for i in range(batch_size):
            # Format input: problem + reasoning
            input_text = f"Problem: {problems[i]}\n\nReasoning: {reasoning[i]}\n\nSolution:"
            
            # Generate solution and compute confidence
            with torch.no_grad():
                # Tokenize input
                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True
                ).to(self.device)
                
                # Generate solution
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=self.temperature,
                    do_sample=True,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                
                # Compute average log probability of generated tokens
                if outputs.scores:
                    log_probs = []
                    for j, score in enumerate(outputs.scores):
                        # Get the generated token at this position
                        token_id = outputs.sequences[0, inputs.input_ids.shape[1] + j]
                        # Get log probability of this token
                        log_prob = F.log_softmax(score[0] / self.temperature, dim=-1)
                        log_probs.append(log_prob[token_id].item())
                    
                    # Average log probability as confidence score
                    avg_log_prob = np.mean(log_probs) if log_probs else -10.0
                    scores.append(avg_log_prob)
                else:
                    scores.append(-10.0)  # Default low score
        
        return np.array(scores)
    
    def get_log_probabilities(
        self,
        inputs: List[str],
        targets: List[str]
    ) -> np.ndarray:
        """
        Compute log probabilities of target sequences given inputs.
        """
        log_probs = []
        
        for input_text, target_text in zip(inputs, targets):
            # Combine input and target
            full_text = input_text + target_text
            
            # Tokenize
            encoding = self.tokenizer(
                full_text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True
            ).to(self.device)
            
            # Get input length to identify target tokens
            input_encoding = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True
            )
            input_length = input_encoding.input_ids.shape[1]
            
            with torch.no_grad():
                outputs = self.model(**encoding)
                logits = outputs.logits
                
                # Compute log probabilities for target tokens
                target_logits = logits[0, input_length-1:-1]  # Shift by 1 for next token prediction
                target_ids = encoding.input_ids[0, input_length:]
                
                # Compute log probabilities
                log_probs_seq = F.log_softmax(target_logits, dim=-1)
                token_log_probs = log_probs_seq.gather(1, target_ids.unsqueeze(1)).squeeze()
                
                # Average log probability
                avg_log_prob = token_log_probs.mean().item()
                log_probs.append(avg_log_prob)
        
        return np.array(log_probs)


class APIStudentEvaluator(BaseStudentEvaluator):
    """
    Student evaluator using API-based models (OpenAI, Anthropic, etc.).
    
    This evaluator uses API calls to assess student understanding.
    """
    
    def __init__(
        self,
        api_type: str = "openai",
        api_key: str = None,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        max_workers: int = 10
    ):
        """
        Initialize API-based student evaluator.
        
        Args:
            api_type: Type of API ("openai", "anthropic", "custom")
            api_key: API key for authentication
            model_name: Model identifier
            temperature: Temperature for generation
            max_workers: Maximum concurrent API calls
        """
        self.api_type = api_type
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        
        # API endpoints
        self.api_endpoints = {
            "openai": "https://api.openai.com/v1/chat/completions",
            "anthropic": "https://api.anthropic.com/v1/messages"
        }
    
    async def _call_api(
        self,
        session: aiohttp.ClientSession,
        prompt: str
    ) -> Dict:
        """Make async API call."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        if self.api_type == "openai":
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a student trying to solve problems."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "logprobs": True,
                "top_logprobs": 1
            }
        else:
            # Anthropic format
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature
            }
        
        async with session.post(
            self.api_endpoints[self.api_type],
            headers=headers,
            json=data
        ) as response:
            return await response.json()
    
    async def _evaluate_batch_async(
        self,
        reasoning: List[str],
        problems: List[str]
    ) -> List[float]:
        """Async batch evaluation."""
        prompts = []
        for prob, reason in zip(problems, reasoning):
            prompt = f"Problem: {prob}\n\nReasoning: {reason}\n\nBased on the reasoning provided, solve the problem step by step."
            prompts.append(prompt)
        
        async with aiohttp.ClientSession() as session:
            tasks = [self._call_api(session, prompt) for prompt in prompts]
            responses = await asyncio.gather(*tasks)
        
        # Extract scores from responses
        scores = []
        for response in responses:
            if self.api_type == "openai" and "choices" in response:
                # Extract log probability if available
                choice = response["choices"][0]
                if "logprobs" in choice and choice["logprobs"]:
                    # Average token log probabilities
                    token_logprobs = choice["logprobs"]["token_logprobs"]
                    avg_logprob = np.mean(token_logprobs) if token_logprobs else -5.0
                    scores.append(avg_logprob)
                else:
                    # Use a heuristic based on response quality
                    scores.append(self._score_response_quality(choice["message"]["content"]))
            else:
                # Fallback scoring
                scores.append(-5.0)
        
        return scores
    
    def evaluate_batch(
        self,
        reasoning: List[str],
        problems: List[str]
    ) -> np.ndarray:
        """Evaluate student understanding using API calls."""
        # Run async evaluation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        scores = loop.run_until_complete(
            self._evaluate_batch_async(reasoning, problems)
        )
        loop.close()
        
        return np.array(scores)
    
    def get_log_probabilities(
        self,
        inputs: List[str],
        targets: List[str]
    ) -> np.ndarray:
        """Get log probabilities using API (if supported)."""
        # Note: This is approximate for APIs that don't return log probs
        self.logger.warning("Exact log probabilities not available for API models")
        
        # Use evaluation as proxy
        combined = [f"{inp}\n{tgt}" for inp, tgt in zip(inputs, targets)]
        scores = self.evaluate_batch(combined, [""] * len(combined))
        
        return scores
    
    def _score_response_quality(self, response: str) -> float:
        """
        Heuristic scoring of response quality when log probs not available.
        """
        score = 0.0
        
        # Check for mathematical expressions
        if re.search(r'\d+\s*[+\-*/]\s*\d+', response):
            score += 1.0
        
        # Check for step-by-step reasoning
        if any(marker in response.lower() for marker in ['step', 'first', 'then', 'finally']):
            score += 1.0
        
        # Check for conclusion
        if any(marker in response.lower() for marker in ['therefore', 'answer', 'result']):
            score += 1.0
        
        # Normalize to log probability scale
        return -5.0 + score  # Maps to [-5, -2] range


class EnsembleStudentEvaluator(BaseStudentEvaluator):
    """
    Ensemble of multiple student evaluators for robust scoring.
    """
    
    def __init__(
        self,
        evaluators: List[BaseStudentEvaluator],
        weights: Optional[List[float]] = None,
        aggregation: str = "mean"
    ):
        """
        Initialize ensemble evaluator.
        
        Args:
            evaluators: List of student evaluators
            weights: Optional weights for weighted averaging
            aggregation: Aggregation method ("mean", "max", "weighted")
        """
        self.evaluators = evaluators
        self.weights = weights or [1.0 / len(evaluators)] * len(evaluators)
        self.aggregation = aggregation
        
        assert len(self.weights) == len(self.evaluators)
        assert sum(self.weights) > 0
        
        # Normalize weights
        self.weights = [w / sum(self.weights) for w in self.weights]
    
    def evaluate_batch(
        self,
        reasoning: List[str],
        problems: List[str]
    ) -> np.ndarray:
        """Evaluate using ensemble of models."""
        all_scores = []
        
        for evaluator in self.evaluators:
            scores = evaluator.evaluate_batch(reasoning, problems)
            all_scores.append(scores)
        
        all_scores = np.array(all_scores)  # Shape: (n_evaluators, batch_size)
        
        if self.aggregation == "mean":
            return np.mean(all_scores, axis=0)
        elif self.aggregation == "max":
            return np.max(all_scores, axis=0)
        elif self.aggregation == "weighted":
            weighted_scores = all_scores * np.array(self.weights).reshape(-1, 1)
            return np.sum(weighted_scores, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
    
    def get_log_probabilities(
        self,
        inputs: List[str],
        targets: List[str]
    ) -> np.ndarray:
        """Get ensemble log probabilities."""
        all_log_probs = []
        
        for evaluator in self.evaluators:
            log_probs = evaluator.get_log_probabilities(inputs, targets)
            all_log_probs.append(log_probs)
        
        all_log_probs = np.array(all_log_probs)
        
        # Log-sum-exp trick for numerical stability
        if self.aggregation == "weighted":
            # Weighted log probability: log(sum(w_i * exp(log_p_i)))
            weighted_log_probs = all_log_probs + np.log(self.weights).reshape(-1, 1)
            return np.logaddexp.reduce(weighted_log_probs, axis=0)
        else:
            # Simple average in log space
            return np.logaddexp.reduce(all_log_probs, axis=0) - np.log(len(self.evaluators))


# Compatibility alias used by training scripts
StudentEvaluator = LocalStudentEvaluator