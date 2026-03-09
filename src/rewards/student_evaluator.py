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
        problems: List[str],
        reference_answers: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Evaluate student understanding given reasoning and problems.

        Args:
            reasoning: List of teacher reasoning texts
            problems: List of problem statements
            reference_answers: If provided, measure correctness against these
                answers instead of using confidence-based scoring.

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
        problems: List[str],
        reference_answers: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Evaluate student understanding.

        When ``reference_answers`` is provided the student generates a greedy
        answer and correctness is checked against the reference (0.0 / 0.5 / 1.0).
        Otherwise falls back to confidence-based scoring (average log-prob of
        generated tokens).
        """
        if reference_answers is not None:
            return self._evaluate_correctness_batch(reasoning, problems, reference_answers)

        batch_size = len(reasoning)
        scores = []

        for i in range(batch_size):
            input_text = f"Problem: {problems[i]}\n\nReasoning: {reasoning[i]}\n\nSolution:"

            with torch.no_grad():
                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True
                ).to(self.device)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=self.temperature,
                    do_sample=True,
                    output_scores=True,
                    return_dict_in_generate=True
                )

                if outputs.scores:
                    log_probs = []
                    for j, score in enumerate(outputs.scores):
                        token_id = outputs.sequences[0, inputs.input_ids.shape[1] + j]
                        log_prob = F.log_softmax(score[0] / self.temperature, dim=-1)
                        log_probs.append(log_prob[token_id].item())
                    avg_log_prob = np.mean(log_probs) if log_probs else -10.0
                    scores.append(avg_log_prob)
                else:
                    scores.append(-10.0)

        return np.array(scores)

    def _evaluate_correctness_batch(
        self,
        reasoning: List[str],
        problems: List[str],
        reference_answers: List[str],
    ) -> np.ndarray:
        """Generate answers greedily and score against reference answers."""
        scores = []

        for i in range(len(reasoning)):
            input_text = f"Problem: {problems[i]}\n\nReasoning: {reasoning[i]}\n\nSolution:"

            with torch.no_grad():
                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                ).to(self.device)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                )
                generated = self.tokenizer.decode(
                    outputs[0, inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )

            score = self._compare_answers(generated, reference_answers[i])
            scores.append(score)

        return np.array(scores)

    @staticmethod
    def _compare_answers(generated: str, reference: str) -> float:
        """Compare a generated answer to a reference answer.

        Returns a continuous score in [0, 1] to provide a dense reward signal:
            1.0  — exact or numerical match
            0.9  — within 1% relative error
            0.7  — within 5% relative error
            0.5  — within 10% relative error
            0.3  — within 25% relative error
            0–0.5 — token overlap for non-numeric answers
            0.0  — no match
        """
        gen_clean = generated.strip().lower()
        ref_clean = reference.strip().lower()

        if gen_clean == ref_clean:
            return 1.0

        # Numerical comparison (for math / quantitative answers)
        gen_nums = re.findall(r'-?\d+\.?\d*', gen_clean)
        ref_nums = re.findall(r'-?\d+\.?\d*', ref_clean)

        if gen_nums and ref_nums:
            try:
                gen_val = float(gen_nums[-1])
                ref_val = float(ref_nums[-1])
                if gen_val == ref_val:
                    return 1.0
                if ref_val != 0:
                    relative_error = abs(gen_val - ref_val) / abs(ref_val)
                    if relative_error <= 0.01:
                        return 0.9
                    elif relative_error <= 0.05:
                        return 0.7
                    elif relative_error <= 0.10:
                        return 0.5
                    elif relative_error <= 0.25:
                        return 0.3
            except ValueError:
                pass

        # Token overlap for non-numeric answers
        gen_tokens = set(gen_clean.split())
        ref_tokens = set(ref_clean.split())
        if ref_tokens:
            overlap = len(gen_tokens & ref_tokens) / len(ref_tokens)
            return min(overlap * 0.5, 0.5)  # Cap at 0.5 for partial token match

        return 0.0
    
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
        problems: List[str],
        reference_answers: Optional[List[str]] = None,
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
        problems: List[str],
        reference_answers: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Evaluate using ensemble of models."""
        all_scores = []

        for evaluator in self.evaluators:
            scores = evaluator.evaluate_batch(reasoning, problems, reference_answers)
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


class SharedModelEvaluator(BaseStudentEvaluator):
    """Evaluator that reuses the student's OptimizedHFModel.

    Avoids loading a second model into memory — critical on
    memory-constrained devices (e.g. 16 GB Apple Silicon).
    """

    def __init__(self, optimized_model):
        self.model_wrapper = optimized_model
        self.model = optimized_model.model
        self.tokenizer = optimized_model.tokenizer
        self.device = optimized_model.device
        self.logger = logging.getLogger(__name__)

    def evaluate_batch(
        self,
        reasoning: List[str],
        problems: List[str],
        reference_answers: Optional[List[str]] = None,
    ) -> np.ndarray:
        if reference_answers is not None:
            return self._evaluate_correctness(reasoning, problems, reference_answers)
        # Confidence-based fallback
        scores = []
        for r, p in zip(reasoning, problems):
            prompt = f"Problem: {p}\n\nReasoning: {r}\n\nSolution:"
            with torch.no_grad():
                inputs = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True,
                    max_length=self.tokenizer.model_max_length,
                ).to(self.device)
                outputs = self.model.generate(
                    **inputs, max_new_tokens=50, do_sample=False,
                )
                gen_ids = outputs[0, inputs.input_ids.shape[1]:]
                if gen_ids.numel() == 0:
                    scores.append(-10.0)
                    continue
                logits = self.model(outputs).logits
                shift_logits = logits[:, inputs.input_ids.shape[1] - 1:-1, :]
                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_lps = log_probs.gather(
                    dim=-1, index=gen_ids.unsqueeze(0).unsqueeze(-1)
                ).squeeze(-1)
                scores.append(token_lps.mean().item())
        return np.array(scores)

    def _evaluate_correctness(self, reasoning, problems, reference_answers):
        scores = []
        for r, p, ref in zip(reasoning, problems, reference_answers):
            prompt = f"Problem: {p}\n\nReasoning: {r}\n\nSolution:"
            with torch.no_grad():
                inputs = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True,
                    max_length=self.tokenizer.model_max_length,
                ).to(self.device)
                outputs = self.model.generate(
                    **inputs, max_new_tokens=50, do_sample=False,
                )
                generated = self.tokenizer.decode(
                    outputs[0, inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )
            score = LocalStudentEvaluator._compare_answers(generated, ref)
            scores.append(score)
        return np.array(scores)

    def get_log_probabilities(self, inputs: List[str], targets: List[str]) -> np.ndarray:
        log_probs = []
        for inp, tgt in zip(inputs, targets):
            lp = self.model_wrapper.compute_log_probs(inp, tgt)
            log_probs.append(lp.detach().item())
        return np.array(log_probs)


# Compatibility alias used by training scripts
StudentEvaluator = LocalStudentEvaluator