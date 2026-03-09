"""
Claude-based Student Evaluator for High-Quality Assessment

This module uses Claude API to evaluate how well a student model understands
and can apply teacher explanations. Provides much more accurate evaluation
than local models, though at higher API cost.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import json
import os
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

from anthropic import Anthropic
from ..teachers.claude_teacher import ClaudeConfig, RateLimiter, RateLimitConfig
from ..utils.cost_tracker import CostTracker


@dataclass
class ClaudeEvaluatorConfig:
    """Configuration for Claude evaluator."""
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.1  # Low temperature for consistent evaluation
    max_tokens: int = 512
    max_workers: int = 5
    use_cache: bool = True  # Cache evaluation results


class ClaudeStudentEvaluator:
    """
    High-quality student evaluator using Claude API.
    
    This evaluator uses Claude to assess how well a student model
    can solve problems given teacher explanations. Provides more
    accurate reward signals than local model evaluation.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cost_tracker: Optional[CostTracker] = None,
        config: Optional[ClaudeEvaluatorConfig] = None
    ):
        """
        Initialize Claude evaluator.
        
        Args:
            api_key: Claude API key. If None, reads from CLAUDE_API_KEY env var.
            cost_tracker: Optional cost tracker for monitoring API usage.
            config: Configuration for Claude evaluator.
        """
        # API key management
        self.api_key = api_key or os.getenv("CLAUDE_API_KEY")
        if not self.api_key:
            raise ValueError("Claude API key required for evaluator")
        
        # Initialize components
        self.client = Anthropic(api_key=self.api_key)
        self.config = config or ClaudeEvaluatorConfig()
        self.cost_tracker = cost_tracker
        self.logger = logging.getLogger(__name__)
        
        # Cache for evaluation results
        self.cache = {} if self.config.use_cache else None
        
        # Rate limiter (shared with teacher if needed)
        self.rate_limiter = RateLimiter(
            RateLimitConfig(requests_per_minute=50, tokens_per_minute=100000)
        )
        
        # Evaluation prompt template
        self.eval_prompt = """You are an expert evaluator assessing student understanding.

Given a problem, a teacher's explanation, and a student's solution attempt, 
evaluate how well the student understood and applied the teacher's reasoning.

Problem: {problem}

Teacher's Explanation: {explanation}

Student's Solution: {student_solution}

Evaluate the student's solution on these criteria:
1. Correctness: Is the final answer correct?
2. Understanding: Does the student show understanding of the teacher's approach?
3. Application: Did the student properly apply the reasoning steps?
4. Clarity: Is the student's work clear and logical?

Provide a score from 0.0 to 1.0 where:
- 1.0 = Perfect understanding and application
- 0.8-0.9 = Good understanding with minor issues
- 0.6-0.7 = Moderate understanding
- 0.4-0.5 = Poor understanding
- 0.0-0.3 = No understanding / incorrect

Return ONLY a JSON object with this format:
{{"score": 0.X, "correct": true/false, "reasoning": "Brief explanation"}}"""
        
        self.logger.info(f"Claude evaluator initialized with model: {self.config.model}")
    
    def evaluate_batch(
        self,
        problems: List[str],
        explanations: List[str], 
        student_solutions: List[str]
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Evaluate batch of student solutions using Claude.
        
        Args:
            problems: List of problem statements
            explanations: List of teacher explanations
            student_solutions: List of student solution attempts
            
        Returns:
            Tuple of (scores array, detailed evaluations)
        """
        batch_size = len(problems)
        if len(explanations) != batch_size or len(student_solutions) != batch_size:
            raise ValueError("All input lists must have the same length")
        
        scores = []
        evaluations = []
        
        # Process in parallel for efficiency
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            
            for i in range(batch_size):
                future = executor.submit(
                    self._evaluate_single,
                    problems[i],
                    explanations[i],
                    student_solutions[i]
                )
                futures.append((i, future))
            
            # Collect results
            for idx, future in futures:
                try:
                    score, evaluation = future.result()
                    scores.append(score)
                    evaluations.append(evaluation)
                except Exception as e:
                    self.logger.error(f"Evaluation failed for item {idx}: {e}")
                    scores.append(0.0)  # Default to low score on error
                    evaluations.append({
                        "score": 0.0,
                        "error": str(e),
                        "correct": False
                    })
        
        return np.array(scores), evaluations
    
    def _evaluate_single(
        self,
        problem: str,
        explanation: str,
        student_solution: str
    ) -> Tuple[float, Dict]:
        """Evaluate a single student solution."""
        # Check cache
        if self.cache is not None:
            cache_key = self._get_cache_key(problem, explanation, student_solution)
            if cache_key in self.cache:
                cached = self.cache[cache_key]
                return cached["score"], cached["evaluation"]
        
        # Format prompt
        prompt = self.eval_prompt.format(
            problem=problem,
            explanation=explanation,
            student_solution=student_solution
        )
        
        try:
            # Wait for rate limit
            self._wait_for_rate_limit()
            
            # Call Claude API
            response = self.client.messages.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # Parse response
            response_text = response.content[0].text
            evaluation = self._parse_evaluation(response_text)
            score = evaluation.get("score", 0.0)
            
            # Track costs
            if self.cost_tracker:
                self.cost_tracker.add_usage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    metadata={
                        'model': self.config.model,
                        'operation': 'evaluate_student'
                    }
                )
            
            # Cache result
            if self.cache is not None:
                self.cache[cache_key] = {
                    "score": score,
                    "evaluation": evaluation
                }
            
            return score, evaluation
            
        except Exception as e:
            self.logger.error(f"Claude evaluation failed: {e}")
            return 0.0, {"score": 0.0, "error": str(e)}
    
    def _parse_evaluation(self, response_text: str) -> Dict:
        """Parse Claude's evaluation response."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback: try to extract score from text
                score_match = re.search(r'score["\s:]+([0-9.]+)', response_text, re.IGNORECASE)
                if score_match:
                    score = float(score_match.group(1))
                    return {"score": score, "reasoning": response_text}
                else:
                    return {"score": 0.5, "reasoning": "Could not parse evaluation"}
        except Exception as e:
            self.logger.warning(f"Failed to parse evaluation: {e}")
            return {"score": 0.5, "error": str(e)}
    
    def _get_cache_key(self, problem: str, explanation: str, solution: str) -> str:
        """Generate cache key for evaluation."""
        import hashlib
        content = f"{problem}|{explanation}|{solution}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limits."""
        can_proceed, wait_time = self.rate_limiter.check_limits()
        if not can_proceed and wait_time:
            import time
            time.sleep(wait_time)
    
    def evaluate_student_model(
        self,
        student_model,
        test_problems: List[Dict],
        use_explanations: bool = True
    ) -> Dict:
        """
        Evaluate a student model's performance on test problems.
        
        Args:
            student_model: The student model to evaluate
            test_problems: List of dicts with 'question' and 'answer' keys
            use_explanations: Whether to provide teacher explanations
            
        Returns:
            Dictionary with evaluation metrics
        """
        total_score = 0.0
        correct_count = 0
        evaluations = []
        
        for problem_data in test_problems:
            question = problem_data['question']
            correct_answer = problem_data['answer']
            
            # Generate student solution
            if use_explanations and 'explanation' in problem_data:
                # Student uses teacher explanation
                student_input = f"Question: {question}\nExplanation: {problem_data['explanation']}\nAnswer:"
            else:
                # Student solves without explanation
                student_input = f"Question: {question}\nAnswer:"
            
            # Get student's solution
            if hasattr(student_model, 'generate_optimized'):
                result = student_model.generate_optimized(
                    student_input,
                    max_new_tokens=200,
                    temperature=0.1
                )
                student_solution = result['generated_texts'][0]
            else:
                # Fallback for other model types
                student_solution = student_model.generate(
                    student_input,
                    max_new_tokens=200
                )
            
            # Evaluate using Claude
            explanation = problem_data.get('explanation', 'No explanation provided')
            score, evaluation = self._evaluate_single(
                question,
                explanation,
                student_solution
            )
            
            total_score += score
            if evaluation.get('correct', False):
                correct_count += 1
            
            evaluations.append({
                'question': question,
                'student_solution': student_solution,
                'score': score,
                'evaluation': evaluation
            })
        
        # Compute metrics
        num_problems = len(test_problems)
        avg_score = total_score / num_problems if num_problems > 0 else 0.0
        accuracy = correct_count / num_problems if num_problems > 0 else 0.0
        
        return {
            'average_score': avg_score,
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_problems': num_problems,
            'evaluations': evaluations
        }
    
    def get_stats(self) -> Dict:
        """Get evaluator statistics."""
        stats = {
            'model': self.config.model,
            'cache_size': len(self.cache) if self.cache is not None else 0,
            'temperature': self.config.temperature
        }
        
        if self.cost_tracker:
            stats['cost_info'] = self.cost_tracker.get_summary()
        
        return stats


class HybridEvaluator:
    """
    Hybrid evaluator that can use both Claude and local models.
    
    Uses Claude for high-stakes evaluations and local models for
    routine evaluations to balance quality and cost.
    """
    
    def __init__(
        self,
        claude_evaluator: ClaudeStudentEvaluator,
        local_evaluator,
        claude_eval_frequency: float = 0.1  # Use Claude 10% of the time
    ):
        """
        Initialize hybrid evaluator.
        
        Args:
            claude_evaluator: Claude-based evaluator
            local_evaluator: Local model evaluator
            claude_eval_frequency: Fraction of evaluations to use Claude
        """
        self.claude_evaluator = claude_evaluator
        self.local_evaluator = local_evaluator
        self.claude_freq = claude_eval_frequency
        self.eval_count = 0
        
    def evaluate_batch(
        self,
        problems: List[str],
        explanations: List[str],
        student_solutions: List[str],
        force_claude: bool = False
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Evaluate batch using hybrid approach.
        
        Args:
            force_claude: Force Claude evaluation regardless of frequency
        """
        self.eval_count += 1
        
        # Decide which evaluator to use
        use_claude = force_claude or (self.eval_count % int(1/self.claude_freq) == 0)
        
        if use_claude:
            return self.claude_evaluator.evaluate_batch(
                problems, explanations, student_solutions
            )
        else:
            # Use local evaluator (needs to be adapted to match interface)
            scores = self.local_evaluator.evaluate_batch(explanations, problems)
            evaluations = [{"score": s, "source": "local"} for s in scores]
            return scores, evaluations