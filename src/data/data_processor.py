"""
Data Processing and Formatting Utilities for RLT

This module provides comprehensive data processing, formatting, and augmentation
utilities for the RLT training pipeline.
"""

import re
import json
import random
import hashlib
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict
import logging

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RLTDataPoint:
    """Data structure for RLT training examples."""
    question: str
    solution: str
    subject: str  # math, physics, chemistry, etc.
    difficulty: str  # easy, medium, hard
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'RLTDataPoint':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TeacherInput:
    """Format for teacher model input: question + solution → explanation"""
    prompt: str
    question: str
    solution: str
    
    def to_string(self) -> str:
        """Convert to formatted string for model input."""
        return self.prompt


@dataclass
class StudentInput:
    """Format for student model input: question + explanation → solution"""
    prompt: str
    question: str
    explanation: str
    
    def to_string(self) -> str:
        """Convert to formatted string for model input."""
        return self.prompt


class DataProcessor:
    """
    Main data processing class for RLT pipeline.
    Handles formatting, preprocessing, and augmentation.
    """
    
    def __init__(self, 
                 teacher_prompt_template: Optional[str] = None,
                 student_prompt_template: Optional[str] = None,
                 enable_augmentation: bool = True):
        """
        Initialize data processor.
        
        Args:
            teacher_prompt_template: Custom template for teacher prompts
            student_prompt_template: Custom template for student prompts
            enable_augmentation: Whether to enable data augmentation
        """
        self.teacher_prompt_template = teacher_prompt_template or self._default_teacher_prompt()
        self.student_prompt_template = student_prompt_template or self._default_student_prompt()
        self.enable_augmentation = enable_augmentation
        
        # Preprocessing functions by subject
        self.preprocessors = {
            'math': self._preprocess_math,
            'physics': self._preprocess_physics,
            'chemistry': self._preprocess_chemistry,
            'default': self._preprocess_default
        }
        
        # Augmentation strategies
        self.augmenters = {
            'paraphrase': self._augment_paraphrase,
            'numerical': self._augment_numerical,
            'difficulty': self._augment_difficulty
        }
    
    def _default_teacher_prompt(self) -> str:
        """Default teacher prompt template."""
        return """You are an expert teacher providing detailed explanations.
Given a question and its solution, explain step-by-step how to reach the solution.
Focus on clarity, logical reasoning, and educational value.

Question: {question}
Solution: {solution}

Explanation:"""
    
    def _default_student_prompt(self) -> str:
        """Default student prompt template."""
        return """Given the following question and explanation, determine the solution.
Apply the reasoning from the explanation to solve the problem.

Question: {question}
Explanation: {explanation}

Solution:"""
    
    def format_teacher_input(self, data_point: RLTDataPoint) -> TeacherInput:
        """
        Format a data point for teacher model input.
        
        Args:
            data_point: RLT data point
            
        Returns:
            Formatted teacher input
        """
        # Preprocess based on subject
        preprocessor = self.preprocessors.get(data_point.subject, self.preprocessors['default'])
        processed_question = preprocessor(data_point.question)
        processed_solution = self._clean_solution(data_point.solution)
        
        # Format prompt
        prompt = self.teacher_prompt_template.format(
            question=processed_question,
            solution=processed_solution
        )
        
        return TeacherInput(
            prompt=prompt,
            question=processed_question,
            solution=processed_solution
        )
    
    def format_student_input(self, question: str, explanation: str) -> StudentInput:
        """
        Format inputs for student model.
        
        Args:
            question: The question
            explanation: Teacher's explanation
            
        Returns:
            Formatted student input
        """
        # Clean inputs
        question = self._clean_text(question)
        explanation = self._clean_text(explanation)
        
        # Format prompt
        prompt = self.student_prompt_template.format(
            question=question,
            explanation=explanation
        )
        
        return StudentInput(
            prompt=prompt,
            question=question,
            explanation=explanation
        )
    
    def batch_format_teacher(self, data_points: List[RLTDataPoint]) -> List[TeacherInput]:
        """Format multiple data points for teacher model."""
        return [self.format_teacher_input(dp) for dp in data_points]
    
    def batch_format_student(self, 
                           questions: List[str], 
                           explanations: List[str]) -> List[StudentInput]:
        """Format multiple inputs for student model."""
        return [
            self.format_student_input(q, e) 
            for q, e in zip(questions, explanations)
        ]
    
    def _clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Fix common encoding issues
        text = text.replace('\\n', '\n').replace('\\t', '\t')
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        
        return text.strip()
    
    def _clean_solution(self, solution: str) -> str:
        """Clean and standardize solution format."""
        solution = self._clean_text(solution)
        
        # Remove common prefixes
        prefixes = ['Answer:', 'Solution:', 'Therefore:', 'Thus:', 'Hence:']
        for prefix in prefixes:
            if solution.startswith(prefix):
                solution = solution[len(prefix):].strip()
        
        return solution
    
    def _preprocess_math(self, text: str) -> str:
        """Preprocess mathematical text."""
        # Standardize math notation
        text = self._clean_text(text)
        
        # Convert common math symbols
        replacements = {
            '×': '*',
            '÷': '/',
            '≤': '<=',
            '≥': '>=',
            '≠': '!=',
            '√': 'sqrt',
            '∑': 'sum',
            '∏': 'product',
            '∫': 'integral'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Ensure proper spacing around operators
        operators = ['+', '-', '*', '/', '=', '<', '>', '(', ')']
        for op in operators:
            text = re.sub(rf'\s*\{op}\s*', f' {op} ', text)
        
        # Clean up multiple spaces
        text = ' '.join(text.split())
        
        return text
    
    def _preprocess_physics(self, text: str) -> str:
        """Preprocess physics text."""
        text = self._clean_text(text)
        
        # Standardize units
        unit_replacements = {
            'metres': 'meters',
            'kilometre': 'kilometer',
            'kg': 'kilograms',
            'g': 'grams',
            's': 'seconds',
            'min': 'minutes',
            'hr': 'hours',
            'deg': 'degrees'
        }
        
        # Apply replacements with word boundaries
        for old, new in unit_replacements.items():
            text = re.sub(rf'\b{old}\b', new, text, flags=re.IGNORECASE)
        
        return text
    
    def _preprocess_chemistry(self, text: str) -> str:
        """Preprocess chemistry text."""
        text = self._clean_text(text)
        
        # Standardize chemical formulas
        # Example: H2O -> H₂O (optional, depends on model preference)
        
        # Ensure proper spacing around chemical equations
        text = re.sub(r'(\w+)\s*\+\s*(\w+)', r'\1 + \2', text)
        text = re.sub(r'(\w+)\s*->\s*(\w+)', r'\1 -> \2', text)
        
        return text
    
    def _preprocess_default(self, text: str) -> str:
        """Default preprocessing for any subject."""
        return self._clean_text(text)
    
    def augment_data_point(self, 
                          data_point: RLTDataPoint, 
                          strategies: List[str] = None) -> List[RLTDataPoint]:
        """
        Apply data augmentation to create variations of a data point.
        
        Args:
            data_point: Original data point
            strategies: List of augmentation strategies to apply
            
        Returns:
            List of augmented data points (including original)
        """
        if not self.enable_augmentation:
            return [data_point]
        
        augmented = [data_point]  # Include original
        
        if strategies is None:
            strategies = list(self.augmenters.keys())
        
        for strategy in strategies:
            if strategy in self.augmenters:
                augmented_point = self.augmenters[strategy](data_point)
                if augmented_point and augmented_point != data_point:
                    augmented.append(augmented_point)
        
        return augmented
    
    def _augment_paraphrase(self, data_point: RLTDataPoint) -> Optional[RLTDataPoint]:
        """Create a paraphrased version of the question."""
        # Simple paraphrasing templates
        paraphrase_templates = {
            'math': [
                "Find {solution} if {question}",
                "Calculate the value when {question}",
                "Determine the answer to: {question}"
            ],
            'physics': [
                "Given that {question}, find {solution}",
                "In a scenario where {question}, what is the result?",
                "Consider: {question}. What is the answer?"
            ]
        }
        
        templates = paraphrase_templates.get(data_point.subject, paraphrase_templates['math'])
        
        # Simple keyword extraction for paraphrasing
        question_lower = data_point.question.lower()
        
        # Don't paraphrase if question is too complex
        if len(data_point.question.split()) > 50:
            return None
        
        # Create simple paraphrase
        augmented = RLTDataPoint(
            question=f"Problem: {data_point.question}",
            solution=data_point.solution,
            subject=data_point.subject,
            difficulty=data_point.difficulty
        )
        
        return augmented
    
    def _augment_numerical(self, data_point: RLTDataPoint) -> Optional[RLTDataPoint]:
        """Create variations with different numbers (for math problems)."""
        if data_point.subject != 'math':
            return None
        
        # Extract numbers from question
        numbers = re.findall(r'\b\d+\.?\d*\b', data_point.question)
        
        if not numbers:
            return None
        
        # Create a variation by slightly modifying numbers
        new_question = data_point.question
        
        for num in numbers[:2]:  # Modify at most 2 numbers
            try:
                value = float(num)
                # Small variation (±10%)
                variation = value * random.uniform(0.9, 1.1)
                
                if '.' in num:
                    new_value = f"{variation:.{len(num.split('.')[1])}f}"
                else:
                    new_value = str(int(variation))
                
                new_question = new_question.replace(num, new_value, 1)
            except:
                continue
        
        # Note: Solution would need to be recalculated in practice
        # This is a simplified example
        return RLTDataPoint(
            question=new_question,
            solution=f"[Requires recalculation based on: {new_question}]",
            subject=data_point.subject,
            difficulty=data_point.difficulty
        )
    
    def _augment_difficulty(self, data_point: RLTDataPoint) -> Optional[RLTDataPoint]:
        """Create a variation with adjusted difficulty."""
        difficulty_modifiers = {
            'easy': {
                'prefix': "For beginners: ",
                'suffix': " (Basic level)"
            },
            'medium': {
                'prefix': "Standard problem: ",
                'suffix': " (Intermediate level)"
            },
            'hard': {
                'prefix': "Challenge question: ",
                'suffix': " (Advanced level)"
            }
        }
        
        modifier = difficulty_modifiers.get(data_point.difficulty, difficulty_modifiers['medium'])
        
        return RLTDataPoint(
            question=modifier['prefix'] + data_point.question + modifier['suffix'],
            solution=data_point.solution,
            subject=data_point.subject,
            difficulty=data_point.difficulty
        )
    
    def prepare_batch(self, 
                     data_points: List[RLTDataPoint],
                     batch_size: int = 32,
                     shuffle: bool = True) -> List[List[RLTDataPoint]]:
        """
        Prepare data points into batches.
        
        Args:
            data_points: List of data points
            batch_size: Size of each batch
            shuffle: Whether to shuffle data
            
        Returns:
            List of batches
        """
        if shuffle:
            data_points = data_points.copy()
            random.shuffle(data_points)
        
        batches = []
        for i in range(0, len(data_points), batch_size):
            batch = data_points[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def filter_by_criteria(self,
                          data_points: List[RLTDataPoint],
                          subjects: Optional[List[str]] = None,
                          difficulties: Optional[List[str]] = None,
                          min_length: Optional[int] = None,
                          max_length: Optional[int] = None) -> List[RLTDataPoint]:
        """
        Filter data points based on various criteria.
        
        Args:
            data_points: List of data points to filter
            subjects: List of subjects to include
            difficulties: List of difficulties to include
            min_length: Minimum question length
            max_length: Maximum question length
            
        Returns:
            Filtered list of data points
        """
        filtered = data_points
        
        if subjects:
            filtered = [dp for dp in filtered if dp.subject in subjects]
        
        if difficulties:
            filtered = [dp for dp in filtered if dp.difficulty in difficulties]
        
        if min_length:
            filtered = [dp for dp in filtered if len(dp.question) >= min_length]
        
        if max_length:
            filtered = [dp for dp in filtered if len(dp.question) <= max_length]
        
        return filtered
    
    def validate_data_point(self, data_point: RLTDataPoint) -> Tuple[bool, List[str]]:
        """
        Validate a data point for quality and completeness.
        
        Args:
            data_point: Data point to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required fields
        if not data_point.question or not data_point.question.strip():
            issues.append("Empty question")
        
        if not data_point.solution or not data_point.solution.strip():
            issues.append("Empty solution")
        
        # Check field lengths
        if len(data_point.question) < 10:
            issues.append("Question too short")
        
        if len(data_point.question) > 5000:
            issues.append("Question too long")
        
        # Check valid subject and difficulty
        valid_subjects = ['math', 'physics', 'chemistry', 'biology', 'general']
        if data_point.subject not in valid_subjects:
            issues.append(f"Invalid subject: {data_point.subject}")
        
        valid_difficulties = ['easy', 'medium', 'hard']
        if data_point.difficulty not in valid_difficulties:
            issues.append(f"Invalid difficulty: {data_point.difficulty}")
        
        # Check for suspicious patterns
        if data_point.question.count('?') > 5:
            issues.append("Too many question marks")
        
        return len(issues) == 0, issues
    
    def get_statistics(self, data_points: List[RLTDataPoint]) -> Dict[str, Any]:
        """
        Calculate statistics about the dataset.
        
        Args:
            data_points: List of data points
            
        Returns:
            Dictionary of statistics
        """
        if not data_points:
            return {}
        
        # Basic counts
        stats = {
            'total_count': len(data_points),
            'subjects': defaultdict(int),
            'difficulties': defaultdict(int),
            'question_lengths': [],
            'solution_lengths': []
        }
        
        # Collect data
        for dp in data_points:
            stats['subjects'][dp.subject] += 1
            stats['difficulties'][dp.difficulty] += 1
            stats['question_lengths'].append(len(dp.question))
            stats['solution_lengths'].append(len(dp.solution))
        
        # Calculate aggregates
        stats['subjects'] = dict(stats['subjects'])
        stats['difficulties'] = dict(stats['difficulties'])
        
        stats['avg_question_length'] = np.mean(stats['question_lengths'])
        stats['avg_solution_length'] = np.mean(stats['solution_lengths'])
        
        stats['min_question_length'] = min(stats['question_lengths'])
        stats['max_question_length'] = max(stats['question_lengths'])
        
        # Remove raw lists for cleaner output
        del stats['question_lengths']
        del stats['solution_lengths']
        
        return stats


# Utility functions
def create_balanced_batches(data_points: List[RLTDataPoint], 
                          batch_size: int = 32,
                          balance_by: str = 'subject') -> List[List[RLTDataPoint]]:
    """
    Create batches that are balanced by a specific attribute.
    
    Args:
        data_points: List of data points
        batch_size: Size of each batch
        balance_by: Attribute to balance by ('subject' or 'difficulty')
        
    Returns:
        List of balanced batches
    """
    # Group by attribute
    groups = defaultdict(list)
    for dp in data_points:
        key = getattr(dp, balance_by)
        groups[key].append(dp)
    
    # Create balanced batches
    batches = []
    current_batch = []
    
    # Round-robin selection from each group
    iterators = {key: iter(items) for key, items in groups.items()}
    
    while True:
        added_any = False
        
        for key, iterator in iterators.items():
            try:
                item = next(iterator)
                current_batch.append(item)
                added_any = True
                
                if len(current_batch) >= batch_size:
                    batches.append(current_batch)
                    current_batch = []
            except StopIteration:
                continue
        
        if not added_any:
            break
    
    # Add remaining items
    if current_batch:
        batches.append(current_batch)
    
    return batches


def merge_datasets(datasets: List[List[RLTDataPoint]], 
                  shuffle: bool = True) -> List[RLTDataPoint]:
    """
    Merge multiple datasets into one.
    
    Args:
        datasets: List of datasets to merge
        shuffle: Whether to shuffle the merged dataset
        
    Returns:
        Merged dataset
    """
    merged = []
    for dataset in datasets:
        merged.extend(dataset)
    
    if shuffle:
        random.shuffle(merged)
    
    return merged