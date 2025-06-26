# RLT Google Colab Development Framework

## Executive Summary
This framework provides a structured approach for implementing Sakana AI's Reinforcement Learning Teachers (RLT) in Google Colab. The implementation focuses on creating a teaching-optimized system where small models learn to generate effective explanations rather than solve problems directly.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Google Colab Environment                  │
├─────────────────────────────────────────────────────────────┤
│  1. Setup & Dependencies                                     │
│  2. Data Pipeline (Questions + Solutions)                    │
│  3. Teacher Model (7B) with Custom Prompting               │
│  4. Student Model for Reward Computation                    │
│  5. Dense Reward System (rSS + rKL)                        │
│  6. GRPO Training Loop                                      │
│  7. Distillation & Evaluation                              │
└─────────────────────────────────────────────────────────────┘
```

## Phase 1: Environment Setup

### 1.1 Colab Configuration
```python
# Cell 1: GPU Setup and Memory Management
!nvidia-smi
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Enable memory efficient attention
!pip install xformers==0.0.22
```

### 1.2 Core Dependencies
```python
# Cell 2: Install Required Libraries
!pip install -q anthropic==0.25.0  # Claude API
!pip install -q transformers==4.36.0
!pip install -q accelerate==0.25.0
!pip install -q datasets==2.16.0
!pip install -q trl==0.7.6  # For GRPO implementation
!pip install -q bitsandbytes==0.41.3  # For quantization
!pip install -q peft==0.7.1  # For LoRA if needed
!pip install -q wandb  # For experiment tracking
!pip install -q python-dotenv  # For environment variables
```

### 1.3 Anthropic API Setup
```python
# Cell 3: API Configuration
from google.colab import userdata
import os
from dotenv import load_dotenv

# Option 1: Use Colab secrets (recommended)
try:
    ANTHROPIC_API_KEY = userdata.get('ANTHROPIC_API_KEY')
    os.environ['ANTHROPIC_API_KEY'] = ANTHROPIC_API_KEY
except:
    print("Please add ANTHROPIC_API_KEY to Colab secrets")
    print("Go to: Tools > Secrets > Add Secret")

# Option 2: Load from .env file
# Create a .env file with: ANTHROPIC_API_KEY=your-key-here
load_dotenv()

# Verify API key is set
if not os.environ.get('ANTHROPIC_API_KEY'):
    raise ValueError("ANTHROPIC_API_KEY not found. Please set it in Colab secrets or .env file")

print("✓ Anthropic API configured")
```

### 1.4 Import Structure
```python
# Cell 4: Imports
import anthropic
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import AutoModelForCausalLMWithValueHead
from datasets import Dataset, load_dataset
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import json
import time
import pickle
from tqdm import tqdm
```

### 1.5 Cost Management Setup
```python
# Cell 5: Cost Tracking and Budget Management
class CostTracker:
    """Track API usage and costs throughout the experiment."""
    
    def __init__(self, budget_limit: float = 10.0):
        self.budget_limit = budget_limit
        self.costs = {
            'input_tokens': 0,
            'output_tokens': 0,
            'total_cost': 0.0,
            'api_calls': 0
        }
        
        # Claude Sonnet 4 pricing (update as needed)
        self.pricing = {
            'input_per_million': 3.0,   # $3 per million input tokens
            'output_per_million': 15.0  # $15 per million output tokens
        }
    
    def add_usage(self, input_tokens: int, output_tokens: int):
        """Add token usage and calculate cost."""
        self.costs['input_tokens'] += input_tokens
        self.costs['output_tokens'] += output_tokens
        self.costs['api_calls'] += 1
        
        # Calculate cost
        input_cost = (input_tokens / 1_000_000) * self.pricing['input_per_million']
        output_cost = (output_tokens / 1_000_000) * self.pricing['output_per_million']
        
        self.costs['total_cost'] += (input_cost + output_cost)
        
        # Check budget
        if self.costs['total_cost'] > self.budget_limit:
            raise ValueError(f"Budget limit exceeded: ${self.costs['total_cost']:.2f} > ${self.budget_limit}")
    
    def get_summary(self) -> Dict:
        """Get cost summary."""
        return {
            'total_cost': f"${self.costs['total_cost']:.4f}",
            'api_calls': self.costs['api_calls'],
            'total_tokens': self.costs['input_tokens'] + self.costs['output_tokens'],
            'average_cost_per_call': f"${self.costs['total_cost'] / max(1, self.costs['api_calls']):.4f}"
        }

# Initialize global cost tracker
cost_tracker = CostTracker(budget_limit=25.0)  # Set your budget
```

## Phase 2: Data Pipeline Development

### 2.1 Data Structure
```python
# Cell 4: Data Classes
@dataclass
class RLTDataPoint:
    question: str
    solution: str
    subject: str  # math, physics, chemistry
    difficulty: str  # easy, medium, hard
    
@dataclass
class TeacherInput:
    """Format: question + solution → explanation"""
    prompt: str
    question: str
    solution: str
    
@dataclass
class StudentInput:
    """Format: question + explanation → solution"""
    prompt: str
    question: str
    explanation: str
```

### 2.2 Dataset Preparation
```python
# Cell 5: Dataset Loading and Formatting
class RLTDataset:
    def __init__(self, data_path: str):
        self.data = self.load_data(data_path)
        self.teacher_prompt = """You are a teacher providing detailed explanations.
Given a question and its solution, explain step-by-step how to reach the solution.

Question: {question}
Solution: {solution}

Explanation:"""
        
        self.student_prompt = """Given the following question and explanation, 
determine the solution.

Question: {question}
Explanation: {explanation}

Solution:"""
    
    def format_teacher_input(self, item: RLTDataPoint) -> str:
        return self.teacher_prompt.format(
            question=item.question,
            solution=item.solution
        )
    
    def format_student_input(self, question: str, explanation: str) -> str:
        return self.student_prompt.format(
            question=question,
            explanation=explanation
        )
```

### 2.3 Data Sources Integration
```python
# Cell 6: Multiple Dataset Integration
# Support for MATH, AIME, GPQA formats
def load_math_dataset():
    """Load and format MATH dataset"""
    dataset = load_dataset("hendrycks/math", split="train[:1000]")
    return [
        RLTDataPoint(
            question=item['problem'],
            solution=item['solution'],
            subject=item['type'],
            difficulty=item['level']
        ) for item in dataset
    ]

def load_custom_dataset(file_path: str):
    """Load custom JSON dataset"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [RLTDataPoint(**item) for item in data]
```

## Phase 3: Model Architecture

### 3.1 Teacher Model Setup with Claude Sonnet 4
```python
# Cell 7: Claude Teacher Model Configuration
import anthropic
import os
from typing import List, Dict, Optional
import asyncio

class ClaudeRLTTeacher:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Claude Sonnet 4 as the teacher model.
        Uses ANTHROPIC_API_KEY environment variable if api_key not provided.
        """
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        
        # Using Claude Sonnet 4 - adjust model name based on latest available
        self.model = "claude-sonnet-4-20250514"  # Update with actual model name
        
        # Teacher-specific system prompt
        self.system_prompt = """You are an expert teacher creating detailed, step-by-step explanations.
Your role is to connect questions to their solutions with clear, logical reasoning that helps students understand.
Focus on:
1. Breaking down complex problems into manageable steps
2. Explaining the reasoning behind each step
3. Making connections between concepts
4. Using clear, accessible language
5. Providing intuitive examples when helpful"""
        
    def generate_explanation(self, 
                           question: str, 
                           solution: str,
                           max_tokens: int = 1024,
                           temperature: float = 0.7) -> Dict:
        """
        Generate explanation using Claude Sonnet 4.
        Returns both the explanation and metadata for reward computation.
        """
        try:
            # Format the prompt
            user_prompt = f"""Question: {question}

Solution: {solution}

Please provide a detailed step-by-step explanation that connects this question to its solution. 
Make your explanation clear and educational, focusing on helping a student understand the reasoning process."""
            
            # Call Claude API
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=self.system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )
            
            # Extract explanation text
            explanation = message.content[0].text
            
            # Return explanation with metadata
            return {
                'explanation': explanation,
                'usage': {
                    'input_tokens': message.usage.input_tokens,
                    'output_tokens': message.usage.output_tokens,
                    'total_cost': self._calculate_cost(message.usage)
                },
                'model': self.model,
                'request_id': message._request_id
            }
            
        except anthropic.APIConnectionError as e:
            print(f"Connection error: {e}")
            raise
        except anthropic.RateLimitError as e:
            print(f"Rate limit reached: {e}")
            raise
        except anthropic.APIStatusError as e:
            print(f"API error: {e.status_code} - {e.response}")
            raise
    
    def _calculate_cost(self, usage) -> float:
        """
        Calculate API cost based on token usage.
        Update these rates based on current Anthropic pricing.
        """
        # Example pricing (update with actual rates)
        input_cost_per_1k = 0.003  # $3 per million input tokens
        output_cost_per_1k = 0.015  # $15 per million output tokens
        
        total_cost = (
            (usage.input_tokens / 1000) * input_cost_per_1k +
            (usage.output_tokens / 1000) * output_cost_per_1k
        )
        
        return round(total_cost, 6)
    
    def generate_batch_explanations(self, 
                                  items: List[RLTDataPoint],
                                  batch_size: int = 5) -> List[Dict]:
        """
        Generate multiple explanations with rate limiting.
        """
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            for item in batch:
                result = self.generate_explanation(
                    question=item.question,
                    solution=item.solution
                )
                results.append(result)
                
                # Small delay to respect rate limits
                time.sleep(0.5)
        
        return results

# Cell 7b: Fallback Teacher Model (for comparison/testing)
class LocalRLTTeacher:
    """
    Local model teacher for development/comparison.
    Uses smaller open-source model when Claude API is not available.
    """
    def __init__(self, model_name: str = "microsoft/phi-2"):
        # Use smaller model for local testing
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def generate_explanation(self, question: str, solution: str, **kwargs):
        prompt = f"""Question: {question}
Solution: {solution}
Explanation:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                **kwargs
            )
        
        explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {
            'explanation': explanation,
            'usage': {'tokens': len(outputs[0])},
            'model': 'local'
        }
```

### 3.2 Student Model for Rewards
```python
# Cell 8: Student Model Setup
class RLTStudent:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        # Keep on CPU to save GPU memory
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to('cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def compute_solution_likelihood(self, 
                                  question: str, 
                                  explanation: str, 
                                  solution: str) -> torch.Tensor:
        """Compute log probability of solution given explanation"""
        student_input = self.format_input(question, explanation)
        full_text = student_input + solution
        
        inputs = self.tokenizer(full_text, return_tensors="pt")
        solution_start = len(self.tokenizer.encode(student_input))
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, solution_start-1:-1]
            solution_ids = inputs.input_ids[0, solution_start:]
            
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(1, solution_ids.unsqueeze(1))
            
        return token_log_probs.squeeze()
```

## Phase 4: Dense Reward System

### 4.1 Reward Components
```python
# Cell 9: RLT Reward Function
class RLTRewardFunction:
    def __init__(self, student_model: RLTStudent, lambda_kl: float = 3.0, alpha: float = 0.01):
        self.student = student_model
        self.lambda_kl = lambda_kl
        self.alpha = alpha
        
    def compute_reward(self, 
                      question: str,
                      solution: str,
                      teacher_explanation: str,
                      teacher_logprobs: torch.Tensor) -> float:
        """
        Compute RLT reward: r = rSS - λ*rKL
        """
        # Solution Score (rSS)
        student_logprobs = self.student.compute_solution_likelihood(
            question, teacher_explanation, solution
        )
        
        rSS = (
            student_logprobs.mean() + 
            self.alpha * student_logprobs.min()
        )
        
        # KL Divergence Score (rKL)
        student_explanation_logprobs = self.student.compute_explanation_likelihood(
            question, teacher_explanation
        )
        
        rKL = self.compute_kl_divergence(
            teacher_logprobs, 
            student_explanation_logprobs
        )
        
        # Combined reward
        reward = rSS - self.lambda_kl * rKL
        
        return reward.item(), {
            'rSS': rSS.item(),
            'rKL': rKL.item(),
            'solution_likelihood': student_logprobs.mean().item()
        }
```

### 4.2 KL Divergence Computation
```python
# Cell 10: KL Divergence Implementation
def compute_kl_divergence(teacher_logprobs: torch.Tensor, 
                         student_logprobs: torch.Tensor) -> torch.Tensor:
    """
    Compute KL(teacher || student) with average and max reduction
    """
    kl_per_token = teacher_logprobs - student_logprobs
    kl_avg = kl_per_token.mean()
    kl_max = kl_per_token.max()
    
    return kl_avg + 0.01 * kl_max
```

## Phase 5: GRPO Training Implementation

### 5.1 Training Loop with Claude Integration
```python
# Cell 11: GRPO Training with Claude
class ClaudeRLTTrainer:
    def __init__(self, 
                 teacher: ClaudeRLTTeacher,
                 reward_fn: RLTRewardFunction,
                 dataset: RLTDataset,
                 config: Dict):
        self.teacher = teacher
        self.reward_fn = reward_fn
        self.dataset = dataset
        self.config = config
        
        # Track API usage and costs
        self.api_stats = {
            'total_calls': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'explanations_generated': []
        }
        
    def collect_explanations(self, batch: List[RLTDataPoint]) -> List[Dict]:
        """
        Collect multiple explanations per question using Claude.
        Implements caching to reduce API calls during development.
        """
        all_explanations = []
        
        for item in batch:
            explanations_for_item = []
            
            # Generate G explanations for GRPO
            for i in range(self.config['group_size']):
                # Check cache first (for development efficiency)
                cache_key = f"{hash(item.question)}_{hash(item.solution)}_{i}"
                
                if cache_key in self.explanation_cache:
                    explanation_data = self.explanation_cache[cache_key]
                else:
                    # Generate new explanation
                    explanation_data = self.teacher.generate_explanation(
                        question=item.question,
                        solution=item.solution,
                        temperature=0.7 + (i * 0.05)  # Vary temperature slightly
                    )
                    
                    # Update API stats
                    self.api_stats['total_calls'] += 1
                    self.api_stats['total_tokens'] += (
                        explanation_data['usage']['input_tokens'] + 
                        explanation_data['usage']['output_tokens']
                    )
                    self.api_stats['total_cost'] += explanation_data['usage']['total_cost']
                    
                    # Cache the explanation
                    self.explanation_cache[cache_key] = explanation_data
                
                # Compute reward
                reward, metrics = self.reward_fn.compute_reward(
                    question=item.question,
                    solution=item.solution,
                    teacher_explanation=explanation_data['explanation'],
                    teacher_logprobs=None  # Claude doesn't provide logprobs
                )
                
                explanations_for_item.append({
                    'explanation': explanation_data['explanation'],
                    'reward': reward,
                    'metrics': metrics,
                    'api_usage': explanation_data['usage']
                })
            
            all_explanations.append(explanations_for_item)
        
        return all_explanations
    
    def train_epoch(self, save_every_n_batches: int = 10):
        """
        Run one epoch of training, collecting explanations and rewards.
        Note: Since Claude is not fine-tunable, we collect high-quality
        explanations for student distillation rather than updating Claude.
        """
        best_explanations = []
        
        for batch_idx, batch in enumerate(self.dataset.get_batches()):
            # Collect explanations
            batch_explanations = self.collect_explanations(batch)
            
            # Process each item's explanations
            for item_idx, explanations in enumerate(batch_explanations):
                # Sort by reward
                sorted_explanations = sorted(
                    explanations, 
                    key=lambda x: x['reward'], 
                    reverse=True
                )
                
                # Keep best explanation
                best = sorted_explanations[0]
                best['question'] = batch[item_idx].question
                best['solution'] = batch[item_idx].solution
                best_explanations.append(best)
                
                # Log progress
                if len(best_explanations) % 10 == 0:
                    avg_reward = np.mean([e['reward'] for e in best_explanations[-10:]])
                    print(f"Processed {len(best_explanations)} items. "
                          f"Avg reward (last 10): {avg_reward:.4f}")
            
            # Save checkpoint
            if (batch_idx + 1) % save_every_n_batches == 0:
                self.save_checkpoint(best_explanations, batch_idx)
        
        return best_explanations
    
    def save_checkpoint(self, explanations: List[Dict], batch_idx: int):
        """Save collected explanations and stats."""
        checkpoint = {
            'explanations': explanations,
            'api_stats': self.api_stats,
            'batch_idx': batch_idx,
            'timestamp': time.time()
        }
        
        path = f"./checkpoints/claude_rlt_batch_{batch_idx}.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"Saved checkpoint: {path}")
        print(f"API Stats - Calls: {self.api_stats['total_calls']}, "
              f"Cost: ${self.api_stats['total_cost']:.4f}")

# Cell 11b: Efficient Training Manager
class EfficientRLTManager:
    """
    Manages the RLT training process with cost optimization.
    """
    def __init__(self, api_key: str, budget_limit: float = 10.0):
        self.teacher = ClaudeRLTTeacher(api_key=api_key)
        self.budget_limit = budget_limit
        self.total_spent = 0.0
        
    def run_training(self, 
                    dataset: RLTDataset,
                    num_samples: int = 100,
                    explanations_per_sample: int = 3):
        """
        Run cost-efficient training within budget constraints.
        """
        # Initialize components
        student = RLTStudent()  # For reward computation
        reward_fn = RLTRewardFunction(student)
        
        # Configure training
        config = {
            'group_size': explanations_per_sample,
            'learning_rate': 1e-6,
            'batch_size': 5  # Small batches for API efficiency
        }
        
        trainer = ClaudeRLTTrainer(
            teacher=self.teacher,
            reward_fn=reward_fn,
            dataset=dataset,
            config=config
        )
        
        # Initialize explanation cache
        trainer.explanation_cache = {}
        
        # Sample subset of data if needed
        if num_samples < len(dataset.data):
            import random
            dataset.data = random.sample(dataset.data, num_samples)
        
        print(f"Starting RLT training with {num_samples} samples")
        print(f"Budget limit: ${self.budget_limit}")
        
        # Run training
        best_explanations = trainer.train_epoch()
        
        # Create distillation dataset
        distillation_data = self.create_distillation_dataset(best_explanations)
        
        return distillation_data, trainer.api_stats
```

### 5.2 Memory-Efficient Training
```python
# Cell 12: Gradient Accumulation for Colab
def train_with_gradient_accumulation(trainer: RLTTrainer, 
                                    num_epochs: int = 1,
                                    accumulation_steps: int = 4):
    """Training loop with gradient accumulation for memory efficiency"""
    
    for epoch in range(num_epochs):
        for i, batch in enumerate(trainer.dataset.get_batches()):
            # Accumulate gradients
            loss = trainer.train_step(batch)
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()
                
            # Log progress
            if i % 10 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}")
```

## Phase 6: Distillation Pipeline

### 6.1 Trace Collection
```python
# Cell 13: Collect Teaching Traces
def collect_distillation_dataset(teacher: RLTTeacher, 
                               test_questions: List[RLTDataPoint],
                               output_path: str):
    """Generate teaching traces for student distillation"""
    
    distillation_data = []
    
    for item in test_questions:
        teacher_input = format_teacher_input(item)
        explanation = teacher.generate_explanation(teacher_input)
        
        # Format for student training
        student_example = {
            'input': f"Question: {item.question}\nExplanation: {explanation}",
            'output': item.solution,
            'metadata': {
                'subject': item.subject,
                'difficulty': item.difficulty
            }
        }
        
        distillation_data.append(student_example)
    
    # Save dataset
    with open(output_path, 'w') as f:
        json.dump(distillation_data, f, indent=2)
```

### 6.2 Student Training
```python
# Cell 14: Train Student Model
from transformers import Trainer, TrainingArguments

def train_student_model(distillation_data_path: str,
                       student_model_name: str = "Qwen/Qwen2.5-7B"):
    """Fine-tune student on teacher's explanations"""
    
    # Load distillation dataset
    dataset = load_dataset('json', data_files=distillation_data_path)
    
    # Initialize student
    model = AutoModelForCausalLM.from_pretrained(student_model_name)
    tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    
    # Training arguments optimized for Colab
    training_args = TrainingArguments(
        output_dir="./student_model",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=100,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=True,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        learning_rate=1e-5
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        tokenizer=tokenizer
    )
    
    trainer.train()
```

## Phase 7: Evaluation Framework

### 7.1 Benchmark Evaluation
```python
# Cell 15: Evaluation Metrics
class RLTEvaluator:
    def __init__(self, benchmarks: Dict[str, Dataset]):
        self.benchmarks = benchmarks
        
    def evaluate_student(self, student_model, tokenizer):
        """Evaluate student on multiple benchmarks"""
        results = {}
        
        for name, dataset in self.benchmarks.items():
            correct = 0
            total = 0
            
            for item in dataset:
                # Generate solution
                input_text = f"Question: {item['question']}\n\nSolution:"
                inputs = tokenizer(input_text, return_tensors="pt")
                
                outputs = student_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1
                )
                
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Check correctness (implement domain-specific checker)
                if self.check_answer(generated, item['solution'], name):
                    correct += 1
                total += 1
            
            results[name] = {
                'accuracy': correct / total,
                'correct': correct,
                'total': total
            }
        
        return results
```

### 7.2 Quality Metrics
```python
# Cell 16: Explanation Quality Analysis
def analyze_explanation_quality(explanations: List[str]):
    """Analyze generated explanations"""
    metrics = {
        'avg_length': np.mean([len(e.split()) for e in explanations]),
        'unique_steps': [],  # Count unique reasoning steps
        'coherence_score': [],  # Measure logical flow
        'completeness': []  # Check if all solution steps covered
    }
    
    # Implement quality metrics
    return metrics
```

## Implementation Checklist for Dev Team

### Week 1: Foundation
- [ ] Set up Colab environment with GPU runtime
- [ ] Install and test all dependencies
- [ ] Implement basic data loading pipeline
- [ ] Create RLTDataPoint and formatting classes

### Week 2: Core Models
- [ ] Implement RLTTeacher with quantization
- [ ] Set up RLTStudent for CPU inference
- [ ] Test teacher explanation generation
- [ ] Verify student likelihood computation

### Week 3: Reward System
- [ ] Implement dense reward function
- [ ] Test rSS and rKL components separately
- [ ] Validate reward scaling and normalization
- [ ] Create debugging visualizations

### Week 4: Training Loop
- [ ] Implement GRPO training step
- [ ] Add gradient accumulation
- [ ] Set up checkpointing system
- [ ] Integrate wandb logging

### Week 5: Distillation
- [ ] Create trace collection pipeline
- [ ] Implement student training
- [ ] Add evaluation metrics
- [ ] Test on small dataset

### Week 6: Optimization & Scaling
- [ ] Profile memory usage
- [ ] Optimize batch processing
- [ ] Implement multi-GPU support if available
- [ ] Add data parallelism

## Best Practices

### Memory Management
1. Use gradient checkpointing for large models
2. Implement CPU offloading for student model
3. Clear cache regularly: `torch.cuda.empty_cache()`
4. Use mixed precision training

### Debugging Tips
1. Start with tiny dataset (10-50 examples)
2. Log all reward components separately
3. Visualize explanation quality over training
4. Monitor GPU memory usage continuously

### Colab-Specific Optimizations
```python
# Cell 17: Colab Utilities
def setup_colab_env():
    """Configure Colab for optimal performance"""
    # Mount drive for persistence
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Set up model cache
    os.environ['HF_HOME'] = '/content/drive/MyDrive/huggingface'
    
    # Enable GPU memory growth
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

## Quick Start Example

### Complete Working Example
```python
# Cell 17: Quick Start - Minimal RLT Implementation
async def quick_start_rlt():
    """
    Minimal example to test the RLT framework with Claude.
    """
    # 1. Initialize teacher
    teacher = ClaudeRLTTeacher()
    
    # 2. Create sample data
    sample_data = [
        RLTDataPoint(
            question="What is 15% of 80?",
            solution="12",
            subject="math",
            difficulty="easy"
        ),
        RLTDataPoint(
            question="Solve for x: 2x + 5 = 13",
            solution="x = 4",
            subject="algebra",
            difficulty="easy"
        )
    ]
    
    # 3. Generate explanations
    print("Generating explanations with Claude Sonnet 4...")
    explanations = []
    
    for item in sample_data:
        result = teacher.generate_explanation(
            question=item.question,
            solution=item.solution
        )
        
        print(f"\nQuestion: {item.question}")
        print(f"Solution: {item.solution}")
        print(f"Explanation: {result['explanation'][:200]}...")
        print(f"Cost: ${result['usage']['total_cost']:.4f}")
        
        explanations.append(result)
    
    # 4. Create student training data
    distillation_data = []
    for item, exp in zip(sample_data, explanations):
        distillation_data.append({
            'input': f"Question: {item.question}\nExplanation: {exp['explanation']}",
            'output': item.solution
        })
    
    # 5. Save for student training
    with open('sample_distillation_data.json', 'w') as f:
        json.dump(distillation_data, f, indent=2)
    
    print(f"\n✓ Generated {len(distillation_data)} training examples")
    print(f"Total API cost: ${sum(e['usage']['total_cost'] for e in explanations):.4f}")
    
    return distillation_data

# Run the example
# await quick_start_rlt()
```

### Development Workflow Example
```python
# Cell 18: Complete Development Pipeline
class RLTDevelopmentPipeline:
    """
    End-to-end pipeline for RLT development in Colab.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.teacher = ClaudeRLTTeacher(api_key)
        self.cost_tracker = CostTracker(budget_limit=5.0)
        
    def run_experiment(self, 
                      dataset_name: str = "gsm8k",
                      num_samples: int = 20,
                      explanations_per_sample: int = 3):
        """
        Run a complete RLT experiment with cost tracking.
        """
        
        # 1. Load dataset
        print(f"Loading {dataset_name} dataset...")
        if dataset_name == "gsm8k":
            dataset = load_dataset("gsm8k", "main", split="train[:100]")
            data_points = [
                RLTDataPoint(
                    question=item['question'],
                    solution=item['answer'].split('####')[-1].strip(),
                    subject="math",
                    difficulty="medium"
                )
                for item in dataset
            ][:num_samples]
        else:
            # Load custom dataset
            data_points = load_custom_dataset(f"{dataset_name}.json")
        
        # 2. Generate explanations with Claude
        print(f"\nGenerating {explanations_per_sample} explanations for {len(data_points)} questions...")
        all_results = []
        
        for idx, item in enumerate(tqdm(data_points)):
            item_results = []
            
            for exp_idx in range(explanations_per_sample):
                try:
                    # Vary temperature for diversity
                    temp = 0.6 + (exp_idx * 0.1)
                    
                    result = self.teacher.generate_explanation(
                        question=item.question,
                        solution=item.solution,
                        temperature=temp
                    )
                    
                    # Track costs
                    self.cost_tracker.add_usage(
                        result['usage']['input_tokens'],
                        result['usage']['output_tokens']
                    )
                    
                    item_results.append(result)
                    
                except Exception as e:
                    print(f"Error on item {idx}: {e}")
                    continue
                
                # Rate limiting
                time.sleep(0.5)
            
            all_results.append({
                'question': item.question,
                'solution': item.solution,
                'explanations': item_results
            })
            
            # Show progress
            if (idx + 1) % 5 == 0:
                print(f"Progress: {idx + 1}/{len(data_points)}")
                print(f"Current cost: {self.cost_tracker.get_summary()['total_cost']}")
        
        # 3. Evaluate and select best explanations
        print("\nEvaluating explanations...")
        best_explanations = self.select_best_explanations(all_results)
        
        # 4. Create distillation dataset
        distillation_dataset = self.create_distillation_dataset(best_explanations)
        
        # 5. Save results
        self.save_results(all_results, distillation_dataset)
        
        # 6. Print summary
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        print(f"Dataset: {dataset_name}")
        print(f"Samples processed: {len(data_points)}")
        print(f"Total explanations: {len(data_points) * explanations_per_sample}")
        print(f"Cost summary: {self.cost_tracker.get_summary()}")
        
        return distillation_dataset
    
    def select_best_explanations(self, results: List[Dict]) -> List[Dict]:
        """
        Select best explanation for each question based on length and clarity.
        In production, this would use the full reward function.
        """
        best = []
        
        for item in results:
            if not item['explanations']:
                continue
                
            # Simple heuristic: prefer medium-length explanations
            sorted_exps = sorted(
                item['explanations'],
                key=lambda x: abs(len(x['explanation']) - 300)  # Target ~300 chars
            )
            
            best.append({
                'question': item['question'],
                'solution': item['solution'],
                'explanation': sorted_exps[0]['explanation'],
                'metadata': sorted_exps[0]['usage']
            })
        
        return best
    
    def create_distillation_dataset(self, best_explanations: List[Dict]) -> List[Dict]:
        """Format data for student training."""
        return [
            {
                'input': f"Question: {item['question']}\n\nExplanation: {item['explanation']}\n\nSolution:",
                'output': f" {item['solution']}",
                'metadata': item.get('metadata', {})
            }
            for item in best_explanations
        ]
    
    def save_results(self, all_results: List[Dict], distillation_dataset: List[Dict]):
        """Save all results and datasets."""
        timestamp = int(time.time())
        
        # Save raw results
        with open(f'rlt_raw_results_{timestamp}.pkl', 'wb') as f:
            pickle.dump(all_results, f)
        
        # Save distillation dataset
        with open(f'distillation_dataset_{timestamp}.json', 'w') as f:
            json.dump(distillation_dataset, f, indent=2)
        
        print(f"\n✓ Results saved with timestamp: {timestamp}")

# Usage example
pipeline = RLTDevelopmentPipeline()
# distillation_data = pipeline.run_experiment(num_samples=10)
```

## Cost Optimization Tips

1. **Start Small**: Test with 5-10 examples first
2. **Cache Responses**: Save all Claude responses to avoid duplicate API calls
3. **Batch Similar Questions**: Group by difficulty/type for consistent explanations
4. **Use Temperature Scaling**: Lower temperature (0.3-0.5) for more consistent outputs
5. **Monitor Usage**: Check cost tracker frequently during development

## Debugging Checklist

- [ ] Verify API key is correctly set
- [ ] Test with single example before batch processing
- [ ] Check rate limits (Claude has per-minute limits)
- [ ] Validate explanation quality manually
- [ ] Monitor memory usage for student model
- [ ] Save checkpoints frequently

## Next Steps

1. **Experiment with Model Sizes**: Start with smaller models (1-3B) for faster iteration
2. **Domain Transfer**: Test zero-shot transfer to new problem types
3. **Ablation Studies**: Systematically test reward component weights
4. **Scale Testing**: Gradually increase dataset and model sizes

## Resources

- Original Paper: https://arxiv.org/abs/2506.08388
- GitHub Implementation: https://github.com/SakanaAI/RLT
- Anthropic API Docs: https://docs.anthropic.com
- Colab Best Practices: https://research.google.com/colaboratory/faq.html

This framework provides a complete roadmap for implementing RLT in Google Colab using Claude Sonnet 4 as the teacher model. The use of Claude's superior reasoning capabilities should produce higher quality explanations compared to smaller open-source models, though at a higher API cost. Start with the quick start example and progressively add complexity as you validate each component works correctly.