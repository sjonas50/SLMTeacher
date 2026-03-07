"""
Group Relative Policy Optimization (GRPO) Trainer for RLT

Trains the STUDENT model to learn from TEACHER explanations.
"""
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # GRPO specific
    group_size: int = 4  # Number of explanations per question
    clip_epsilon: float = 0.2  # PPO-style clipping
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Reward normalization
    normalize_rewards: bool = True
    reward_baseline: str = "mean"  # "mean" or "min"
    
    # Checkpointing
    save_steps: int = 100
    eval_steps: int = 50
    checkpoint_dir: str = "./checkpoints"


class GRPOTrainer:
    """
    Implements Group Relative Policy Optimization for RLT.
    
    Key difference from original: This trains the STUDENT model to learn
    from TEACHER explanations, not the other way around.
    """
    
    def __init__(
        self,
        teacher,  # Claude API or any explanation generator
        student_model,  # The model we're training
        student_evaluator,  # Evaluates student understanding for rewards
        reward_function,
        config: GRPOConfig
    ):
        self.teacher = teacher
        self.student_model = student_model  # This is what we train!
        self.student_evaluator = student_evaluator
        self.reward_fn = reward_function
        self.config = config
        
        # Setup optimizer for STUDENT model parameters
        if hasattr(self.student_model, 'model'):
            # For OptimizedHFModel wrapper
            model_params = self.student_model.model.parameters()
        else:
            # For direct model
            model_params = self.student_model.parameters()
            
        self.optimizer = AdamW(
            model_params,
            lr=config.learning_rate
        )
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_reward = float('-inf')
        
        # Metrics tracking
        self.metrics = {
            'rewards': [],
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'learning_rate': []
        }
    
    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None
    ):
        """Main training loop."""
        num_epochs = num_epochs or self.config.num_epochs
        
        # Setup scheduler
        total_steps = len(train_dataloader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info("Starting GRPO training for %d epochs", num_epochs)
        logger.info("Training STUDENT model to learn from TEACHER explanations")
        logger.info("Total steps: %d", total_steps)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_metrics = self.train_epoch(train_dataloader)
            
            logger.info("Epoch %d/%d completed", epoch + 1, num_epochs)
            logger.info("Average reward: %.4f", epoch_metrics['avg_reward'])
            logger.info("Average loss: %.4f", epoch_metrics['avg_loss'])
            
            # Evaluation
            if eval_dataloader and (epoch + 1) % self.config.eval_steps == 0:
                eval_metrics = self.evaluate(eval_dataloader)
                logger.info("Evaluation reward: %.4f", eval_metrics['avg_reward'])
                
                # Save best model
                if eval_metrics['avg_reward'] > self.best_reward:
                    self.best_reward = eval_metrics['avg_reward']
                    self.save_checkpoint(is_best=True)
    
    def train_epoch(self, dataloader: DataLoader) -> Dict:
        """Train for one epoch."""
        # Set student model to training mode
        if hasattr(self.student_model, 'model'):
            self.student_model.model.train()
        else:
            self.student_model.train()
        
        epoch_rewards = []
        epoch_losses = []
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Generate teacher explanations and compute rewards
            batch_results = self.generate_and_score_explanations(batch)
            
            # Train student model on these explanations
            loss, metrics = self.train_student_on_explanations(batch_results)
            
            # Backward pass
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients
                if hasattr(self.student_model, 'model'):
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.model.parameters(),
                        self.config.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(),
                        self.config.max_grad_norm
                    )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Track metrics
            epoch_rewards.extend(metrics['rewards'])
            epoch_losses.append(loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'reward': f"{np.mean(metrics['rewards']):.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Checkpoint
            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint()
        
        return {
            'avg_reward': np.mean(epoch_rewards),
            'avg_loss': np.mean(epoch_losses),
            'total_steps': len(dataloader)
        }
    
    def generate_and_score_explanations(self, batch: Dict) -> Dict:
        """
        Generate teacher explanations and compute rewards.
        Teacher generates explanations, evaluator scores them.
        """
        questions = batch['questions']
        answers = batch['answers']
        batch_size = len(questions)
        
        all_explanations = []
        all_rewards = []
        
        # Generate explanations from teacher (Claude API)
        for i in range(batch_size):
            question = questions[i]
            answer = answers[i]
            
            explanations_for_item = []
            rewards_for_item = []
            
            # Generate multiple explanations with different temperatures
            for g in range(self.config.group_size):
                temperature = 0.6 + (g * 0.1)  # Vary temperature
                
                # Teacher generates explanation
                if hasattr(self.teacher, 'generate_explanation'):
                    # Claude teacher
                    explanation = self.teacher.generate_explanation(
                        question=question,
                        answer=answer,
                        temperature=temperature
                    )
                else:
                    # HF teacher model
                    result = self.teacher.generate_explanation(
                        question=question,
                        answer=answer,
                        temperature=temperature
                    )
                    explanation = result.get('explanation', result)
                
                # Compute reward based on how well student would understand
                reward_result = self.reward_fn.compute_reward(
                    [explanation],
                    [question]
                )
                reward = reward_result['rewards'][0]
                
                explanations_for_item.append(explanation)
                rewards_for_item.append(reward)
            
            all_explanations.append(explanations_for_item)
            all_rewards.append(rewards_for_item)
        
        return {
            'explanations': all_explanations,
            'rewards': all_rewards,
            'questions': questions,
            'answers': answers
        }
    
    def train_student_on_explanations(self, batch_results: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Train the student model to generate good responses given teacher explanations.
        This is where the actual learning happens!
        """
        all_rewards = batch_results['rewards']
        all_explanations = batch_results['explanations']
        questions = batch_results['questions']
        answers = batch_results['answers']
        
        total_loss = 0
        all_advantages = []
        flat_rewards = []
        
        for item_idx, (item_rewards, item_explanations) in enumerate(zip(all_rewards, all_explanations)):
            # Convert to tensors
            rewards = torch.tensor(item_rewards, dtype=torch.float32)
            
            # Compute advantages within group (GRPO)
            if self.config.normalize_rewards:
                if self.config.reward_baseline == "mean":
                    baseline = rewards.mean()
                else:  # min
                    baseline = rewards.min()
                advantages = rewards - baseline
                
                # Normalize advantages
                if rewards.std() > 1e-6:
                    advantages = advantages / (rewards.std() + 1e-6)
            else:
                advantages = rewards
            
            # Train student on each explanation weighted by advantage
            question = questions[item_idx]
            answer = answers[item_idx]
            
            for i, (explanation, advantage) in enumerate(zip(item_explanations, advantages)):
                # Create training input for student:
                # Student should learn to solve problems using teacher's explanation
                student_input = f"Question: {question}\nExplanation: {explanation}\nAnswer:"
                student_target = answer
                
                # Compute student loss (standard language modeling loss weighted by advantage)
                if hasattr(self.student_model, 'compute_loss'):
                    # For models with compute_loss method
                    loss = self.student_model.compute_loss(
                        input_text=student_input,
                        target_text=student_target
                    )
                else:
                    # For OptimizedHFModel, we need to implement this
                    loss = self._compute_student_loss(student_input, student_target)
                
                # Weight loss by advantage (GRPO key insight)
                weighted_loss = loss * advantage.item()
                total_loss += weighted_loss
            
            all_advantages.extend(advantages.tolist())
            flat_rewards.extend(item_rewards)
        
        # Average loss
        num_items = len(flat_rewards)
        total_loss = total_loss / max(1, num_items)
        
        metrics = {
            'rewards': flat_rewards,
            'advantages': all_advantages,
            'num_items': num_items
        }
        
        return total_loss, metrics
    
    def _compute_student_loss(self, input_text: str, target_text: str) -> torch.Tensor:
        """Compute language modeling loss for student model."""
        # Tokenize input and target
        if hasattr(self.student_model, 'tokenizer'):
            tokenizer = self.student_model.tokenizer
            model = self.student_model.model
        else:
            raise ValueError("Student model must have tokenizer attribute")
        
        # Prepare input
        full_text = input_text + " " + target_text
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # Get the position where target starts
        input_ids = inputs.input_ids
        input_tokens = tokenizer(input_text, return_tensors="pt").input_ids
        target_start_pos = input_tokens.shape[1]
        
        # Create labels (mask out the input part)
        labels = input_ids.clone()
        labels[0, :target_start_pos] = -100  # Ignore input part in loss
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels
        )
        
        return outputs.loss
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict:
        """Evaluate student model performance."""
        if hasattr(self.student_model, 'model'):
            self.student_model.model.eval()
        else:
            self.student_model.eval()
        
        all_rewards = []
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch_results = self.generate_and_score_explanations(batch)
                
                # Flatten rewards
                for item_rewards in batch_results['rewards']:
                    all_rewards.extend(item_rewards)
        
        return {
            'avg_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'max_reward': np.max(all_rewards),
            'num_samples': len(all_rewards)
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': self.metrics,
            'config': self.config.__dict__
        }
        
        # Save student model state
        if hasattr(self.student_model, 'save_model'):
            model_path = os.path.join(self.config.checkpoint_dir, f'student_model_step_{self.global_step}')
            self.student_model.save_model(model_path)
        else:
            checkpoint['model_state_dict'] = self.student_model.state_dict()
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f'checkpoint_step_{self.global_step}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save as best if specified
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info("Saved best model with reward: %.4f", self.best_reward)
        
        # Save metrics
        metrics_path = os.path.join(self.config.checkpoint_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)