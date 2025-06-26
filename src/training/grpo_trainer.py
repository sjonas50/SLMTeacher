"""
Group Relative Policy Optimization (GRPO) Trainer for RLT
"""
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import os
import json
from dataclasses import dataclass


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
    """Implements Group Relative Policy Optimization for RLT."""
    
    def __init__(
        self,
        teacher_model,
        student_evaluator,
        reward_function,
        config: GRPOConfig
    ):
        self.teacher = teacher_model
        self.student = student_evaluator
        self.reward_fn = reward_function
        self.config = config
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.teacher.model.parameters(),
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
        
        print(f"Starting GRPO training for {num_epochs} epochs")
        print(f"Total steps: {total_steps}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_metrics = self.train_epoch(train_dataloader)
            
            print(f"\nEpoch {epoch+1}/{num_epochs} completed")
            print(f"Average reward: {epoch_metrics['avg_reward']:.4f}")
            print(f"Average loss: {epoch_metrics['avg_loss']:.4f}")
            
            # Evaluation
            if eval_dataloader and (epoch + 1) % self.config.eval_steps == 0:
                eval_metrics = self.evaluate(eval_dataloader)
                print(f"Evaluation reward: {eval_metrics['avg_reward']:.4f}")
                
                # Save best model
                if eval_metrics['avg_reward'] > self.best_reward:
                    self.best_reward = eval_metrics['avg_reward']
                    self.save_checkpoint(is_best=True)
    
    def train_epoch(self, dataloader: DataLoader) -> Dict:
        """Train for one epoch."""
        self.teacher.model.train()
        
        epoch_rewards = []
        epoch_losses = []
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Generate multiple explanations per question
            batch_results = self.generate_and_score_explanations(batch)
            
            # Compute GRPO loss
            loss, metrics = self.compute_grpo_loss(batch_results)
            
            # Backward pass
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.teacher.model.parameters(),
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
        """Generate multiple explanations and compute rewards."""
        questions = batch['questions']
        answers = batch['answers']
        batch_size = len(questions)
        
        all_explanations = []
        all_rewards = []
        all_logprobs = []
        
        # Generate G explanations for each question
        for i in range(batch_size):
            question = questions[i]
            answer = answers[i]
            
            explanations_for_item = []
            rewards_for_item = []
            logprobs_for_item = []
            
            # Generate multiple explanations with different temperatures
            for g in range(self.config.group_size):
                temperature = 0.6 + (g * 0.1)  # Vary temperature
                
                # Generate explanation
                result = self.teacher.generate_explanation(
                    question=question,
                    answer=answer,
                    temperature=temperature,
                    return_logprobs=True
                )
                
                explanation = result['explanation']
                logprobs = result.get('logprobs', None)
                
                # Compute reward
                reward_result = self.reward_fn.compute_reward(
                    [explanation],
                    [question]
                )
                reward = reward_result['rewards'][0]
                
                explanations_for_item.append(explanation)
                rewards_for_item.append(reward)
                logprobs_for_item.append(logprobs)
            
            all_explanations.append(explanations_for_item)
            all_rewards.append(rewards_for_item)
            all_logprobs.append(logprobs_for_item)
        
        return {
            'explanations': all_explanations,
            'rewards': all_rewards,
            'logprobs': all_logprobs,
            'questions': questions,
            'answers': answers
        }
    
    def compute_grpo_loss(self, batch_results: Dict) -> Tuple[torch.Tensor, Dict]:
        """Compute GRPO loss with group-based advantages."""
        all_rewards = batch_results['rewards']
        all_logprobs = batch_results['logprobs']
        
        total_loss = 0
        all_advantages = []
        flat_rewards = []
        
        for item_rewards, item_logprobs in zip(all_rewards, all_logprobs):
            # Convert to tensors
            rewards = torch.tensor(item_rewards, dtype=torch.float32)
            
            # Compute advantages within group
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
            
            # Compute policy loss for this group
            for i, (logprob, advantage) in enumerate(zip(item_logprobs, advantages)):
                if logprob is not None:
                    # PPO-style clipped loss
                    policy_loss = -logprob.mean() * advantage.item()
                    total_loss += policy_loss
            
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
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict:
        """Evaluate model on validation set."""
        self.teacher.model.eval()
        
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
            'model_state_dict': self.teacher.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': self.metrics,
            'config': self.config.__dict__
        }
        
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
            print(f"Saved best model with reward: {self.best_reward:.4f}")
        
        # Save metrics
        metrics_path = os.path.join(self.config.checkpoint_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        
        self.teacher.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.metrics = checkpoint['metrics']
        
        print(f"Loaded checkpoint from step {self.global_step}")