"""
Group Relative Policy Optimization (GRPO) Trainer for RLT

Trains the STUDENT model to learn from TEACHER explanations.
"""
import copy
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_steps: int = 0  # Explicit warmup steps (overrides warmup_ratio when > 0)
    warmup_ratio: float = 0.1  # Fraction of total steps used for warmup
    lr_scheduler_type: str = "cosine"  # "cosine" or "linear"
    max_grad_norm: float = 1.0

    # GRPO specific
    group_size: int = 6  # Number of explanations per question
    clip_epsilon: float = 0.2  # PPO-style clipping
    kl_coef: float = 0.1  # KL penalty coefficient (policy vs reference)
    entropy_coef: float = 0.01
    ref_update_freq: int = 1  # Re-snapshot reference policy every N epochs

    # Reward normalization
    normalize_rewards: bool = True
    reward_baseline: str = "mean"  # "mean" or "min"

    # Checkpointing
    save_steps: int = 100
    eval_steps: int = 50
    checkpoint_dir: str = "./checkpoints"

    # Experiment tracking (None, "wandb", or "tensorboard")
    tracker: Optional[str] = None
    tracker_project: str = "slmteacher"
    tracker_run_name: Optional[str] = None


class ExperimentTracker:
    """Thin wrapper around W&B or TensorBoard for experiment tracking."""

    def __init__(self, config: GRPOConfig, extra_config: Optional[Dict] = None):
        self.backend = config.tracker
        self._writer = None
        self._run = None

        if self.backend == "wandb":
            try:
                import wandb
                self._run = wandb.init(
                    project=config.tracker_project,
                    name=config.tracker_run_name,
                    config={**(extra_config or {}), **config.__dict__},
                    reinit=True,
                )
                logger.info("W&B tracking enabled: %s/%s",
                             config.tracker_project, config.tracker_run_name)
            except ImportError:
                logger.warning("wandb not installed. Install with: pip install wandb")
                self.backend = None

        elif self.backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = os.path.join(config.checkpoint_dir, "tensorboard")
                self._writer = SummaryWriter(log_dir=log_dir)
                logger.info("TensorBoard tracking enabled: %s", log_dir)
            except ImportError:
                logger.warning("tensorboard not installed. Install with: pip install tensorboard")
                self.backend = None

    def log(self, metrics: Dict[str, float], step: int):
        """Log scalar metrics."""
        if self.backend == "wandb":
            import wandb
            wandb.log(metrics, step=step)
        elif self.backend == "tensorboard" and self._writer:
            for key, value in metrics.items():
                self._writer.add_scalar(key, value, global_step=step)

    def finish(self):
        """Close the tracker."""
        if self.backend == "wandb":
            import wandb
            wandb.finish()
        elif self.backend == "tensorboard" and self._writer:
            self._writer.close()


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
        config: GRPOConfig,
        explanation_dataset=None,  # Pre-generated ExplanationDataset
    ):
        self.teacher = teacher
        self.student_model = student_model  # This is what we train!
        self.student_evaluator = student_evaluator
        self.reward_fn = reward_function
        self.config = config
        self.explanation_dataset = explanation_dataset

        # Setup optimizer for STUDENT model parameters
        if hasattr(self.student_model, 'model'):
            model_params = self.student_model.model.parameters()
        else:
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

        # Experiment tracker
        self.tracker = ExperimentTracker(config) if config.tracker else None

    def _create_scheduler(self, total_steps: int, warmup_override: Optional[int] = None):
        """Create a learning rate scheduler based on config.

        Args:
            total_steps: Total number of optimizer steps.
            warmup_override: If set, use this for warmup steps instead of config.
        """
        if warmup_override is not None:
            warmup_steps = warmup_override
        elif self.config.warmup_steps > 0:
            warmup_steps = self.config.warmup_steps
        else:
            warmup_steps = int(total_steps * self.config.warmup_ratio)

        if self.config.lr_scheduler_type == "cosine":
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None
    ):
        """Main training loop."""
        num_epochs = num_epochs or self.config.num_epochs

        # Reset optimizer LR to configured value before creating a fresh scheduler.
        # This is needed because a prior SFT warmup scheduler may have decayed it.
        for group in self.optimizer.param_groups:
            group['lr'] = self.config.learning_rate
            group.pop('initial_lr', None)  # Force LambdaLR to re-read current lr

        # Setup scheduler
        total_steps = len(train_dataloader) * num_epochs
        self.scheduler = self._create_scheduler(total_steps)

        logger.info("Starting GRPO training for %d epochs", num_epochs)
        logger.info("Training STUDENT model to learn from TEACHER explanations")
        logger.info("Total steps: %d", total_steps)

        # Snapshot reference policy before training begins
        self._snapshot_reference_policy()

        try:
            for epoch in range(num_epochs):
                self.current_epoch = epoch

                # Re-snapshot reference policy based on frequency
                if epoch > 0 and epoch % self.config.ref_update_freq == 0:
                    self._snapshot_reference_policy()

                epoch_metrics = self.train_epoch(train_dataloader)

                logger.info("Epoch %d/%d completed", epoch + 1, num_epochs)
                logger.info("Average reward: %.4f", epoch_metrics['avg_reward'])
                logger.info("Average loss: %.4f", epoch_metrics['avg_loss'])

                if self.tracker:
                    self.tracker.log({
                        "epoch/avg_reward": epoch_metrics['avg_reward'],
                        "epoch/avg_loss": epoch_metrics['avg_loss'],
                        "epoch": epoch + 1,
                    }, step=self.global_step)

                # Evaluation
                if eval_dataloader:
                    eval_metrics = self.evaluate(eval_dataloader)
                    logger.info("Evaluation reward: %.4f", eval_metrics['avg_reward'])

                    if self.tracker:
                        self.tracker.log({
                            "eval/avg_reward": eval_metrics['avg_reward'],
                            "eval/std_reward": eval_metrics['std_reward'],
                            "eval/max_reward": eval_metrics['max_reward'],
                        }, step=self.global_step)

                    if eval_metrics['avg_reward'] > self.best_reward:
                        self.best_reward = eval_metrics['avg_reward']
                        self.save_checkpoint(is_best=True)
        finally:
            if self.tracker:
                self.tracker.finish()

    def train_epoch(self, dataloader: DataLoader) -> Dict:
        """Train for one epoch."""
        if hasattr(self.student_model, 'model'):
            self.student_model.model.train()
        else:
            self.student_model.train()

        epoch_rewards = []
        epoch_losses = []

        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch+1}")

        for batch_idx, batch in enumerate(progress_bar):
            batch_results = self.generate_and_score_explanations(batch)
            loss, metrics = self.train_student_on_explanations(batch_results)

            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
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

                # Log step-level metrics
                if self.tracker:
                    log_data = {
                        "train/loss": loss.item(),
                        "train/reward_mean": float(np.mean(metrics['rewards'])),
                        "train/reward_std": float(np.std(metrics['rewards'])),
                        "train/lr": self.scheduler.get_last_lr()[0],
                        "train/advantage_mean": float(np.mean(metrics['advantages'])),
                    }
                    if 'mean_ratio' in metrics:
                        log_data["train/policy_ratio"] = metrics['mean_ratio']
                    self.tracker.log(log_data, step=self.global_step)

            epoch_rewards.extend(metrics['rewards'])
            epoch_losses.append(loss.item())

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'reward': f"{np.mean(metrics['rewards']):.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

            if self.global_step > 0 and self.global_step % self.config.save_steps == 0:
                self.save_checkpoint()

        return {
            'avg_reward': np.mean(epoch_rewards),
            'avg_loss': np.mean(epoch_losses),
            'total_steps': len(dataloader)
        }

    def generate_and_score_explanations(self, batch: Dict) -> Dict:
        """Generate teacher explanations and compute rewards.

        When ``self.explanation_dataset`` is set, pre-generated explanations
        are used instead of calling the teacher API.
        """
        questions = batch['questions']
        answers = batch['answers']
        subjects = batch.get('subjects', ['default'] * len(questions))
        difficulties = batch.get('difficulties', ['medium'] * len(questions))
        batch_size = len(questions)

        all_explanations = []
        all_rewards = []

        for i in range(batch_size):
            question = questions[i]
            answer = answers[i]
            subject = subjects[i] if i < len(subjects) else 'default'
            difficulty = difficulties[i] if i < len(difficulties) else 'medium'

            explanations_for_item = []
            rewards_for_item = []

            # Try pre-generated explanations first
            if self.explanation_dataset is not None:
                cached = self.explanation_dataset.get_explanations(
                    question, self.config.group_size
                )
            else:
                cached = []

            for g in range(self.config.group_size):
                if g < len(cached):
                    explanation = cached[g]["explanation"]
                else:
                    temperature = min(1.0, 0.6 + (g * 0.1))
                    explanation = self.teacher.generate_explanation(
                        question=question,
                        answer=answer,
                        temperature=temperature,
                        subject=subject,
                        difficulty=difficulty,
                    )
                    if isinstance(explanation, dict):
                        explanation = explanation.get('explanation', str(explanation))

                reward_result = self.reward_fn.compute_reward(
                    [explanation],
                    [question],
                    answers=[answer],
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

    # ------------------------------------------------------------------
    # Reference policy for PPO-clipped GRPO
    # ------------------------------------------------------------------
    def _snapshot_reference_policy(self):
        """Freeze a copy of the current student model as the reference policy."""
        if hasattr(self.student_model, 'model'):
            model = self.student_model.model
        else:
            model = self.student_model

        # For PEFT models, copy only the adapter weights (much cheaper)
        try:
            from peft import PeftModel
            if isinstance(model, PeftModel):
                self._ref_adapter_state = copy.deepcopy(
                    {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}
                )
                self._ref_is_peft = True
                logger.info("Reference policy snapshot: PEFT adapter (%d params)",
                            len(self._ref_adapter_state))
                return
        except ImportError:
            pass

        # Full model copy — freeze all parameters
        self._ref_model = copy.deepcopy(model)
        for p in self._ref_model.parameters():
            p.requires_grad = False
        self._ref_model.eval()
        self._ref_is_peft = False
        logger.info("Reference policy snapshot: full model copy")

    def _compute_ref_log_probs(self, input_text: str, target_text: str) -> torch.Tensor:
        """Compute log probs under the reference policy (no gradients)."""
        # For PEFT adapters, temporarily swap weights
        if getattr(self, '_ref_is_peft', False):
            model = self.student_model.model if hasattr(self.student_model, 'model') else self.student_model
            # Save current adapter weights, load reference
            current_state = {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in self._ref_adapter_state:
                        param.copy_(self._ref_adapter_state[name])

            ref_log_prob = self.student_model.compute_log_probs(input_text, target_text).detach()

            # Restore current weights
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in current_state:
                        param.copy_(current_state[name])

            return ref_log_prob

        # For full model copy, use the reference directly
        if hasattr(self.student_model, 'tokenizer'):
            tokenizer = self.student_model.tokenizer
            device = self.student_model.device
        else:
            raise ValueError("Student model must have tokenizer attribute")

        full_text = input_text + target_text
        prefix_encoding = tokenizer(
            input_text, return_tensors="pt", truncation=True,
            max_length=tokenizer.model_max_length,
            add_special_tokens=True,
        )
        prefix_len = prefix_encoding.input_ids.shape[1]

        full_encoding = tokenizer(
            full_text, return_tensors="pt", truncation=True,
            max_length=tokenizer.model_max_length,
            add_special_tokens=True,
        ).to(device)

        input_ids = full_encoding.input_ids
        with torch.no_grad():
            outputs = self._ref_model(input_ids=input_ids)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(
                dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            target_start = max(0, min(prefix_len - 1, token_log_probs.shape[1]))
            target_log_probs = token_log_probs[:, target_start:]

            if target_log_probs.numel() == 0:
                return torch.tensor(-100.0, device=device)
            return target_log_probs.sum()

    def train_student_on_explanations(self, batch_results: Dict) -> Tuple[torch.Tensor, Dict]:
        """Train the student with PPO-clipped policy gradient (proper GRPO).

        For each (question, explanation, advantage) triple:
          1. Compute log π_θ(answer | question + explanation)      (current policy, with grad)
          2. Compute log π_ref(answer | question + explanation)     (reference policy, no grad)
          3. ratio = exp(log π_θ - log π_ref)
          4. clipped_ratio = clamp(ratio, 1-ε, 1+ε)
          5. policy_loss = -min(ratio * A, clipped_ratio * A)
          6. kl_penalty = kl_coef * (log π_θ - log π_ref)
          7. entropy_bonus = -entropy_coef * log π_θ
          8. loss = policy_loss + kl_penalty + entropy_bonus
        """
        all_rewards = batch_results['rewards']
        all_explanations = batch_results['explanations']
        questions = batch_results['questions']
        answers = batch_results['answers']

        total_loss = torch.tensor(0.0, requires_grad=True)
        all_advantages = []
        flat_rewards = []
        all_ratios = []

        has_ref = hasattr(self, '_ref_adapter_state') or hasattr(self, '_ref_model')

        for item_idx, (item_rewards, item_explanations) in enumerate(zip(all_rewards, all_explanations)):
            rewards = torch.tensor(item_rewards, dtype=torch.float32)

            if self.config.normalize_rewards:
                if self.config.reward_baseline == "mean":
                    baseline = rewards.mean()
                else:
                    baseline = rewards.min()
                advantages = rewards - baseline
                if rewards.std() > 1e-6:
                    advantages = advantages / (rewards.std() + 1e-6)
            else:
                advantages = rewards

            question = questions[item_idx]
            answer = answers[item_idx]

            for explanation, advantage in zip(item_explanations, advantages):
                student_input = f"Question: {question}\nExplanation: {explanation}\nAnswer: "
                student_target = answer

                # Current policy log prob (with gradients)
                if hasattr(self.student_model, 'compute_log_probs'):
                    log_prob = self.student_model.compute_log_probs(student_input, student_target)
                else:
                    # Fallback: use negative loss as proxy
                    loss = self._compute_student_loss(student_input, student_target)
                    log_prob = -loss

                if has_ref:
                    ref_log_prob = self._compute_ref_log_probs(student_input, student_target)

                    ratio = torch.exp(log_prob - ref_log_prob)
                    clipped_ratio = torch.clamp(
                        ratio,
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    )

                    adv_tensor = torch.tensor(advantage.item() if hasattr(advantage, 'item') else advantage,
                                              device=log_prob.device)
                    policy_loss = -torch.min(ratio * adv_tensor, clipped_ratio * adv_tensor)
                    kl_penalty = self.config.kl_coef * (log_prob - ref_log_prob)
                    entropy_bonus = -self.config.entropy_coef * log_prob

                    step_loss = policy_loss + kl_penalty + entropy_bonus
                    all_ratios.append(ratio.item())
                else:
                    # Before first snapshot: fall back to reward-weighted loss
                    if hasattr(self.student_model, 'compute_loss'):
                        ce_loss = self.student_model.compute_loss(student_input, student_target)
                    else:
                        ce_loss = self._compute_student_loss(student_input, student_target)
                    adv_val = advantage.item() if hasattr(advantage, 'item') else advantage
                    step_loss = ce_loss * adv_val

                total_loss = total_loss + step_loss

            all_advantages.extend(advantages.tolist())
            flat_rewards.extend(item_rewards)

        num_items = max(1, len(flat_rewards))
        total_loss = total_loss / num_items

        metrics = {
            'rewards': flat_rewards,
            'advantages': all_advantages,
            'num_items': num_items,
        }
        if all_ratios:
            metrics['mean_ratio'] = float(np.mean(all_ratios))

        return total_loss, metrics

    def _compute_student_loss(self, input_text: str, target_text: str) -> torch.Tensor:
        """Compute language modeling loss for student model."""
        if hasattr(self.student_model, 'compute_loss'):
            return self.student_model.compute_loss(input_text, target_text)

        if hasattr(self.student_model, 'tokenizer'):
            tokenizer = self.student_model.tokenizer
            model = self.student_model.model
        else:
            raise ValueError("Student model must have tokenizer attribute")

        full_text = input_text + target_text
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length,
            add_special_tokens=True,
        ).to(model.device)

        input_ids = inputs.input_ids
        prefix_encoding = tokenizer(
            input_text, return_tensors="pt", truncation=True,
            max_length=tokenizer.model_max_length,
            add_special_tokens=True,
        )
        target_start_pos = prefix_encoding.input_ids.shape[1]

        labels = input_ids.clone()
        labels[0, :target_start_pos] = -100

        outputs = model(
            input_ids=input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels
        )

        return outputs.loss

    def sft_warmup(
        self,
        train_dataloader: DataLoader,
        num_epochs: int = 2,
    ):
        """Supervised fine-tuning warmup before GRPO.

        Trains the student to predict the answer given (question + explanation)
        using standard cross-entropy loss.  No rewards, no policy gradient.
        This gives the model a baseline ability to follow explanations before
        GRPO refines it.
        """
        if hasattr(self.student_model, 'model'):
            self.student_model.model.train()
        else:
            self.student_model.train()

        total_steps = len(train_dataloader) * num_epochs
        sft_scheduler = self._create_scheduler(
            total_steps,
            warmup_override=min(
                max(1, int(total_steps * self.config.warmup_ratio)),
                total_steps // 4,
            ),
        )

        logger.info("Starting SFT warmup for %d epochs (%d steps)", num_epochs, total_steps)

        for epoch in range(num_epochs):
            epoch_losses = []
            progress_bar = tqdm(train_dataloader, desc=f"SFT Warmup {epoch+1}/{num_epochs}")

            for batch_idx, batch in enumerate(progress_bar):
                batch_results = self.generate_and_score_explanations(batch)

                batch_loss = torch.tensor(0.0, requires_grad=True)
                n_items = 0

                for item_idx in range(len(batch_results['questions'])):
                    question = batch_results['questions'][item_idx]
                    answer = batch_results['answers'][item_idx]
                    # Use the first explanation (typically highest quality)
                    explanation = batch_results['explanations'][item_idx][0]

                    student_input = f"Question: {question}\nExplanation: {explanation}\nAnswer: "
                    loss = self.student_model.compute_loss(student_input, answer)
                    batch_loss = batch_loss + loss
                    n_items += 1

                batch_loss = batch_loss / max(1, n_items)
                batch_loss = batch_loss / self.config.gradient_accumulation_steps

                # Skip NaN losses to prevent gradient corruption
                if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                    logger.warning("SFT batch %d: NaN/Inf loss, skipping", batch_idx)
                    self.optimizer.zero_grad()
                    epoch_losses.append(0.0)
                    progress_bar.set_postfix({'loss': 'NaN(skip)'})
                    continue

                batch_loss.backward()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if hasattr(self.student_model, 'model'):
                        torch.nn.utils.clip_grad_norm_(
                            self.student_model.model.parameters(),
                            self.config.max_grad_norm,
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.student_model.parameters(),
                            self.config.max_grad_norm,
                        )
                    self.optimizer.step()
                    sft_scheduler.step()
                    self.optimizer.zero_grad()

                epoch_losses.append(batch_loss.item())
                progress_bar.set_postfix({'loss': f"{batch_loss.item():.4f}"})

            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            logger.info("SFT Warmup epoch %d/%d — avg loss: %.4f", epoch + 1, num_epochs, avg_loss)

        # Reset optimizer state for GRPO phase
        self.optimizer.zero_grad()
        logger.info("SFT warmup complete.")

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

        if hasattr(self.student_model, 'save_model'):
            model_path = os.path.join(self.config.checkpoint_dir, f'student_model_step_{self.global_step}')
            self.student_model.save_model(model_path)
        else:
            checkpoint['model_state_dict'] = self.student_model.state_dict()

        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f'checkpoint_step_{self.global_step}.pt'
        )
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info("Saved best model with reward: %.4f", self.best_reward)

        metrics_path = os.path.join(self.config.checkpoint_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
