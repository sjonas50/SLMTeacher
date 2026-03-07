#!/usr/bin/env python3
"""
RLT Training Script

Trains a STUDENT model to learn from TEACHER (Claude) explanations
using Group Relative Policy Optimization (GRPO).

Supports three evaluation modes:
  - local:  Use a local HF model for reward evaluation (fastest, cheapest)
  - claude: Use Claude API for evaluation (highest quality, most expensive)
  - hybrid: Use Claude for a fraction of evaluations (balanced)
"""
import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import set_seed

# Import our modules
from src.teachers.claude_teacher import ClaudeRLTTeacher, ClaudeConfig, CacheConfig
from src.models.optimized_model import OptimizedHFModel, OptimizedModelConfig
from src.rewards.reward_function import RewardFunction
from src.rewards.student_evaluator import StudentEvaluator
from src.training.grpo_trainer import GRPOTrainer, GRPOConfig
from src.data.data_processor import DataProcessor
from src.utils.cost_tracker import CostTracker


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RLTTrainer:
    """Main class for RLT training with configurable evaluation modes."""

    def __init__(self, config: Dict):
        self.config = config
        self.setup_directories()

        # Set random seed
        set_seed(config.get('seed', 42))

        # Initialize components
        self.setup_teacher()
        self.setup_student()
        self.setup_evaluator()
        self.setup_training()

        eval_mode = self.config.get('evaluation', {}).get('mode', 'local')
        logger.info("RLT Trainer initialized successfully")
        logger.info("Teacher: Claude API (generates explanations)")
        logger.info("Student: Local HF model (learns from explanations)")
        logger.info("Evaluator: %s mode", eval_mode)

    def setup_directories(self):
        """Create necessary directories."""
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

    def setup_teacher(self):
        """Initialize Claude teacher for generating explanations."""
        budget_limit = self.config.get('budget_limit',
                                       self.config.get('teacher', {}).get('budget_limit', 50.0))
        self.cost_tracker = CostTracker(budget_limit=budget_limit)

        claude_config = ClaudeConfig(
            model=self.config['teacher'].get('model', 'claude-sonnet-4-6-20250514'),
            temperature=self.config['teacher'].get('temperature', 0.7),
            max_tokens=self.config['teacher'].get('max_tokens', 1024)
        )

        cache_config = CacheConfig(
            enabled=True,
            max_size=self.config['teacher'].get('cache_size', 10000),
            ttl_hours=self.config['teacher'].get('cache_ttl_hours', 168)
        )

        self.teacher = ClaudeRLTTeacher(
            api_key=os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
            cost_tracker=self.cost_tracker,
            claude_config=claude_config,
            cache_config=cache_config
        )

        logger.info("Claude teacher initialized (model: %s)", claude_config.model)

    def setup_student(self):
        """Initialize student model that will learn from teacher explanations."""
        model_config = OptimizedModelConfig(
            model_name=self.config['student']['model_name'],
            use_4bit=self.config['student'].get('use_4bit', True),
            use_8bit=self.config['student'].get('use_8bit', False),
            bnb_4bit_compute_dtype=self.config['student'].get('compute_dtype', 'bfloat16'),
            use_flash_attention_2=self.config['student'].get('use_flash_attention', True),
            use_peft=self.config['student'].get('use_peft', True),
            peft_method=self.config['student'].get('peft_method', 'adalora'),
            lora_r=self.config['student'].get('lora_r', 32),
            lora_alpha=self.config['student'].get('lora_alpha', 64),
            adalora_init_r=self.config['student'].get('adalora_init_r', 12),
            adalora_target_r=self.config['student'].get('adalora_target_r', 32),
            gradient_checkpointing=self.config['student'].get('gradient_checkpointing', True),
            mixed_precision=self.config['student'].get('mixed_precision', 'bf16')
        )

        self.student_model = OptimizedHFModel(model_config)

        logger.info("Student model initialized: %s", model_config.model_name)
        logger.info("Optimizations: 4bit=%s, PEFT=%s, FlashAttn=%s",
                     model_config.use_4bit, model_config.peft_method,
                     model_config.use_flash_attention_2)

    def setup_evaluator(self):
        """Initialize evaluator with local, claude, or hybrid mode."""
        eval_mode = self.config.get('evaluation', {}).get('mode', 'local')

        if eval_mode == 'claude':
            from src.rewards.claude_evaluator import ClaudeStudentEvaluator, ClaudeEvaluatorConfig
            eval_config = ClaudeEvaluatorConfig(
                model=self.config['evaluation'].get('model', 'claude-sonnet-4-6-20250514'),
                temperature=self.config['evaluation'].get('temperature', 0.1),
                max_tokens=self.config['evaluation'].get('max_tokens', 512)
            )
            self.student_evaluator = ClaudeStudentEvaluator(
                api_key=os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
                cost_tracker=self.cost_tracker,
                config=eval_config
            )
            logger.info("Using Claude for student evaluation (highest quality)")

        elif eval_mode == 'hybrid':
            from src.rewards.claude_evaluator import (
                ClaudeStudentEvaluator, ClaudeEvaluatorConfig, HybridEvaluator
            )
            eval_config = ClaudeEvaluatorConfig(
                model=self.config['evaluation'].get('model', 'claude-sonnet-4-6-20250514'),
                temperature=self.config['evaluation'].get('temperature', 0.1)
            )
            claude_evaluator = ClaudeStudentEvaluator(
                api_key=os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
                cost_tracker=self.cost_tracker,
                config=eval_config
            )
            eval_device = self._get_device()
            local_evaluator = StudentEvaluator(
                model_name=self.config['evaluation'].get('local_model',
                                                          self.config['student']['model_name']),
                device=eval_device
            )
            claude_freq = self.config['evaluation'].get('claude_frequency', 0.1)
            self.student_evaluator = HybridEvaluator(
                claude_evaluator=claude_evaluator,
                local_evaluator=local_evaluator,
                claude_eval_frequency=claude_freq
            )
            logger.info("Using hybrid evaluation (Claude %.0f%% of the time)", claude_freq * 100)

        else:  # local (default)
            eval_device = self._get_device()
            evaluation_model = self.config.get('evaluation', {}).get(
                'model_name', self.config['student']['model_name'])
            self.student_evaluator = StudentEvaluator(
                model_name=evaluation_model,
                device=eval_device,
                batch_size=self.config.get('evaluation', {}).get('batch_size', 8)
            )
            logger.info("Using local model for student evaluation")

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def setup_training(self):
        """Initialize training components."""
        self.reward_function = RewardFunction(
            student_evaluator=self.student_evaluator,
            kl_weight=self.config['training'].get('kl_weight', 0.1),
            correctness_weight=self.config['training'].get('correctness_weight', 0.5),
            confidence_weight=self.config['training'].get('confidence_weight', 0.3),
            length_penalty_weight=self.config['training'].get('length_penalty_weight', 0.1)
        )

        grpo_config = GRPOConfig(
            learning_rate=self.config['training'].get('learning_rate', 2e-5),
            batch_size=self.config['training'].get('batch_size', 1),
            gradient_accumulation_steps=self.config['training'].get('gradient_accumulation_steps', 8),
            num_epochs=self.config['training'].get('num_epochs', 3),
            group_size=self.config['training'].get('group_size', 4),
            checkpoint_dir=str(self.checkpoint_dir),
            # Experiment tracking
            tracker=self.config.get('tracking', {}).get('backend'),
            tracker_project=self.config.get('tracking', {}).get('project', 'slmteacher'),
            tracker_run_name=self.config.get('tracking', {}).get('run_name'),
        )

        self.trainer = GRPOTrainer(
            teacher=self.teacher,
            student_model=self.student_model,
            student_evaluator=self.student_evaluator,
            reward_function=self.reward_function,
            config=grpo_config
        )

        logger.info("Training components initialized")

    def prepare_data(self):
        """Prepare training and evaluation data."""
        data_processor = DataProcessor(
            cache_dir=self.config['data'].get('cache_dir', './data_cache')
        )

        dataset_name = self.config['data'].get('dataset', 'gsm8k')
        train_data = data_processor.load_dataset(dataset_name, split='train')
        eval_data = data_processor.load_dataset(dataset_name, split='test')

        train_processed = data_processor.process_for_rlt(
            train_data,
            max_samples=self.config['data'].get('max_train_samples', 1000)
        )
        eval_processed = data_processor.process_for_rlt(
            eval_data,
            max_samples=self.config['data'].get('max_eval_samples', 100)
        )

        train_processed.sort(key=lambda x: len(x['question']))
        eval_processed.sort(key=lambda x: len(x['question']))

        self.train_loader = DataLoader(
            train_processed,
            batch_size=self.trainer.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )

        self.eval_loader = DataLoader(
            eval_processed,
            batch_size=self.trainer.config.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )

        logger.info("Data prepared: %d train, %d eval samples",
                     len(train_processed), len(eval_processed))

    @staticmethod
    def _collate_fn(batch):
        return {
            'questions': [item['question'] for item in batch],
            'answers': [item['answer'] for item in batch]
        }

    def train(self):
        """Run the training loop."""
        logger.info("Starting RLT training...")

        initial_memory = self.student_model.memory_monitor.get_memory_stats()
        logger.info("Initial memory state: %s", initial_memory)

        self.prepare_data()
        self._log_cost_estimate()

        try:
            self.trainer.train(
                train_dataloader=self.train_loader,
                eval_dataloader=self.eval_loader,
                num_epochs=self.config['training'].get('num_epochs', 3)
            )
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error("Training error: %s", e)
            raise
        finally:
            self.cleanup()

    def _log_cost_estimate(self):
        """Log estimated API costs."""
        eval_mode = self.config.get('evaluation', {}).get('mode', 'local')
        num_train = len(self.train_loader.dataset)
        group_size = self.config['training'].get('group_size', 4)
        num_epochs = self.config['training'].get('num_epochs', 3)

        teacher_calls = num_train * group_size * num_epochs

        if eval_mode == 'claude':
            eval_calls = teacher_calls
        elif eval_mode == 'hybrid':
            claude_freq = self.config['evaluation'].get('claude_frequency', 0.1)
            eval_calls = int(teacher_calls * claude_freq)
        else:
            eval_calls = 0

        total_calls = teacher_calls + eval_calls
        estimated_cost = total_calls * 0.002

        logger.info("Cost estimate: %d teacher calls + %d eval calls = ~$%.2f",
                     teacher_calls, eval_calls, estimated_cost)

    def cleanup(self):
        """Clean up resources and save final artifacts."""
        final_stats = {
            'training_completed': datetime.now().isoformat(),
            'teacher_stats': self.teacher.get_stats(),
            'cost_summary': self.cost_tracker.get_summary(),
            'final_memory': self.student_model.memory_monitor.get_memory_stats()
        }
        if hasattr(self.student_evaluator, 'get_stats'):
            final_stats['evaluator_stats'] = self.student_evaluator.get_stats()

        with open(self.output_dir / "final_stats.json", 'w') as f:
            json.dump(final_stats, f, indent=2)

        final_model_path = self.output_dir / "final_student_model"
        self.student_model.save_model(str(final_model_path))
        self.student_model.cleanup()

        logger.info("Final student model saved to: %s", final_model_path)
        logger.info("Total API cost: $%.2f", self.cost_tracker.get_summary()['total_cost'])


def create_default_config(eval_mode: str = "local") -> Dict:
    """Create default configuration."""
    config = {
        "output_dir": "./rlt_output",
        "seed": 42,
        "budget_limit": 100.0,
        "teacher": {
            "model": "claude-sonnet-4-6-20250514",
            "temperature": 0.7,
            "max_tokens": 1024,
            "cache_size": 10000,
            "cache_ttl_hours": 168,
        },
        "student": {
            "model_name": "meta-llama/Llama-3.2-3B-Instruct",
            "use_4bit": True,
            "use_flash_attention": True,
            "use_peft": True,
            "peft_method": "adalora",
            "lora_r": 32,
            "adalora_init_r": 12,
            "adalora_target_r": 32,
            "gradient_checkpointing": True,
            "mixed_precision": "bf16",
        },
        "evaluation": {
            "mode": eval_mode,
        },
        "training": {
            "learning_rate": 2e-5,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "num_epochs": 3,
            "group_size": 4,
            "kl_weight": 0.1,
            "correctness_weight": 0.5,
            "confidence_weight": 0.3,
            "length_penalty_weight": 0.1,
        },
        "data": {
            "dataset": "gsm8k",
            "max_train_samples": 1000,
            "max_eval_samples": 100,
            "cache_dir": "./data_cache",
        },
        "tracking": {
            "backend": None,
            "project": "slmteacher",
            "run_name": None,
        },
    }

    # Add eval-mode specific defaults
    if eval_mode == "claude":
        config["evaluation"].update({
            "model": "claude-sonnet-4-6-20250514",
            "temperature": 0.1,
            "max_tokens": 512,
        })
    elif eval_mode == "hybrid":
        config["evaluation"].update({
            "model": "claude-sonnet-4-6-20250514",
            "temperature": 0.1,
            "claude_frequency": 0.2,
            "local_model": "meta-llama/Llama-3.2-3B-Instruct",
        })
    else:
        config["evaluation"].update({
            "model_name": "meta-llama/Llama-3.2-3B-Instruct",
            "batch_size": 8,
        })

    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RLT Training - Student learns from Teacher (Claude) explanations"
    )
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--create-config", action="store_true",
                        help="Create a default configuration file and exit")
    parser.add_argument("--student-model", type=str, help="Override student model name")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument("--eval-mode", choices=["local", "claude", "hybrid"],
                        default=None,
                        help="Evaluation mode: local (default), claude, or hybrid")
    parser.add_argument("--budget-limit", type=float, help="API budget limit in USD")
    parser.add_argument("--dataset", type=str, help="Dataset name (e.g. gsm8k, math)")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--tracker", choices=["wandb", "tensorboard"],
                        default=None, help="Experiment tracking backend")
    parser.add_argument("--tracker-project", type=str,
                        help="Project name for experiment tracker")
    parser.add_argument("--tracker-run-name", type=str,
                        help="Run name for experiment tracker")

    args = parser.parse_args()

    # Create default config if requested
    if args.create_config:
        eval_mode = args.eval_mode or "local"
        config = create_default_config(eval_mode)
        config_path = "rlt_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Default configuration saved to {config_path}")
        print(f"\nEvaluation mode: {eval_mode}")
        print("  local:  Local HF model evaluation (cheapest)")
        print("  claude: Claude API evaluation (highest quality)")
        print("  hybrid: Mix of local + Claude (balanced)")
        return

    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config(args.eval_mode or "local")

    # Apply CLI overrides
    if args.student_model:
        config['student']['model_name'] = args.student_model
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.eval_mode:
        config.setdefault('evaluation', {})['mode'] = args.eval_mode
    if args.budget_limit is not None:
        config['budget_limit'] = args.budget_limit
    if args.dataset:
        config.setdefault('data', {})['dataset'] = args.dataset
    if args.epochs is not None:
        config.setdefault('training', {})['num_epochs'] = args.epochs
    if args.batch_size is not None:
        config.setdefault('training', {})['batch_size'] = args.batch_size
    if args.lr is not None:
        config.setdefault('training', {})['learning_rate'] = args.lr
    if args.seed is not None:
        config['seed'] = args.seed
    if args.tracker:
        config.setdefault('tracking', {})['backend'] = args.tracker
    if args.tracker_project:
        config.setdefault('tracking', {})['project'] = args.tracker_project
    if args.tracker_run_name:
        config.setdefault('tracking', {})['run_name'] = args.tracker_run_name

    # Check for API key
    api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "API key not set. Set CLAUDE_API_KEY or ANTHROPIC_API_KEY environment variable."
        )

    # Log configuration
    logger.info("Configuration:\n%s", json.dumps(config, indent=2))

    # Initialize and run trainer
    trainer = RLTTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
