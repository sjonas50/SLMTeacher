#!/usr/bin/env python3
"""
RLT Training Script

Trains a STUDENT model to learn from TEACHER (Claude) explanations
using Group Relative Policy Optimization (GRPO).
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


class OptimizedRLTTrainer:
    """Main class for optimized RLT training."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_directories()
        
        # Set random seed
        set_seed(config.get('seed', 42))
        
        # Initialize components
        self.setup_teacher()  # Claude API - generates explanations
        self.setup_student()  # HF Model - learns from explanations
        self.setup_evaluator()  # Evaluates student understanding for rewards
        self.setup_training()
        
        logger.info("Corrected Optimized RLT Trainer initialized successfully")
        logger.info("Teacher: Claude API (generates explanations)")
        logger.info("Student: Local HF model (learns from explanations)")
    
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
        # Create cost tracker
        budget_limit = self.config.get('teacher', {}).get('budget_limit', 50.0)
        self.cost_tracker = CostTracker(budget_limit=budget_limit)
        
        # Teacher configuration
        claude_config = ClaudeConfig(
            model=self.config['teacher'].get('model', 'claude-sonnet-4-6-20250514'),
            temperature=self.config['teacher'].get('temperature', 0.7),
            max_tokens=self.config['teacher'].get('max_tokens', 1024)
        )
        
        # Enhanced caching for API efficiency
        cache_config = CacheConfig(
            enabled=True,
            max_size=self.config['teacher'].get('cache_size', 10000),
            ttl_hours=self.config['teacher'].get('cache_ttl_hours', 168)  # 1 week
        )
        
        # Initialize teacher
        self.teacher = ClaudeRLTTeacher(
            api_key=os.getenv("CLAUDE_API_KEY"),
            cost_tracker=self.cost_tracker,
            claude_config=claude_config,
            cache_config=cache_config
        )
        
        logger.info(f"Claude teacher initialized (model: {claude_config.model})")
        logger.info("Teacher role: Generate high-quality explanations")
    
    def setup_student(self):
        """Initialize student model that will learn from teacher explanations."""
        # Student model configuration
        model_config = OptimizedModelConfig(
            model_name=self.config['student']['model_name'],
            # Quantization
            use_4bit=self.config['student'].get('use_4bit', True),
            use_8bit=self.config['student'].get('use_8bit', False),
            bnb_4bit_compute_dtype=self.config['student'].get('compute_dtype', 'bfloat16'),
            # Flash Attention
            use_flash_attention_2=self.config['student'].get('use_flash_attention', True),
            # PEFT
            use_peft=self.config['student'].get('use_peft', True),
            peft_method=self.config['student'].get('peft_method', 'adalora'),
            lora_r=self.config['student'].get('lora_r', 32),
            lora_alpha=self.config['student'].get('lora_alpha', 64),
            adalora_init_r=self.config['student'].get('adalora_init_r', 12),
            adalora_target_r=self.config['student'].get('adalora_target_r', 32),
            # Memory optimization
            gradient_checkpointing=self.config['student'].get('gradient_checkpointing', True),
            mixed_precision=self.config['student'].get('mixed_precision', 'bf16')
        )
        
        # Initialize student model - THIS IS WHAT WE TRAIN
        self.student_model = OptimizedHFModel(model_config)
        
        logger.info(f"Student model initialized: {model_config.model_name}")
        logger.info("Student role: Learn to solve problems using teacher explanations")
        logger.info(f"Optimizations: Flash Attention={model_config.use_flash_attention_2}, "
                   f"QLoRA={model_config.use_4bit}, PEFT={model_config.peft_method}")
    
    def setup_evaluator(self):
        """Initialize evaluator to assess student understanding."""
        # The evaluator judges how well a student can solve problems
        # It's used to compute rewards, not for training
        evaluation_model = self.config['evaluation'].get('model_name', 
                                                         self.config['student']['model_name'])
        if torch.cuda.is_available():
            eval_device = "cuda"
        elif torch.backends.mps.is_available():
            eval_device = "mps"
        else:
            eval_device = "cpu"
        self.student_evaluator = StudentEvaluator(
            model_name=evaluation_model,
            device=eval_device,
            batch_size=self.config['evaluation'].get('batch_size', 8)
        )
        
        logger.info("Student evaluator initialized (local model)")
        logger.info("Evaluator role: Assess student understanding for reward computation")
        logger.info("Note: For higher quality evaluation, see train_optimized_rlt_claude_eval.py")
    
    def setup_training(self):
        """Initialize training components."""
        # Reward function
        self.reward_function = RewardFunction(
            student_evaluator=self.student_evaluator,
            kl_weight=self.config['training'].get('kl_weight', 0.1),
            correctness_weight=self.config['training'].get('correctness_weight', 0.5),
            confidence_weight=self.config['training'].get('confidence_weight', 0.3),
            length_penalty_weight=self.config['training'].get('length_penalty_weight', 0.1)
        )
        
        # GRPO configuration
        grpo_config = GRPOConfig(
            learning_rate=self.config['training'].get('learning_rate', 2e-5),
            batch_size=self.config['training'].get('batch_size', 1),
            gradient_accumulation_steps=self.config['training'].get('gradient_accumulation_steps', 8),
            num_epochs=self.config['training'].get('num_epochs', 3),
            group_size=self.config['training'].get('group_size', 4),
            checkpoint_dir=str(self.checkpoint_dir)
        )
        
        # Initialize GRPO trainer with CORRECTED parameter order
        self.trainer = GRPOTrainer(
            teacher=self.teacher,  # Generates explanations
            student_model=self.student_model,  # LEARNS from explanations (trainable)
            student_evaluator=self.student_evaluator,  # Evaluates for rewards
            reward_function=self.reward_function,
            config=grpo_config
        )
        
        logger.info("Training components initialized")
        logger.info("Training objective: Student learns to solve problems using teacher explanations")
    
    def prepare_data(self):
        """Prepare training and evaluation data."""
        # Initialize data processor
        data_processor = DataProcessor(
            cache_dir=self.config['data'].get('cache_dir', './data_cache')
        )
        
        # Load datasets
        dataset_name = self.config['data'].get('dataset', 'gsm8k')
        train_data = data_processor.load_dataset(dataset_name, split='train')
        eval_data = data_processor.load_dataset(dataset_name, split='test')
        
        # Process data
        train_processed = data_processor.process_for_rlt(
            train_data,
            max_samples=self.config['data'].get('max_train_samples', 1000)
        )
        eval_processed = data_processor.process_for_rlt(
            eval_data,
            max_samples=self.config['data'].get('max_eval_samples', 100)
        )
        
        # Sort by length for efficient batching
        train_processed.sort(key=lambda x: len(x['question']))
        eval_processed.sort(key=lambda x: len(x['question']))
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_processed,
            batch_size=self.trainer.config.batch_size,
            shuffle=True,
            collate_fn=self.custom_collate_fn
        )
        
        self.eval_loader = DataLoader(
            eval_processed,
            batch_size=self.trainer.config.batch_size,
            shuffle=False,
            collate_fn=self.custom_collate_fn
        )
        
        logger.info(f"Data prepared: {len(train_processed)} train, {len(eval_processed)} eval samples")
    
    def custom_collate_fn(self, batch):
        """Custom collate function for batching."""
        return {
            'questions': [item['question'] for item in batch],
            'answers': [item['answer'] for item in batch]
        }
    
    def train(self):
        """Run the optimized training loop."""
        logger.info("Starting corrected RLT training...")
        logger.info("Process: Teacher generates explanations → Student learns from them")
        
        # Log initial memory state
        initial_memory = self.student_model.memory_monitor.get_memory_stats()
        logger.info(f"Initial memory state: {initial_memory}")
        
        # Prepare data
        self.prepare_data()
        
        # Run training
        try:
            self.trainer.train(
                train_dataloader=self.train_loader,
                eval_dataloader=self.eval_loader,
                num_epochs=self.config['training'].get('num_epochs', 3)
            )
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
        finally:
            # Final cleanup
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        # Save final statistics
        final_stats = {
            'training_completed': datetime.now().isoformat(),
            'teacher_stats': self.teacher.get_stats(),
            'cost_summary': self.cost_tracker.get_summary(),
            'final_memory': self.student_model.memory_monitor.get_memory_stats()
        }
        
        with open(self.output_dir / "final_stats.json", 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        # Save final student model
        final_model_path = self.output_dir / "final_student_model"
        self.student_model.save_model(str(final_model_path))
        
        # Clean up memory
        self.student_model.cleanup()
        
        logger.info("Training completed and resources cleaned up")
        logger.info(f"Final student model saved to: {final_model_path}")


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_default_config() -> Dict:
    """Create default configuration."""
    return {
        "output_dir": "./optimized_rlt_output",
        "seed": 42,
        "teacher": {
            "model": "claude-sonnet-4-6-20250514",
            "temperature": 0.7,
            "max_tokens": 1024,
            "budget_limit": 50.0,
            "cache_size": 10000,
            "cache_ttl_hours": 168,
            "comment": "Teacher generates explanations (not trainable)"
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
            "comment": "Student learns from teacher explanations (trainable)"
        },
        "evaluation": {
            "model_name": "meta-llama/Llama-3.2-3B-Instruct",
            "batch_size": 8,
            "comment": "Evaluates student understanding for reward computation"
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
            "length_penalty_weight": 0.1
        },
        "data": {
            "dataset": "gsm8k",
            "max_train_samples": 1000,
            "max_eval_samples": 100,
            "cache_dir": "./data_cache"
        }
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Corrected RLT Training - Student learns from Teacher explanations"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create a default configuration file"
    )
    parser.add_argument(
        "--student-model",
        type=str,
        help="Override student model name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory"
    )
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        config = create_default_config()
        config_path = "corrected_rlt_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Default configuration saved to {config_path}")
        print("\nKey points in corrected implementation:")
        print("- Teacher (Claude): Generates explanations only")
        print("- Student (HF model): Learns from explanations (trainable)")
        print("- Evaluator: Assesses student understanding for rewards")
        return
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()
    
    # Override with command line arguments
    if args.student_model:
        config['student']['model_name'] = args.student_model
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Check for API key
    if not os.getenv("CLAUDE_API_KEY"):
        raise ValueError("CLAUDE_API_KEY environment variable not set")
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(json.dumps(config, indent=2))
    
    # Initialize and run trainer
    trainer = OptimizedRLTTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()