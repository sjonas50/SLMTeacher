#!/usr/bin/env python3
"""
Optimized RLT Training with Claude as Both Teacher and Evaluator

This version uses Claude for both:
1. Teacher: Generates high-quality explanations
2. Evaluator: Provides accurate assessment of student understanding

This gives the highest quality training signals but uses more API calls.
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
from src.rewards.claude_evaluator import ClaudeStudentEvaluator, ClaudeEvaluatorConfig, HybridEvaluator
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


class OptimizedRLTTrainerWithClaudeEval:
    """RLT training with Claude as both teacher and evaluator."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_directories()
        
        # Set random seed
        set_seed(config.get('seed', 42))
        
        # Initialize components
        self.setup_teacher()  # Claude API - generates explanations
        self.setup_student()  # HF Model - learns from explanations
        self.setup_evaluator()  # Claude API or Hybrid - evaluates understanding
        self.setup_training()
        
        logger.info("RLT Trainer with Claude Evaluation initialized")
        logger.info("Teacher: Claude API (generates explanations)")
        logger.info("Student: Local HF model (learns from explanations)")
        logger.info(f"Evaluator: {self.config['evaluation'].get('mode', 'claude')} mode")
    
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
        # Create cost tracker (shared between teacher and evaluator)
        budget_limit = self.config.get('budget_limit', 100.0)
        self.cost_tracker = CostTracker(budget_limit=budget_limit)
        
        # Teacher configuration
        claude_config = ClaudeConfig(
            model=self.config['teacher'].get('model', 'claude-sonnet-4-6-20250514'),
            temperature=self.config['teacher'].get('temperature', 0.7),
            max_tokens=self.config['teacher'].get('max_tokens', 1024)
        )
        
        # Enhanced caching
        cache_config = CacheConfig(
            enabled=True,
            max_size=self.config['teacher'].get('cache_size', 10000),
            ttl_hours=self.config['teacher'].get('cache_ttl_hours', 168)
        )
        
        # Initialize teacher
        self.teacher = ClaudeRLTTeacher(
            api_key=os.getenv("CLAUDE_API_KEY"),
            cost_tracker=self.cost_tracker,
            claude_config=claude_config,
            cache_config=cache_config
        )
        
        logger.info(f"Claude teacher initialized (model: {claude_config.model})")
    
    def setup_student(self):
        """Initialize student model that will learn from teacher explanations."""
        # Student model configuration
        model_config = OptimizedModelConfig(
            model_name=self.config['student']['model_name'],
            use_4bit=self.config['student'].get('use_4bit', True),
            use_flash_attention_2=self.config['student'].get('use_flash_attention', True),
            use_peft=self.config['student'].get('use_peft', True),
            peft_method=self.config['student'].get('peft_method', 'adalora'),
            lora_r=self.config['student'].get('lora_r', 32),
            adalora_init_r=self.config['student'].get('adalora_init_r', 12),
            adalora_target_r=self.config['student'].get('adalora_target_r', 32),
            gradient_checkpointing=self.config['student'].get('gradient_checkpointing', True),
            mixed_precision=self.config['student'].get('mixed_precision', 'bf16')
        )
        
        # Initialize student model
        self.student_model = OptimizedHFModel(model_config)
        
        logger.info(f"Student model initialized: {model_config.model_name}")
    
    def setup_evaluator(self):
        """Initialize evaluator with Claude, local, or hybrid mode."""
        eval_mode = self.config['evaluation'].get('mode', 'claude')
        
        if eval_mode == 'claude':
            # Pure Claude evaluation
            eval_config = ClaudeEvaluatorConfig(
                model=self.config['evaluation'].get('model', 'claude-sonnet-4-6-20250514'),
                temperature=self.config['evaluation'].get('temperature', 0.1),
                max_tokens=self.config['evaluation'].get('max_tokens', 512)
            )
            
            self.student_evaluator = ClaudeStudentEvaluator(
                api_key=os.getenv("CLAUDE_API_KEY"),
                cost_tracker=self.cost_tracker,
                config=eval_config
            )
            
            logger.info("Using Claude for student evaluation (highest quality)")
            
        elif eval_mode == 'hybrid':
            # Hybrid evaluation (Claude + local)
            # Claude evaluator
            eval_config = ClaudeEvaluatorConfig(
                model=self.config['evaluation'].get('model', 'claude-sonnet-4-6-20250514'),
                temperature=self.config['evaluation'].get('temperature', 0.1)
            )
            
            claude_evaluator = ClaudeStudentEvaluator(
                api_key=os.getenv("CLAUDE_API_KEY"),
                cost_tracker=self.cost_tracker,
                config=eval_config
            )
            
            # Local evaluator
            local_evaluator = StudentEvaluator(
                model_name=self.config['evaluation'].get('local_model', 
                                                         self.config['student']['model_name']),
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Hybrid evaluator
            claude_freq = self.config['evaluation'].get('claude_frequency', 0.1)
            self.student_evaluator = HybridEvaluator(
                claude_evaluator=claude_evaluator,
                local_evaluator=local_evaluator,
                claude_eval_frequency=claude_freq
            )
            
            logger.info(f"Using hybrid evaluation (Claude {claude_freq*100}% of the time)")
            
        else:  # local
            # Local evaluation only
            self.student_evaluator = StudentEvaluator(
                model_name=self.config['evaluation'].get('model_name', 
                                                         self.config['student']['model_name']),
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            logger.info("Using local model for evaluation (lower cost)")
    
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
        
        # Initialize GRPO trainer
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
        logger.info("Starting RLT training with Claude evaluation...")
        logger.info("Note: This uses more API calls but provides highest quality training")
        
        # Log initial state
        initial_memory = self.student_model.memory_monitor.get_memory_stats()
        logger.info(f"Initial memory state: {initial_memory}")
        
        # Prepare data
        self.prepare_data()
        
        # Estimate costs
        self._estimate_training_costs()
        
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
    
    def _estimate_training_costs(self):
        """Estimate API costs for training."""
        num_train = len(self.train_loader.dataset)
        num_eval = len(self.eval_loader.dataset)
        group_size = self.config['training'].get('group_size', 4)
        num_epochs = self.config['training'].get('num_epochs', 3)
        
        # Estimate API calls
        teacher_calls = num_train * group_size * num_epochs
        
        if self.config['evaluation'].get('mode') == 'claude':
            eval_calls = teacher_calls  # Same number for evaluation
        elif self.config['evaluation'].get('mode') == 'hybrid':
            claude_freq = self.config['evaluation'].get('claude_frequency', 0.1)
            eval_calls = teacher_calls * claude_freq
        else:
            eval_calls = 0
        
        total_calls = teacher_calls + eval_calls
        
        # Rough cost estimate (adjust based on actual token usage)
        estimated_cost = total_calls * 0.002  # ~$0.002 per call estimate
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Training Cost Estimate:")
        logger.info(f"- Teacher API calls: {teacher_calls:,}")
        logger.info(f"- Evaluator API calls: {eval_calls:,}")
        logger.info(f"- Total API calls: {total_calls:,}")
        logger.info(f"- Estimated cost: ${estimated_cost:.2f}")
        logger.info(f"- Budget limit: ${self.config.get('budget_limit', 100.0):.2f}")
        logger.info(f"{'='*50}\n")
    
    def cleanup(self):
        """Clean up resources."""
        # Save final statistics
        final_stats = {
            'training_completed': datetime.now().isoformat(),
            'teacher_stats': self.teacher.get_stats(),
            'evaluator_stats': getattr(self.student_evaluator, 'get_stats', lambda: {})(),
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
        logger.info(f"Total API cost: ${self.cost_tracker.get_summary()['total_cost']:.2f}")


def create_default_config() -> Dict:
    """Create default configuration with Claude evaluation."""
    return {
        "output_dir": "./claude_eval_rlt_output",
        "seed": 42,
        "budget_limit": 100.0,
        "teacher": {
            "model": "claude-sonnet-4-6-20250514",
            "temperature": 0.7,
            "max_tokens": 1024,
            "cache_size": 10000,
            "cache_ttl_hours": 168,
            "comment": "Teacher: Generates high-quality explanations"
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
            "comment": "Student: Learns from teacher explanations"
        },
        "evaluation": {
            "mode": "hybrid",  # "claude", "local", or "hybrid"
            "model": "claude-sonnet-4-6-20250514",
            "temperature": 0.1,
            "max_tokens": 512,
            "claude_frequency": 0.2,  # Use Claude 20% of time in hybrid mode
            "local_model": "meta-llama/Llama-3.2-3B-Instruct",
            "comment": "Evaluator: High-quality assessment of student understanding"
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
        description="RLT Training with Claude as Teacher and Evaluator"
    )
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--create-config", action="store_true", help="Create default config")
    parser.add_argument("--student-model", type=str, help="Override student model")
    parser.add_argument("--eval-mode", choices=['claude', 'local', 'hybrid'], 
                       help="Evaluation mode")
    parser.add_argument("--budget-limit", type=float, help="API budget limit in USD")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        config = create_default_config()
        config_path = "claude_eval_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Default configuration saved to {config_path}")
        print("\nEvaluation modes:")
        print("- claude: Always use Claude (highest quality, highest cost)")
        print("- hybrid: Use Claude periodically (balanced quality/cost)")
        print("- local: Use local model only (lowest cost)")
        return
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # Override with command line arguments
    if args.student_model:
        config['student']['model_name'] = args.student_model
    if args.eval_mode:
        config['evaluation']['mode'] = args.eval_mode
    if args.budget_limit:
        config['budget_limit'] = args.budget_limit
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Check for API key
    if not os.getenv("CLAUDE_API_KEY"):
        raise ValueError("CLAUDE_API_KEY environment variable not set")
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(json.dumps(config, indent=2))
    
    # Initialize and run trainer
    trainer = OptimizedRLTTrainerWithClaudeEval(config)
    trainer.train()


if __name__ == "__main__":
    main()