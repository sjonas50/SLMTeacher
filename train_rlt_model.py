#!/usr/bin/env python3
"""
Train a Hugging Face model using RLT methodology
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List
import json

# Add src to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.hf_teacher_model import HFTeacherModel
from src.training.grpo_trainer import GRPOTrainer, GRPOConfig
from src.rewards import RLTRewardFunction, LocalStudentEvaluator, TransformerKLCalculator
from src.data import DataLoader as RLTDataLoader, DataProcessor
from src.utils.cost_tracker import CostTracker


class RLTDataset(Dataset):
    """PyTorch dataset for RLT training."""
    
    def __init__(self, data_points, processor):
        self.data = data_points
        self.processor = processor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'questions': item.question,
            'answers': item.answer,
            'subject': item.subject,
            'difficulty': item.difficulty
        }


def collate_fn(batch):
    """Custom collate function for batching."""
    return {
        'questions': [item['questions'] for item in batch],
        'answers': [item['answers'] for item in batch],
        'subjects': [item['subject'] for item in batch],
        'difficulties': [item['difficulty'] for item in batch]
    }


def main():
    parser = argparse.ArgumentParser(description='Train HF model with RLT')
    
    # Model arguments
    parser.add_argument('--teacher-model', type=str, default='microsoft/phi-2',
                       help='Hugging Face model to train as teacher')
    parser.add_argument('--student-model', type=str, default='microsoft/phi-2',
                       help='Student model for reward computation')
    parser.add_argument('--use-lora', action='store_true', default=True,
                       help='Use LoRA for efficient training')
    
    # Data arguments
    parser.add_argument('--datasets', nargs='+', default=['gsm8k'],
                       choices=['gsm8k', 'math', 'arc-c'],
                       help='Datasets to use for training')
    parser.add_argument('--max-samples', type=int, default=1000,
                       help='Maximum samples to use')
    parser.add_argument('--train-split', type=float, default=0.9,
                       help='Train/eval split ratio')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--num-epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--group-size', type=int, default=4,
                       help='Number of explanations per question')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./rlt_output',
                       help='Output directory for model and logs')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Directory for checkpoints')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("="*60)
    print("RLT Model Training")
    print("="*60)
    print(f"Teacher Model: {args.teacher_model}")
    print(f"Student Model: {args.student_model}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Max Samples: {args.max_samples}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Group Size: {args.group_size}")
    print(f"Use LoRA: {args.use_lora}")
    print("="*60)
    
    # Initialize components
    print("\n1. Initializing models...")
    
    # Teacher model (trainable)
    teacher = HFTeacherModel(
        model_name=args.teacher_model,
        use_lora=args.use_lora,
        use_8bit=True
    )
    
    # Student evaluator (for rewards)
    student_evaluator = LocalStudentEvaluator(
        model_name=args.student_model
    )
    
    # KL calculator
    kl_calculator = TransformerKLCalculator(
        model_name=args.teacher_model
    )
    
    # Reward function
    reward_function = RLTRewardFunction(
        student_evaluator=student_evaluator,
        kl_calculator=kl_calculator
    )
    
    print("✅ Models initialized")
    
    # Load data
    print("\n2. Loading datasets...")
    data_loader = RLTDataLoader()
    data_processor = DataProcessor()
    
    all_data = []
    for dataset_name in args.datasets:
        data = data_loader.load_dataset(
            dataset_name,
            split='train',
            max_samples=args.max_samples // len(args.datasets)
        )
        all_data.extend(data)
        print(f"  Loaded {len(data)} samples from {dataset_name}")
    
    # Split into train/eval
    split_idx = int(len(all_data) * args.train_split)
    train_data = all_data[:split_idx]
    eval_data = all_data[split_idx:]
    
    print(f"✅ Total samples: {len(all_data)} (train: {len(train_data)}, eval: {len(eval_data)})")
    
    # Create PyTorch datasets
    train_dataset = RLTDataset(train_data, data_processor)
    eval_dataset = RLTDataset(eval_data, data_processor)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Configure training
    print("\n3. Configuring training...")
    grpo_config = GRPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        group_size=args.group_size,
        checkpoint_dir=args.checkpoint_dir,
        gradient_accumulation_steps=4,
        save_steps=100,
        eval_steps=1
    )
    
    # Initialize trainer
    trainer = GRPOTrainer(
        teacher_model=teacher,
        student_evaluator=student_evaluator,
        reward_function=reward_function,
        config=grpo_config
    )
    
    print("✅ Training configured")
    
    # Train model
    print("\n4. Starting training...")
    print("="*60)
    
    try:
        trainer.train(
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            num_epochs=args.num_epochs
        )
        
        print("\n✅ Training completed!")
        
        # Save final model
        print("\n5. Saving final model...")
        final_path = os.path.join(args.output_dir, "final_model")
        teacher.save_model(final_path)
        print(f"✅ Model saved to: {final_path}")
        
        # Save training config
        config_path = os.path.join(args.output_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump({
                'teacher_model': args.teacher_model,
                'student_model': args.student_model,
                'datasets': args.datasets,
                'max_samples': args.max_samples,
                'batch_size': args.batch_size,
                'num_epochs': args.num_epochs,
                'learning_rate': args.learning_rate,
                'group_size': args.group_size,
                'use_lora': args.use_lora
            }, f, indent=2)
        
        print(f"✅ Config saved to: {config_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        save_checkpoint = input("Save checkpoint? (y/n): ")
        if save_checkpoint.lower() == 'y':
            trainer.save_checkpoint()
            print("✅ Checkpoint saved")
    
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise
    
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Final checkpoint: {os.path.join(args.checkpoint_dir, 'best_model.pt')}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()