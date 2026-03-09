#!/usr/bin/env python3
"""Minimal live test: loads a real model, makes real API calls, trains 1 batch.

Usage:
    python3 test_live_run.py

Requirements:
    - CLAUDE_API_KEY or ANTHROPIC_API_KEY environment variable set
    - ~6GB free RAM (TinyLlama 1.1B in fp32 + training overhead)
    - Internet connection (downloads model + GSM8K on first run)

Expected cost: ~$0.01-0.02 (a handful of Claude API calls)
Expected time: 2-5 minutes (dominated by model download on first run)
"""
import logging
import os
import sys
import time

from dotenv import load_dotenv
import torch

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("live_test")


def main():
    # 0. Pre-flight checks
    api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set CLAUDE_API_KEY or ANTHROPIC_API_KEY")
        sys.exit(1)

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | RAM: %.1f GB", device,
                 torch.mps.current_allocated_memory() / 1e9 if device == "mps" else 0)

    # 1. Load a small dataset (10 GSM8K problems)
    logger.info("=== Step 1: Loading data ===")
    sys.stdout.flush()
    from src.data.data_loader import DataLoader as RLTDataLoader
    loader = RLTDataLoader(data_dir="./data_cache", auto_download=True)
    train_data = loader.load_dataset("gsm8k", split="train", max_samples=10, use_cache=False)
    eval_data = loader.load_dataset("gsm8k", split="test", max_samples=5, use_cache=False)
    logger.info("Loaded %d train, %d eval problems", len(train_data), len(eval_data))
    logger.info("Sample: Q=%s  A=%s", train_data[0].question[:60], train_data[0].solution)

    # 2. Load student model
    logger.info("=== Step 2: Loading student model ===")
    from src.models.optimized_model import OptimizedHFModel, OptimizedModelConfig
    model_config = OptimizedModelConfig(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        use_4bit=False,   # No CUDA
        use_8bit=False,
        use_flash_attention_2=False,
        use_peft=True,
        peft_method="lora",  # Standard LoRA (more stable than AdaLoRA for testing)
        lora_r=8,             # Small rank for fast test
        lora_alpha=16,
        gradient_checkpointing=True,
        device_map=None,      # Explicit placement (MPS-safe)
        mixed_precision="no",
    )
    student_model = OptimizedHFModel(model_config)
    logger.info("Student model loaded on %s", student_model.device)

    # 3. Test compute_loss and compute_log_probs
    logger.info("=== Step 3: Smoke-testing compute_loss / compute_log_probs ===")
    test_input = "Question: What is 2+2?\nExplanation: 2+2=4.\nAnswer: "
    test_target = "4"

    loss = student_model.compute_loss(test_input, test_target)
    log_prob = student_model.compute_log_probs(test_input, test_target)
    logger.info("compute_loss: %.4f  |  compute_log_probs (sum): %.4f", loss.item(), log_prob.item())
    assert loss.item() > 0, "Loss should be positive"
    assert log_prob.item() < 0, "Log prob should be negative"

    # 4. Setup teacher + evaluator + reward function
    logger.info("=== Step 4: Setting up teacher + evaluator ===")
    from src.teachers.claude_teacher import ClaudeRLTTeacher, ClaudeConfig, CacheConfig
    from src.utils.cost_tracker import CostTracker
    from src.rewards.student_evaluator import SharedModelEvaluator
    from src.rewards.reward_function import RewardFunction

    cost_tracker = CostTracker(budget_limit=1.0)
    teacher = ClaudeRLTTeacher(
        api_key=api_key,
        cost_tracker=cost_tracker,
        claude_config=ClaudeConfig(model="claude-sonnet-4-6", max_tokens=512),
        cache_config=CacheConfig(enabled=True),
    )

    evaluator = SharedModelEvaluator(student_model)
    reward_fn = RewardFunction(student_evaluator=evaluator)

    # 5. Test teacher explanation generation (1 real API call)
    logger.info("=== Step 5: Generating 1 teacher explanation (live API call) ===")
    q, a = train_data[0].question, train_data[0].solution
    t0 = time.time()
    explanation = teacher.generate_explanation(
        question=q, answer=a, temperature=0.7, subject="math", difficulty="easy",
    )
    if isinstance(explanation, dict):
        explanation = explanation.get("explanation", str(explanation))
    logger.info("Explanation (%.1fs): %s...", time.time() - t0, explanation[:120])
    logger.info("Cost so far: $%.4f", cost_tracker.get_summary()["total_cost"])

    # 6. Test reward computation
    logger.info("=== Step 6: Computing reward ===")
    reward_result = reward_fn.compute_reward([explanation], [q], answers=[a])
    logger.info("Reward: %s", reward_result["rewards"])

    # 7. Setup GRPOTrainer and run 1 SFT warmup step
    logger.info("=== Step 7: SFT warmup (1 batch) ===")
    from src.training.grpo_trainer import GRPOTrainer, GRPOConfig
    from torch.utils.data import DataLoader as TorchDataLoader

    grpo_config = GRPOConfig(
        learning_rate=2e-5,
        batch_size=2,
        gradient_accumulation_steps=1,
        group_size=2,   # Small for test speed
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        checkpoint_dir="./test_checkpoints",
    )

    trainer = GRPOTrainer(
        teacher=teacher,
        student_model=student_model,
        student_evaluator=evaluator,
        reward_function=reward_fn,
        config=grpo_config,
    )

    def collate_fn(batch):
        return {
            "questions": [dp.question for dp in batch],
            "answers": [dp.solution for dp in batch],
            "subjects": [dp.subject for dp in batch],
            "difficulties": [dp.difficulty for dp in batch],
        }

    train_loader = TorchDataLoader(
        train_data[:4], batch_size=2, shuffle=False, collate_fn=collate_fn,
    )

    # SFT warmup: 1 epoch over 2 batches
    trainer.sft_warmup(train_loader, num_epochs=1)
    logger.info("SFT warmup complete")

    # 8. Run 1 GRPO training epoch
    logger.info("=== Step 8: GRPO training (1 epoch, 2 batches) ===")
    trainer.train(train_loader, num_epochs=1)
    logger.info("GRPO training complete")

    # 9. Summary
    cost_summary = cost_tracker.get_summary()
    logger.info("=== DONE ===")
    logger.info("Total API cost: $%.4f", cost_summary["total_cost"])
    logger.info("Total API calls: %d", cost_summary.get("total_calls", 0))
    logger.info("All steps completed successfully!")

    # Cleanup
    import shutil
    if os.path.exists("./test_checkpoints"):
        shutil.rmtree("./test_checkpoints")


if __name__ == "__main__":
    main()
