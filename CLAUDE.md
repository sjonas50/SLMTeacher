# SLMTeacher — Claude Code Configuration

## Project Overview

SLMTeacher implements **Reinforcement Learning from Teachers (RLT)** based on Sakana AI's paper (arXiv:2412.15192). A **teacher** (Claude API) generates explanations, a **student** (local HF model with QLoRA) learns from them via GRPO training, and a **reward function** scores explanation quality.

**Critical architectural rule:** The STUDENT model is what gets trained. The teacher (Claude) only generates explanations — never optimize teacher parameters.

## Commands

```bash
# Install
pip install -e ".[dev]"          # development install
pip install -e ".[dev,tracking]" # with W&B/tensorboard

# Test
python3 -m pytest tests/ -v     # run all tests (52 tests)
python3 -m pytest tests/ -v -x  # stop on first failure

# Lint
ruff check src/ tests/

# Type check
mypy src/ --ignore-missing-imports

# Train (requires CLAUDE_API_KEY or ANTHROPIC_API_KEY)
python3 train.py --help
python3 train.py --create-config
python3 train.py --eval-mode local          # cheapest
python3 train.py --eval-mode hybrid         # balanced
python3 train.py --eval-mode claude          # highest quality
python3 train.py --tracker wandb             # with experiment tracking
```

## Project Structure

```
train.py                          # Main entry point (all eval modes)
src/
  teachers/claude_teacher.py      # Claude API wrapper (teacher)
  models/optimized_model.py       # QLoRA/AdaLoRA student model (requires peft)
  training/grpo_trainer.py        # GRPO trainer + ExperimentTracker
  rewards/
    reward_function.py            # RewardFunction wrapper
    student_evaluator.py          # Local HF model evaluator
    claude_evaluator.py           # Claude API evaluator
    kl_divergence.py              # KL divergence calculator
    reward_utils.py               # Normalization, GAE, logging
  data/
    data_processor.py             # RLTDataPoint, formatting, augmentation
    data_loader.py                # Dataset loading (gsm8k, math, arc-c)
    cache_manager.py              # Dataset/explanation cache
  utils/cost_tracker.py           # API cost tracking with per-model pricing
tests/                            # All tests (pytest)
```

## Key Conventions

- **Python 3.10+**, no TypeScript/JS in this repo
- **Dependencies** are defined in `pyproject.toml` (requirements.txt just points there)
- **Tests** go in `tests/`, never in `src/`
- Use `logger = logging.getLogger(__name__)` — no `print()` in library code
- Use `python3` not `python` (macOS compatibility)
- Optional heavy deps (`peft`, `bitsandbytes`, `wandb`) use lazy imports or `try/except`
- MPS (Apple Silicon) support: always check `torch.backends.mps.is_available()` alongside CUDA
- `torch.amp.autocast("cuda")` not deprecated `torch.cuda.amp.autocast`
- API key: accept both `CLAUDE_API_KEY` and `ANTHROPIC_API_KEY`
- Model IDs: use `claude-sonnet-4-6-20250514` (current as of 2025-05)

## Common Pitfalls

- `optimized_model.py` imports `peft` at module level — tests that touch it need `pytest.importorskip("peft")`
- `use_cache=True` conflicts with `gradient_checkpointing=True` in HF models — always disable cache when checkpointing
- `CostTracker.get_summary()` returns **numeric floats**, not formatted strings
- `RateLimiter` takes a `RateLimitConfig` object, not raw kwargs
- `data_loader.py` imports `requests` and `pandas` at module level — these are required deps
