# SLMTeacher — Reinforcement Learning from Teachers (RLT)

A production implementation of [Sakana AI's RLT methodology](https://arxiv.org/abs/2412.15192) for training small language models using explanations from a large teacher model (Claude). A **teacher** (Claude API) generates step-by-step explanations, a **student** (local HuggingFace model with LoRA/QLoRA) learns from them via GRPO training, and a **reward function** scores explanation quality.

**Key principle:** The STUDENT model is what gets trained. The teacher (Claude) only generates explanations.

## Features

- **Teacher-Student Architecture** — Claude generates explanations; a local HF model learns from them
- **GRPO Training** — Group Relative Policy Optimization with SFT warmup
- **Curriculum-Based Training** — Describe what you want the model to learn in plain text (`--teach`)
- **Continuous Training** — Adaptive rounds with student assessment and data selection
- **LoRA / QLoRA** — Parameter-efficient fine-tuning (AdaLoRA with automatic fallback)
- **MPS + CUDA Support** — Runs on Apple Silicon (MPS) and NVIDIA GPUs
- **Cost Management** — Per-model API pricing, budget limits, automatic caching
- **Cosine LR Scheduling** — Configurable warmup ratio and scheduler type
- **Multi-Dataset** — GSM8K, MATH, ARC-C via HuggingFace datasets

## Quick Start

### Install

```bash
pip install -e ".[dev]"

# With experiment tracking (optional)
pip install -e ".[dev,tracking]"
```

### Set API Key

```bash
# Option 1: Environment variable
export ANTHROPIC_API_KEY="your-key"

# Option 2: .env file (recommended)
cp .env.example .env
# Edit .env with your key
```

### Train on a Dataset

```bash
# Local evaluation (cheapest)
python3 train.py --eval-mode local

# Hybrid evaluation (balanced cost/quality)
python3 train.py --eval-mode hybrid

# Claude evaluation (highest quality)
python3 train.py --eval-mode claude --budget-limit 10.0

# Custom student model
python3 train.py --student-model "mistralai/Mistral-7B-Instruct-v0.3"
```

### Train on a Custom Topic

Describe what the model should learn — no dataset needed:

```bash
python3 train.py \
    --teach "become an expert in analyzing quarterly financial reports" \
    --curriculum-size 20 \
    --student-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --eval-mode local \
    --budget-limit 5.0
```

The system automatically generates a curriculum of questions and answers, then trains the student using teacher-generated explanations.

### Full Options

```bash
python3 train.py --help
python3 train.py --create-config  # Generate example config file
```

## Project Structure

```
train.py                              # Main entry point
src/
  teachers/claude_teacher.py          # Claude API wrapper (teacher)
  models/optimized_model.py           # LoRA/QLoRA student model
  training/
    grpo_trainer.py                   # GRPO trainer + SFT warmup
    continuous_trainer.py             # Multi-round adaptive training
    student_assessor.py               # Tracks student accuracy per category
    adaptive_data_selector.py         # Selects training data based on weaknesses
  rewards/
    reward_function.py                # Reward computation wrapper
    student_evaluator.py              # Local model evaluator + SharedModelEvaluator
    claude_evaluator.py               # Claude API evaluator
    kl_divergence.py                  # KL divergence calculator
  data/
    data_loader.py                    # Dataset loading (gsm8k, math, arc-c)
    data_processor.py                 # RLTDataPoint, formatting, augmentation
    curriculum_generator.py           # Generate Q&A curriculum from text description
    explanation_generator.py          # Pre-generate teacher explanations
    cache_manager.py                  # Dataset/explanation cache
  utils/cost_tracker.py              # API cost tracking with per-model pricing
tests/                                # 137 tests (pytest)
```

## Training Pipeline

1. **Curriculum Generation** (optional) — Claude decomposes a text description into topics and generates Q&A pairs
2. **SFT Warmup** — Supervised fine-tuning so the student can follow explanations before policy gradient
3. **GRPO Training** — For each question, the teacher generates multiple explanations at varying temperatures; the student is trained via policy gradient weighted by reward
4. **Evaluation** — Student accuracy assessed on held-out problems
5. **Adaptive Rounds** — Repeat with targeted data selection until convergence or patience exhausted

## Configuration

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--student-model` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | HuggingFace model ID |
| `--eval-mode` | `local` | Evaluation mode: `local`, `hybrid`, `claude` |
| `--budget-limit` | `10.0` | Max API spend in USD |
| `--batch-size` | `2` | Training batch size |
| `--sft-epochs` | `2` | SFT warmup epochs (0 to disable) |
| `--lr-scheduler` | `cosine` | LR scheduler: `cosine` or `linear` |
| `--warmup-ratio` | `0.1` | Fraction of steps for LR warmup |
| `--teach` | — | Natural language description for curriculum |
| `--curriculum-size` | `30` | Number of problems to generate |
| `--target-accuracy` | `0.8` | Stop when accuracy reaches this |
| `--patience` | `3` | Rounds without improvement before stopping |
| `--tracker` | — | `wandb` or `tensorboard` |

### JSON Config

```bash
python3 train.py --create-config  # Creates optimized_rlt_config_example.json
python3 train.py --config my_config.json
```

## Supported Models

### Teacher
- `claude-sonnet-4-6` (Claude Sonnet 4.6, recommended)

### Student (local)

| Model | Parameters | Notes |
|-------|------------|-------|
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B | Fast testing, MPS-friendly |
| meta-llama/Llama-3.2-3B-Instruct | 3B | Good balance |
| mistralai/Mistral-7B-Instruct-v0.3 | 7B | Strong performance |

On CUDA: QLoRA (4-bit) enables training larger models. On MPS/CPU: quantization is auto-disabled, runs in fp32.

## Development

```bash
# Run tests
python3 -m pytest tests/ -v

# Lint
ruff check src/ tests/

# Type check
mypy src/ --ignore-missing-imports
```

## Citation

```bibtex
@article{sakana2024rlt,
  title={Reinforcement Learning Teachers},
  author={Sakana AI},
  year={2024},
  journal={arXiv preprint arXiv:2412.15192}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
