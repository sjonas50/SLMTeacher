# Training module
from .grpo_trainer import GRPOTrainer, GRPOConfig
try:
    from .grpo_trainer_corrected import GRPOTrainer as CorrectedGRPOTrainer
except ImportError:
    CorrectedGRPOTrainer = None

__all__ = ['GRPOTrainer', 'GRPOConfig', 'CorrectedGRPOTrainer']