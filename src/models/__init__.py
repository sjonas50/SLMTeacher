# Models module
try:
    from .hf_teacher_model import HFTeacherModel
except ImportError:
    HFTeacherModel = None

try:
    from .optimized_model import OptimizedHFModel, OptimizedModelConfig, MemoryMonitor
except ImportError:
    OptimizedHFModel = None
    OptimizedModelConfig = None
    MemoryMonitor = None

__all__ = ['HFTeacherModel', 'OptimizedHFModel', 'OptimizedModelConfig', 'MemoryMonitor']
