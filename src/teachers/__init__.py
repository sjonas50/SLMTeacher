# Teachers module
import os
from typing import Optional

def create_teacher_from_env(cost_tracker=None):
    """Create a teacher based on environment configuration."""
    api_key = os.environ.get('CLAUDE_API_KEY')
    
    if api_key:
        from .claude_teacher import ClaudeRLTTeacher
        return ClaudeRLTTeacher(api_key=api_key, cost_tracker=cost_tracker)
    else:
        # Fallback to HF model
        from ..models.hf_teacher_model import HFTeacherModel
        return HFTeacherModel(model_name="microsoft/phi-2")

__all__ = ['create_teacher_from_env']