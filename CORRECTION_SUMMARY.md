# Important Correction: Teacher-Student Architecture

## The Issue

The original implementation had a fundamental misunderstanding about which model gets trained:
- ❌ **Original**: Tried to train the teacher model
- ✅ **Corrected**: Trains the student model (as intended by RLT methodology)

## What Changed

### 1. **Training Target**
- **Before**: `optimizer = AdamW(self.teacher.model.parameters())`
- **After**: `optimizer = AdamW(self.student_model.parameters())`

### 2. **Architecture Clarification**
```
Teacher (Claude API) → Generates explanations → NOT trainable
                                                      ↓
Student (HF Model) ← Learns from explanations ← THIS gets trained!
```

### 3. **New Files**
- `train_optimized_rlt_corrected.py` - Corrected training script
- `src/training/grpo_trainer_corrected.py` - Corrected GRPO implementation
- `ARCHITECTURE.md` - Clear explanation of roles

## How to Use the Corrected Version

```bash
# Use the corrected script
python train_optimized_rlt_corrected.py --create-config
python train_optimized_rlt_corrected.py
```

## Key Points to Remember

1. **Teacher (Claude)**: Only generates explanations, never trained
2. **Student (HF Model)**: Learns from explanations, this is what we train
3. **Evaluator**: Measures student performance for reward computation
4. **Training Goal**: Student learns to solve problems using teacher's reasoning

## Why This Matters

The RLT methodology's power comes from:
- Small student models learning from high-quality teacher explanations
- Dense reward signals based on student understanding
- Efficient knowledge transfer from teacher to student

Training the teacher would defeat the purpose - we want to leverage Claude's existing capabilities to teach smaller models!