# RLT Teacher-Student Training Pipeline

## Overview

This notebook implements **Sakana AI's Reinforcement Learning Teachers (RLT)** approach for training small language models to reason effectively. Based on their groundbreaking research, this pipeline enables efficient teacher-student training that outperforms much larger models.

## 🔬 **Research Background**

Sakana AI's RLT methodology introduces a revolutionary approach:

- **Dense Rewards**: Teachers are rewarded based on how well students understand their explanations
- **Question-Answer Prompting**: Teachers receive both questions and answers, focusing on explanation quality
- **Efficient Training**: 7B models outperform 671B models (DeepSeek R1) on reasoning benchmarks
- **No Post-Processing**: Raw outputs are directly usable without filtering

### Key Results from Sakana AI:
- **26.3%** performance vs **18.9%** from DeepSeek R1 on reasoning tasks
- **7B teachers** successfully train **32B students**
- Training completed in **hours** vs **months** for traditional RL

## 🚀 **Quick Start for Google Colab**

1. **Open in Colab**: Upload `RLT_Teacher_Student_Training.ipynb` to Google Colab
2. **Enable GPU**: Runtime → Change runtime type → GPU (T4 or better)
3. **Run All Cells**: Execute the notebook step by step
4. **Monitor Training**: Watch the dense reward signals and explanation quality

## 📋 **What the Notebook Does**

### Phase 1: Setup
- Installs required libraries (transformers, peft, trl, etc.)
- Configures models with efficient quantization for Colab
- Loads sample mathematical reasoning problems

### Phase 2: Teacher Training
- Generates explanations for question-answer pairs
- Calculates dense reward signals from student understanding
- Filters high-quality explanations for training
- Fine-tunes teacher model using LoRA

### Phase 3: Evaluation
- Visualizes training metrics and reward distributions
- Tests trained teacher on new problems
- Compares performance against baselines

### Phase 4: Student Distillation (Optional)
- Trains student model using teacher explanations
- Implements knowledge distillation pipeline

## ⚙️ **Configuration Options**

The notebook is configured for Colab's free tier but can be scaled:

```python
# Current settings (Colab-friendly)
teacher_model = "microsoft/DialoGPT-medium"  # Small for demo
student_model = "microsoft/DialoGPT-small"
batch_size = 2
max_length = 512

# Production settings (for better GPUs)
teacher_model = "Qwen/Qwen2.5-7B-Instruct"
student_model = "Qwen/Qwen2.5-1.5B" 
batch_size = 8
max_length = 1024
```

## 📊 **Expected Results**

The notebook demonstrates:
- **Dense reward signals** improving over training
- **Quality filtering** selecting best explanations
- **Temperature diversity** in explanation generation
- **Understanding scores** correlating with explanation quality

## 🔄 **Scaling to Production**

### Recommended Upgrades:

1. **Better Models**:
   - Teacher: `Qwen2.5-7B-Instruct`, `Llama-3.1-7B-Instruct`
   - Student: `Qwen2.5-1.5B`, `Llama-3.2-3B`

2. **Larger Datasets**:
   - GSM8k (8,000+ math problems)
   - MATH dataset (12,000+ competition problems)
   - ARC-C (challenging reasoning)

3. **Advanced Training**:
   - Full reinforcement learning (PPO/DPO)
   - Multi-task reasoning datasets
   - Curriculum learning

4. **Proper Evaluation**:
   - AIME 2024 benchmarks
   - Competition MATH
   - GPQA Diamond

## 🛠 **Troubleshooting**

### Common Issues:

**Out of Memory**: 
- Reduce `batch_size` to 1
- Use smaller models
- Enable gradient checkpointing

**Slow Training**:
- Ensure GPU is enabled
- Use mixed precision (fp16=True)
- Increase gradient accumulation

**Poor Explanations**:
- Increase training data
- Adjust temperature range
- Filter higher quality examples

## 📚 **Key References**

- [Sakana AI RLT Paper](https://arxiv.org/abs/2506.08388)
- [Sakana AI Blog Post](https://sakana.ai/rlt/)
- [Original GitHub Code](https://github.com/SakanaAI/RLT)

## 🤝 **Contributing**

This implementation is based on Sakana AI's research. For production use:

1. Scale to proper datasets (GSM8k, MATH)
2. Implement full RL training pipeline  
3. Add comprehensive evaluation benchmarks
4. Optimize for larger model sizes

## 📝 **License**

This notebook is for educational and research purposes. Please refer to Sakana AI's original work for commercial applications.

---

**Built for the future of efficient AI reasoning! 🚀** 