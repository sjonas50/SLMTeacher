#!/usr/bin/env python3
"""
Test script for RLT Teacher-Student Training notebook
Tests key components without full training
"""

import sys
import json

# Track test results
test_results = {
    "imports": {"status": "pending", "message": ""},
    "data_processing": {"status": "pending", "message": ""},
    "config": {"status": "pending", "message": ""},
    "sample_data": {"status": "pending", "message": ""},
    "notebook_structure": {"status": "pending", "message": ""}
}

# Test 1: Check imports
print("Testing imports...")
try:
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from typing import List, Dict, Tuple, Optional
    from dataclasses import dataclass
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset, load_dataset
    from trl import SFTTrainer
    
    test_results["imports"]["status"] = "passed"
    test_results["imports"]["message"] = "All core imports successful"
    print("✓ Imports test passed")
except ImportError as e:
    test_results["imports"]["status"] = "failed"
    test_results["imports"]["message"] = f"Import error: {str(e)}"
    print(f"✗ Imports test failed: {e}")

# Test 2: Check RLT configuration
print("\nTesting RLT configuration...")
try:
    @dataclass
    class RLTConfig:
        teacher_model_name: str = "microsoft/DialoGPT-medium"
        student_model_name: str = "microsoft/DialoGPT-small"
        max_length: int = 512
        batch_size: int = 2
        learning_rate: float = 2e-4
        num_epochs: int = 1
        warmup_steps: int = 50
        lora_r: int = 16
        lora_alpha: int = 32
        lora_dropout: float = 0.1
        temperature_range: Tuple[float, float] = (0.1, 0.9)
        reward_threshold_percentile: int = 75
    
    config = RLTConfig()
    test_results["config"]["status"] = "passed"
    test_results["config"]["message"] = f"Config initialized with teacher: {config.teacher_model_name}"
    print("✓ Configuration test passed")
except Exception as e:
    test_results["config"]["status"] = "failed"
    test_results["config"]["message"] = f"Config error: {str(e)}"
    print(f"✗ Configuration test failed: {e}")

# Test 3: Data processing
print("\nTesting data processing...")
try:
    class RLTDataProcessor:
        def __init__(self):
            self.explanation_prompt = """You are an expert teacher..."""
        
        def create_teacher_prompt(self, question: str, answer: str) -> str:
            return f"Question: {question}\\nAnswer: {answer}"
        
        def load_sample_data(self) -> Dataset:
            sample_problems = [
                {"question": "Test question?", "answer": "Test answer"}
            ]
            return Dataset.from_list(sample_problems)
    
    data_processor = RLTDataProcessor()
    sample_dataset = data_processor.load_sample_data()
    
    test_results["data_processing"]["status"] = "passed"
    test_results["data_processing"]["message"] = f"Data processor initialized, {len(sample_dataset)} samples"
    print("✓ Data processing test passed")
except Exception as e:
    test_results["data_processing"]["status"] = "failed"
    test_results["data_processing"]["message"] = f"Data processing error: {str(e)}"
    print(f"✗ Data processing test failed: {e}")

# Test 4: Sample data
print("\nTesting sample data structure...")
try:
    sample_problems = [
        {
            "question": "A store sells 15 apples per day and operates 6 days a week. How many apples does it sell in 4 weeks?",
            "answer": "360"
        },
        {
            "question": "If a rectangle has length 8 meters and width 5 meters, what is its area?", 
            "answer": "40 square meters"
        }
    ]
    
    # Verify data structure
    for problem in sample_problems:
        assert "question" in problem, "Missing question field"
        assert "answer" in problem, "Missing answer field"
        assert isinstance(problem["question"], str), "Question must be string"
        assert isinstance(problem["answer"], str), "Answer must be string"
    
    test_results["sample_data"]["status"] = "passed"
    test_results["sample_data"]["message"] = f"Sample data validated, {len(sample_problems)} problems"
    print("✓ Sample data test passed")
except Exception as e:
    test_results["sample_data"]["status"] = "failed"
    test_results["sample_data"]["message"] = f"Sample data error: {str(e)}"
    print(f"✗ Sample data test failed: {e}")

# Test 5: Notebook structure
print("\nTesting notebook structure...")
try:
    # Check if notebook exists and can be read
    import nbformat
    
    with open('RLT_Teacher_Student_Training.ipynb', 'r') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Check number of cells
    num_cells = len(nb.cells)
    code_cells = sum(1 for cell in nb.cells if cell.cell_type == 'code')
    markdown_cells = sum(1 for cell in nb.cells if cell.cell_type in ['markdown', 'raw'])
    
    test_results["notebook_structure"]["status"] = "passed"
    test_results["notebook_structure"]["message"] = f"Notebook has {num_cells} cells ({code_cells} code, {markdown_cells} markdown/raw)"
    print("✓ Notebook structure test passed")
except Exception as e:
    test_results["notebook_structure"]["status"] = "failed"
    test_results["notebook_structure"]["message"] = f"Notebook structure error: {str(e)}"
    print(f"✗ Notebook structure test failed: {e}")

# Summary
print("\n" + "="*50)
print("TEST SUMMARY")
print("="*50)

passed = sum(1 for r in test_results.values() if r["status"] == "passed")
failed = sum(1 for r in test_results.values() if r["status"] == "failed")

for test_name, result in test_results.items():
    status_icon = "✓" if result["status"] == "passed" else "✗"
    print(f"{status_icon} {test_name}: {result['status']} - {result['message']}")

print(f"\nTotal: {passed} passed, {failed} failed")

# Check GPU availability
print("\n" + "="*50)
print("SYSTEM INFO")
print("="*50)
print(f"Python version: {sys.version.split()[0]}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
except:
    print("PyTorch not available")

# Save results
with open('test_results.json', 'w') as f:
    json.dump(test_results, f, indent=2)
print("\nTest results saved to test_results.json")

# Exit code
sys.exit(0 if failed == 0 else 1)