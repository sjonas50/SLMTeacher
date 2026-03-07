#!/usr/bin/env python3
"""
Setup verification script for RLT implementation.
Checks that all required dependencies and configurations are properly set up.
"""
import os
import sys
from pathlib import Path


def check_python_version():
    """Check Python version is 3.10+"""
    print("Checking Python version...", end=" ")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"OK: Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"FAIL: Python {version.major}.{version.minor} (requires 3.10+)")
        return False


def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def check_dependencies():
    """Check all required dependencies"""
    print("\nChecking dependencies:")

    dependencies = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("anthropic", "anthropic"),
        ("peft", "peft"),
        ("bitsandbytes", "bitsandbytes"),
        ("accelerate", "accelerate"),
        ("datasets", "datasets"),
        ("trl", "trl"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
        ("psutil", "psutil"),
    ]

    all_installed = True
    for package, import_name in dependencies:
        installed = check_package(package, import_name)
        status = "OK" if installed else "MISSING"
        print(f"  [{status}] {package}")
        if not installed:
            all_installed = False

    return all_installed


def check_accelerator():
    """Check GPU/MPS/CPU availability"""
    print("\nChecking compute device...", end=" ")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"OK: CUDA {torch.version.cuda} with {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_mem / (1024**3)
                print(f"     GPU {i}: {props.name} ({mem_gb:.1f} GB)")
            return True
        elif torch.backends.mps.is_available():
            print("OK: Apple MPS (Metal Performance Shaders)")
            return True
        else:
            print("WARN: No GPU detected (CPU mode only - training will be slow)")
            return True  # Not a failure, just a warning
    except Exception as e:
        print(f"FAIL: Could not check compute device: {e}")
        return False


def check_environment_variables():
    """Check required environment variables"""
    print("\nChecking environment variables:")

    claude_key = os.getenv("CLAUDE_API_KEY")
    if claude_key:
        masked = claude_key[:8] + "..." + claude_key[-4:]
        print(f"  [OK] CLAUDE_API_KEY is set ({masked})")
        return True
    else:
        print("  [WARN] CLAUDE_API_KEY not set (required for Claude teacher)")
        print("         Set with: export CLAUDE_API_KEY='your-key-here'")
        return True  # Warning, not failure


def check_directories():
    """Check project structure"""
    print("\nChecking project structure:")

    required_dirs = [
        "src",
        "src/models",
        "src/teachers",
        "src/training",
        "src/rewards",
        "src/data",
        "src/utils",
    ]

    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        status = "OK" if path.exists() else "MISSING"
        print(f"  [{status}] {dir_path}/")
        if status == "MISSING":
            all_exist = False

    return all_exist


def check_models():
    """Test loading a small model"""
    print("\nTesting model loading...", end=" ")
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("gpt2")
        print("OK: Can load models from HuggingFace")
        return True
    except Exception as e:
        print(f"FAIL: Error loading test model: {e}")
        return False


def main():
    """Run all checks"""
    print("RLT Setup Verification")
    print("=" * 50)

    checks = [
        check_python_version(),
        check_dependencies(),
        check_accelerator(),
        check_environment_variables(),
        check_directories(),
        check_models(),
    ]

    print("\n" + "=" * 50)
    if all(checks):
        print("All checks passed! Ready to train.")
        print("\nNext steps:")
        print("1. Set CLAUDE_API_KEY if using Claude teacher")
        print("2. Run: python train.py --create-config")
        print("3. Run: python train.py")
    else:
        print("Some checks failed. Please install missing dependencies:")
        print("   pip install -r requirements.txt")
        print("   # or: pip install -e .")

    print("\nFor development setup:")
    print("   pip install -e '.[dev]'")


if __name__ == "__main__":
    main()
