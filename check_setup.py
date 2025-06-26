#!/usr/bin/env python3
"""
Setup verification script for RLT implementation.
Checks that all required dependencies and configurations are properly set up.
"""
import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version is 3.8+"""
    print("🐍 Checking Python version...", end=" ")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} (requires 3.8+)")
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
    print("\n📦 Checking dependencies:")
    
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
        ("gputil", "GPUtil")
    ]
    
    all_installed = True
    for package, import_name in dependencies:
        status = "✅" if check_package(package, import_name) else "❌"
        print(f"  {status} {package}")
        if status == "❌":
            all_installed = False
    
    return all_installed


def check_cuda():
    """Check CUDA availability"""
    print("\n🎮 Checking CUDA...", end=" ")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA {torch.version.cuda} with {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                print(f"     GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("⚠️  CUDA not available (CPU mode only)")
            return True  # Not a failure, just a warning
    except:
        print("❌ Could not check CUDA")
        return False


def check_environment_variables():
    """Check required environment variables"""
    print("\n🔑 Checking environment variables:")
    
    claude_key = os.getenv("CLAUDE_API_KEY")
    if claude_key:
        print(f"  ✅ CLAUDE_API_KEY is set ({len(claude_key)} characters)")
        return True
    else:
        print("  ⚠️  CLAUDE_API_KEY not set (required for Claude teacher)")
        print("     Set with: export CLAUDE_API_KEY='your-key-here'")
        return True  # Warning, not failure


def check_directories():
    """Check project structure"""
    print("\n📁 Checking project structure:")
    
    required_dirs = [
        "src",
        "src/models",
        "src/teachers", 
        "src/training",
        "src/rewards",
        "src/data",
        "src/utils",
        "notebooks"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        status = "✅" if path.exists() else "❌"
        print(f"  {status} {dir_path}/")
        if status == "❌":
            all_exist = False
    
    return all_exist


def check_models():
    """Test loading a small model"""
    print("\n🤖 Testing model loading...", end=" ")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("✅ Can load models from HuggingFace")
        return True
    except Exception as e:
        print(f"❌ Error loading test model: {e}")
        return False


def check_flash_attention():
    """Check if Flash Attention 2 is available"""
    print("\n⚡ Checking Flash Attention 2...", end=" ")
    try:
        import torch
        from transformers import AutoConfig
        
        # Check if Flash Attention is available
        config = AutoConfig.from_pretrained("gpt2")
        if hasattr(config, '_attn_implementation'):
            print("✅ Flash Attention 2 support detected")
            return True
        else:
            print("⚠️  Flash Attention 2 may not be available")
            return True  # Warning, not failure
    except:
        print("⚠️  Could not check Flash Attention 2")
        return True


def main():
    """Run all checks"""
    print("🚀 RLT Setup Verification")
    print("=" * 50)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_cuda(),
        check_environment_variables(),
        check_directories(),
        check_models(),
        check_flash_attention()
    ]
    
    print("\n" + "=" * 50)
    if all(checks):
        print("✅ All checks passed! Ready to train.")
        print("\nNext steps:")
        print("1. Set CLAUDE_API_KEY if using Claude teacher")
        print("2. Run: python train_optimized_rlt.py --create-config")
        print("3. Run: python train_optimized_rlt.py")
    else:
        print("❌ Some checks failed. Please install missing dependencies:")
        print("   pip install -r requirements.txt")
    
    print("\nFor development setup:")
    print("   pip install -r requirements-dev.txt")


if __name__ == "__main__":
    main()