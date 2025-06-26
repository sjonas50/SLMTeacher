#!/usr/bin/env python3
"""
Setup script for RLT Data Pipeline

This script helps set up and verify the data pipeline installation.
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.8 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, but found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor} detected")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False


def verify_imports():
    """Verify that all modules can be imported."""
    print("\n🔍 Verifying imports...")
    
    modules = [
        ("Core modules", ["requests", "tqdm", "numpy"]),
        ("Data pipeline", ["src.data"])
    ]
    
    all_good = True
    
    for category, module_list in modules:
        print(f"\n{category}:")
        for module in module_list:
            try:
                __import__(module)
                print(f"  ✅ {module}")
            except ImportError as e:
                print(f"  ❌ {module}: {e}")
                all_good = False
    
    return all_good


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    dirs = [
        Path.home() / ".cache" / "rlt_data",
        Path.home() / ".rlt_data",
        Path("checkpoints"),
        Path("exports")
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {dir_path}")


def run_quick_test():
    """Run a quick test of the data pipeline."""
    print("\n🧪 Running quick test...")
    
    try:
        from src.data import RLTDataPoint, DataProcessor, CacheManager
        
        # Test data point
        dp = RLTDataPoint(
            question="What is 2 + 2?",
            solution="4",
            subject="math",
            difficulty="easy"
        )
        print(f"  ✅ Created data point: {dp.question}")
        
        # Test processor
        processor = DataProcessor()
        teacher_input = processor.format_teacher_input(dp)
        print(f"  ✅ Formatted teacher input (length: {len(teacher_input.prompt)})")
        
        # Test cache
        cache = CacheManager()
        stats = cache.get_cache_stats()
        print(f"  ✅ Cache initialized (entries: {stats['total_entries']})")
        
        print("\n✨ Data pipeline is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 RLT Data Pipeline Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    # Create directories
    create_directories()
    
    # Verify imports
    if not verify_imports():
        print("\n⚠️  Some imports failed. The pipeline may not work correctly.")
        print("Try installing missing dependencies manually.")
    
    # Run quick test
    if run_quick_test():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run example: python src/data/example_usage.py")
        print("2. Run tests: python -m pytest src/data/test_data_pipeline.py")
        print("3. Start using: from src.data import quick_start")
        return 0
    else:
        print("\n⚠️  Setup completed with warnings.")
        return 1


if __name__ == "__main__":
    sys.exit(main())