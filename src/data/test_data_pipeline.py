#!/usr/bin/env python3
"""
Test suite for RLT Data Pipeline

Run with: python -m pytest src/data/test_data_pipeline.py
"""

import unittest
import tempfile
import json
from pathlib import Path

from src.data import (
    RLTDataPoint,
    DataProcessor,
    CacheManager,
    DataLoader,
    BatchGenerator
)


class TestRLTDataPoint(unittest.TestCase):
    """Test RLTDataPoint class."""
    
    def test_creation(self):
        """Test data point creation."""
        dp = RLTDataPoint(
            question="What is 2 + 2?",
            solution="4",
            subject="math",
            difficulty="easy"
        )
        
        self.assertEqual(dp.question, "What is 2 + 2?")
        self.assertEqual(dp.solution, "4")
        self.assertEqual(dp.subject, "math")
        self.assertEqual(dp.difficulty, "easy")
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        dp = RLTDataPoint("Q", "A", "math", "easy")
        d = dp.to_dict()
        
        self.assertEqual(d['question'], "Q")
        self.assertEqual(d['solution'], "A")
        self.assertEqual(d['subject'], "math")
        self.assertEqual(d['difficulty'], "easy")
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {'question': 'Q', 'solution': 'A', 'subject': 'math', 'difficulty': 'hard'}
        dp = RLTDataPoint.from_dict(d)
        
        self.assertEqual(dp.question, "Q")
        self.assertEqual(dp.difficulty, "hard")


class TestDataProcessor(unittest.TestCase):
    """Test DataProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = DataProcessor()
        self.sample = RLTDataPoint(
            question="What is 15% of 80?",
            solution="12",
            subject="math",
            difficulty="easy"
        )
    
    def test_format_teacher_input(self):
        """Test teacher input formatting."""
        teacher_input = self.processor.format_teacher_input(self.sample)
        
        self.assertIn("What is 15% of 80?", teacher_input.prompt)
        self.assertIn("12", teacher_input.prompt)
        self.assertEqual(teacher_input.question, "What is 15% of 80?")
        self.assertEqual(teacher_input.solution, "12")
    
    def test_format_student_input(self):
        """Test student input formatting."""
        explanation = "To find 15% of 80, multiply 80 by 0.15"
        student_input = self.processor.format_student_input(
            self.sample.question,
            explanation
        )
        
        self.assertIn("What is 15% of 80?", student_input.prompt)
        self.assertIn(explanation, student_input.prompt)
    
    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        dirty_text = "  Hello   world\n\ntest  "
        clean = self.processor._clean_text(dirty_text)
        
        self.assertEqual(clean, "Hello world\n\ntest")
    
    def test_math_preprocessing(self):
        """Test math text preprocessing."""
        text = "Calculate 5×3÷2"
        processed = self.processor._preprocess_math(text)
        
        self.assertIn("*", processed)
        self.assertIn("/", processed)
        self.assertNotIn("×", processed)
        self.assertNotIn("÷", processed)
    
    def test_validation(self):
        """Test data point validation."""
        # Valid data point
        is_valid, issues = self.processor.validate_data_point(self.sample)
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
        
        # Invalid data point
        invalid = RLTDataPoint("", "answer", "unknown", "invalid")
        is_valid, issues = self.processor.validate_data_point(invalid)
        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 0)
    
    def test_batch_preparation(self):
        """Test batch preparation."""
        data_points = [self.sample] * 10
        batches = self.processor.prepare_batch(data_points, batch_size=3)
        
        self.assertEqual(len(batches), 4)  # 3 full batches + 1 partial
        self.assertEqual(len(batches[0]), 3)
        self.assertEqual(len(batches[-1]), 1)
    
    def test_statistics(self):
        """Test statistics calculation."""
        data_points = [
            RLTDataPoint("Q1", "A1", "math", "easy"),
            RLTDataPoint("Q2", "A2", "math", "medium"),
            RLTDataPoint("Q3", "A3", "physics", "hard")
        ]
        
        stats = self.processor.get_statistics(data_points)
        
        self.assertEqual(stats['total_count'], 3)
        self.assertEqual(stats['subjects']['math'], 2)
        self.assertEqual(stats['subjects']['physics'], 1)
        self.assertIn('avg_question_length', stats)


class TestCacheManager(unittest.TestCase):
    """Test CacheManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = CacheManager(cache_dir=self.temp_dir)
    
    def test_dataset_caching(self):
        """Test dataset caching."""
        # Cache a dataset
        test_data = [{"question": "Q1", "answer": "A1"}]
        key = self.cache.cache_dataset("test_dataset", test_data)
        
        self.assertIsNotNone(key)
        
        # Retrieve from cache
        cached = self.cache.get_dataset("test_dataset")
        self.assertEqual(cached, test_data)
        
        # Cache hit
        self.assertEqual(self.cache.stats['hits'], 1)
    
    def test_explanation_caching(self):
        """Test explanation caching."""
        # Cache an explanation
        key = self.cache.cache_explanation(
            question="Q",
            solution="S",
            explanation="E",
            model="test_model",
            temperature=0.7
        )
        
        # Retrieve from cache
        cached = self.cache.get_explanation(
            question="Q",
            solution="S",
            model="test_model",
            temperature=0.7
        )
        
        self.assertIsNotNone(cached)
        self.assertEqual(cached['explanation'], "E")
    
    def test_cache_stats(self):
        """Test cache statistics."""
        # Add some data
        self.cache.cache_dataset("test", [1, 2, 3])
        self.cache.get_dataset("test")
        self.cache.get_dataset("nonexistent")
        
        stats = self.cache.get_cache_stats()
        
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        self.assertGreater(stats['hit_rate'], 0)
        self.assertEqual(stats['total_entries'], 1)
    
    def test_cache_clearing(self):
        """Test cache clearing."""
        # Add data
        self.cache.cache_dataset("test1", [1])
        self.cache.cache_explanation("Q", "S", "E", "model", 0.5)
        
        # Clear specific type
        self.cache.clear_cache('dataset')
        stats = self.cache.get_cache_stats()
        self.assertEqual(stats['datasets_cached'], 0)
        self.assertGreater(stats['explanations_cached'], 0)
        
        # Clear all
        self.cache.clear_cache()
        stats = self.cache.get_cache_stats()
        self.assertEqual(stats['total_entries'], 0)


class TestDataLoader(unittest.TestCase):
    """Test DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = DataLoader(data_dir=self.temp_dir, auto_download=False)
    
    def test_custom_json_loading(self):
        """Test loading custom JSON dataset."""
        # Create test JSON file
        test_data = [
            {
                "question": "What is 2 + 2?",
                "answer": "4",
                "subject": "math",
                "difficulty": "easy"
            }
        ]
        
        json_path = Path(self.temp_dir) / "test.json"
        with open(json_path, 'w') as f:
            json.dump(test_data, f)
        
        # Load dataset
        data = self.loader.load_dataset(str(json_path))
        
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0].question, "What is 2 + 2?")
        self.assertEqual(data[0].solution, "4")
    
    def test_export_formats(self):
        """Test dataset export in different formats."""
        data = [
            RLTDataPoint("Q1", "A1", "math", "easy"),
            RLTDataPoint("Q2", "A2", "physics", "hard")
        ]
        
        # Test JSON export
        json_path = Path(self.temp_dir) / "export.json"
        self.loader.export_dataset(data, str(json_path), format='json')
        self.assertTrue(json_path.exists())
        
        # Test JSONL export
        jsonl_path = Path(self.temp_dir) / "export.jsonl"
        self.loader.export_dataset(data, str(jsonl_path), format='jsonl')
        self.assertTrue(jsonl_path.exists())
        
        # Verify JSONL format
        with open(jsonl_path, 'r') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)
    
    def test_dataset_merging(self):
        """Test merging multiple datasets."""
        # Create test datasets
        data1 = [RLTDataPoint(f"Q{i}", f"A{i}", "math", "easy") for i in range(10)]
        data2 = [RLTDataPoint(f"P{i}", f"B{i}", "physics", "hard") for i in range(10)]
        
        # Save as JSON files
        path1 = Path(self.temp_dir) / "data1.json"
        path2 = Path(self.temp_dir) / "data2.json"
        
        self.loader.export_dataset(data1, str(path1))
        self.loader.export_dataset(data2, str(path2))
        
        # Merge datasets
        merged = self.loader.merge_datasets(
            [str(path1), str(path2)],
            proportions=[0.7, 0.3],
            total_samples=10
        )
        
        self.assertEqual(len(merged), 10)
        
        # Check approximate proportions
        math_count = sum(1 for dp in merged if dp.subject == "math")
        self.assertGreaterEqual(math_count, 5)  # Should be around 7


class TestBatchGenerator(unittest.TestCase):
    """Test BatchGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = DataLoader(data_dir=self.temp_dir, auto_download=False)
        
        # Create test dataset
        test_data = [
            RLTDataPoint(f"Q{i}", f"A{i}", "math", ["easy", "medium", "hard"][i % 3])
            for i in range(30)
        ]
        
        json_path = Path(self.temp_dir) / "test.json"
        self.loader.export_dataset(test_data, str(json_path))
        self.dataset_path = str(json_path)
    
    def test_basic_batch_generation(self):
        """Test basic batch generation."""
        generator = BatchGenerator(self.loader, batch_size=10)
        
        batches = list(generator.generate_batches(
            self.dataset_path,
            num_epochs=1,
            shuffle=False
        ))
        
        self.assertEqual(len(batches), 3)  # 30 samples / 10 batch_size
        self.assertEqual(len(batches[0][0]), 10)  # First batch size
        self.assertEqual(batches[0][1]['epoch'], 0)
    
    def test_curriculum_learning(self):
        """Test curriculum learning batch generation."""
        generator = BatchGenerator(
            self.loader,
            batch_size=10,
            curriculum_learning=True
        )
        
        batches = list(generator.generate_batches(
            self.dataset_path,
            num_epochs=1,
            shuffle=False
        ))
        
        # First batches should contain easier examples
        first_batch = batches[0][0]
        difficulties = [dp.difficulty for dp in first_batch]
        
        # Most should be easy
        easy_count = difficulties.count('easy')
        self.assertGreaterEqual(easy_count, 7)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], exit=False)


if __name__ == "__main__":
    run_tests()