"""
Data Loader for RLT Training Pipeline

This module provides comprehensive data loading functionality for various datasets
including GSM8K, MATH, ARC-C, and custom JSON formats. Features automatic downloading,
caching, and efficient batch generation.
"""

import os
import json
import requests
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Iterator, Tuple, Any
from dataclasses import dataclass
import random
import logging
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

# Import from other modules
from .data_processor import RLTDataPoint, DataProcessor
from .cache_manager import CacheManager, get_default_cache

# Configure logging
logger = logging.getLogger(__name__)

# Dataset URLs and configurations
DATASET_CONFIGS = {
    'gsm8k': {
        'url': 'https://huggingface.co/datasets/gsm8k/resolve/main/data/',
        'files': ['train.jsonl', 'test.jsonl'],
        'format': 'jsonl',
        'loader': 'load_gsm8k'
    },
    'math': {
        'url': 'https://huggingface.co/datasets/hendrycks/competition_math/resolve/main/',
        'format': 'json',
        'loader': 'load_math'
    },
    'arc-c': {
        'url': 'https://huggingface.co/datasets/ai2_arc/resolve/main/data/',
        'files': ['ARC-Challenge-Train.jsonl', 'ARC-Challenge-Dev.jsonl', 'ARC-Challenge-Test.jsonl'],
        'format': 'jsonl',
        'loader': 'load_arc'
    }
}


class DataLoader:
    """
    Main data loader class for RLT training pipeline.
    Handles multiple dataset formats with automatic downloading and caching.
    """
    
    def __init__(self,
                 data_dir: str = "~/.rlt_data",
                 cache_manager: Optional[CacheManager] = None,
                 auto_download: bool = True,
                 verify_checksums: bool = True):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory for storing downloaded datasets
            cache_manager: Cache manager instance (uses default if None)
            auto_download: Whether to automatically download missing datasets
            verify_checksums: Whether to verify dataset checksums
        """
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_manager = cache_manager or get_default_cache()
        self.auto_download = auto_download
        self.verify_checksums = verify_checksums
        
        # Data processor for formatting
        self.processor = DataProcessor()
        
        # Dataset loaders
        self.loaders = {
            'gsm8k': self._load_gsm8k,
            'math': self._load_math,
            'arc-c': self._load_arc,
            'custom': self._load_custom_json
        }
        
        # Track loaded datasets
        self.loaded_datasets = {}
    
    def load_dataset(self,
                    dataset_name: str,
                    split: str = 'train',
                    subset: Optional[str] = None,
                    max_samples: Optional[int] = None,
                    use_cache: bool = True) -> List[RLTDataPoint]:
        """
        Load a dataset by name.
        
        Args:
            dataset_name: Name of the dataset (gsm8k, math, arc-c, or path to custom)
            split: Dataset split to load (train, test, val)
            subset: Optional subset of the dataset
            max_samples: Maximum number of samples to load
            use_cache: Whether to use cached data
            
        Returns:
            List of RLTDataPoint objects
        """
        # Check cache first
        cache_key = f"{dataset_name}_{split}_{subset}_{max_samples}"
        
        if use_cache:
            cached_data = self.cache_manager.get_dataset(cache_key)
            if cached_data is not None:
                logger.info(f"Loaded {dataset_name} from cache")
                return cached_data
        
        # Load dataset
        if dataset_name in self.loaders:
            data = self.loaders[dataset_name](split, subset, max_samples)
        elif os.path.exists(dataset_name):
            # Load custom dataset from file
            data = self._load_custom_json(dataset_name, max_samples)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Cache the loaded data
        if use_cache and data:
            self.cache_manager.cache_dataset(cache_key, data)
        
        # Store reference
        self.loaded_datasets[dataset_name] = data
        
        logger.info(f"Loaded {len(data)} samples from {dataset_name}")
        return data
    
    def _download_file(self, url: str, destination: Path, chunk_size: int = 8192):
        """Download a file with progress bar."""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        with open(destination, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    
    def _ensure_dataset_downloaded(self, dataset_name: str) -> Path:
        """Ensure a dataset is downloaded and return its path."""
        dataset_path = self.data_dir / dataset_name
        
        if dataset_path.exists():
            return dataset_path
        
        if not self.auto_download:
            raise FileNotFoundError(f"Dataset {dataset_name} not found at {dataset_path}")
        
        # Download dataset
        logger.info(f"Downloading {dataset_name}...")
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        config = DATASET_CONFIGS.get(dataset_name, {})
        
        if 'files' in config:
            # Download individual files
            for file_name in config['files']:
                url = config['url'] + file_name
                destination = dataset_path / file_name
                
                if not destination.exists():
                    self._download_file(url, destination)
        
        return dataset_path
    
    def _load_gsm8k(self, 
                   split: str = 'train',
                   subset: Optional[str] = None,
                   max_samples: Optional[int] = None) -> List[RLTDataPoint]:
        """Load GSM8K dataset."""
        dataset_path = self._ensure_dataset_downloaded('gsm8k')
        
        # Determine file to load
        if split == 'train':
            file_path = dataset_path / 'train.jsonl'
        elif split == 'test':
            file_path = dataset_path / 'test.jsonl'
        else:
            raise ValueError(f"Invalid split for GSM8K: {split}")
        
        # Try HuggingFace datasets first
        try:
            from datasets import load_dataset
            dataset = load_dataset('gsm8k', 'main', split=split)
            
            data_points = []
            for idx, item in enumerate(dataset):
                if max_samples and idx >= max_samples:
                    break
                
                # Extract answer from the format "#### answer"
                answer = item['answer'].split('####')[-1].strip()
                
                data_point = RLTDataPoint(
                    question=item['question'],
                    solution=answer,
                    subject='math',
                    difficulty=self._estimate_difficulty(item['question'])
                )
                
                data_points.append(data_point)
            
            return data_points
            
        except ImportError:
            logger.info("Datasets library not available, loading from file")
        
        # Fallback to manual loading
        if not file_path.exists():
            logger.warning(f"GSM8K file not found: {file_path}")
            return []
        
        data_points = []
        
        with open(file_path, 'r') as f:
            for idx, line in enumerate(f):
                if max_samples and idx >= max_samples:
                    break
                
                item = json.loads(line)
                
                # Extract answer from the format "#### answer"
                answer = item['answer'].split('####')[-1].strip()
                
                data_point = RLTDataPoint(
                    question=item['question'],
                    solution=answer,
                    subject='math',
                    difficulty=self._estimate_difficulty(item['question'])
                )
                
                data_points.append(data_point)
        
        return data_points
    
    def _load_math(self,
                  split: str = 'train',
                  subset: Optional[str] = None,
                  max_samples: Optional[int] = None) -> List[RLTDataPoint]:
        """Load MATH dataset."""
        # Try HuggingFace datasets first
        try:
            from datasets import load_dataset
            
            # MATH dataset has different subsets (algebra, geometry, etc.)
            if subset:
                dataset = load_dataset('hendrycks/competition_math', subset, split=split)
            else:
                # Load all subsets
                subsets = ['algebra', 'counting_and_probability', 'geometry', 
                          'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
                
                all_data = []
                for sub in subsets:
                    try:
                        ds = load_dataset('hendrycks/competition_math', sub, split=split)
                        all_data.extend(list(ds))
                    except:
                        logger.warning(f"Failed to load MATH subset: {sub}")
                
                dataset = all_data
            
            data_points = []
            for idx, item in enumerate(dataset):
                if max_samples and idx >= max_samples:
                    break
                
                data_point = RLTDataPoint(
                    question=item['problem'],
                    solution=item['solution'],
                    subject='math',
                    difficulty=item.get('level', 'medium')
                )
                
                data_points.append(data_point)
            
            return data_points
            
        except Exception as e:
            logger.error(f"Failed to load MATH dataset: {e}")
            return []
    
    def _load_arc(self,
                 split: str = 'train',
                 subset: Optional[str] = None,
                 max_samples: Optional[int] = None) -> List[RLTDataPoint]:
        """Load ARC (AI2 Reasoning Challenge) dataset."""
        # Try HuggingFace datasets first
        try:
            from datasets import load_dataset
            
            # ARC-Challenge is the harder subset
            dataset = load_dataset('ai2_arc', 'ARC-Challenge', split=split)
            
            data_points = []
            for idx, item in enumerate(dataset):
                if max_samples and idx >= max_samples:
                    break
                
                # Format multiple choice question
                question = item['question']
                choices = item['choices']
                
                # Add choices to question
                question_with_choices = question + "\n\nChoices:\n"
                for i, (choice_id, choice_text) in enumerate(zip(choices['label'], choices['text'])):
                    question_with_choices += f"{choice_id}. {choice_text}\n"
                
                # Get correct answer
                answer_idx = choices['label'].index(item['answerKey'])
                correct_answer = f"{item['answerKey']}. {choices['text'][answer_idx]}"
                
                data_point = RLTDataPoint(
                    question=question_with_choices,
                    solution=correct_answer,
                    subject='science',
                    difficulty='hard'  # ARC-Challenge is considered hard
                )
                
                data_points.append(data_point)
            
            return data_points
            
        except Exception as e:
            logger.error(f"Failed to load ARC dataset: {e}")
            return []
    
    def _load_custom_json(self, 
                         file_path: str,
                         max_samples: Optional[int] = None) -> List[RLTDataPoint]:
        """Load custom JSON dataset."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Custom dataset not found: {file_path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and 'data' in data:
            items = data['data']
        elif isinstance(data, dict) and 'examples' in data:
            items = data['examples']
        else:
            raise ValueError(f"Unsupported JSON format in {file_path}")
        
        data_points = []
        
        for idx, item in enumerate(items):
            if max_samples and idx >= max_samples:
                break
            
            # Try to extract fields with various common names
            question = (item.get('question') or item.get('problem') or 
                       item.get('prompt') or item.get('input'))
            
            solution = (item.get('solution') or item.get('answer') or 
                       item.get('output') or item.get('response'))
            
            subject = (item.get('subject') or item.get('category') or 
                      item.get('type') or 'general')
            
            difficulty = (item.get('difficulty') or item.get('level') or 
                         self._estimate_difficulty(question))
            
            if question and solution:
                data_point = RLTDataPoint(
                    question=str(question),
                    solution=str(solution),
                    subject=subject,
                    difficulty=difficulty
                )
                
                data_points.append(data_point)
        
        return data_points
    
    def _estimate_difficulty(self, text: str) -> str:
        """Estimate difficulty based on text complexity."""
        if not text:
            return 'medium'
        
        # Simple heuristics
        word_count = len(text.split())
        
        if word_count < 50:
            return 'easy'
        elif word_count < 150:
            return 'medium'
        else:
            return 'hard'
    
    def create_data_iterator(self,
                           dataset_name: str,
                           batch_size: int = 32,
                           shuffle: bool = True,
                           repeat: bool = True,
                           **load_kwargs) -> Iterator[List[RLTDataPoint]]:
        """
        Create an iterator over batches of data.
        
        Args:
            dataset_name: Name of the dataset
            batch_size: Size of each batch
            shuffle: Whether to shuffle data
            repeat: Whether to repeat the dataset indefinitely
            **load_kwargs: Additional arguments for load_dataset
            
        Yields:
            Batches of RLTDataPoint objects
        """
        # Load dataset
        data = self.load_dataset(dataset_name, **load_kwargs)
        
        if not data:
            raise ValueError(f"No data loaded from {dataset_name}")
        
        # Create batches
        while True:
            if shuffle:
                random.shuffle(data)
            
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                yield batch
            
            if not repeat:
                break
    
    def merge_datasets(self,
                      dataset_names: List[str],
                      proportions: Optional[List[float]] = None,
                      total_samples: Optional[int] = None,
                      **load_kwargs) -> List[RLTDataPoint]:
        """
        Merge multiple datasets with optional proportions.
        
        Args:
            dataset_names: List of dataset names to merge
            proportions: Optional proportions for each dataset (must sum to 1)
            total_samples: Total number of samples in merged dataset
            **load_kwargs: Additional arguments for load_dataset
            
        Returns:
            Merged dataset
        """
        if proportions:
            if len(proportions) != len(dataset_names):
                raise ValueError("Number of proportions must match number of datasets")
            if abs(sum(proportions) - 1.0) > 1e-6:
                raise ValueError("Proportions must sum to 1")
        else:
            # Equal proportions
            proportions = [1.0 / len(dataset_names)] * len(dataset_names)
        
        merged_data = []
        
        for dataset_name, proportion in zip(dataset_names, proportions):
            # Load dataset
            data = self.load_dataset(dataset_name, **load_kwargs)
            
            # Calculate samples to take
            if total_samples:
                n_samples = int(total_samples * proportion)
            else:
                n_samples = len(data)
            
            # Sample data
            if n_samples < len(data):
                sampled_data = random.sample(data, n_samples)
            else:
                sampled_data = data
            
            merged_data.extend(sampled_data)
        
        # Shuffle merged data
        random.shuffle(merged_data)
        
        return merged_data
    
    def export_dataset(self,
                      data: List[RLTDataPoint],
                      output_path: str,
                      format: str = 'json'):
        """
        Export dataset to file.
        
        Args:
            data: List of data points to export
            output_path: Output file path
            format: Export format (json, jsonl, csv)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump([dp.to_dict() for dp in data], f, indent=2)
        
        elif format == 'jsonl':
            with open(output_path, 'w') as f:
                for dp in data:
                    f.write(json.dumps(dp.to_dict()) + '\n')
        
        elif format == 'csv':
            df = pd.DataFrame([dp.to_dict() for dp in data])
            df.to_csv(output_path, index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported {len(data)} samples to {output_path}")
    
    def get_dataset_stats(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about loaded datasets.
        
        Args:
            dataset_name: Specific dataset name or None for all
            
        Returns:
            Dictionary of statistics
        """
        if dataset_name:
            if dataset_name not in self.loaded_datasets:
                return {}
            datasets = {dataset_name: self.loaded_datasets[dataset_name]}
        else:
            datasets = self.loaded_datasets
        
        stats = {}
        
        for name, data in datasets.items():
            dataset_stats = self.processor.get_statistics(data)
            stats[name] = dataset_stats
        
        return stats


class BatchGenerator:
    """
    Efficient batch generator with advanced features like curriculum learning
    and dynamic batching.
    """
    
    def __init__(self, 
                 data_loader: DataLoader,
                 batch_size: int = 32,
                 curriculum_learning: bool = False,
                 dynamic_batching: bool = False):
        """
        Initialize batch generator.
        
        Args:
            data_loader: DataLoader instance
            batch_size: Base batch size
            curriculum_learning: Whether to use curriculum learning
            dynamic_batching: Whether to use dynamic batch sizing
        """
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.curriculum_learning = curriculum_learning
        self.dynamic_batching = dynamic_batching
        
        self.current_difficulty = 'easy'
        self.difficulty_progression = ['easy', 'medium', 'hard']
    
    def generate_batches(self,
                        dataset_names: Union[str, List[str]],
                        num_epochs: int = 1,
                        shuffle: bool = True,
                        **load_kwargs) -> Iterator[Tuple[List[RLTDataPoint], Dict[str, Any]]]:
        """
        Generate batches with optional curriculum learning.
        
        Args:
            dataset_names: Dataset name(s) to load
            num_epochs: Number of epochs to generate
            shuffle: Whether to shuffle data
            **load_kwargs: Additional arguments for dataset loading
            
        Yields:
            Tuple of (batch, metadata)
        """
        # Ensure dataset_names is a list
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        
        # Load and merge datasets
        all_data = []
        for dataset_name in dataset_names:
            data = self.data_loader.load_dataset(dataset_name, **load_kwargs)
            all_data.extend(data)
        
        # Apply curriculum learning if enabled
        if self.curriculum_learning:
            all_data = self._apply_curriculum(all_data)
        
        # Generate batches
        for epoch in range(num_epochs):
            if shuffle:
                random.shuffle(all_data)
            
            for i in range(0, len(all_data), self.batch_size):
                batch = all_data[i:i + self.batch_size]
                
                # Dynamic batching based on sequence length
                if self.dynamic_batching:
                    batch = self._dynamic_batch_size(batch)
                
                metadata = {
                    'epoch': epoch,
                    'batch_idx': i // self.batch_size,
                    'batch_size': len(batch),
                    'dataset_names': dataset_names
                }
                
                yield batch, metadata
    
    def _apply_curriculum(self, data: List[RLTDataPoint]) -> List[RLTDataPoint]:
        """Apply curriculum learning by sorting by difficulty."""
        # Group by difficulty
        difficulty_groups = defaultdict(list)
        for dp in data:
            difficulty_groups[dp.difficulty].append(dp)
        
        # Create curriculum order
        curriculum_data = []
        for difficulty in self.difficulty_progression:
            if difficulty in difficulty_groups:
                curriculum_data.extend(difficulty_groups[difficulty])
        
        return curriculum_data
    
    def _dynamic_batch_size(self, batch: List[RLTDataPoint]) -> List[RLTDataPoint]:
        """Adjust batch size based on sequence lengths."""
        # Calculate average sequence length
        avg_length = sum(len(dp.question) + len(dp.solution) for dp in batch) / len(batch)
        
        # Adjust batch size inversely proportional to length
        # Longer sequences -> smaller batch
        if avg_length > 500:
            max_batch_size = self.batch_size // 2
        elif avg_length > 200:
            max_batch_size = int(self.batch_size * 0.75)
        else:
            max_batch_size = self.batch_size
        
        return batch[:max_batch_size]


# Convenience functions
def load_dataset(dataset_name: str, **kwargs) -> List[RLTDataPoint]:
    """Load a dataset using the default data loader."""
    loader = DataLoader()
    return loader.load_dataset(dataset_name, **kwargs)


def create_batch_iterator(dataset_name: str, 
                         batch_size: int = 32, 
                         **kwargs) -> Iterator[List[RLTDataPoint]]:
    """Create a batch iterator using the default data loader."""
    loader = DataLoader()
    return loader.create_data_iterator(dataset_name, batch_size, **kwargs)