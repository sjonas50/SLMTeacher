"""
Cache Manager for RLT Data Pipeline

This module provides efficient caching functionality for datasets and explanations,
reducing API calls and improving performance during development and training.
"""

import os
import json
import pickle
import hashlib
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timedelta
import tempfile
import gzip
from contextlib import contextmanager
import threading
import logging

# Configure logging
logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages caching for datasets and API responses with features like:
    - Automatic cache invalidation
    - Compression support
    - Thread-safe operations
    - Memory-mapped files for large datasets
    - Hierarchical cache organization
    """
    
    def __init__(self, 
                 cache_dir: str = "~/.cache/rlt_data",
                 max_cache_size_gb: float = 10.0,
                 enable_compression: bool = True,
                 cache_ttl_days: int = 30):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Base directory for cache storage
            max_cache_size_gb: Maximum cache size in GB
            enable_compression: Whether to compress cached data
            cache_ttl_days: Time-to-live for cache entries in days
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.max_cache_size_bytes = max_cache_size_gb * 1024 * 1024 * 1024
        self.enable_compression = enable_compression
        self.cache_ttl = timedelta(days=cache_ttl_days)
        
        # Create cache directories
        self.datasets_dir = self.cache_dir / "datasets"
        self.explanations_dir = self.cache_dir / "explanations"
        self.metadata_dir = self.cache_dir / "metadata"
        
        for dir_path in [self.datasets_dir, self.explanations_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Thread lock for safe concurrent access
        self._lock = threading.Lock()
        
        # Load cache metadata
        self.metadata_file = self.metadata_dir / "cache_index.json"
        self.metadata = self._load_metadata()
        
        # Initialize cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'bytes_saved': 0,
            'api_calls_saved': 0
        }
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        return {'entries': {}, 'total_size': 0}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        with self._lock:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
    
    def _generate_key(self, data: Union[str, Dict, List]) -> str:
        """Generate a unique cache key from input data."""
        if isinstance(data, (dict, list)):
            data = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _get_cache_path(self, key: str, cache_type: str) -> Path:
        """Get the file path for a cache entry."""
        subdir = key[:2]  # Use first 2 chars for directory sharding
        if cache_type == "dataset":
            base_dir = self.datasets_dir / subdir
        elif cache_type == "explanation":
            base_dir = self.explanations_dir / subdir
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
        
        base_dir.mkdir(exist_ok=True)
        
        ext = ".pkl.gz" if self.enable_compression else ".pkl"
        return base_dir / f"{key}{ext}"
    
    def _check_ttl(self, key: str) -> bool:
        """Check if a cache entry is still valid based on TTL."""
        if key not in self.metadata['entries']:
            return False
        
        created_time = datetime.fromisoformat(self.metadata['entries'][key]['created'])
        return datetime.now() - created_time < self.cache_ttl
    
    def _cleanup_old_entries(self):
        """Remove expired cache entries."""
        with self._lock:
            expired_keys = []
            for key, entry in self.metadata['entries'].items():
                if not self._check_ttl(key):
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
    
    def _remove_entry(self, key: str):
        """Remove a cache entry and update metadata."""
        if key not in self.metadata['entries']:
            return
        
        entry = self.metadata['entries'][key]
        cache_path = Path(entry['path'])
        
        if cache_path.exists():
            file_size = cache_path.stat().st_size
            cache_path.unlink()
            self.metadata['total_size'] -= file_size
        
        del self.metadata['entries'][key]
    
    def _ensure_cache_size_limit(self):
        """Ensure cache doesn't exceed size limit using LRU eviction."""
        if self.metadata['total_size'] <= self.max_cache_size_bytes:
            return
        
        # Sort entries by last access time
        entries = sorted(
            self.metadata['entries'].items(),
            key=lambda x: x[1].get('last_accessed', x[1]['created'])
        )
        
        # Remove oldest entries until under limit
        while self.metadata['total_size'] > self.max_cache_size_bytes and entries:
            key, _ = entries.pop(0)
            self._remove_entry(key)
    
    def cache_dataset(self, 
                     dataset_name: str, 
                     data: Any, 
                     metadata: Optional[Dict] = None) -> str:
        """
        Cache a dataset with metadata.
        
        Args:
            dataset_name: Name of the dataset
            data: Dataset to cache
            metadata: Optional metadata about the dataset
            
        Returns:
            Cache key for the dataset
        """
        # Generate cache key
        key_data = {'name': dataset_name, 'metadata': metadata}
        key = self._generate_key(key_data)
        
        # Check if already cached
        if key in self.metadata['entries'] and self._check_ttl(key):
            logger.info(f"Dataset '{dataset_name}' already cached with key: {key}")
            return key
        
        # Get cache path
        cache_path = self._get_cache_path(key, "dataset")
        
        # Save data
        with self._lock:
            if self.enable_compression:
                with gzip.open(cache_path, 'wb') as f:
                    pickle.dump({'data': data, 'metadata': metadata}, f)
            else:
                with open(cache_path, 'wb') as f:
                    pickle.dump({'data': data, 'metadata': metadata}, f)
            
            # Update metadata
            file_size = cache_path.stat().st_size
            self.metadata['entries'][key] = {
                'type': 'dataset',
                'name': dataset_name,
                'path': str(cache_path),
                'size': file_size,
                'created': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'metadata': metadata
            }
            self.metadata['total_size'] += file_size
            
            # Ensure cache size limit
            self._ensure_cache_size_limit()
            self._save_metadata()
        
        logger.info(f"Cached dataset '{dataset_name}' ({file_size / 1024 / 1024:.2f} MB)")
        return key
    
    def get_dataset(self, dataset_name: str, metadata: Optional[Dict] = None) -> Optional[Any]:
        """
        Retrieve a cached dataset.
        
        Args:
            dataset_name: Name of the dataset
            metadata: Optional metadata to match
            
        Returns:
            Cached dataset or None if not found
        """
        # Generate cache key
        key_data = {'name': dataset_name, 'metadata': metadata}
        key = self._generate_key(key_data)
        
        # Check if cached and valid
        if key not in self.metadata['entries'] or not self._check_ttl(key):
            self.stats['misses'] += 1
            return None
        
        # Load from cache
        cache_path = Path(self.metadata['entries'][key]['path'])
        
        try:
            with self._lock:
                if self.enable_compression:
                    with gzip.open(cache_path, 'rb') as f:
                        cached_data = pickle.load(f)
                else:
                    with open(cache_path, 'rb') as f:
                        cached_data = pickle.load(f)
                
                # Update access time
                self.metadata['entries'][key]['last_accessed'] = datetime.now().isoformat()
                self._save_metadata()
            
            self.stats['hits'] += 1
            logger.info(f"Retrieved dataset '{dataset_name}' from cache")
            return cached_data['data']
            
        except Exception as e:
            logger.error(f"Failed to load cached dataset: {e}")
            self._remove_entry(key)
            return None
    
    def cache_explanation(self,
                         question: str,
                         solution: str,
                         explanation: str,
                         model: str,
                         temperature: float,
                         additional_data: Optional[Dict] = None) -> str:
        """
        Cache an explanation generated by a model.
        
        Args:
            question: The question
            solution: The solution
            explanation: Generated explanation
            model: Model that generated the explanation
            temperature: Temperature used for generation
            additional_data: Additional data to cache (e.g., usage stats)
            
        Returns:
            Cache key for the explanation
        """
        # Generate cache key based on inputs
        key_data = {
            'question': question,
            'solution': solution,
            'model': model,
            'temperature': temperature
        }
        key = self._generate_key(key_data)
        
        # Prepare cache data
        cache_data = {
            'question': question,
            'solution': solution,
            'explanation': explanation,
            'model': model,
            'temperature': temperature,
            'timestamp': datetime.now().isoformat(),
            'additional_data': additional_data
        }
        
        # Get cache path
        cache_path = self._get_cache_path(key, "explanation")
        
        # Save data
        with self._lock:
            if self.enable_compression:
                with gzip.open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
            else:
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
            
            # Update metadata
            file_size = cache_path.stat().st_size
            self.metadata['entries'][key] = {
                'type': 'explanation',
                'model': model,
                'path': str(cache_path),
                'size': file_size,
                'created': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat()
            }
            self.metadata['total_size'] += file_size
            
            # Track API calls saved
            if additional_data and 'usage' in additional_data:
                self.stats['api_calls_saved'] += 1
                self.stats['bytes_saved'] += file_size
            
            self._ensure_cache_size_limit()
            self._save_metadata()
        
        return key
    
    def get_explanation(self,
                       question: str,
                       solution: str,
                       model: str,
                       temperature: float) -> Optional[Dict]:
        """
        Retrieve a cached explanation.
        
        Args:
            question: The question
            solution: The solution
            model: Model that generated the explanation
            temperature: Temperature used for generation
            
        Returns:
            Cached explanation data or None if not found
        """
        # Generate cache key
        key_data = {
            'question': question,
            'solution': solution,
            'model': model,
            'temperature': temperature
        }
        key = self._generate_key(key_data)
        
        # Check if cached and valid
        if key not in self.metadata['entries'] or not self._check_ttl(key):
            self.stats['misses'] += 1
            return None
        
        # Load from cache
        cache_path = Path(self.metadata['entries'][key]['path'])
        
        try:
            with self._lock:
                if self.enable_compression:
                    with gzip.open(cache_path, 'rb') as f:
                        cached_data = pickle.load(f)
                else:
                    with open(cache_path, 'rb') as f:
                        cached_data = pickle.load(f)
                
                # Update access time
                self.metadata['entries'][key]['last_accessed'] = datetime.now().isoformat()
                self._save_metadata()
            
            self.stats['hits'] += 1
            self.stats['api_calls_saved'] += 1
            return cached_data
            
        except Exception as e:
            logger.error(f"Failed to load cached explanation: {e}")
            self._remove_entry(key)
            return None
    
    @contextmanager
    def batch_operation(self):
        """Context manager for batch operations to reduce metadata saves."""
        # Temporarily disable auto-saving
        original_save = self._save_metadata
        self._save_metadata = lambda: None
        
        try:
            yield
        finally:
            # Restore and save once
            self._save_metadata = original_save
            self._save_metadata()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics and usage information."""
        self._cleanup_old_entries()
        
        stats = {
            **self.stats,
            'total_entries': len(self.metadata['entries']),
            'total_size_mb': self.metadata['total_size'] / 1024 / 1024,
            'cache_limit_mb': self.max_cache_size_bytes / 1024 / 1024,
            'usage_percent': (self.metadata['total_size'] / self.max_cache_size_bytes) * 100,
            'hit_rate': self.stats['hits'] / max(1, self.stats['hits'] + self.stats['misses']),
            'datasets_cached': sum(1 for e in self.metadata['entries'].values() if e['type'] == 'dataset'),
            'explanations_cached': sum(1 for e in self.metadata['entries'].values() if e['type'] == 'explanation')
        }
        
        return stats
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """
        Clear cache entries.
        
        Args:
            cache_type: Optional type to clear ('dataset', 'explanation', or None for all)
        """
        with self._lock:
            if cache_type:
                # Clear specific type
                keys_to_remove = [
                    key for key, entry in self.metadata['entries'].items()
                    if entry['type'] == cache_type
                ]
            else:
                # Clear all
                keys_to_remove = list(self.metadata['entries'].keys())
            
            for key in keys_to_remove:
                self._remove_entry(key)
            
            self._save_metadata()
            
        logger.info(f"Cleared {len(keys_to_remove)} cache entries")
    
    def export_cache_summary(self, output_file: str):
        """Export a summary of cached items to a file."""
        summary = {
            'stats': self.get_cache_stats(),
            'entries': []
        }
        
        for key, entry in self.metadata['entries'].items():
            summary['entries'].append({
                'key': key,
                'type': entry['type'],
                'name': entry.get('name', 'N/A'),
                'model': entry.get('model', 'N/A'),
                'size_mb': entry['size'] / 1024 / 1024,
                'created': entry['created'],
                'last_accessed': entry.get('last_accessed', entry['created'])
            })
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Exported cache summary to {output_file}")


# Convenience functions
_default_cache = None

def get_default_cache() -> CacheManager:
    """Get or create the default cache manager instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = CacheManager()
    return _default_cache


def cache_dataset(dataset_name: str, data: Any, metadata: Optional[Dict] = None) -> str:
    """Cache a dataset using the default cache manager."""
    return get_default_cache().cache_dataset(dataset_name, data, metadata)


def get_cached_dataset(dataset_name: str, metadata: Optional[Dict] = None) -> Optional[Any]:
    """Get a cached dataset using the default cache manager."""
    return get_default_cache().get_dataset(dataset_name, metadata)


def cache_explanation(question: str, solution: str, explanation: str, 
                     model: str, temperature: float, **kwargs) -> str:
    """Cache an explanation using the default cache manager."""
    return get_default_cache().cache_explanation(
        question, solution, explanation, model, temperature, kwargs
    )


def get_cached_explanation(question: str, solution: str, 
                          model: str, temperature: float) -> Optional[Dict]:
    """Get a cached explanation using the default cache manager."""
    return get_default_cache().get_explanation(question, solution, model, temperature)