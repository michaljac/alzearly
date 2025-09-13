"""
Parallel and Distributed Processing Module

This module provides utilities for parallel and distributed data processing
using Dask, Ray, and joblib. It supports both data processing and model training.
"""

import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import warnings

# Dask disabled due to IPython conflicts
DASK_AVAILABLE = False
print("INFO: Dask disabled - using alternative parallel processing methods.")

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    warnings.warn("Ray not available. Install with: pip install ray")

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    warnings.warn("Joblib not available. Install with: pip install joblib")

logger = logging.getLogger(__name__)

class ParallelProcessor:
    """Handles parallel and distributed data processing."""
    
    def __init__(self, backend="auto", n_jobs=None, memory_limit="2GB", dask_cluster=None):
        """
        Initialize parallel processor.
        
        Args:
            backend: Processing backend ("dask", "ray", "joblib", "auto")
            n_jobs: Number of parallel jobs (None for auto-detection)
            memory_limit: Memory limit for Dask workers
            dask_cluster: Pre-configured Dask cluster
        """
        self.backend = backend
        self.n_jobs = n_jobs or min(mp.cpu_count(), 8)  # Cap at 8 by default
        self.memory_limit = memory_limit
        self.dask_cluster = dask_cluster
        
        # Auto-select backend
        if backend == "auto":
            if DASK_AVAILABLE and self.n_jobs > 4:
                self.backend = "dask"
            elif RAY_AVAILABLE:
                self.backend = "ray"
            elif JOBLIB_AVAILABLE:
                self.backend = "joblib"
            else:
                self.backend = "multiprocessing"
        
        self._setup_backend()
    
    def _setup_backend(self):
        """Setup the selected backend."""
        if self.backend == "dask" and DASK_AVAILABLE:
            if self.dask_cluster is None:
                cluster = LocalCluster(
                    n_workers=self.n_jobs,
                    memory_limit=self.memory_limit,
                    threads_per_worker=2
                )
                self.client = Client(cluster)
                # Started Dask cluster
            else:
                self.client = self.dask_cluster
                # Using provided Dask cluster
        
        elif self.backend == "ray" and RAY_AVAILABLE:
            if not ray.is_initialized():
                ray.init(num_cpus=self.n_jobs)
                # Initialized Ray
            else:
                # Using existing Ray cluster
                pass
        
        elif self.backend == "joblib" and JOBLIB_AVAILABLE:
            # Using Joblib
            pass
        else:
            # Using multiprocessing
            pass
    
    def process_dataframe_parallel(self, df, func, chunk_size=None, **kwargs):
        """
        Process DataFrame in parallel using the selected backend.
        
        Args:
            df: Input DataFrame
            func: Function to apply to each chunk
            chunk_size: Size of chunks to process
            **kwargs: Additional arguments for func
        
        Returns:
            Processed DataFrame
        """
        if chunk_size is None:
            chunk_size = max(1, len(df) // (self.n_jobs * 4))
        
        if self.backend == "dask" and DASK_AVAILABLE:
            return self._process_with_dask(df, func, chunk_size, **kwargs)
        elif self.backend == "ray" and RAY_AVAILABLE:
            return self._process_with_ray(df, func, chunk_size, **kwargs)
        elif self.backend == "joblib" and JOBLIB_AVAILABLE:
            return self._process_with_joblib(df, func, chunk_size, **kwargs)
        else:
            return self._process_with_multiprocessing(df, func, chunk_size, **kwargs)
    
    def _process_with_dask(self, df, func, chunk_size, **kwargs):
        """Process DataFrame using Dask."""
        # Convert to Dask DataFrame
        ddf = dd.from_pandas(df, npartitions=self.n_jobs)
        
        # Apply function
        result_ddf = ddf.map_partitions(func, **kwargs)
        
        # Compute result
        result = result_ddf.compute()
        return result
    
    def _process_with_ray(self, df, func, chunk_size, **kwargs):
        """Process DataFrame using Ray."""
        # Split DataFrame into chunks
        chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
        
        # Define remote function
        @ray.remote
        def process_chunk(chunk, func, **kwargs):
            return func(chunk, **kwargs)
        
        # Process chunks in parallel
        from tqdm import tqdm
        futures = [process_chunk.remote(chunk, func, **kwargs) for chunk in chunks]
        
        # Show progress while waiting for results
        results = []
        for future in tqdm(ray.as_completed(futures), 
                          total=len(futures), 
                          desc="Processing chunks", 
                          unit="chunk",
                          bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                          ncols=80, ascii=True, position=0, dynamic_ncols=False, 
                          mininterval=0.1, maxinterval=1.0):
            results.append(ray.get(future))
        
        # Combine results
        return pd.concat(results, ignore_index=True)
    
    def _process_with_joblib(self, df, func, chunk_size, **kwargs):
        """Process DataFrame using Joblib."""
        # Split DataFrame into chunks
        chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
        
        # Process chunks in parallel
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(func)(chunk, **kwargs) for chunk in chunks
        )
        
        # Combine results
        return pd.concat(results, ignore_index=True)
    
    def _process_with_multiprocessing(self, df, func, chunk_size, **kwargs):
        """Process DataFrame using multiprocessing."""
        # Split DataFrame into chunks
        chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
        
        # Process chunks in parallel
        from tqdm import tqdm
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(func, chunk, **kwargs) for chunk in chunks]
            results = []
            for future in tqdm(futures, 
                             total=len(futures), 
                             desc="Processing chunks", 
                             unit="chunk",
                             bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                             ncols=80, ascii=True, position=0, dynamic_ncols=False, 
                             mininterval=0.1, maxinterval=1.0):
                results.append(future.result())
        
        # Combine results
        return pd.concat(results, ignore_index=True)
    
    def train_models_parallel(self, X, y, models, cv_folds=5):
        """
        Train multiple models in parallel.
        
        Args:
            X: Feature matrix
            y: Target vector
            models: List of model instances
            cv_folds: Number of cross-validation folds
        
        Returns:
            Dictionary with trained models and scores
        """
        if self.backend == "ray" and RAY_AVAILABLE:
            return self._train_with_ray(X, y, models, cv_folds)
        elif self.backend == "joblib" and JOBLIB_AVAILABLE:
            return self._train_with_joblib(X, y, models, cv_folds)
        else:
            return self._train_with_multiprocessing(X, y, models, cv_folds)
    
    def _train_with_ray(self, X, y, models, cv_folds):
        """Train models using Ray."""
        from sklearn.model_selection import cross_val_score
        
        @ray.remote
        def train_single_model(model, X, y, cv_folds):
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
            model.fit(X, y)  # Fit on full dataset
            return {
                'model': model,
                'cv_scores': scores,
                'cv_mean': scores.mean(),
                'cv_std': scores.std()
            }
        
        # Train models in parallel
        from tqdm import tqdm
        futures = [train_single_model.remote(model, X, y, cv_folds) for model in models]
        
        # Show progress while waiting for results
        results = []
        for future in tqdm(ray.as_completed(futures), 
                          total=len(futures), 
                          desc="Training models", 
                          unit="model",
                          bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                          ncols=80, ascii=True, position=0, dynamic_ncols=False, 
                          mininterval=0.1, maxinterval=1.0):
            results.append(ray.get(future))
        
        return {f"model_{i}": result for i, result in enumerate(results)}
    
    def _train_with_joblib(self, X, y, models, cv_folds):
        """Train models using Joblib."""
        from sklearn.model_selection import cross_val_score
        
        def train_single_model(model, X, y, cv_folds):
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
            model.fit(X, y)  # Fit on full dataset
            return {
                'model': model,
                'cv_scores': scores,
                'cv_mean': scores.mean(),
                'cv_std': scores.std()
            }
        
        # Train models in parallel
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(train_single_model)(model, X, y, cv_folds) for model in models
        )
        
        return {f"model_{i}": result for i, result in enumerate(results)}
    
    def _train_with_multiprocessing(self, X, y, models, cv_folds):
        """Train models using multiprocessing."""
        from sklearn.model_selection import cross_val_score
        
        def train_single_model(model, X, y, cv_folds):
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
            model.fit(X, y)  # Fit on full dataset
            return {
                'model': model,
                'cv_scores': scores,
                'cv_mean': scores.mean(),
                'cv_std': scores.std()
            }
        
        # Train models in parallel
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(train_single_model, model, X, y, cv_folds) for model in models]
            results = [future.result() for future in futures]
        
        return {f"model_{i}": result for i, result in enumerate(results)}
    
    def cleanup(self):
        """Cleanup resources."""
        if self.backend == "dask" and DASK_AVAILABLE and hasattr(self, 'client'):
            self.client.close()
            # Closed Dask client
        
        elif self.backend == "ray" and RAY_AVAILABLE and ray.is_initialized():
            ray.shutdown()
            # Shutdown Ray

# Convenience functions
def parallel_process_dataframe(df, func, backend="auto", n_jobs=None, **kwargs):
    """Convenience function for parallel DataFrame processing."""
    processor = ParallelProcessor(backend=backend, n_jobs=n_jobs)
    try:
        return processor.process_dataframe_parallel(df, func, **kwargs)
    finally:
        processor.cleanup()

def parallel_train_models(X, y, models, backend="auto", n_jobs=None, cv_folds=5):
    """Convenience function for parallel model training."""
    processor = ParallelProcessor(backend=backend, n_jobs=n_jobs)
    try:
        return processor.train_models_parallel(X, y, models, cv_folds)
    finally:
        processor.cleanup()

