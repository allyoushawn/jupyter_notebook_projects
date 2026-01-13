"""
PyArrow Dataset Wrapper for PyTorch

A PyTorch IterableDataset that wraps PyArrow's modern dataset API,
providing efficient reading of Hive-partitioned parquet files with
support for filtering, shuffling, and multi-worker DataLoader.
"""

import pyarrow.dataset as ds
import torch
from torch.utils.data import IterableDataset
import numpy as np
import random


class PyArrowParquetDataset(IterableDataset):
    """
    PyTorch IterableDataset wrapper around PyArrow's modern dataset API.
    Handles Hive-partitioned parquet files efficiently, supporting filtering,
    shuffling, and multi-worker DataLoader.
    
    Note: ds and h columns should be stored directly in parquet files for best results.
    """
    
    def __init__(self, path, batch_size=1024, filters=None, 
                 shuffle_row_groups=False, shuffle_rows=False, seed=None):
        """
        Args:
            path: Path to parquet directory (Hive-partitioned)
            batch_size: Number of rows per batch
            filters: PyArrow filter expression (e.g., ds.field("ds") >= 20260101)
            shuffle_row_groups: Whether to shuffle the order of parquet fragments
            shuffle_rows: Whether to shuffle rows within each batch
            seed: Random seed for reproducibility
        """
        self.path = path
        self.batch_size = batch_size
        self.filters = filters
        self.shuffle_row_groups = shuffle_row_groups
        self.shuffle_rows = shuffle_rows
        self.seed = seed
        
        # Initialize PyArrow dataset with Hive partitioning
        self.dataset = ds.dataset(
            str(path), 
            format="parquet", 
            partitioning="hive"
        )
        
        # Store schema for reference
        self.schema = self.dataset.schema
        
    def _get_fragments(self):
        """Get parquet fragments, optionally shuffled."""
        fragments = list(self.dataset.get_fragments(filter=self.filters))
        
        if self.shuffle_row_groups:
            rng = random.Random(self.seed) if self.seed is not None else random
            fragments = list(fragments)
            rng.shuffle(fragments)
        
        return fragments
    
    def _get_fragments_for_worker(self, worker_info):
        """Split fragments across workers for multi-worker DataLoader."""
        fragments = self._get_fragments()
        num_workers = worker_info.num_workers
        worker_id = worker_info.id
        
        # Distribute fragments across workers
        worker_fragments = [
            frag for idx, frag in enumerate(fragments)
            if idx % num_workers == worker_id
        ]
        
        return worker_fragments
    
    def _shuffle_batch(self, batch_dict):
        """Shuffle rows within a batch."""
        if not self.shuffle_rows:
            return batch_dict
        
        batch_size = len(next(iter(batch_dict.values())))
        
        # Create indices for shuffling
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            indices = torch.randperm(batch_size, generator=generator)
        else:
            indices = torch.randperm(batch_size)
        
        return {key: value[indices] for key, value in batch_dict.items()}
    
    def __iter__(self):
        """Iterate over batches of data."""
        # Handle multi-worker DataLoader
        worker_info = torch.utils.data.get_worker_info()
        
        # Get fragments (with optional shuffling and worker sharding)
        if worker_info is not None:
            fragments = self._get_fragments_for_worker(worker_info)
        else:
            fragments = self._get_fragments()
        
        # Set random seed for this worker if provided
        if self.seed is not None:
            worker_seed = self.seed
            if worker_info is not None:
                worker_seed = self.seed + worker_info.id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
        
        # Simplified: fragment.to_table() includes all columns (ds/h stored in files)
        for fragment in fragments:
            # Read fragment data
            table = fragment.to_table(filter=self.filters)
            
            # Convert to batches
            for batch in table.to_batches(max_chunksize=self.batch_size):
                # Convert Arrow batch to dict of numpy arrays
                batch_dict = {}
                for col in batch.schema.names:
                    arr = batch[col].to_numpy(zero_copy_only=False)  # Handles nulls
                    batch_dict[col] = arr
                
                # Convert to torch tensors
                torch_batch = {
                    key: torch.as_tensor(value) 
                    for key, value in batch_dict.items()
                }
                
                # Shuffle rows if requested
                if self.shuffle_rows:
                    torch_batch = self._shuffle_batch(torch_batch)
                
                yield torch_batch
