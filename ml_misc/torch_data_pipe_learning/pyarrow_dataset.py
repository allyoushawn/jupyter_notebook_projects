"""
PyArrow Dataset Wrapper for PyTorch

A PyTorch IterableDataset that wraps PyArrow's modern dataset API,
providing efficient reading of Hive-partitioned parquet files with
support for filtering, shuffling, and multi-worker DataLoader.
"""

import pyarrow.dataset as ds
import pyarrow as pa
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
        
        # Convert to Python list for use with both tensors and lists
        indices_list = indices.tolist()
        
        result = {}
        for key, value in batch_dict.items():
            if isinstance(value, list):
                # For Python lists (e.g., var_len_sparse string features), use list indexing
                result[key] = [value[i] for i in indices_list]
            else:
                # For tensors, use tensor indexing
                result[key] = value[indices]
        
        return result
    
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
                    col_type = batch.schema.field(col).type
                    # Handle list columns (e.g., embeddings stored as list<float32>, or list<string>)
                    if pa.types.is_list(col_type):
                        list_arr = batch[col]
                        value_type = col_type.value_type
                        
                        # Check if it's a list of strings (for var_len_sparse features)
                        if pa.types.is_string(value_type) or pa.types.is_large_string(value_type):
                            # For list<string>, convert to Python list of lists
                            # This preserves variable lengths for var_len_sparse features
                            list_of_lists = list_arr.to_pylist()
                            # Convert None to empty list for consistency
                            list_of_lists = [lst if lst is not None else [] for lst in list_of_lists]
                            batch_dict[col] = np.array(list_of_lists, dtype=object)
                        else:
                            # For numeric list columns (e.g., embeddings stored as list<float32>)
                            # Use values+offsets to build (N, emb_dim) array.
                            # This avoids .as_py() (DoubleScalar/float()) and ensures uniform shapes
                            # for null/empty/variable-length lists.
                            n = len(list_arr)
                            # Convert offsets and values to numpy for efficient indexing
                            offsets = list_arr.offsets.to_numpy()
                            values = list_arr.values.to_numpy(zero_copy_only=False)
                            # Get null mask as numpy array for efficient checking
                            null_mask = list_arr.is_null().to_numpy(zero_copy_only=False)
                            # emb_dim from first non-null row
                            emb_dim = None
                            for i in range(n):
                                if not null_mask[i]:
                                    emb_dim = offsets[i + 1] - offsets[i]
                                    break
                            if emb_dim is None:
                                arr = np.empty((n, 0), dtype=np.float32)
                            else:
                                arr = np.full((n, emb_dim), np.nan, dtype=np.float32)
                                for i in range(n):
                                    if null_mask[i]:
                                        continue  # row already NaN
                                    start = offsets[i]
                                    stop = offsets[i + 1]
                                    row_np = values[start:stop].astype(np.float32)
                                    ncopy = min(len(row_np), emb_dim)
                                    arr[i, :ncopy] = row_np[:ncopy]
                            batch_dict[col] = arr
                    else:
                        arr = batch[col].to_numpy(zero_copy_only=False)  # Handles nulls
                        batch_dict[col] = arr
                
                # Convert to torch tensors - use copy() to ensure arrays are writable
                # Handle object dtype arrays (list<string>) specially - keep as Python lists
                torch_batch = {}
                for key, value in batch_dict.items():
                    if hasattr(value, 'dtype') and value.dtype == object:
                        # For object dtype (list<string>), convert to Python list
                        # transform_batch expects lists for VAR_LEN_SPARSE features
                        torch_batch[key] = value.tolist()
                    else:
                        torch_batch[key] = torch.as_tensor(value.copy())
                
                # Shuffle rows if requested
                if self.shuffle_rows:
                    torch_batch = self._shuffle_batch(torch_batch)
                
                yield torch_batch
