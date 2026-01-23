from enum import Enum
from dataclasses import dataclass
from typing import Optional, List

class FeatureType(Enum):
    DENSE = "dense"           # Scalar float features
    EMBEDDING = "embedding"   # Fixed-dim vector features
    ID = "id"                 # Integer ID features
    PARTITION = "partition"   # Partition columns (ds, h)
    LABEL = "label"           # Binary or multi-class labels
    SPARSE = "sparse"                    # String -> hashed bucket index
    VAR_LEN_SPARSE = "var_len_sparse"   # List[String] -> padded indices

@dataclass
class FeatureConfig:
    name: str
    type: FeatureType
    dim: Optional[int] = None              # Required for EMBEDDING type
    bucket_edges: Optional[List[float]] = None  # Optional bucket edges for DENSE type
    num_buckets: Optional[int] = None   # For SPARSE/VAR_LEN_SPARSE
    max_len: Optional[int] = None       # For VAR_LEN_SPARSE
    combiner: Optional[str] = None      # For VAR_LEN_SPARSE: 'mean', 'sum', 'max'

FEATURE_CONFIGS = {
    # Dense features (scalar floats)
    # feat1 with bucketization: values map to buckets [<0.1, 0.1-0.2, 0.2-0.3, 0.3-0.4, >=0.4]
    "feat1": FeatureConfig("feat1", FeatureType.DENSE, bucket_edges=[0.1, 0.2, 0.3, 0.4]),
    "feat2": FeatureConfig("feat2", FeatureType.DENSE),
    "feat3": FeatureConfig("feat3", FeatureType.DENSE),
    "feat4": FeatureConfig("feat4", FeatureType.DENSE),
    "feat5": FeatureConfig("feat5", FeatureType.DENSE),
    
    # Embedding features
    "emb_1": FeatureConfig("emb_1", FeatureType.EMBEDDING, dim=32),
    "emb_2": FeatureConfig("emb_2", FeatureType.EMBEDDING, dim=32),
    
    # ID features
    "swiper_id": FeatureConfig("swiper_id", FeatureType.ID),
    "swipee_id": FeatureConfig("swipee_id", FeatureType.ID),
    
    # Partition features
    "ds": FeatureConfig("ds", FeatureType.PARTITION),
    "h": FeatureConfig("h", FeatureType.PARTITION),
    
    # Label
    "label": FeatureConfig("label", FeatureType.LABEL),
    
    # Sparse features
    "employer": FeatureConfig("employer", FeatureType.SPARSE, num_buckets=1000),
    "school_name": FeatureConfig("school_name", FeatureType.SPARSE, num_buckets=1000),
    
    # VarLen sparse features
    "interests": FeatureConfig("interests", FeatureType.VAR_LEN_SPARSE, 
                               num_buckets=500, max_len=10, combiner="mean"),
    "skills": FeatureConfig("skills", FeatureType.VAR_LEN_SPARSE,
                            num_buckets=500, max_len=10, combiner="mean"),
}
