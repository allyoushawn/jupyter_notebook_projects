#!/usr/bin/env python3
"""
Dummy Data Generator for PyTorch Data Pipeline Learning

Generates synthetic parquet files in a partitioned directory structure:
data/ds=YYYYMMDD/h=HH/<uuid>.parquet

Each parquet file contains rows with columns:
- ds: int32 (date in YYYYMMDD format)
- h: int32 (hour 0-23)
- swiper_id: int64
- swipee_id: int64
- feat1-featN: float64 (configurable number of dense features)
- emb_1-emb_N: list of float32 (configurable number and dimension of embeddings)
- employer: string (sparse feature; nullable)
- school_name: string (sparse feature; nullable)
- interests: list<string> (var_len_sparse feature; can be empty list or null)
- skills: list<string> (var_len_sparse feature; can be empty list or null)
- label: int32 (binary label: 0 or 1, no null values)

Configuration can be provided via:
1. Command-line arguments
2. Configuration file (data_gen.cfg) via --config option
CLI arguments take precedence over config file values.
"""

import argparse
import configparser
import shutil
import uuid
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


# Default configuration values
DEFAULT_CONFIG = {
    'num_dense_features': 5,
    'embedding_dim': 32,
    'num_embeddings': 2,
    'start_date': '20260101',
    'end_date': '20260101',
    'rows_per_file': 500,
    'files_per_hour': 4,
    'null_probability': 0.0,
    'output_dir': 'data',
    'clean_existing': True,
}

# Vocabulary constants for sparse features
EMPLOYERS = [
    "Google", "Meta", "Amazon", "Apple", "Microsoft", "Netflix",
    "Uber", "Airbnb", "Stripe", "Spotify", "Twitter", "LinkedIn",
    "Tesla", "Oracle", "IBM", "Salesforce", "Adobe", "Intel",
    "Nvidia", "AMD", "Cisco", "VMware", "PayPal", "eBay",
    "Goldman Sachs", "JPMorgan", "Morgan Stanley", "Bank of America",
    "Wells Fargo", "Citigroup", "BlackRock", "Vanguard"
]

SCHOOLS = [
    "MIT", "Stanford", "Harvard", "Berkeley", "CMU", "Princeton",
    "Yale", "Columbia", "Cornell", "Penn", "Dartmouth", "Brown",
    "Caltech", "UCLA", "USC", "NYU", "Duke", "Northwestern",
    "Chicago", "Michigan", "Virginia", "UNC", "Georgia Tech",
    "UT Austin", "Washington", "Illinois", "Purdue", "Texas A&M"
]

INTERESTS = [
    "hiking", "reading", "gaming", "cooking", "photography",
    "traveling", "music", "fitness", "art", "coding",
    "movies", "sports", "dancing", "yoga", "cycling",
    "swimming", "running", "basketball", "soccer", "tennis",
    "chess", "puzzles", "gardening", "painting", "writing",
    "podcasts", "comedy", "theater", "concerts", "festivals"
]

SKILLS = [
    "python", "java", "sql", "machine_learning", "data_analysis",
    "communication", "leadership", "project_management", "javascript",
    "react", "nodejs", "aws", "docker", "kubernetes", "git",
    "agile", "scrum", "product_management", "design", "marketing",
    "sales", "finance", "accounting", "consulting", "research"
]


def load_config(config_path: str) -> dict:
    """
    Load configuration from a .cfg file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary with configuration values
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    
    return {
        # Data section
        'num_dense_features': config.getint('data', 'num_dense_features', fallback=DEFAULT_CONFIG['num_dense_features']),
        'embedding_dim': config.getint('data', 'embedding_dim', fallback=DEFAULT_CONFIG['embedding_dim']),
        'num_embeddings': config.getint('data', 'num_embeddings', fallback=DEFAULT_CONFIG['num_embeddings']),
        
        # Generation section
        'start_date': config.get('generation', 'start_date', fallback=DEFAULT_CONFIG['start_date']),
        'end_date': config.get('generation', 'end_date', fallback=DEFAULT_CONFIG['end_date']),
        'rows_per_file': config.getint('generation', 'rows_per_file', fallback=DEFAULT_CONFIG['rows_per_file']),
        'files_per_hour': config.getint('generation', 'files_per_hour', fallback=DEFAULT_CONFIG['files_per_hour']),
        'null_probability': config.getfloat('generation', 'null_probability', fallback=DEFAULT_CONFIG['null_probability']),
        'output_dir': config.get('generation', 'output_dir', fallback=DEFAULT_CONFIG['output_dir']),
        
        # Options section
        'clean_existing': config.getboolean('options', 'clean_existing', fallback=DEFAULT_CONFIG['clean_existing']),
    }


def clean_partitions(output_dir: str, start_date: str, end_date: str) -> None:
    """
    Remove existing partition directories for the specified date range.
    
    Only removes data/ds=YYYYMMDD/ directories within the date range,
    preserving data from other dates.
    
    Args:
        output_dir: Base output directory (e.g., 'data')
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
    """
    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date)
    
    output_path = Path(output_dir)
    if not output_path.exists():
        return
    
    current_date = start_dt
    removed_count = 0
    while current_date <= end_dt:
        date_str = current_date.strftime('%Y%m%d')
        date_partition = output_path / f"ds={date_str}"
        
        if date_partition.exists():
            print(f"Removing existing partition: {date_partition}")
            shutil.rmtree(date_partition)
            removed_count += 1
        
        current_date += timedelta(days=1)
    
    if removed_count > 0:
        print(f"Removed {removed_count} existing partition(s)\n")


def generate_dataframe(
    num_rows: int,
    ds: int,
    h: int,
    num_dense_features: int = 5,
    embedding_dim: int = 32,
    num_embeddings: int = 2,
    null_probability: float = 0.0
) -> pd.DataFrame:
    """
    Generate a DataFrame with synthetic data.
    
    Args:
        num_rows: Number of rows to generate
        ds: Date in YYYYMMDD format (e.g., 20260101)
        h: Hour (0-23)
        num_dense_features: Number of dense features to generate (feat1, feat2, ..., featN)
        embedding_dim: Dimension of each embedding vector
        num_embeddings: Number of embedding columns to generate (emb_1, emb_2, ..., emb_N)
        null_probability: Probability (0.0 to 1.0) that each feature value will be null
        
    Returns:
        DataFrame with columns: ds, h, swiper_id, swipee_id, feat1-featN, emb_1-emb_N,
        employer, school_name, interests, skills, label
    """
    # Generate sparse features (employer, school_name)
    employer_list = [random.choice(EMPLOYERS + [None]) for _ in range(num_rows)]
    school_list = [random.choice(SCHOOLS + [None]) for _ in range(num_rows)]
    
    # Generate var_len_sparse features (interests, skills)
    # interests: 0-15 items per row
    interests_list = [
        random.sample(INTERESTS, k=random.randint(0, 15)) if random.random() > 0.1 else []
        for _ in range(num_rows)
    ]
    # skills: 0-12 items per row
    skills_list = [
        random.sample(SKILLS, k=random.randint(0, 12)) if random.random() > 0.1 else []
        for _ in range(num_rows)
    ]
    
    data = {
        'ds': np.full(num_rows, ds, dtype=np.int32),
        'h': np.full(num_rows, h, dtype=np.int32),
        'swiper_id': np.random.randint(1, 1000000, size=num_rows, dtype=np.int64),
        'swipee_id': np.random.randint(1, 1000000, size=num_rows, dtype=np.int64),
    }
    
    # Generate dynamic dense features (feat1, feat2, ..., featN)
    for i in range(1, num_dense_features + 1):
        data[f'feat{i}'] = np.random.randn(num_rows).astype(np.float64)
    
    # Generate dynamic embeddings (emb_1, emb_2, ..., emb_N)
    for i in range(1, num_embeddings + 1):
        emb_arr = np.random.randn(num_rows, embedding_dim).astype(np.float32)
        data[f'emb_{i}'] = [emb_arr[j].tolist() for j in range(num_rows)]
    
    # Add sparse and var_len_sparse features
    data['employer'] = employer_list
    data['school_name'] = school_list
    data['interests'] = interests_list
    data['skills'] = skills_list
    data['label'] = np.random.randint(0, 2, size=num_rows, dtype=np.int32)
    
    df = pd.DataFrame(data)
    
    # Randomly set null values in feature columns if null_probability > 0
    if null_probability > 0.0:
        # For dense features (feat1, feat2, ..., featN): set to NaN
        for i in range(1, num_dense_features + 1):
            null_mask = np.random.random(num_rows) < null_probability
            df.loc[null_mask, f'feat{i}'] = np.nan
        # For embeddings (emb_1, emb_2, ..., emb_N): set entire vector to null
        for i in range(1, num_embeddings + 1):
            emb_null_mask = np.random.random(num_rows) < null_probability
            df.loc[emb_null_mask, f'emb_{i}'] = None
        # For sparse features (employer, school_name): set to None with same probability
        for sparse_col in ['employer', 'school_name']:
            sparse_null_mask = np.random.random(num_rows) < null_probability
            df.loc[sparse_null_mask, sparse_col] = None
        # For var_len_sparse features (interests, skills): set to empty list or None with same probability
        for varlen_col in ['interests', 'skills']:
            varlen_null_mask = np.random.random(num_rows) < null_probability
            df.loc[varlen_null_mask, varlen_col] = None
    
    return df


def parse_date(date_str: str) -> datetime:
    """
    Parse date string in YYYYMMDD format.
    
    Args:
        date_str: Date string in YYYYMMDD format
        
    Returns:
        datetime object
        
    Raises:
        ValueError: If date string is not in correct format
    """
    try:
        return datetime.strptime(date_str, '%Y%m%d')
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYYMMDD format.")


def generate_data(
    start_date: str,
    end_date: str,
    rows_per_file: int = 500,
    output_dir: str = 'data',
    files_per_hour: int = 4,
    null_probability: float = 0.0,
    num_dense_features: int = 5,
    embedding_dim: int = 32,
    num_embeddings: int = 2,
    clean_existing: bool = True
):
    """
    Generate parquet files for the specified date range.
    
    Args:
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        rows_per_file: Number of rows per parquet file
        output_dir: Output directory path
        files_per_hour: Number of parquet files to generate per hour
        null_probability: Probability (0.0 to 1.0) that each feature value will be null
        num_dense_features: Number of dense features to generate (feat1, feat2, ..., featN)
        embedding_dim: Dimension of each embedding vector
        num_embeddings: Number of embedding columns to generate (emb_1, emb_2, ..., emb_N)
        clean_existing: If True, remove existing partition directories before generating
    """
    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date)
    
    if start_dt > end_dt:
        raise ValueError(f"Start date {start_date} must be <= end date {end_date}")
    
    # Clean existing partitions if requested
    if clean_existing:
        clean_partitions(output_dir, start_date, end_date)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_files = 0
    current_date = start_dt
    
    print(f"Generating data from {start_date} to {end_date}")
    print(f"Rows per file: {rows_per_file}")
    print(f"Files per hour: {files_per_hour}")
    print(f"Null probability: {null_probability}")
    print(f"Dense features: {num_dense_features} (feat1-feat{num_dense_features})")
    print(f"Embeddings: {num_embeddings} x {embedding_dim}d (emb_1-emb_{num_embeddings})")
    print(f"Output directory: {output_dir}\n")
    
    # Iterate through each date
    while current_date <= end_dt:
        date_str = current_date.strftime('%Y%m%d')
        date_dir = output_path / f"ds={date_str}"
        
        # Iterate through each hour (00-23)
        for hour in range(24):
            hour_str = f"{hour:02d}"
            hour_dir = date_dir / f"h={hour_str}"
            hour_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate files_per_hour parquet files for this hour
            date_int = int(date_str)  # Convert YYYYMMDD string to integer
            for file_idx in range(files_per_hour):
                # Generate UUID for filename
                file_uuid = uuid.uuid4()
                file_path = hour_dir / f"{file_uuid}.parquet"
                
                # Generate and save DataFrame with ds and h columns
                df = generate_dataframe(
                    rows_per_file,
                    ds=date_int,
                    h=hour,
                    num_dense_features=num_dense_features,
                    embedding_dim=embedding_dim,
                    num_embeddings=num_embeddings,
                    null_probability=null_probability
                )
                df.to_parquet(file_path, index=False)
                total_files += 1
                
                if total_files % 100 == 0:
                    print(f"Generated {total_files} files...")
        
        current_date += timedelta(days=1)
    
    print(f"\n✅ Successfully generated {total_files} parquet files")
    print(f"   Date range: {start_date} to {end_date}")
    print(f"   Total dates: {(end_dt - start_dt).days + 1}")
    print(f"   Files per hour: {files_per_hour}")
    print(f"   Rows per file: {rows_per_file}")
    print(f"   Dense features: {num_dense_features}")
    print(f"   Embeddings: {num_embeddings} x {embedding_dim}d")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Generate dummy parquet files for data pipeline learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using command-line arguments
  python dummy_data_gen.py --start-date 20260101 --end-date 20260107
  python dummy_data_gen.py --start-date 20260101 --end-date 20260101 --rows-per-file 1000
  python dummy_data_gen.py --start-date 20260101 --end-date 20260107 --output-dir custom_data
  python dummy_data_gen.py --start-date 20260101 --end-date 20260107 --null-probability 0.1
  python dummy_data_gen.py --num-dense-features 10 --num-embeddings 3 --embedding-dim 64
  
  # Using config file (CLI args override config values)
  python dummy_data_gen.py --config data_gen.cfg
  python dummy_data_gen.py --config data_gen.cfg --start-date 20260115
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (data_gen.cfg). CLI args override config values.'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date in YYYYMMDD format (default: 20260101)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date in YYYYMMDD format (default: 20260101)'
    )
    
    parser.add_argument(
        '--rows-per-file',
        type=int,
        default=None,
        help='Number of rows per parquet file (default: 500)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory path (default: data/)'
    )
    
    parser.add_argument(
        '--files-per-hour',
        type=int,
        default=None,
        help='Number of parquet files to generate per hour (default: 4)'
    )
    
    parser.add_argument(
        '--null-probability',
        type=float,
        default=None,
        help='Probability (0.0 to 1.0) that each feature value will be null (default: 0.0)'
    )
    
    parser.add_argument(
        '--num-dense-features',
        type=int,
        default=None,
        help='Number of dense features to generate: feat1, feat2, ..., featN (default: 5)'
    )
    
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=None,
        help='Dimension of each embedding vector (default: 32)'
    )
    
    parser.add_argument(
        '--num-embeddings',
        type=int,
        default=None,
        help='Number of embedding columns: emb_1, emb_2, ..., emb_N (default: 2)'
    )
    
    parser.add_argument(
        '--no-clean',
        action='store_true',
        help='Do not remove existing partition directories before generating'
    )
    
    args = parser.parse_args()
    
    # Start with default config
    cfg = DEFAULT_CONFIG.copy()
    
    # Load config file if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Error: Config file not found: {args.config}")
            return 1
        print(f"Loading configuration from: {args.config}")
        file_cfg = load_config(args.config)
        cfg.update(file_cfg)
    
    # Override with CLI arguments if provided (CLI takes precedence)
    if args.start_date is not None:
        cfg['start_date'] = args.start_date
    if args.end_date is not None:
        cfg['end_date'] = args.end_date
    if args.rows_per_file is not None:
        cfg['rows_per_file'] = args.rows_per_file
    if args.output_dir is not None:
        cfg['output_dir'] = args.output_dir
    if args.files_per_hour is not None:
        cfg['files_per_hour'] = args.files_per_hour
    if args.null_probability is not None:
        cfg['null_probability'] = args.null_probability
    if args.num_dense_features is not None:
        cfg['num_dense_features'] = args.num_dense_features
    if args.embedding_dim is not None:
        cfg['embedding_dim'] = args.embedding_dim
    if args.num_embeddings is not None:
        cfg['num_embeddings'] = args.num_embeddings
    if args.no_clean:
        cfg['clean_existing'] = False
    
    # Validate null_probability
    if cfg['null_probability'] < 0.0 or cfg['null_probability'] > 1.0:
        print(f"❌ Error: null-probability must be between 0.0 and 1.0, got {cfg['null_probability']}")
        return 1
    
    # Validate positive integers
    if cfg['num_dense_features'] < 1:
        print(f"❌ Error: num-dense-features must be >= 1, got {cfg['num_dense_features']}")
        return 1
    if cfg['embedding_dim'] < 1:
        print(f"❌ Error: embedding-dim must be >= 1, got {cfg['embedding_dim']}")
        return 1
    if cfg['num_embeddings'] < 1:
        print(f"❌ Error: num-embeddings must be >= 1, got {cfg['num_embeddings']}")
        return 1
    
    try:
        generate_data(
            start_date=cfg['start_date'],
            end_date=cfg['end_date'],
            rows_per_file=cfg['rows_per_file'],
            output_dir=cfg['output_dir'],
            files_per_hour=cfg['files_per_hour'],
            null_probability=cfg['null_probability'],
            num_dense_features=cfg['num_dense_features'],
            embedding_dim=cfg['embedding_dim'],
            num_embeddings=cfg['num_embeddings'],
            clean_existing=cfg['clean_existing']
        )
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
