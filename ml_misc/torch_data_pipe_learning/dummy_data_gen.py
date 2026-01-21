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
- feat1-feat5: float64
- emb_1: list of 32 float32 (embedding; null_probability sets entire vector to null)
- emb_2: list of 32 float32 (embedding; null_probability sets entire vector to null)
- label: int32 (binary label: 0 or 1, no null values)
"""

EMB_DIM = 32

import argparse
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def generate_dataframe(num_rows: int, ds: int, h: int, null_probability: float = 0.0) -> pd.DataFrame:
    """
    Generate a DataFrame with synthetic data.
    
    Args:
        num_rows: Number of rows to generate
        ds: Date in YYYYMMDD format (e.g., 20260101)
        h: Hour (0-23)
        null_probability: Probability (0.0 to 1.0) that each feature value will be null
        
    Returns:
        DataFrame with columns: ds, h, swiper_id, swipee_id, feat1-feat5, emb_1, emb_2, label
    """
    # Embedding: (num_rows, EMB_DIM) float32, stored as list of lists for Parquet
    emb_1_arr = np.random.randn(num_rows, EMB_DIM).astype(np.float32)
    emb_1_list = [emb_1_arr[i].tolist() for i in range(num_rows)]
    emb_2_arr = np.random.randn(num_rows, EMB_DIM).astype(np.float32)
    emb_2_list = [emb_2_arr[i].tolist() for i in range(num_rows)]

    data = {
        'ds': np.full(num_rows, ds, dtype=np.int32),
        'h': np.full(num_rows, h, dtype=np.int32),
        'swiper_id': np.random.randint(1, 1000000, size=num_rows, dtype=np.int64),
        'swipee_id': np.random.randint(1, 1000000, size=num_rows, dtype=np.int64),
        'feat1': np.random.randn(num_rows).astype(np.float64),
        'feat2': np.random.randn(num_rows).astype(np.float64),
        'feat3': np.random.randn(num_rows).astype(np.float64),
        'feat4': np.random.randn(num_rows).astype(np.float64),
        'feat5': np.random.randn(num_rows).astype(np.float64),
        'emb_1': emb_1_list,
        'emb_2': emb_2_list,
        'label': np.random.randint(0, 2, size=num_rows, dtype=np.int32),
    }
    
    df = pd.DataFrame(data)
    
    # Randomly set null values in feature columns if null_probability > 0
    if null_probability > 0.0:
        feature_cols = ['feat1', 'feat2', 'feat3', 'feat4', 'feat5']
        for col in feature_cols:
            # Create a mask for null values based on probability
            null_mask = np.random.random(num_rows) < null_probability
            df.loc[null_mask, col] = np.nan
        # For emb_1, emb_2: set entire vector to null with same probability
        for emb_col in ['emb_1', 'emb_2']:
            emb_null_mask = np.random.random(num_rows) < null_probability
            df.loc[emb_null_mask, emb_col] = None
    
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
    null_probability: float = 0.0
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
    """
    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date)
    
    if start_dt > end_dt:
        raise ValueError(f"Start date {start_date} must be <= end date {end_date}")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_files = 0
    current_date = start_dt
    
    print(f"Generating data from {start_date} to {end_date}")
    print(f"Rows per file: {rows_per_file}")
    print(f"Files per hour: {files_per_hour}")
    print(f"Null probability: {null_probability}")
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
                df = generate_dataframe(rows_per_file, ds=date_int, h=hour, null_probability=null_probability)
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


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Generate dummy parquet files for data pipeline learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dummy_data_gen.py --start-date 20260101 --end-date 20260107
  python dummy_data_gen.py --start-date 20260101 --end-date 20260101 --rows-per-file 1000
  python dummy_data_gen.py --start-date 20260101 --end-date 20260107 --output-dir custom_data
  python dummy_data_gen.py --start-date 20260101 --end-date 20260107 --null-probability 0.1
        """
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='20260101',
        help='Start date in YYYYMMDD format (default: 20260101)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default='20260101',
        help='End date in YYYYMMDD format (default: 20260101)'
    )
    
    parser.add_argument(
        '--rows-per-file',
        type=int,
        default=500,
        help='Number of rows per parquet file (default: 500)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory path (default: data/)'
    )
    
    parser.add_argument(
        '--null-probability',
        type=float,
        default=0.0,
        help='Probability (0.0 to 1.0) that each feature value will be null (default: 0.0)'
    )
    
    args = parser.parse_args()
    
    # Validate null_probability
    if args.null_probability < 0.0 or args.null_probability > 1.0:
        print(f"❌ Error: null-probability must be between 0.0 and 1.0, got {args.null_probability}")
        return 1
    
    try:
        generate_data(
            start_date=args.start_date,
            end_date=args.end_date,
            rows_per_file=args.rows_per_file,
            output_dir=args.output_dir,
            files_per_hour=4,  # Fixed at 4 files per hour as per requirements
            null_probability=args.null_probability
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
