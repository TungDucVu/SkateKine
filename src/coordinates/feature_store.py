"""
Feature Store Serializer (Phase 1, Step 1.5)
Persists smoothed spatial trajectories, kinematic derivatives, and provenance states
into Apache Parquet format.
"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def save_trajectory_parquet(df_trajectory: pd.DataFrame, clip_id: str, output_dir: str = 'features/v1_trajectories') -> str:
    """
    Serializes a single clip's trajectory dataframe into a typed Apache Parquet file.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'{clip_id}.parquet').replace(os.sep, '/')
    
    # Write using pyarrow for optimal compression and schema typing
    table = pa.Table.from_pandas(df_trajectory)
    pq.write_table(table, out_path, compression='snappy')
    return out_path


def load_trajectory_parquet(clip_id: str, feature_dir: str = 'features/v1_trajectories') -> pd.DataFrame:
    path = os.path.join(feature_dir, f'{clip_id}.parquet')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Parquet feature file not found: {path}')
    return pd.read_parquet(path)
