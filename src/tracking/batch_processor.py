"""
Phase 3.5 Step 3.5.1: High-Throughput Batch Trajectory Ingestion.
Batch-processes tracking-eligible canonical video clips with progress tracking,
automatic checkpointing, and class/skater stratification balancing.
"""

import os
import sys
import glob
import time
import argparse
import pandas as pd
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.tracking.phase1_pipeline import Phase1Pipeline
from src.classification.dataset_audit import CANONICAL_9_CLASSES


def select_batch_clips(
    manifest_path: str = "data/metadata/video_manifest.csv",
    trajectory_dir: str = "features/v1_trajectories",
    target_per_class: int = 25,
    max_total_clips: int = 150
) -> pd.DataFrame:
    """
    Selects canonical clips to process, skipping existing parquet files and
    prioritizing multi-skater coverage across all 9 classes.
    """
    df = pd.read_csv(manifest_path)
    df['canonical_class'] = df['trick_name'].map(CANONICAL_9_CLASSES)
    df = df[df['canonical_class'].notna() & (df['tracking_eligible'] == True)].copy()

    # Identify existing parquet files
    existing = set()
    if os.path.exists(trajectory_dir):
        for f in glob.glob(os.path.join(trajectory_dir, "*.parquet")):
            existing.add(os.path.splitext(os.path.basename(f))[0])

    df['already_processed'] = df['clip_id'].isin(existing)
    candidates = df[~df['already_processed']].copy()

    # Sort to prioritize non-player_a skaters first to maximize diversity
    candidates['is_player_a'] = candidates['skater_id'] == 'player_a'
    candidates = candidates.sort_values(by=['is_player_a', 'canonical_class', 'duration_sec'])

    # Sample balanced set up to target_per_class per class
    selected_indices = []
    for cname in sorted(set(CANONICAL_9_CLASSES.values())):
        c_sub = candidates[candidates['canonical_class'] == cname]
        n_needed = max(0, target_per_class - sum((df['canonical_class'] == cname) & df['already_processed']))
        take = c_sub.head(n_needed)
        selected_indices.extend(take.index.tolist())

    selected_df = candidates.loc[selected_indices]
    if len(selected_df) > max_total_clips:
        selected_df = selected_df.head(max_total_clips)

    return selected_df


def run_batch_processing(
    clips_df: pd.DataFrame,
    output_dir: str = "features/v1_trajectories"
) -> Dict[str, Any]:
    """
    Runs Phase 1 tracking on selected clips.
    """
    os.makedirs(output_dir, exist_ok=True)
    pipeline = Phase1Pipeline()

    total = len(clips_df)
    print(f"Starting batch processing of {total} clips into {output_dir}...")

    successes = 0
    failures = 0
    t0_all = time.time()

    for idx, (_, row) in enumerate(clips_df.iterrows(), 1):
        cid = row['clip_id']
        fp = row['filepath']
        cname = row['canonical_class']
        sk = row['skater_id']

        if not os.path.exists(fp):
            print(f"[{idx}/{total}] Skipping {cid}: file not found ({fp})")
            failures += 1
            continue

        print(f"[{idx}/{total}] Processing {cid} ({cname} - {sk}, {row['total_frames']} frames)...", end="", flush=True)
        t0 = time.time()
        try:
            res = pipeline.process_clip(filepath=fp, clip_id=cid, output_dir=output_dir)
            t_el = time.time() - t0
            fps_proc = res.total_frames / max(0.01, t_el)
            print(f" Done ({t_el:.1f}s, {fps_proc:.1f} FPS)")
            successes += 1
        except Exception as e:
            print(f" FAILED ({e})")
            failures += 1

    total_time = time.time() - t0_all
    summary = {
        'total_attempted': total,
        'successes': successes,
        'failures': failures,
        'total_time_sec': total_time,
        'avg_time_per_clip': total_time / max(1, successes)
    }

    print("=" * 80)
    print(f"BATCH PROCESSING COMPLETE: {successes} succeeded, {failures} failed in {total_time:.1f}s")
    print("=" * 80)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch trajectory processor for Phase 3.5")
    parser.add_argument("--max_clips", type=int, default=50, help="Maximum number of clips to process in this run")
    parser.add_argument("--per_class", type=int, default=15, help="Target clips per class")
    args = parser.parse_args()

    selected = select_batch_clips(target_per_class=args.per_class, max_total_clips=args.max_clips)
    print(f"Selected {len(selected)} candidate clips across classes:")
    print(selected['canonical_class'].value_counts())
    print("\nSkater representation:")
    print(selected['skater_id'].value_counts())

    if len(selected) > 0:
        run_batch_processing(selected)
    else:
        print("No new clips to process matching criteria.")
