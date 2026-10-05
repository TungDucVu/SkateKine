"""
Phase 3.5 Step 3.5.1: Targeted Trajectory Store Expansion.
Expands the trajectory store to resolve zero-F1 class collapse,
augment under-represented classes (Varial/Hardflip, FS180, BS180, Pop Shove-it, FS Shove-it),
and ingest balanced bail attempts across multiple skaters.
"""

import os
import sys
import glob
import time
import pandas as pd
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.tracking.phase1_pipeline import Phase1Pipeline
from src.classification.dataset_audit import CANONICAL_9_CLASSES


def select_phase35_targeted_clips(
    manifest_path: str = "data/metadata/video_manifest.csv",
    trajectory_dir: str = "features/v1_trajectories"
) -> pd.DataFrame:
    """
    Selects balanced targeted clips focusing on:
    1. Zero-F1 classes: Varial / Hardflip, Pop Shove-it, FS 180, BS 180, FS Shove-it
    2. Balanced bail attempts across diverse professional skaters.
    """
    files = glob.glob(os.path.join(trajectory_dir, "*.parquet"))
    existing = set([os.path.splitext(os.path.basename(f))[0] for f in files])

    df = pd.read_csv(manifest_path)
    df['canonical_class'] = df['trick_name'].map(CANONICAL_9_CLASSES)
    df['in_store'] = df['clip_id'].isin(existing)

    cands = df[~df['in_store'] & (df['tracking_eligible'] == True)].copy()

    selected_dfs = []

    # 1. Varial / Hardflip across diverse pro skaters
    vh_cands = cands[cands['canonical_class'] == 'Varial / Hardflip'].copy()
    vh_sel = vh_cands.sort_values(by=['duration_sec']).groupby('skater_id').head(2).head(12)
    selected_dfs.append(vh_sel)

    # 2. Pop Shove-it (5 clips)
    ps_cands = cands[cands['canonical_class'] == 'Pop Shove-it'].copy()
    ps_sel = ps_cands.sort_values(by=['duration_sec']).head(5)
    selected_dfs.append(ps_sel)

    # 3. FS 180 (5 clips)
    fs180_cands = cands[cands['canonical_class'] == 'Frontside 180'].copy()
    fs180_sel = fs180_cands.sort_values(by=['duration_sec']).head(5)
    selected_dfs.append(fs180_sel)

    # 4. BS 180 (5 clips)
    bs180_cands = cands[cands['canonical_class'] == 'Backside 180'].copy()
    bs180_sel = bs180_cands.sort_values(by=['duration_sec']).head(5)
    selected_dfs.append(bs180_sel)

    # 5. FS Shove-it (5 clips)
    fss_cands = cands[cands['canonical_class'] == 'Frontside Shove-it'].copy()
    fss_sel = fss_cands.sort_values(by=['duration_sec']).head(5)
    selected_dfs.append(fss_sel)

    # 6. Diverse bails to balance Gate 4 (15 clips across diverse skaters)
    already_picked_ids = set().union(*[set(d['clip_id']) for d in selected_dfs])
    bail_cands = cands[~cands['clip_id'].isin(already_picked_ids) & (cands['outcome'] == 'bail')].copy()
    bail_sel = bail_cands.sort_values(by=['duration_sec']).groupby('skater_id').head(2).head(15)
    selected_dfs.append(bail_sel)

    batch_df = pd.concat(selected_dfs).drop_duplicates(subset=['clip_id'])
    return batch_df


def run_phase35_batch():
    batch_df = select_phase35_targeted_clips()
    print("=" * 80)
    print(f"PHASE 3.5 TARGETED BATCH EXPANSION: {len(batch_df)} clips selected")
    print("=" * 80)
    print("Class breakdown:\n", batch_df['canonical_class'].value_counts(dropna=False))
    print("\nOutcome breakdown:\n", batch_df['outcome'].value_counts(dropna=False))
    print("\nSkater breakdown:\n", batch_df['skater_id'].value_counts())
    print("=" * 80)

    output_dir = "features/v1_trajectories"
    pipeline = Phase1Pipeline()

    successes = 0
    failures = 0
    t0_start = time.time()

    for idx, (_, row) in enumerate(batch_df.iterrows(), 1):
        cid = row['clip_id']
        fp = row['filepath']
        cname = str(row['canonical_class'])
        outcome = row['outcome']
        sk = row['skater_id']

        if not os.path.exists(fp):
            print(f"[{idx}/{len(batch_df)}] SKIP {cid}: {fp} not found")
            failures += 1
            continue

        print(f"[{idx}/{len(batch_df)}] Processing {cid} ({cname}, {outcome}, {sk}, {row['total_frames']}f)...", end="", flush=True)
        t0 = time.time()
        try:
            res = pipeline.process_clip(filepath=fp, clip_id=cid, output_dir=output_dir)
            t_el = time.time() - t0
            fps_proc = res.total_frames / max(0.01, t_el)
            print(f" OK ({t_el:.1f}s, {fps_proc:.1f} FPS)")
            successes += 1
        except Exception as e:
            print(f" FAILED ({e})")
            failures += 1

    total_time = time.time() - t0_start
    print("=" * 80)
    print(f"PHASE 3.5 BATCH COMPLETED: {successes} succeeded, {failures} failed in {total_time:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    run_phase35_batch()
