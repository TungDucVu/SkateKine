"""
Phase 3.5.9 Ingestion & Tracking Pipeline:
Ingests and validates the 15 audited Ollie clips across 7 distinct skater sessions,
extracts Phase 1 2D/3D kinematic trajectories, performs Phase 2 event segmentation,
and registers them into video_manifest.csv with strict session-clustered skater IDs.
"""

import os
import sys
import json
import time
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.tracking.phase1_pipeline import Phase1Pipeline
from src.segmentation.phase2_pipeline import Phase2Pipeline


def ingest_curated_ollies(
    curated_csv: str = "data/metadata/curated_ollie_15.csv",
    manifest_csv: str = "data/metadata/video_manifest.csv",
    trajectory_dir: str = "features/v1_trajectories"
):
    print("=" * 80)
    print("PHASE 3.5.9: INGESTING 15 AUDITED OLLIE CLIPS")
    print("=" * 80)

    df_curated = pd.read_csv(curated_csv)
    print(f"Loaded {len(df_curated)} clips from {curated_csv}")
    print("Skater session breakdown:")
    print(df_curated['skater_cluster'].value_counts())
    print("=" * 80)

    os.makedirs(trajectory_dir, exist_ok=True)
    p1 = Phase1Pipeline()
    p2 = Phase2Pipeline()

    results = []
    t0_all = time.time()

    for idx, row in df_curated.iterrows():
        cid = row['clip_id']
        fp = row['filepath']
        sk = row['skater_cluster']
        fn = row['filename']

        print(f"[{idx+1}/{len(df_curated)}] Tracking {cid} ({fn}, {sk})...", end="", flush=True)
        t0 = time.time()

        try:
            p1_res = p1.process_clip(filepath=fp, clip_id=cid, output_dir=trajectory_dir)
            t_el = time.time() - t0
            fps_proc = p1_res.total_frames / max(0.01, t_el)

            # Read trajectory for Phase 2 validation
            pq_path = os.path.join(trajectory_dir, f"{cid}.parquet")
            df_traj = pd.read_parquet(pq_path)
            fps_vid = row['fps']

            p2_res = p2.process_trajectory(df_traj, fps=fps_vid, clip_id=cid)
            events = p2_res.events

            print(f" OK ({t_el:.1f}s, {fps_proc:.1f} FPS) -> Pop: {events.t_pop}, Apex: {events.t_apex}, Land: {events.t_land}, Valid: {events.is_valid_monotonic}")

            results.append({
                'clip_id': cid,
                'source_dataset': 'skateboard_ml',
                'filepath': fp,
                'skater_id': sk,
                'stance': 'regular',  # Default canonical stance
                'original_label': 'Ollie',
                'trick_name': 'ollie',
                'category': 'linear',
                'phase1_eligible': True,
                'outcome': row['outcome'],
                'outcome_source': 'visual_audit',
                'width': int(row['resolution'].split('x')[0]),
                'height': int(row['resolution'].split('x')[1]),
                'fps': float(row['fps']),
                'total_frames': p1_res.total_frames,
                'duration_sec': float(row['duration_sec']),
                'quality_flag': 'pass' if events.is_valid_monotonic else 'low_confidence',
                'split': 'train',
                'tracking_eligible': True,
                'temporal_high_speed': False,
                'pilot_set': False,
                't_pop': events.t_pop,
                't_apex': events.t_apex,
                't_land': events.t_land,
                'flight_ms': events.flight_duration_ms
            })

        except Exception as e:
            print(f" FAILED ({e})")

    elapsed_all = time.time() - t0_all
    print("=" * 80)
    print(f"TRACKING COMPLETE: {len(results)}/{len(df_curated)} succeeded in {elapsed_all:.1f}s")
    print("=" * 80)

    # Register into video_manifest.csv
    if results:
        df_new = pd.DataFrame(results)
        manifest_cols = [
            'clip_id', 'source_dataset', 'filepath', 'skater_id', 'stance',
            'original_label', 'trick_name', 'category', 'phase1_eligible',
            'outcome', 'outcome_source', 'width', 'height', 'fps',
            'total_frames', 'duration_sec', 'quality_flag', 'split',
            'tracking_eligible', 'temporal_high_speed', 'pilot_set'
        ]

        if os.path.exists(manifest_csv):
            df_manifest = pd.read_csv(manifest_csv)
            # Remove any previous entries with these clip_ids
            df_manifest = df_manifest[~df_manifest['clip_id'].isin(df_new['clip_id'])]
            df_combined = pd.concat([df_manifest, df_new[manifest_cols]], ignore_index=True)
        else:
            df_combined = df_new[manifest_cols]

        df_combined.to_csv(manifest_csv, index=False)
        print(f"Updated {manifest_csv}: now contains {len(df_combined)} total clips.")

        # Save event audit log
        event_log_path = "data/metadata/curated_ollie_15_events.json"
        event_dict = {
            r['clip_id']: {
                't_pop': r['t_pop'],
                't_apex': r['t_apex'],
                't_land': r['t_land'],
                'flight_ms': r['flight_ms'],
                'skater_id': r['skater_id']
            }
            for r in results
        }
        with open(event_log_path, "w", encoding="utf-8") as f:
            json.dump(event_dict, f, indent=2)
        print(f"Saved temporal event metadata to {event_log_path}")

    return results


if __name__ == "__main__":
    ingest_curated_ollies()
