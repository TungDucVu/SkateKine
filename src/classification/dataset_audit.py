"""
Phase 3 Step 3.1: Pre-Training Corpus Audit (P3-A8).
Inspects trick class distribution, skater diversity, viewpoint/stance characteristics,
and Land/Bail outcomes across video_manifest.csv and existing trajectory stores.
"""

import os
import json
import glob
import pandas as pd
import numpy as np
from typing import Dict, Any, List

CANONICAL_9_CLASSES = {
    'ollie': 'Ollie',
    'frontside180': 'Frontside 180',
    'frontside_180': 'Frontside 180',
    'backside180': 'Backside 180',
    'backside_180': 'Backside 180',
    'pop_shuvit': 'Pop Shove-it',
    'frontside_shuvit': 'Frontside Shove-it',
    'kickflip': 'Kickflip',
    'heelflip': 'Heelflip',
    'hardflip': 'Varial / Hardflip',
    'varial_kickflip': 'Varial / Hardflip',
    '360_flip': '360 Flip'
}

CLASS_MECHANICAL_SPECS = {
    'Ollie': {'yaw_deg': 0, 'roll_deg': 0, 'body_yaw_deg': 0, 'shuvit_dir': 'none', 'flip_dir': 'none'},
    'Frontside 180': {'yaw_deg': 180, 'roll_deg': 0, 'body_yaw_deg': 180, 'shuvit_dir': 'frontside', 'flip_dir': 'none'},
    'Backside 180': {'yaw_deg': -180, 'roll_deg': 0, 'body_yaw_deg': -180, 'shuvit_dir': 'backside', 'flip_dir': 'none'},
    'Pop Shove-it': {'yaw_deg': -180, 'roll_deg': 0, 'body_yaw_deg': 0, 'shuvit_dir': 'backside', 'flip_dir': 'none'},
    'Frontside Shove-it': {'yaw_deg': 180, 'roll_deg': 0, 'body_yaw_deg': 0, 'shuvit_dir': 'frontside', 'flip_dir': 'none'},
    'Kickflip': {'yaw_deg': 0, 'roll_deg': 360, 'body_yaw_deg': 0, 'shuvit_dir': 'none', 'flip_dir': 'toe_side'},
    'Heelflip': {'yaw_deg': 0, 'roll_deg': 360, 'body_yaw_deg': 0, 'shuvit_dir': 'none', 'flip_dir': 'heel_side'},
    'Varial / Hardflip': {'yaw_deg': 180, 'roll_deg': 360, 'body_yaw_deg': 0, 'shuvit_dir': 'variable', 'flip_dir': 'toe_side'},
    '360 Flip': {'yaw_deg': -360, 'roll_deg': 360, 'body_yaw_deg': 0, 'shuvit_dir': 'backside', 'flip_dir': 'toe_side'}
}


class DatasetAuditor:
    """Audits the SkateKine dataset manifest against Phase 3 requirements."""

    def __init__(self, manifest_path: str = "data/metadata/video_manifest.csv", trajectory_dir: str = "features/v1_trajectories"):
        self.manifest_path = manifest_path
        self.trajectory_dir = trajectory_dir
        self.df = pd.read_csv(manifest_path)

    def run_audit(self) -> Dict[str, Any]:
        df = self.df.copy()

        # Map canonical trick label
        df['canonical_class'] = df['trick_name'].map(CANONICAL_9_CLASSES)
        df['is_canonical_9'] = df['canonical_class'].notna()

        # Available trajectory files
        existing_trajs = set()
        if os.path.exists(self.trajectory_dir):
            for f in glob.glob(os.path.join(self.trajectory_dir, "*.parquet")):
                existing_trajs.add(os.path.splitext(os.path.basename(f))[0])

        df['has_trajectory'] = df['clip_id'].isin(existing_trajs)

        # 1. Total clip counts
        total_clips = len(df)
        eligible_clips = int(df['tracking_eligible'].sum())
        canonical_clips = int(df['is_canonical_9'].sum())
        canonical_eligible = int((df['is_canonical_9'] & df['tracking_eligible']).sum())

        # 2. Canonical Class Distribution
        class_counts = df[df['is_canonical_9']]['canonical_class'].value_counts().to_dict()
        class_eligible_counts = df[df['is_canonical_9'] & df['tracking_eligible']]['canonical_class'].value_counts().to_dict()

        # 3. Skater Diversity per Class
        skater_by_class = {}
        for cname in sorted(set(CANONICAL_9_CLASSES.values())):
            sub = df[df['canonical_class'] == cname]
            skaters = sub['skater_id'].unique().tolist()
            skater_by_class[cname] = {
                'total_clips': len(sub),
                'unique_skaters': len(skaters),
                'skater_list': skaters,
                'max_single_skater_share': float(sub['skater_id'].value_counts().iloc[0] / len(sub)) if len(sub) > 0 else 0.0
            }

        # 4. Viewpoint & Stance Distribution
        stance_dist = df['stance'].value_counts().to_dict()
        source_dist = df['source_dataset'].value_counts().to_dict()

        # FPS buckets
        def bucket_fps(fps):
            if fps < 45:
                return "30_fps"
            elif fps < 90:
                return "60_fps"
            else:
                return "high_speed_100_120_fps"

        df['fps_bucket'] = df['fps'].apply(bucket_fps)
        fps_dist = df['fps_bucket'].value_counts().to_dict()

        # 5. Outcome (Land / Bail) distribution across canonical classes
        outcome_by_class = {}
        for cname in sorted(set(CANONICAL_9_CLASSES.values())):
            sub = df[df['canonical_class'] == cname]
            outcome_by_class[cname] = sub['outcome'].value_counts().to_dict()

        # 6. Current Trajectory Store Status
        traj_summary = {
            'total_parquet_files': len(existing_trajs),
            'pilot_clips_with_traj': int(df[df['pilot_set']]['has_trajectory'].sum()),
            'canonical_clips_with_traj': int((df['is_canonical_9'] & df['has_trajectory']).sum()),
            'classes_in_traj_store': df[df['has_trajectory']]['canonical_class'].value_counts().to_dict(),
            'skaters_in_traj_store': df[df['has_trajectory']]['skater_id'].value_counts().to_dict(),
            'outcomes_in_traj_store': df[df['has_trajectory']]['outcome'].value_counts().to_dict()
        }

        audit_results = {
            'overview': {
                'total_manifest_clips': total_clips,
                'tracking_eligible_clips': eligible_clips,
                'canonical_9_total_clips': canonical_clips,
                'canonical_9_tracking_eligible': canonical_eligible
            },
            'canonical_class_distribution': class_eligible_counts,
            'skater_diversity_per_class': skater_by_class,
            'stance_distribution': stance_dist,
            'source_dataset_distribution': source_dist,
            'fps_bucket_distribution': fps_dist,
            'outcome_by_canonical_class': outcome_by_class,
            'trajectory_store_status': traj_summary,
            'mechanical_specifications': CLASS_MECHANICAL_SPECS
        }

        return audit_results

    def save_and_print_report(self, output_path: str = "docs/reports/phase3_dataset_audit.json") -> Dict[str, Any]:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results = self.run_audit()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print("=" * 80)
        print("SKATEKINE PHASE 3 PRE-TRAINING CORPUS AUDIT (P3-A8)")
        print("=" * 80)
        print(f"Total Clips in Manifest: {results['overview']['total_manifest_clips']}")
        print(f"Tracking Eligible Clips: {results['overview']['tracking_eligible_clips']}")
        print(f"Canonical 9-Class Eligible Clips: {results['overview']['canonical_9_tracking_eligible']}")
        print("-" * 80)
        print("CANONICAL 9-CLASS DISTRIBUTION (TRACKING ELIGIBLE):")
        for cname, cnt in sorted(results['canonical_class_distribution'].items(), key=lambda x: -x[1]):
            sk_info = results['skater_diversity_per_class'][cname]
            print(f"  • {cname:20s}: {cnt:4d} clips | {sk_info['unique_skaters']:2d} skaters | top-skater share: {sk_info['max_single_skater_share']*100:4.1f}%")
        print("-" * 80)
        print("CURRENT TRAJECTORY STORE (features/v1_trajectories):")
        ts = results['trajectory_store_status']
        print(f"  • Existing Parquet files: {ts['total_parquet_files']}")
        print(f"  • Canonical 9 clips in store: {ts['canonical_clips_with_traj']}")
        print(f"  • Unique skaters in store: {len(ts['skaters_in_traj_store'])}")
        print(f"  • Outcomes in store: {ts['outcomes_in_traj_store']}")
        print("=" * 80)
        print(f"Audit report saved to: {output_path}")

        return results


if __name__ == "__main__":
    auditor = DatasetAuditor()
    auditor.save_and_print_report()
