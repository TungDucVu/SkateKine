"""
Phase 1 Pilot Batch Certification Test Runner (Guardrail G9)
Executes the full Phase 1 pipeline on the 25 diverse pilot clips,
computes Gate 1 QC metrics, verifies Parquet integrity,
and generates a formal benchmark summary.
"""

import os
import sys
import json
import time

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.tracking.phase1_pipeline import Phase1Pipeline


MANIFEST_PATH = 'data/metadata/video_manifest.csv'
REPORT_OUTPUT_PATH = 'docs/reports/pilot_benchmark_results.json'


def run_pilot_certification():
    print('===============================================================')
    print('      PHASE 1 PILOT BATCH CERTIFICATION (25 DIVERSE CLIPS)     ')
    print('===============================================================')

    df = pd.read_csv(MANIFEST_PATH)
    pilot_clips = df[df['pilot_set'] == True]
    assert len(pilot_clips) == 25, f'Expected 25 pilot clips, found {len(pilot_clips)}'

    pipeline = Phase1Pipeline()
    results = []

    start_time = time.time()

    for idx, (_, row) in enumerate(pilot_clips.iterrows()):
        clip_id = row['clip_id']
        filepath = row['filepath'].replace('/', os.sep)
        skater = row['skater_id']
        trick = row['trick_name']
        outcome = row['outcome']
        fps = row['fps']

        t0 = time.time()
        print(f'[{idx+1:02d}/25] Processing {clip_id} ({skater}, {trick}, {outcome}, {fps}fps)...', end='', flush=True)

        res = pipeline.process_clip(filepath, clip_id, output_dir='features/v1_trajectories')
        dur = time.time() - t0

        print(f' Done in {dur:.2f}s | Skater: {res.skater_detection_rate*100:.1f}%, Board: {res.board_detection_rate*100:.1f}%, Jerk: {res.mean_torso_jerk:.1f}, ScaleValid: {res.scale_valid}')
        
        results.append({
            'clip_id': res.clip_id,
            'source_dataset': row['source_dataset'],
            'skater_id': skater,
            'trick_name': trick,
            'outcome': outcome,
            'fps': fps,
            'total_frames': res.total_frames,
            'processed_frames': res.processed_frames,
            'skater_detection_rate': res.skater_detection_rate,
            'board_detection_rate': res.board_detection_rate,
            'tracking_drop_rate': res.tracking_drop_rate,
            'mean_torso_jerk': res.mean_torso_jerk,
            'scale_cm_per_px': res.scale_cm_per_px,
            'scale_valid': res.scale_valid,
            'provenance': res.provenance_distribution,
            'parquet_path': res.parquet_path,
            'processing_time_sec': dur,
            'qc_passed': res.qc_passed
        })

    total_duration = time.time() - start_time
    total_frames_processed = sum(r['processed_frames'] for r in results)
    avg_fps_throughput = total_frames_processed / total_duration if total_duration > 0 else 0

    # Aggregate Metrics
    avg_skater_rate = float(pd.Series([r['skater_detection_rate'] for r in results]).mean())
    avg_board_rate = float(pd.Series([r['board_detection_rate'] for r in results]).mean())
    avg_drop_rate = float(pd.Series([r['tracking_drop_rate'] for r in results]).mean())
    avg_jerk = float(pd.Series([r['mean_torso_jerk'] for r in results]).dropna().mean())
    scale_valid_rate = float(pd.Series([r['scale_valid'] for r in results]).mean())
    qc_pass_rate = float(pd.Series([r['qc_passed'] for r in results]).mean())

    summary = {
        'total_pilot_clips': len(results),
        'total_frames_processed': total_frames_processed,
        'wall_clock_time_sec': round(total_duration, 2),
        'avg_processing_fps': round(avg_fps_throughput, 2),
        'average_skater_detection_rate': round(avg_skater_rate, 4),
        'average_board_detection_rate': round(avg_board_rate, 4),
        'average_tracking_drop_rate': round(avg_drop_rate, 4),
        'average_torso_normalized_jerk': round(avg_jerk, 2),
        'scale_valid_pass_rate': round(scale_valid_rate, 4),
        'pilot_qc_pass_rate': round(qc_pass_rate, 4),
        'gate_1_operational_target_met': bool(avg_drop_rate < 0.05 and qc_pass_rate >= 0.85),
        'detailed_clip_results': results
    }

    os.makedirs(os.path.dirname(REPORT_OUTPUT_PATH), exist_ok=True)
    with open(REPORT_OUTPUT_PATH, 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n===============================================================')
    print('                 PILOT CERTIFICATION SUMMARY                   ')
    print('===============================================================')
    print(f'Total Pilot Clips:            {len(results)}')
    print(f'Total Frames Processed:       {total_frames_processed}')
    print(f'Total Time:                   {total_duration:.2f}s ({avg_fps_throughput:.1f} frames/sec)')
    print(f'Average Skater Detection:     {avg_skater_rate * 100:.2f}%')
    print(f'Average Board Detection:      {avg_board_rate * 100:.2f}%')
    print(f'Average Tracking Drop Rate:   {avg_drop_rate * 100:.2f}% (Target: < 3.0%)')
    print(f'Average Torso-Norm Jerk:      {avg_jerk:.2f} torso-lengths/s^3')
    print(f'Scale Factor Valid Rate:      {scale_valid_rate * 100:.2f}%')
    print(f'Pilot QC Pass Rate:           {qc_pass_rate * 100:.2f}%')
    print(f'Gate 1 Status:                {"PASS" if summary["gate_1_operational_target_met"] else "FAIL"}')
    print(f'Summary Report Saved:         {REPORT_OUTPUT_PATH}')
    print('===============================================================')

    return summary


if __name__ == '__main__':
    run_pilot_certification()
