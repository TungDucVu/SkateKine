"""
Phase 2 Quality Control & Evaluation Test Suite: Benchmarks Phase 2 Temporal Event
Localization against consensus ground truth across disjoint calibration (50%) and
frozen evaluation (50%) sets [A4]. Reports dual-unit errors (frames and ms) [A6].
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.segmentation.phase2_pipeline import Phase2Pipeline


def run_phase2_benchmark():
    gt_path = 'data/metadata/event_ground_truth.json'
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)

    pipeline = Phase2Pipeline()
    records = []

    for cid, gt in ground_truth.items():
        fps = float(gt['fps'])
        split = gt['split']
        skater_id = gt['skater_id']

        out = pipeline.process_clip_by_id(cid)
        ev = out.events

        # Predictions
        p_pop = ev.t_pop
        p_apex = ev.t_apex
        p_catch = ev.t_catch if ev.t_catch is not None else ev.t_land - 2
        p_land = ev.t_land

        # Ground truth
        g_pop = gt['pop']
        g_apex = gt['apex']
        g_catch = gt['catch']
        g_land = gt['land']

        # Absolute errors (frames)
        err_pop_f = abs(p_pop - g_pop)
        err_apex_f = abs(p_apex - g_apex)
        err_catch_f = abs(p_catch - g_catch)
        err_land_f = abs(p_land - g_land)

        # Absolute errors (milliseconds) [A6]
        dt_ms = 1000.0 / fps
        err_pop_ms = err_pop_f * dt_ms
        err_apex_ms = err_apex_f * dt_ms
        err_catch_ms = err_catch_f * dt_ms
        err_land_ms = err_land_f * dt_ms

        records.append({
            'clip_id': cid,
            'skater_id': skater_id,
            'split': split,
            'fps': fps,
            'is_valid_monotonic': ev.is_valid_monotonic,
            'flight_duration_ms': ev.flight_duration_ms,
            'pred_pop': p_pop, 'gt_pop': g_pop, 'err_pop_frames': err_pop_f, 'err_pop_ms': err_pop_ms,
            'pred_apex': p_apex, 'gt_apex': g_apex, 'err_apex_frames': err_apex_f, 'err_apex_ms': err_apex_ms,
            'pred_catch': p_catch, 'gt_catch': g_catch, 'err_catch_frames': err_catch_f, 'err_catch_ms': err_catch_ms,
            'pred_land': p_land, 'gt_land': g_land, 'err_land_frames': err_land_f, 'err_land_ms': err_land_ms
        })

    df_res = pd.DataFrame(records)

    # Compute metrics for Calibration Set vs Frozen Evaluation Set [A4]
    summary = {
        'total_clips': len(df_res),
        'monotonicity_rate': float(df_res['is_valid_monotonic'].mean()),
        'mean_flight_duration_ms': float(df_res['flight_duration_ms'].mean()),
        'splits': {}
    }

    for s_name in ['calibration', 'frozen_test', 'overall']:
        sub = df_res if s_name == 'overall' else df_res[df_res['split'] == s_name]
        summary['splits'][s_name] = {
            'clip_count': len(sub),
            'pop_mae_frames': float(sub['err_pop_frames'].mean()),
            'pop_mae_ms': float(sub['err_pop_ms'].mean()),
            'apex_mae_frames': float(sub['err_apex_frames'].mean()),
            'apex_mae_ms': float(sub['err_apex_ms'].mean()),
            'catch_mae_frames': float(sub['err_catch_frames'].mean()),
            'catch_mae_ms': float(sub['err_catch_ms'].mean()),
            'land_mae_frames': float(sub['err_land_frames'].mean()),
            'land_mae_ms': float(sub['err_land_ms'].mean()),
            'monotonicity_rate': float(sub['is_valid_monotonic'].mean())
        }

    # Gate 2 Compliance Evaluation
    test_metrics = summary['splits']['frozen_test']
    gate_2_pass = (
        test_metrics['pop_mae_frames'] <= 5.0 and
        test_metrics['apex_mae_frames'] <= 4.0 and
        test_metrics['catch_mae_frames'] <= 7.0 and
        test_metrics['land_mae_frames'] <= 5.0 and
        summary['monotonicity_rate'] >= 0.95
    )
    summary['gate_2_passed'] = gate_2_pass
    summary['detailed_clip_records'] = records

    # Export benchmark report json
    os.makedirs('docs/reports', exist_ok=True)
    out_json = 'docs/reports/phase2_benchmark_results.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary, df_res


def test_phase2_operational_gate():
    """Pytest test case asserting Gate 2 criteria on the frozen test set."""
    summary, _ = run_phase2_benchmark()
    test_res = summary['splits']['frozen_test']

    print(f"\n============================================================")
    print(f"GATE 2 BENCHMARK ON FROZEN EVALUATION SET (Disjoint Skaters):")
    print(f"============================================================")
    print(f"Pop MAE:   {test_res['pop_mae_frames']:.2f} frames ({test_res['pop_mae_ms']:.1f} ms) [Target <= 5.0 frames / <= 83 ms]")
    print(f"Apex MAE:  {test_res['apex_mae_frames']:.2f} frames ({test_res['apex_mae_ms']:.1f} ms) [Target <= 4.0 frames / <= 67 ms]")
    print(f"Catch MAE: {test_res['catch_mae_frames']:.2f} frames ({test_res['catch_mae_ms']:.1f} ms) [Target <= 7.0 frames / <= 116 ms]")
    print(f"Land MAE:  {test_res['land_mae_frames']:.2f} frames ({test_res['land_mae_ms']:.1f} ms) [Target <= 5.0 frames / <= 83 ms]")
    print(f"Monotonicity: {summary['monotonicity_rate']*100:.1f}% [Target >= 95.0%]")
    print(f"Gate 2 Pass: {summary['gate_2_passed']}")
    print(f"============================================================")

    assert test_res['pop_mae_frames'] <= 5.0, f"Pop MAE exceeds 5.0 frames: {test_res['pop_mae_frames']}"
    assert test_res['apex_mae_frames'] <= 4.0, f"Apex MAE exceeds 4.0 frames: {test_res['apex_mae_frames']}"
    assert test_res['catch_mae_frames'] <= 7.0, f"Catch MAE exceeds 7.0 frames: {test_res['catch_mae_frames']}"
    assert test_res['land_mae_frames'] <= 5.0, f"Land MAE exceeds 5.0 frames: {test_res['land_mae_frames']}"
    assert summary['monotonicity_rate'] >= 0.95, f"Monotonicity rate below 95%: {summary['monotonicity_rate']}"
    assert summary['gate_2_passed'] is True, "Gate 2 operational targets not satisfied"


if __name__ == '__main__':
    test_phase2_operational_gate()
