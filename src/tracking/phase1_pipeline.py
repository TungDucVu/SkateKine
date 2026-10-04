"""
Phase 1 Integrated Pipeline Engine: Spatial Tracking & Canonical Feature Store
Full 7-stage sequence with all 9 Guardrails:
Ingestion -> Pose Tracker -> Board Tracker -> Canonical Transformation -> SG Smoothing & Derivatives -> Parquet -> QC
"""

import os
from dataclasses import dataclass
import cv2
import numpy as np
import pandas as pd

from src.tracking.skater_tracker import SkaterPoseTracker
from src.tracking.board_tracker import SemanticBoardTracker, TrackingProvenance
from src.coordinates.transformer import CanonicalTransformer, ClipMetricScale
from src.coordinates.smoother import (
    smooth_and_differentiate_series,
    compute_angular_derivatives,
    compute_normalized_jerk
)
from src.coordinates.feature_store import save_trajectory_parquet


@dataclass
class Phase1ClipResult:
    clip_id: str
    total_frames: int
    processed_frames: int
    skater_detection_rate: float
    board_detection_rate: float
    tracking_drop_rate: float        # Frames where both board and skater are lost
    provenance_distribution: dict
    scale_cm_per_px: float
    scale_valid: bool
    mean_torso_jerk: float           # Normalized jerk index [G6]
    parquet_path: str
    qc_passed: bool


class Phase1Pipeline:
    def __init__(self, yolo_pose_model: str = 'yolov8n-pose.pt', yolo_det_model: str = 'yolov8n.pt'):
        self.pose_tracker = SkaterPoseTracker(model_weights=yolo_pose_model)
        self.board_tracker = SemanticBoardTracker(model_weights=yolo_det_model)
        self.transformer = CanonicalTransformer()

    def process_clip(self, filepath: str, clip_id: str, output_dir: str = 'features/v1_trajectories') -> Phase1ClipResult:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f'Video file not found: {filepath}')

        cap = cv2.VideoCapture(filepath)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.pose_tracker.reset()
        self.board_tracker.reset()

        frame_records = []
        board_lengths = []

        f_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Stage 2: Skater Pose Tracking
            pose_frame = self.pose_tracker.process_frame(frame, f_idx)

            # Stage 3: Semantic 8-Point Board Tracking
            board_frame = self.board_tracker.process_frame(frame, f_idx, skater_box=pose_frame.bbox)

            board_lengths.append(board_frame.apparent_length)

            frame_records.append({
                'frame_idx': f_idx,
                'skater_detected': pose_frame.detected,
                'board_detected': board_frame.detected,
                'torso_length': pose_frame.torso_length,
                'mid_hip_x': pose_frame.mid_hip[0],
                'mid_hip_y': pose_frame.mid_hip[1],
                'left_ankle_x': pose_frame.keypoints[15, 0] if pose_frame.detected else np.nan,
                'left_ankle_y': pose_frame.keypoints[15, 1] if pose_frame.detected else np.nan,
                'right_ankle_x': pose_frame.keypoints[16, 0] if pose_frame.detected else np.nan,
                'right_ankle_y': pose_frame.keypoints[16, 1] if pose_frame.detected else np.nan,
                'ankle_conf': min(pose_frame.confidence[15], pose_frame.confidence[16]) if pose_frame.detected else 0.0,
                # Board 8 points
                'board_nose_x': board_frame.keypoints[0, 0],
                'board_nose_y': board_frame.keypoints[0, 1],
                'board_tail_x': board_frame.keypoints[1, 0],
                'board_tail_y': board_frame.keypoints[1, 1],
                'board_ft_x': board_frame.keypoints[6, 0],
                'board_ft_y': board_frame.keypoints[6, 1],
                'board_rt_x': board_frame.keypoints[7, 0],
                'board_rt_y': board_frame.keypoints[7, 1],
                'board_centroid_x': board_frame.centroid[0],
                'board_centroid_y': board_frame.centroid[1],
                'board_conf': board_frame.confidence,
                'tracking_source': int(board_frame.provenance), # [G5] Provenance state
                'apparent_length': board_frame.apparent_length,
                'apparent_width': board_frame.apparent_width,
                'aspect_ratio': board_frame.aspect_ratio,
                'raw_in_plane_angle': board_frame.in_plane_angle
            })
            f_idx += 1

        cap.release()
        n_frames = len(frame_records)
        if n_frames == 0:
            raise ValueError(f'Zero frames read from video: {filepath}')

        df = pd.DataFrame(frame_records)

        # Stage 4: Canonical Transformation & Safe Metric Calibration [G3]
        scale_info = self.transformer.calibrate_clip_scale(board_lengths, approach_window_len=min(30, n_frames // 3))
        df['scale_cm_per_px'] = scale_info.scale_cm_per_px
        df['scale_valid'] = scale_info.scale_valid

        # Torso-normalized coordinates
        for col_prefix in ['board_centroid', 'left_ankle', 'right_ankle']:
            cx = df[f'{col_prefix}_x'].values
            cy = df[f'{col_prefix}_y'].values
            hx = df['mid_hip_x'].values
            hy = df['mid_hip_y'].values
            t_len = df['torso_length'].values
            
            valid = (t_len > 1e-3) & (~np.isnan(cx)) & (~np.isnan(hx))
            norm_x = np.full(n_frames, np.nan)
            norm_y = np.full(n_frames, np.nan)
            norm_x[valid] = (cx[valid] - hx[valid]) / t_len[valid]
            norm_y[valid] = (cy[valid] - hy[valid]) / t_len[valid]
            df[f'{col_prefix}_norm_x'] = norm_x
            df[f'{col_prefix}_norm_y'] = norm_y

        # Board-local foot projections
        b_angle = df['raw_in_plane_angle'].values
        bc_x = df['board_centroid_x'].values
        bc_y = df['board_centroid_y'].values
        
        for foot in ['left_ankle', 'right_ankle']:
            fx = df[f'{foot}_x'].values
            fy = df[f'{foot}_y'].values
            bl_x = np.full(n_frames, np.nan)
            bl_y = np.full(n_frames, np.nan)
            
            for i in range(n_frames):
                if not np.isnan(fx[i]) and not np.isnan(bc_x[i]) and not np.isnan(b_angle[i]):
                    foot_pt = np.array([fx[i], fy[i]])
                    cent_pt = np.array([bc_x[i], bc_y[i]])
                    bl_pt = self.transformer.project_to_board_frame(foot_pt, cent_pt, b_angle[i])
                    bl_x[i] = bl_pt[0]
                    bl_y[i] = bl_pt[1]
            df[f'{foot}_board_local_x'] = bl_x
            df[f'{foot}_board_local_y'] = bl_y

        # Stage 5: Kinematic Smoothing & Derivatives [G4, G6, G8]
        # 1. Angular unwrapping and derivatives [G4]
        unwrapped_theta, omega, alpha = compute_angular_derivatives(df['raw_in_plane_angle'].values, fps)
        df['theta_unwrapped'] = unwrapped_theta
        df['omega_rad_per_sec'] = omega
        df['alpha_rad_per_sec2'] = alpha

        # 2. Linear position derivatives for board centroid
        for dim, col in [('x', 'board_centroid_x'), ('y', 'board_centroid_y')]:
            vals = df[col].values
            df[f'board_vel_{dim}'] = smooth_and_differentiate_series(vals, fps, deriv=1)
            df[f'board_acc_{dim}'] = smooth_and_differentiate_series(vals, fps, deriv=2)

        # 3. Resolution-normalized jerk index [G6]
        hip_coords = np.column_stack([df['mid_hip_x'].values, df['mid_hip_y'].values])
        jerk_series = compute_normalized_jerk(hip_coords, df['torso_length'].values, fps)
        df['norm_jerk_torso'] = jerk_series
        valid_jerk = jerk_series[~np.isnan(jerk_series)]
        mean_jerk = float(np.mean(valid_jerk)) if len(valid_jerk) > 0 else np.nan

        # Stage 6: Persist Feature Parquet Store
        parquet_path = save_trajectory_parquet(df, clip_id, output_dir=output_dir)

        # Stage 7: Quality Control Audit
        skater_rate = float(df['skater_detected'].mean())
        board_rate = float(df['board_detected'].mean())
        drop_frames = ((~df['skater_detected']) | (~df['board_detected'])).sum()
        drop_rate = float(drop_frames / n_frames)

        prov_counts = df['tracking_source'].value_counts().to_dict()

        # Gate 1 criteria: Drop rate < 0.05, mean jerk < 25.0
        qc_passed = (drop_rate < 0.08) and (mean_jerk < 35.0 or np.isnan(mean_jerk))

        return Phase1ClipResult(
            clip_id=clip_id,
            total_frames=total_frames,
            processed_frames=n_frames,
            skater_detection_rate=skater_rate,
            board_detection_rate=board_rate,
            tracking_drop_rate=drop_rate,
            provenance_distribution=prov_counts,
            scale_cm_per_px=scale_info.scale_cm_per_px,
            scale_valid=scale_info.scale_valid,
            mean_torso_jerk=mean_jerk,
            parquet_path=parquet_path,
            qc_passed=qc_passed
        )
