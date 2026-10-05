"""
Phase 3 Step 3.3: Post-Impact Land/Bail Verification Gate (P3-A5, P3-A6).
Evaluates stability persistence over [t_land, t_land + 30] to classify attempts as
LANDED, BAILED, or UNCERTAIN. Strictly decoupled from execution cleanliness.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BailVerificationResult:
    clip_id: str
    status: str              # 'LANDED', 'BAILED', or 'UNCERTAIN'
    confidence: float        # [0.0, 1.0]
    stability_score: float   # [0.0, 1.0] (1.0 = rock solid rollout)
    indicators: Dict[str, float]
    passed_gate: bool        # True only if LANDED


class PostImpactBailGatekeeper:
    """
    Verifies attempt completion over the post-touchdown interval [t_land, t_land + 30].
    """

    def __init__(self, post_window_frames: int = 30, stability_threshold: float = 0.50):
        self.post_window_frames = post_window_frames
        self.stability_threshold = stability_threshold

    def evaluate_rollout(
        self,
        df_traj: pd.DataFrame,
        t_land: int,
        fps: float,
        clip_id: str = "unknown"
    ) -> BailVerificationResult:
        """
        Ingests the post-landing trajectory slice and computes physical rollout indicators.
        """
        N = len(df_traj)
        if t_land is None or np.isnan(t_land) or t_land >= N - 2:
            return BailVerificationResult(
                clip_id=clip_id,
                status="UNCERTAIN",
                confidence=0.5,
                stability_score=0.0,
                indicators={'available_post_frames': 0},
                passed_gate=False
            )

        t_start = int(t_land)
        t_end = min(N - 1, t_start + self.post_window_frames)
        post_df = df_traj.iloc[t_start:t_end + 1]
        available_frames = len(post_df)

        # If clip terminates abruptly post-impact (< 5 frames), label UNCERTAIN [P3-A5]
        if available_frames < 5:
            return BailVerificationResult(
                clip_id=clip_id,
                status="UNCERTAIN",
                confidence=0.4,
                stability_score=0.3,
                indicators={'available_post_frames': available_frames},
                passed_gate=False
            )

        # 1. Foot-to-Deck Proximity Persistence (Normalized by Apparent Board Length)
        board_len = float(post_df['apparent_length'].median()) if 'apparent_length' in post_df else 200.0
        if board_len <= 10.0 or np.isnan(board_len):
            board_len = 200.0

        foot_dist_mean = 0.0
        foot_dist_frac = 0.3
        max_foot_frac = 0.3
        if 'left_ankle_board_local_y' in post_df and 'right_ankle_board_local_y' in post_df:
            left_d = np.sqrt(post_df['left_ankle_board_local_x']**2 + post_df['left_ankle_board_local_y']**2)
            right_d = np.sqrt(post_df['right_ankle_board_local_x']**2 + post_df['right_ankle_board_local_y']**2)
            mean_dist = (left_d + right_d) / 2.0
            max_dist = np.maximum(left_d, right_d)
            foot_dist_mean = float(mean_dist.mean())
            foot_dist_frac = float(foot_dist_mean / board_len)
            max_foot_frac = float(max_dist.mean() / board_len)
        else:
            foot_dist_frac = 0.3
            max_foot_frac = 0.3

        # Normalized foot proximity score: combines mean and single-foot maximum separation
        combined_dist_frac = 0.5 * foot_dist_frac + 0.5 * max_foot_frac
        foot_proximity_score = float(np.clip(1.0 - (combined_dist_frac / 0.70), 0.0, 1.0))

        # 2. Skater Center-of-Mass vs. Board Velocity Coherence
        vel_coherence_score = 1.0
        if 'board_vel_x' in post_df and 'mid_hip_x' in post_df:
            skater_vx = np.gradient(post_df['mid_hip_x'].values) * fps
            board_vx = post_df['board_vel_x'].fillna(0.0).values
            vx_diff = np.abs(skater_vx - board_vx)
            mean_vx_diff = float(np.mean(vx_diff))
            vel_coherence_score = float(np.clip(1.0 - (mean_vx_diff / 500.0), 0.0, 1.0))

        # 3. Board Motion Persistence & Detection Stability
        board_det_rate = float(post_df['board_detected'].mean()) if 'board_detected' in post_df else 1.0
        skater_det_rate = float(post_df['skater_detected'].mean()) if 'skater_detected' in post_df else 1.0
        detection_stability = float((board_det_rate + skater_det_rate) / 2.0)

        # 4. Aspect Ratio Stability (Board does not flip wildly or tumble away)
        aspect_stability = 1.0
        if 'aspect_ratio' in post_df:
            ar_std = float(post_df['aspect_ratio'].std()) if len(post_df) > 1 else 0.0
            aspect_stability = float(np.clip(1.0 - (ar_std / 0.5), 0.0, 1.0))

        # Aggregate Stability Score (Weighted synthesis)
        stability_score = float(
            0.45 * foot_proximity_score +
            0.25 * vel_coherence_score +
            0.15 * detection_stability +
            0.15 * aspect_stability
        )

        indicators = {
            'available_post_frames': available_frames,
            'foot_proximity_score': foot_proximity_score,
            'vel_coherence_score': vel_coherence_score,
            'detection_stability': detection_stability,
            'aspect_stability': aspect_stability,
            'foot_dist_mean': foot_dist_mean,
            'foot_dist_frac': foot_dist_frac
        }

        # Tri-state decision [P3-A5]
        if detection_stability < 0.20 or available_frames < 8:
            status = "UNCERTAIN"
            confidence = 0.55
            passed = False
        elif stability_score >= self.stability_threshold:
            status = "LANDED"
            confidence = float(np.clip(0.5 + (stability_score - self.stability_threshold) * 1.0, 0.51, 0.99))
            passed = True
        else:
            status = "BAILED"
            confidence = float(np.clip(0.5 + (self.stability_threshold - stability_score) * 1.0, 0.51, 0.99))
            passed = False

        return BailVerificationResult(
            clip_id=clip_id,
            status=status,
            confidence=confidence,
            stability_score=stability_score,
            indicators=indicators,
            passed_gate=passed
        )
