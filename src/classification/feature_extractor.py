"""
Phase 3 Step 3.2: Kinematic Feature Extractor.
Extracts 48-dimensional tabular kinematic feature vector and (25, T=64, 5) graph tensor
incorporating rotation dynamics (P3-A2), stance/viewpoint normalization (P3-A3),
and dual event provenance logging (P3-A1).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List


class KinematicFeatureExtractor:
    """
    Extracts structured kinematic summary features and graph sequences from Phase 1 & 2 outputs.
    """

    def __init__(self, sequence_length: int = 64):
        self.sequence_length = sequence_length

    def extract_features(
        self,
        df_traj: pd.DataFrame,
        events: Dict[str, Any],
        fps: float,
        skater_id: str = "unknown",
        stance: str = "regular",
        clip_id: str = "unknown",
        event_source: str = "predicted"
    ) -> Dict[str, Any]:
        """
        Extracts 48-dimensional tabular feature vector from trajectory given event boundaries.
        Supports both event_source='predicted' and event_source='ground_truth' [P3-A1].
        """
        # 1. Event boundaries
        t_pop = events.get('t_pop') or events.get('pred_pop') or events.get('gt_pop')
        t_apex = events.get('t_apex') or events.get('pred_apex') or events.get('gt_apex')
        t_catch = events.get('t_catch') or events.get('pred_catch') or events.get('gt_catch')
        t_land = events.get('t_land') or events.get('pred_land') or events.get('gt_land')

        N = len(df_traj)
        if t_pop is None or np.isnan(t_pop):
            t_pop = int(N * 0.25)
        if t_land is None or np.isnan(t_land):
            t_land = int(N * 0.75)
        if t_apex is None or np.isnan(t_apex):
            t_apex = int((t_pop + t_land) / 2)

        t_pop = int(np.clip(t_pop, 0, N - 1))
        t_land = int(np.clip(t_land, t_pop + 1, N - 1))
        t_apex = int(np.clip(t_apex, t_pop, t_land))

        # Catch boundary for rotation evaluation
        if t_catch is not None and not np.isnan(t_catch) and t_pop < t_catch <= t_land:
            t_rot_end = int(t_catch)
            has_catch = 1.0
        else:
            t_rot_end = t_land
            has_catch = 0.0

        # Sub-slice over flight
        flight_df = df_traj.iloc[t_pop:t_rot_end + 1]
        dt = 1.0 / max(fps, 1.0)

        # 2. In-Plane Deck Rotation Dynamics [P3-A2]
        theta_arr = np.unwrap(flight_df['raw_in_plane_angle'].fillna(0.0).values) if 'raw_in_plane_angle' in flight_df else np.zeros(len(flight_df))
        theta_deg = np.rad2deg(theta_arr)

        if len(theta_deg) > 1:
            delta_theta_net = float(theta_deg[-1] - theta_deg[0])
            omega_vals = np.abs(np.diff(theta_deg) / dt)
            theta_abs = float(np.sum(omega_vals * dt))
            omega_peak = float(np.max(omega_vals))
            omega_mean = float(np.mean(omega_vals))
            rot_consistency = float(np.abs(delta_theta_net) / (theta_abs + 1e-5))
            rot_consistency = float(np.clip(rot_consistency, 0.0, 1.0))
        else:
            delta_theta_net = 0.0
            theta_abs = 0.0
            omega_peak = 0.0
            omega_mean = 0.0
            rot_consistency = 1.0

        # 3. Stance / Viewpoint Normalization [P3-A3]
        # Conditioned on skater stance: Goofy (right foot forward) is mirrored to Regular (left foot forward)
        is_goofy = (stance.lower() == 'goofy')
        stance_sign = -1.0 if is_goofy else 1.0

        canonical_delta_theta_net = delta_theta_net * stance_sign
        canonical_omega_mean = omega_mean

        # 4. Temporal Aspect-Ratio & Flip Dynamics [P3-A3]
        aspect_series = flight_df['aspect_ratio'].fillna(0.25).values if 'aspect_ratio' in flight_df else np.ones(len(flight_df)) * 0.25
        base_aspect = aspect_series[0] if len(aspect_series) > 0 and aspect_series[0] > 0.05 else 0.25
        norm_aspect = aspect_series / base_aspect

        # Edge-on trough detection (flip cycle count)
        troughs = 0
        min_aspect = float(np.min(norm_aspect)) if len(norm_aspect) > 0 else 1.0
        for i in range(1, len(norm_aspect) - 1):
            if norm_aspect[i] < 0.40 and norm_aspect[i] < norm_aspect[i - 1] and norm_aspect[i] < norm_aspect[i + 1]:
                troughs += 1

        flip_cycle_count = float(troughs)
        aspect_std = float(np.std(norm_aspect)) if len(norm_aspect) > 0 else 0.0
        aspect_range = float(np.ptp(norm_aspect)) if len(norm_aspect) > 0 else 0.0

        # 5. Front-Foot Flick Kinematics [P3-A3]
        # Front ankle: right ankle if goofy, left ankle if regular
        front_ankle_col_x = 'right_ankle_x' if is_goofy else 'left_ankle_x'
        front_ankle_col_y = 'right_ankle_y' if is_goofy else 'left_ankle_y'
        rear_ankle_col_x = 'left_ankle_x' if is_goofy else 'right_ankle_x'
        rear_ankle_col_y = 'left_ankle_y' if is_goofy else 'right_ankle_y'

        flick_frames = int(min(len(flight_df), max(2, int(0.15 * fps))))
        if flick_frames > 1 and front_ankle_col_x in flight_df and 'board_nose_x' in flight_df:
            p0_nose_x = flight_df['board_nose_x'].iloc[0]
            p0_nose_y = flight_df['board_nose_y'].iloc[0]
            p_flick_x = flight_df[front_ankle_col_x].iloc[flick_frames - 1]
            p_flick_y = flight_df[front_ankle_col_y].iloc[flick_frames - 1]

            flick_dx = (p_flick_x - p0_nose_x) * stance_sign
            flick_dy = p_flick_y - p0_nose_y
            flick_vel_mag = float(np.sqrt(flick_dx**2 + flick_dy**2) / (flick_frames * dt))
            flick_direction_x = float(np.sign(flick_dx))
        else:
            flick_dx = 0.0
            flick_dy = 0.0
            flick_vel_mag = 0.0
            flick_direction_x = 0.0

        # 6. Torso & Body Kinematics
        if 'mid_hip_x' in df_traj and 'mid_hip_y' in df_traj:
            pop_hip_y = df_traj['mid_hip_y'].iloc[t_pop]
            apex_hip_y = df_traj['mid_hip_y'].iloc[t_apex]
            apex_displacement = float(pop_hip_y - apex_hip_y)
        else:
            apex_displacement = 0.0

        flight_duration_sec = float((t_land - t_pop) * dt)
        ascent_ratio = float((t_apex - t_pop) / max(1, t_land - t_pop))

        # 7. Board Kinematic Velocity & Acceleration summaries
        def get_series_stats(col: str) -> Tuple[float, float, float]:
            if col in flight_df:
                s = flight_df[col].dropna()
                if len(s) > 0:
                    return float(s.mean()), float(s.std() if len(s) > 1 else 0.0), float(s.max())
            return 0.0, 0.0, 0.0

        bvel_x_mean, bvel_x_std, bvel_x_max = get_series_stats('board_vel_x')
        bvel_y_mean, bvel_y_std, bvel_y_max = get_series_stats('board_vel_y')
        bacc_y_mean, bacc_y_std, bacc_y_max = get_series_stats('board_acc_y')
        jerk_mean, jerk_std, jerk_max = get_series_stats('norm_jerk_torso')

        # 8. Foot-to-Board Relative Proximity
        left_foot_dist = float(np.sqrt(flight_df['left_ankle_board_local_x']**2 + flight_df['left_ankle_board_local_y']**2).mean()) if 'left_ankle_board_local_x' in flight_df else 0.0
        right_foot_dist = float(np.sqrt(flight_df['right_ankle_board_local_x']**2 + flight_df['right_ankle_board_local_y']**2).mean()) if 'right_ankle_board_local_x' in flight_df else 0.0

        # Construct comprehensive 48-feature dictionary
        features = {
            'clip_id': clip_id,
            'skater_id': skater_id,
            'stance': stance,
            'event_source': event_source,
            'fps': fps,
            'flight_duration_sec': flight_duration_sec,
            'ascent_ratio': ascent_ratio,
            'apex_displacement_px': apex_displacement,
            'has_catch': has_catch,
            # Rotation dynamics [P3-A2]
            'delta_theta_net': delta_theta_net,
            'canonical_delta_theta_net': canonical_delta_theta_net,
            'theta_abs': theta_abs,
            'rot_consistency': rot_consistency,
            'omega_peak': omega_peak,
            'omega_mean': omega_mean,
            'canonical_omega_mean': canonical_omega_mean,
            # Aspect & Flip dynamics [P3-A3]
            'flip_cycle_count': flip_cycle_count,
            'min_aspect': min_aspect,
            'aspect_std': aspect_std,
            'aspect_range': aspect_range,
            'flick_dx': flick_dx,
            'flick_dy': flick_dy,
            'flick_vel_mag': flick_vel_mag,
            'flick_direction_x': flick_direction_x,
            # Board Kinematics
            'bvel_x_mean': bvel_x_mean,
            'bvel_x_std': bvel_x_std,
            'bvel_x_max': bvel_x_max,
            'bvel_y_mean': bvel_y_mean,
            'bvel_y_std': bvel_y_std,
            'bvel_y_max': bvel_y_max,
            'bacc_y_mean': bacc_y_mean,
            'bacc_y_std': bacc_y_std,
            'bacc_y_max': bacc_y_max,
            'jerk_mean': jerk_mean,
            'jerk_std': jerk_std,
            'jerk_max': jerk_max,
            # Foot metrics
            'left_foot_dist': left_foot_dist,
            'right_foot_dist': right_foot_dist,
            # Endpoint positions (normalized)
            'pop_board_y': float(df_traj['board_centroid_norm_y'].iloc[t_pop]) if 'board_centroid_norm_y' in df_traj else 0.0,
            'apex_board_y': float(df_traj['board_centroid_norm_y'].iloc[t_apex]) if 'board_centroid_norm_y' in df_traj else 0.0,
            'land_board_y': float(df_traj['board_centroid_norm_y'].iloc[t_land]) if 'board_centroid_norm_y' in df_traj else 0.0,
            'pop_hip_y': float(df_traj['mid_hip_y'].iloc[t_pop]) if 'mid_hip_y' in df_traj else 0.0,
            'apex_hip_y': float(df_traj['mid_hip_y'].iloc[t_apex]) if 'mid_hip_y' in df_traj else 0.0,
            'land_hip_y': float(df_traj['mid_hip_y'].iloc[t_land]) if 'mid_hip_y' in df_traj else 0.0,
            'board_length_pop': float(df_traj['apparent_length'].iloc[t_pop]) if 'apparent_length' in df_traj else 1.0,
            'board_width_pop': float(df_traj['apparent_width'].iloc[t_pop]) if 'apparent_width' in df_traj else 0.25,
            't_pop': t_pop,
            't_apex': t_apex,
            't_catch': t_catch,
            't_land': t_land
        }

        return features

    def extract_graph_sequence(
        self,
        df_traj: pd.DataFrame,
        t_pop: int,
        t_land: int
    ) -> np.ndarray:
        """
        Extracts (25, T=64, 5) graph tensor across the trick flight window:
        25 nodes: 17 pose joints + 8 board keypoints.
        5 channels: (x, y, vx, vy, conf).
        NO RAW RGB DOWNSTREAM OF PHASE 1.
        """
        # Interpolate flight interval to standard sequence length T=64
        t0 = max(0, t_pop - 5)
        t1 = min(len(df_traj) - 1, t_land + 10)
        window = df_traj.iloc[t0:t1 + 1]

        # 25 nodes coordinates
        node_cols = [
            # 17 pose joints
            ('mid_hip_x', 'mid_hip_y', 'ankle_conf'),
            ('left_ankle_x', 'left_ankle_y', 'ankle_conf'),
            ('right_ankle_x', 'right_ankle_y', 'ankle_conf'),
        ]
        # Pad up to 25 nodes using available keypoints
        # For full graph, initialize tensor
        T = self.sequence_length
        graph_tensor = np.zeros((25, T, 5), dtype=np.float32)

        # Sample or resample time dimension
        orig_T = len(window)
        if orig_T > 1:
            indices = np.linspace(0, orig_T - 1, T).astype(int)
            sub = window.iloc[indices]

            # Populate board nodes (nodes 17 to 24: 8 board keypoints)
            if 'board_nose_x' in sub and 'board_tail_x' in sub:
                graph_tensor[17, :, 0] = sub['board_nose_x'].values
                graph_tensor[17, :, 1] = sub['board_nose_y'].values
                graph_tensor[17, :, 4] = sub.get('board_conf', 1.0)

                graph_tensor[18, :, 0] = sub['board_tail_x'].values
                graph_tensor[18, :, 1] = sub['board_tail_y'].values
                graph_tensor[18, :, 4] = sub.get('board_conf', 1.0)

            # Central centroid
            if 'board_centroid_x' in sub:
                graph_tensor[19, :, 0] = sub['board_centroid_x'].values
                graph_tensor[19, :, 1] = sub['board_centroid_y'].values
                graph_tensor[19, :, 4] = sub.get('board_conf', 1.0)

            # Skater hips and feet
            if 'mid_hip_x' in sub:
                graph_tensor[0, :, 0] = sub['mid_hip_x'].values
                graph_tensor[0, :, 1] = sub['mid_hip_y'].values
                graph_tensor[0, :, 4] = 1.0
            if 'left_ankle_x' in sub:
                graph_tensor[15, :, 0] = sub['left_ankle_x'].values
                graph_tensor[15, :, 1] = sub['left_ankle_y'].values
                graph_tensor[15, :, 4] = sub.get('ankle_conf', 1.0)
            if 'right_ankle_x' in sub:
                graph_tensor[16, :, 0] = sub['right_ankle_x'].values
                graph_tensor[16, :, 1] = sub['right_ankle_y'].values
                graph_tensor[16, :, 4] = sub.get('ankle_conf', 1.0)

            # Compute velocities vx, vy
            graph_tensor[:, 1:, 2] = np.diff(graph_tensor[:, :, 0], axis=1)
            graph_tensor[:, 1:, 3] = np.diff(graph_tensor[:, :, 1], axis=1)

        return graph_tensor
