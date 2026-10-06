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

        # Front ankle local Y displacement (toe vs heel side)
        front_ankle_local_y_col = 'right_ankle_board_local_y' if is_goofy else 'left_ankle_board_local_y'
        if front_ankle_local_y_col in flight_df and len(flight_df[front_ankle_local_y_col].dropna()) > 1:
            pop_ankle_local_y = float(flight_df[front_ankle_local_y_col].iloc[0])
            f_end = min(len(flight_df) - 1, max(1, flick_frames))
            flick_ankle_local_y = float(flight_df[front_ankle_local_y_col].iloc[f_end])
            flick_local_y_delta = float((flick_ankle_local_y - pop_ankle_local_y) * stance_sign)
        else:
            flick_local_y_delta = 0.0

        # 6. Torso & Body Kinematics
        if 'mid_hip_x' in df_traj and 'mid_hip_y' in df_traj:
            pop_hip_y = df_traj['mid_hip_y'].iloc[t_pop]
            apex_hip_y = df_traj['mid_hip_y'].iloc[t_apex]
            apex_displacement = float(pop_hip_y - apex_hip_y)
        else:
            apex_displacement = 0.0

        flight_duration_sec = float((t_land - t_pop) * dt)
        ascent_ratio = float((t_apex - t_pop) / max(1, t_land - t_pop))

        # 7. Skater Body Yaw & Stance Switch Kinematics (Feet Inversion in 180s)
        # In a 180 (FS 180 or BS 180), the skater rotates body yaw and lands switch.
        pop_w = df_traj.iloc[max(0, t_pop - 5):min(len(df_traj), t_pop + 5)]
        land_w = df_traj.iloc[max(0, t_land - 5):min(len(df_traj), t_land + 5)]

        l_pop = max(20.0, float(pop_w['apparent_length'].median())) if 'apparent_length' in pop_w and len(pop_w['apparent_length'].dropna()) > 0 else 100.0
        l_land = max(20.0, float(land_w['apparent_length'].median())) if 'apparent_length' in land_w and len(land_w['apparent_length'].dropna()) > 0 else 100.0

        pop_feet_dx = float((pop_w['left_ankle_x'] - pop_w['right_ankle_x']).median()) if 'left_ankle_x' in pop_w else 0.0
        land_feet_dx = float((land_w['left_ankle_x'] - land_w['right_ankle_x']).median()) if 'left_ankle_x' in land_w else 0.0

        if not np.isnan(pop_feet_dx) and not np.isnan(land_feet_dx) and abs(pop_feet_dx) > 1.0:
            feet_swap = 1.0 if (pop_feet_dx * land_feet_dx) < 0 else 0.0
            delta_feet_dx = float((land_feet_dx - pop_feet_dx) * stance_sign)
            delta_feet_norm = float(((land_feet_dx / l_land) - (pop_feet_dx / l_pop)) * stance_sign)
        else:
            feet_swap = 0.0
            delta_feet_dx = 0.0
            delta_feet_norm = 0.0
        skater_yaw_swap = feet_swap

        # 8. Board Yaw & Foreshortening Trough Dynamics (Shove-it discrimination)
        lens = flight_df['apparent_length'].dropna() if 'apparent_length' in flight_df else pd.Series([])
        min_norm_length = float(lens.min() / l_pop) if len(lens) > 0 else 1.0
        board_length_std = float(lens.std()) if len(lens) > 1 else 0.0

        # Board nose-to-tail inversion
        pop_b_dx = float((pop_w['board_nose_x'] - pop_w['board_tail_x']).median()) if 'board_nose_x' in pop_w else 0.0
        land_b_dx = float((land_w['board_nose_x'] - land_w['board_tail_x']).median()) if 'board_nose_x' in land_w else 0.0
        pop_b_dy = float((pop_w['board_nose_y'] - pop_w['board_tail_y']).median()) if 'board_nose_y' in pop_w else 0.0
        land_b_dy = float((land_w['board_nose_y'] - land_w['board_tail_y']).median()) if 'board_nose_y' in land_w else 0.0

        if not np.isnan(pop_b_dx) and not np.isnan(land_b_dx) and abs(pop_b_dx) > 1.0:
            board_swap = 1.0 if (pop_b_dx * land_b_dx) < 0 else 0.0
            delta_board_norm = float(((land_b_dx / l_land) - (pop_b_dx / l_pop)) * stance_sign)
        else:
            board_swap = 0.0
            delta_board_norm = 0.0
        board_yaw_swap = board_swap

        diff_yaw_norm = float(delta_board_norm - delta_feet_norm)

        # --- Phase 3.5.7 Tier 1 Physics Invariant Features ---
        # 8A. Deck Inversion Invariant (Cosine Similarity between pop & land deck vectors)
        pop_v_mag = np.hypot(pop_b_dx, pop_b_dy)
        land_v_mag = np.hypot(land_b_dx, land_b_dy)
        if pop_v_mag > 1.0 and land_v_mag > 1.0:
            c_inv_cosine = float((pop_b_dx * land_b_dx + pop_b_dy * land_b_dy) / (pop_v_mag * land_v_mag))
            c_inv_cosine = float(np.clip(c_inv_cosine, -1.0, 1.0))
        else:
            c_inv_cosine = 1.0

        # 8B. Foreshortening Curve Dynamics
        if len(lens) > 2 and l_pop > 10.0:
            lens_norm = lens.values / l_pop
            trough_depth = float(1.0 - min_norm_length)
            trough_idx = int(np.argmin(lens_norm))
            trough_symmetry = float(trough_idx / max(1, len(lens_norm) - 1))
            trough_duration = float(np.mean(lens_norm < 0.65))
            length_ratio_land_pop = float(l_land / l_pop)
            recovery_slope = float((lens_norm[-1] - min_norm_length) / max(1, len(lens_norm) - 1 - trough_idx))
        else:
            trough_depth = 0.0
            trough_symmetry = 0.5
            trough_duration = 0.0
            length_ratio_land_pop = 1.0
            recovery_slope = 0.0

        # 8C. Body vs. Board Rotational Decoupling & Rotational Coupling
        if 'left_ankle_x' in flight_df and 'right_ankle_x' in flight_df and len(flight_df) > 1:
            body_dx = flight_df['left_ankle_x'].values - flight_df['right_ankle_x'].values
            body_dy = flight_df['left_ankle_y'].values - flight_df['right_ankle_y'].values
            theta_body_raw = np.arctan2(body_dy, body_dx)
            theta_body_unwrapped = np.unwrap(theta_body_raw)
        elif 'left_hip_x' in flight_df and 'right_hip_x' in flight_df and len(flight_df) > 1:
            body_dx = flight_df['left_hip_x'].values - flight_df['right_hip_x'].values
            body_dy = flight_df['left_hip_y'].values - flight_df['right_hip_y'].values
            theta_body_raw = np.arctan2(body_dy, body_dx)
            theta_body_unwrapped = np.unwrap(theta_body_raw)
        else:
            theta_body_unwrapped = np.zeros(len(flight_df))

        if 'board_nose_x' in flight_df and 'board_tail_x' in flight_df and len(flight_df) > 1:
            board_dx_arr = flight_df['board_nose_x'].values - flight_df['board_tail_x'].values
            board_dy_arr = flight_df['board_nose_y'].values - flight_df['board_tail_y'].values
            theta_board_raw = np.arctan2(board_dy_arr, board_dx_arr)
            theta_board_unwrapped = np.unwrap(theta_board_raw)
        else:
            theta_board_unwrapped = np.zeros(len(flight_df))

        if len(flight_df) > 1:
            delta_theta_body = float((theta_body_unwrapped[-1] - theta_body_unwrapped[0]) * stance_sign)
            delta_theta_board = float((theta_board_unwrapped[-1] - theta_board_unwrapped[0]) * stance_sign)
            delta_theta_relative = float(delta_theta_board - delta_theta_body)
            rot_ratio = float(abs(delta_theta_board) / (abs(delta_theta_body) + 0.25))

            omega_body = np.diff(theta_body_unwrapped) / dt
            omega_board = np.diff(theta_board_unwrapped) / dt
            num_coupling = np.sum(omega_body * omega_board)
            denom_coupling = np.sqrt(np.sum(omega_body**2) * np.sum(omega_board**2) + 1e-6)
            rotational_coupling = float(np.clip(num_coupling / denom_coupling, -1.0, 1.0))
        else:
            delta_theta_body = 0.0
            delta_theta_board = 0.0
            delta_theta_relative = 0.0
            rot_ratio = 0.0
            rotational_coupling = 0.0

        # 8D. Ballistic Trajectory Parabolic Residuals & Hip-Board Clearance
        if 'board_centroid_norm_y' in flight_df and len(flight_df['board_centroid_norm_y'].dropna()) >= 5:
            y_vals = flight_df['board_centroid_norm_y'].dropna().values
            t_steps = np.arange(len(y_vals)) * dt
            poly_coeffs = np.polyfit(t_steps, y_vals, deg=2)
            y_pred = np.polyval(poly_coeffs, t_steps)
            ss_res = np.sum((y_vals - y_pred)**2)
            ss_tot = np.sum((y_vals - np.mean(y_vals))**2) + 1e-6
            ballistic_r2 = float(np.clip(1.0 - (ss_res / ss_tot), 0.0, 1.0))
            ballistic_rmse = float(np.sqrt(np.mean((y_vals - y_pred)**2)))
            effective_acc_y = float(2.0 * poly_coeffs[0])
        elif 'board_center_y' in flight_df and len(flight_df['board_center_y'].dropna()) >= 5:
            y_vals = flight_df['board_center_y'].dropna().values / l_pop
            t_steps = np.arange(len(y_vals)) * dt
            poly_coeffs = np.polyfit(t_steps, y_vals, deg=2)
            y_pred = np.polyval(poly_coeffs, t_steps)
            ss_res = np.sum((y_vals - y_pred)**2)
            ss_tot = np.sum((y_vals - np.mean(y_vals))**2) + 1e-6
            ballistic_r2 = float(np.clip(1.0 - (ss_res / ss_tot), 0.0, 1.0))
            ballistic_rmse = float(np.sqrt(np.mean((y_vals - y_pred)**2)))
            effective_acc_y = float(2.0 * poly_coeffs[0])
        else:
            ballistic_r2 = 0.0
            ballistic_rmse = 0.0
            effective_acc_y = 0.0

        if 'mid_hip_y' in flight_df and 'board_center_y' in flight_df:
            clearance_vals = (flight_df['board_center_y'] - flight_df['mid_hip_y']).dropna() / l_pop
            hip_board_clearance_mean = float(clearance_vals.mean()) if len(clearance_vals) > 0 else 0.0
            hip_board_clearance_max = float(clearance_vals.max()) if len(clearance_vals) > 0 else 0.0
        else:
            hip_board_clearance_mean = 0.0
            hip_board_clearance_max = 0.0

        # 8E. Pop Impulse Proxy
        idx_pre_pop = max(0, t_pop - 3)
        idx_post_pop = min(len(df_traj) - 1, t_pop + 3)
        if 'board_vel_y' in df_traj:
            v_pre = float(df_traj['board_vel_y'].iloc[idx_pre_pop])
            v_post = float(df_traj['board_vel_y'].iloc[idx_post_pop])
            pop_impulse_dy = float((v_post - v_pre) / l_pop)
        else:
            pop_impulse_dy = 0.0

        # 8F. Flick Acceleration & Timing
        flick_acc_mag = float(flick_vel_mag / max(dt, flick_frames * dt)) if flick_frames > 1 else 0.0
        flick_timing_ratio = float(flick_frames / max(1, t_land - t_pop))

        # 9. Flip-Yaw Composite Interaction (Varial / Hardflip / 360 Flip vs pure flips/shuvs)
        flip_depth = float(np.clip(1.0 - min_aspect, 0.0, 1.0))
        yaw_depth = float(np.clip(1.0 - min_norm_length, 0.0, 1.0))
        flip_yaw_product = float(flip_depth * yaw_depth)

        # 10. Board Kinematic Velocity & Acceleration summaries
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

        # Construct comprehensive feature dictionary
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
            # Rotational axis & yaw discrimination [Phase 3.5]
            'skater_yaw_swap': skater_yaw_swap,
            'feet_swap': feet_swap,
            'delta_feet_dx': delta_feet_dx,
            'delta_feet_norm': delta_feet_norm,
            'min_norm_length': min_norm_length,
            'board_length_std': board_length_std,
            'board_yaw_swap': board_yaw_swap,
            'board_swap': board_swap,
            'delta_board_norm': delta_board_norm,
            'diff_yaw_norm': diff_yaw_norm,
            'flip_yaw_product': flip_yaw_product,
            # Phase 3.5.7 Tier 1 Physics Invariant Features
            'c_inv_cosine': c_inv_cosine,
            'trough_depth': trough_depth,
            'trough_symmetry': trough_symmetry,
            'trough_duration': trough_duration,
            'length_ratio_land_pop': length_ratio_land_pop,
            'recovery_slope': recovery_slope,
            'delta_theta_body': delta_theta_body,
            'delta_theta_board': delta_theta_board,
            'delta_theta_relative': delta_theta_relative,
            'rot_ratio': rot_ratio,
            'rotational_coupling': rotational_coupling,
            'ballistic_r2': ballistic_r2,
            'ballistic_rmse': ballistic_rmse,
            'effective_acc_y': effective_acc_y,
            'hip_board_clearance_mean': hip_board_clearance_mean,
            'hip_board_clearance_max': hip_board_clearance_max,
            'pop_impulse_dy': pop_impulse_dy,
            'flick_acc_mag': flick_acc_mag,
            'flick_timing_ratio': flick_timing_ratio,
            # Aspect & Flip dynamics [P3-A3]
            'flip_cycle_count': flip_cycle_count,
            'min_aspect': min_aspect,
            'aspect_std': aspect_std,
            'aspect_range': aspect_range,
            'flick_dx': flick_dx,
            'flick_dy': flick_dy,
            'flick_vel_mag': flick_vel_mag,
            'flick_direction_x': flick_direction_x,
            'flick_local_y_delta': flick_local_y_delta,
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
