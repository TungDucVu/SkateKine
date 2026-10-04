"""
Kinematic Signal Conditioning Pipeline: Implements canonical Z-up coordinate framing,
time-scaled continuous differentiation, and biomechanical signal conditioning for Phase 2.
Adheres strictly to Amendment [A2] (Canonical Z-Up: z = -y, positive z_dot = ascent).
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import Dict, Any, Optional, Tuple


class KinematicSignalProcessor:
    """
    Conditions kinematic trajectories into canonical Z-up world convention and computes
    continuous first and second-order derivatives scaled to clip frame rate.
    """

    def __init__(self, filter_window_sec: float = 0.075):
        """
        :param filter_window_sec: Savitzky-Golay smoothing window in seconds (default 75ms) [G8]
        """
        self.filter_window_sec = filter_window_sec

    def _get_window_length(self, fps: float, max_len: int) -> int:
        """Computes odd window length in frames proportional to continuous time tau."""
        w = int(round(self.filter_window_sec * fps))
        if w % 2 == 0:
            w += 1
        w = max(5, w)
        if w >= max_len:
            w = max_len - 1 if max_len % 2 == 0 else max_len - 2
            w = max(3, w)
        return w

    def _smooth_derivative(self, series: pd.Series, fps: float, deriv_order: int = 0) -> np.ndarray:
        """Computes smooth derivative using Savitzky-Golay with gap-handling."""
        arr = series.values.astype(float)
        n = len(arr)
        if n < 5:
            # Fallback to simple gradient if sequence is too short
            if deriv_order == 0:
                return arr
            elif deriv_order == 1:
                return np.gradient(arr, 1.0 / fps)
            else:
                return np.gradient(np.gradient(arr, 1.0 / fps), 1.0 / fps)

        # Interpolate NaNs temporarily for filtering
        s = pd.Series(arr).interpolate(method='linear', limit_direction='both')
        clean_arr = s.fillna(0.0).values

        w = self._get_window_length(fps, n)
        poly = 2 if deriv_order < 2 else 3
        poly = min(poly, w - 1)

        dt = 1.0 / fps
        smoothed = savgol_filter(clean_arr, window_length=w, polyorder=poly, deriv=deriv_order, delta=dt)

        # Restore original NaN positions
        smoothed[np.isnan(arr)] = np.nan
        return smoothed

    def process(self, df_traj: pd.DataFrame, fps: float) -> pd.DataFrame:
        """
        Transforms Phase 1 trajectory DataFrame into conditioned canonical Z-up state space.
        """
        df = df_traj.copy()
        n = len(df)

        # 1. Canonical Z-Up Framing [A2]: z(t) = -y(t)
        # In pixel space, y increases downwards. Canonical z increases upwards.
        df['z_deck'] = -df['board_centroid_y']
        df['z_nose'] = -df['board_nose_y']
        df['z_tail'] = -df['board_tail_y']
        df['z_ft'] = -df['board_ft_y']
        df['z_rt'] = -df['board_rt_y']
        df['z_left_ankle'] = -df['left_ankle_y']
        df['z_right_ankle'] = -df['right_ankle_y']
        df['z_mid_hip'] = -df['mid_hip_y']

        # 2. Time-scaled continuous vertical velocities (dz/dt) and accelerations (d2z/dt2)
        # Positive vel_z = ascending; Negative vel_z = descending
        df['vel_z_deck'] = self._smooth_derivative(df['z_deck'], fps, deriv_order=1)
        df['acc_z_deck'] = self._smooth_derivative(df['z_deck'], fps, deriv_order=2)

        df['vel_z_nose'] = self._smooth_derivative(df['z_nose'], fps, deriv_order=1)
        df['vel_z_tail'] = self._smooth_derivative(df['z_tail'], fps, deriv_order=1)

        df['vel_z_left_ankle'] = self._smooth_derivative(df['z_left_ankle'], fps, deriv_order=1)
        df['acc_z_left_ankle'] = self._smooth_derivative(df['z_left_ankle'], fps, deriv_order=2)

        df['vel_z_right_ankle'] = self._smooth_derivative(df['z_right_ankle'], fps, deriv_order=1)
        df['acc_z_right_ankle'] = self._smooth_derivative(df['z_right_ankle'], fps, deriv_order=2)

        df['vel_z_mid_hip'] = self._smooth_derivative(df['z_mid_hip'], fps, deriv_order=1)
        df['acc_z_mid_hip'] = self._smooth_derivative(df['z_mid_hip'], fps, deriv_order=2)

        # 3. Deck Angular Dynamics
        # omega (rad/s) and angular acceleration alpha (rad/s^2)
        if 'omega_rad_per_sec' in df.columns and not df['omega_rad_per_sec'].isna().all():
            omega = df['omega_rad_per_sec']
        elif 'theta_unwrapped' in df.columns:
            omega = self._smooth_derivative(df['theta_unwrapped'], fps, deriv_order=1)
            df['omega_rad_per_sec'] = omega
        else:
            omega = pd.Series(np.zeros(n))
            df['omega_rad_per_sec'] = omega

        df['alpha_rad_per_sec2'] = self._smooth_derivative(pd.Series(omega), fps, deriv_order=1)

        # 4. Ground Plane Baseline Estimation (Approach Phase)
        # Sample early valid frames where skater rolls flat
        valid_z = df['z_deck'].dropna()
        if len(valid_z) > 10:
            sample_len = min(30, len(valid_z) // 3)
            early_z = valid_z.iloc[:sample_len]
            ground_z_ref = float(np.percentile(early_z, 15))  # Lower percentile of flat rolling
        else:
            ground_z_ref = float(valid_z.min()) if len(valid_z) > 0 else 0.0

        df['ground_z_ref'] = ground_z_ref
        df['deck_altitude'] = df['z_deck'] - ground_z_ref
        df['tail_altitude'] = df['z_tail'] - ground_z_ref
        df['nose_altitude'] = df['z_nose'] - ground_z_ref

        # 5. Dual Ankle-Deck Proximity and Relative Velocities
        # Euclidean distance from ankles to board centroid
        df['dist_la_deck'] = np.sqrt((df['left_ankle_x'] - df['board_centroid_x'])**2 + (df['left_ankle_y'] - df['board_centroid_y'])**2)
        df['dist_ra_deck'] = np.sqrt((df['right_ankle_x'] - df['board_centroid_x'])**2 + (df['right_ankle_y'] - df['board_centroid_y'])**2)
        df['min_ankle_dist'] = np.fmin(df['dist_la_deck'], df['dist_ra_deck'])

        # Relative vertical velocity between feet and deck
        df['rel_vel_la_deck'] = np.abs(df['vel_z_left_ankle'] - df['vel_z_deck'])
        df['rel_vel_ra_deck'] = np.abs(df['vel_z_right_ankle'] - df['vel_z_deck'])
        df['min_rel_vel'] = np.fmin(df['rel_vel_la_deck'], df['rel_vel_ra_deck'])

        return df
