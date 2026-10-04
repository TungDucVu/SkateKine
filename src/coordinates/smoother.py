"""
Kinematic Signal Conditioning & Derivative Engine (Phase 1, Step 1.5)
Guardrails implemented:
- [G4] Derivative Integrity: np.unwrap(theta) & segment over tracking gaps >= 3 frames.
- [G6] Resolution-Agnostic Jerk: Normalize jerk by torso length [torso-lengths / sec^3].
- [G7] Formal PCK Normalization: Denominator = ground-truth apparent length.
- [G8] FPS-Scaled Filter Windows: Window size in continuous time (tau ~ 60-80ms), not fixed frames.
"""

import numpy as np
from scipy.signal import savgol_filter


DEFAULT_TAU_SEC = 0.075 # 75 ms continuous time window [G8]


def compute_fps_window_length(fps: float, tau_sec: float = DEFAULT_TAU_SEC, polyorder: int = 2) -> int:
    """
    [G8] Computes odd integer window length scaled by exact video FPS.
    """
    nominal = int(round(tau_sec * fps))
    if nominal % 2 == 0:
        nominal += 1
    # Window length must be at least polyorder + 2 and odd
    return max(polyorder + 2 if (polyorder + 2) % 2 != 0 else polyorder + 3, nominal)


def smooth_and_differentiate_series(
    values: np.ndarray,
    fps: float,
    polyorder: int = 2,
    deriv: int = 0,
    gap_threshold: int = 3,
    delta: float = None
) -> np.ndarray:
    """
    [G4] Savitzky-Golay filtering partitioned over tracking gaps.
    Never interpolates or filters across gaps >= gap_threshold.
    """
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or fps <= 0:
        return out

    win_len = compute_fps_window_length(fps, polyorder=polyorder)
    dt = (1.0 / fps) if delta is None else delta

    # Find contiguous non-nan intervals
    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        return out

    # Find continuous segment boundaries
    segments = []
    start = None
    gap_count = 0

    for i in range(n):
        if valid_mask[i]:
            if start is None:
                start = i
            gap_count = 0
        else:
            if start is not None:
                gap_count += 1
                if gap_count >= gap_threshold:
                    seg_end = i - gap_count + 1
                    if seg_end - start >= win_len:
                        segments.append((start, seg_end))
                    start = None
                    gap_count = 0

    if start is not None:
        seg_end = n - gap_count
        if seg_end - start >= win_len:
            segments.append((start, seg_end))

    # Apply filter on each isolated continuous segment
    for s_start, s_end in segments:
        seg_data = values[s_start:s_end].copy()
        nans = np.isnan(seg_data)
        if np.any(nans):
            x = np.arange(len(seg_data))
            seg_data = np.interp(x, x[~nans], seg_data[~nans])
            
        smoothed = savgol_filter(seg_data, window_length=win_len, polyorder=polyorder, deriv=deriv, delta=dt)
        out[s_start:s_end] = smoothed

    return out


def compute_angular_derivatives(theta_angles: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    [G4] Unwraps phase before computing angular velocity and acceleration.
    Returns: (unwrapped_theta, angular_velocity_rad_per_sec, angular_accel_rad_per_sec2)
    """
    valid = ~np.isnan(theta_angles)
    if not np.any(valid):
        n = len(theta_angles)
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

    # Unwrap phase across valid points
    unwrapped = theta_angles.copy()
    valid_indices = np.where(valid)[0]
    unwrapped[valid_indices] = np.unwrap(theta_angles[valid_indices])

    # Smooth unwrapped angle and compute derivatives
    smooth_theta = smooth_and_differentiate_series(unwrapped, fps, deriv=0)
    omega = smooth_and_differentiate_series(unwrapped, fps, deriv=1)
    alpha = smooth_and_differentiate_series(unwrapped, fps, deriv=2)

    return smooth_theta, omega, alpha


def compute_normalized_jerk(coords: np.ndarray, torso_lengths: np.ndarray, fps: float, in_frames: bool = True) -> np.ndarray:
    """
    [G6] Resolution-Agnostic Jerk Index:
    Computes 3rd derivative of trajectory and normalizes by torso length.
    If in_frames=True, returns in torso-lengths / frame^3 (operational target: < 0.15).
    """
    n = len(coords)
    jerk_norm = np.full(n, np.nan, dtype=np.float64)
    dt = 1.0 if in_frames else (1.0 / fps)

    jerk_sq = np.zeros(n)
    for dim in range(2):
        pos_dim = coords[:, dim]
        jerk_dim = smooth_and_differentiate_series(pos_dim, fps, polyorder=4, deriv=3, delta=dt)
        jerk_sq += np.nan_to_num(jerk_dim**2, 0.0)

    jerk_mag = np.sqrt(jerk_sq)
    valid_torso = (torso_lengths > 1e-3) & (~np.isnan(coords[:, 0]))
    jerk_norm[valid_torso] = jerk_mag[valid_torso] / torso_lengths[valid_torso]
    return jerk_norm


def compute_normalized_pck_error(pred_pts: np.ndarray, gt_pts: np.ndarray, gt_nose: np.ndarray, gt_tail: np.ndarray) -> np.ndarray:
    """
    [G7] Formal PCK Normalization:
    Error divided by ground-truth apparent deck length: ||p_i^pred - p_i^gt|| / ||p_N^gt - p_T^gt||.
    """
    denom = np.linalg.norm(gt_nose - gt_tail, axis=-1)
    denom = np.maximum(denom, 1e-3)
    num = np.linalg.norm(pred_pts - gt_pts, axis=-1)
    return num / denom
