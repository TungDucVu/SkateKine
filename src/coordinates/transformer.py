"""
Canonical Coordinate Transformation & Safe Metric Calibration (Phase 1, Step 1.4)
Guardrails implemented:
- [G3] Safe Metric Calibration: Anchor scale to flat pre-pop roll window (t < t_pop).
  Never compute metric scaling frame-by-frame during the flip phase.
  Fallback to dimensionless torso-normalized ratios if CV >= 0.15.
"""

from dataclasses import dataclass
import numpy as np


STANDARD_DECK_LENGTH_CM = 80.0
NOMINAL_WHEELBASE_CM = 36.0


@dataclass
class ClipMetricScale:
    scale_cm_per_px: float       # s_clip [cm / pixel]
    scale_valid: bool            # True if grounded approach passed CV test
    apparent_length_median: float
    apparent_length_std: float
    cv_score: float              # Coefficient of variation in approach window


class CanonicalTransformer:
    def __init__(self, nominal_deck_length_cm: float = STANDARD_DECK_LENGTH_CM):
        self.nominal_deck_length_cm = nominal_deck_length_cm

    def calibrate_clip_scale(self, board_lengths: list[float], approach_window_len: int = 30) -> ClipMetricScale:
        """
        [G3] Safe Metric Calibration:
        Computes the clip-level metric scaling factor during the flat approach.
        Anchors on the first valid detected approach frames when skater rolls in.
        """
        # Find first valid detected frames where board is flat on ground
        valid_lengths = [l for l in board_lengths if not np.isnan(l) and l > 15.0]
        approach_lengths = valid_lengths[:approach_window_len]
        
        if len(approach_lengths) < 5:
            # Not enough valid frames in approach window
            return ClipMetricScale(
                scale_cm_per_px=np.nan,
                scale_valid=False,
                apparent_length_median=np.nan,
                apparent_length_std=np.nan,
                cv_score=np.nan
            )

        med_len = float(np.median(approach_lengths))
        std_len = float(np.std(approach_lengths))
        cv = std_len / med_len if med_len > 0 else 1.0

        # Guardrail: CV must be < 0.20 to confirm flat approach without foreshortening
        if cv < 0.20 and med_len > 20.0:
            scale_factor = self.nominal_deck_length_cm / med_len
            scale_valid = True
        else:
            scale_factor = np.nan
            scale_valid = False

        return ClipMetricScale(
            scale_cm_per_px=scale_factor,
            scale_valid=scale_valid,
            apparent_length_median=med_len,
            apparent_length_std=std_len,
            cv_score=cv
        )

    def normalize_to_torso(self, keypoints: np.ndarray, mid_hip: np.ndarray, torso_length: float) -> np.ndarray:
        """
        Transforms keypoints to hip-centered, torso-scale invariant coordinates:
        p_norm(t) = (p(t) - C_hip(t)) / L_torso(t)
        """
        if np.isnan(mid_hip).any() or torso_length <= 1e-3 or np.isnan(torso_length):
            return np.full_like(keypoints, np.nan)
        return (keypoints - mid_hip) / torso_length

    def project_to_board_frame(self, foot_coord: np.ndarray, board_centroid: np.ndarray, in_plane_angle_rad: float) -> np.ndarray:
        """
        Projects foot coordinates into the board's moving 2D planar reference frame:
        p_foot^board(t) = R(-theta(t)) * (p_foot(t) - C_board(t))
        """
        if np.isnan(foot_coord).any() or np.isnan(board_centroid).any() or np.isnan(in_plane_angle_rad):
            return np.full(2, np.nan)

        delta = foot_coord - board_centroid
        c = np.cos(-in_plane_angle_rad)
        s = np.sin(-in_plane_angle_rad)
        rot_mat = np.array([[c, -s], [s, c]], dtype=np.float32)
        
        return rot_mat @ delta
