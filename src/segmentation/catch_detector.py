"""
Persistent Probabilistic Catch Detector: Implements Amendment [A3]
(Temporally Persistent Catch Detection over K consecutive frames) to eliminate
false single-frame optical crossing artifacts.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional


class PersistentCatchDetector:
    """
    Detects t_catch within the descent window [t_apex, t_land] by evaluating
    foot-deck proximity, relative velocity convergence, and angular deceleration,
    enforcing sustained multi-frame physical contact [A3].
    """

    def __init__(
        self,
        w_dist: float = 0.40,
        w_vel: float = 0.35,
        w_alpha: float = 0.25,
        sigma_dist: float = 40.0,
        sigma_vel: float = 150.0,
        sigma_alpha: float = 50.0,
        tau_persist_sec: float = 0.04,  # Sustained contact duration (~40ms) [A3]
        tau_catch_score: float = 0.55
    ):
        self.w_dist = w_dist
        self.w_vel = w_vel
        self.w_alpha = w_alpha
        self.sigma_dist = sigma_dist
        self.sigma_vel = sigma_vel
        self.sigma_alpha = sigma_alpha
        self.tau_persist_sec = tau_persist_sec
        self.tau_catch_score = tau_catch_score

    def detect_catch(
        self,
        df: pd.DataFrame,
        fps: float,
        t_apex: int,
        t_land_cand: int
    ) -> Tuple[Optional[int], float]:
        """
        Locates t_catch between t_apex and t_land_cand.
        Returns: (t_catch_frame, catch_confidence_score)
        """
        n = len(df)
        k_persist = max(2, int(round(self.tau_persist_sec * fps)))

        search_start = min(n - 1, t_apex + 2)
        search_end = min(n - 1, max(search_start + 1, t_land_cand))

        if search_start >= search_end:
            return None, 0.0

        min_ankle_dist = df['min_ankle_dist'].values
        min_rel_vel = df['min_rel_vel'].values
        alpha = np.abs(df['alpha_rad_per_sec2'].values)

        scores = np.zeros(n)

        # 1. Compute instantaneous catch score S_catch(t)
        for t in range(search_start, search_end + 1):
            d = min_ankle_dist[t] if not np.isnan(min_ankle_dist[t]) else 100.0
            v = min_rel_vel[t] if not np.isnan(min_rel_vel[t]) else 300.0
            a = alpha[t] if not np.isnan(alpha[t]) else 100.0

            s_d = np.exp(- (d**2) / (2.0 * (self.sigma_dist**2)))
            s_v = np.exp(- (v**2) / (2.0 * (self.sigma_vel**2)))
            s_a = np.exp(- (a**2) / (2.0 * (self.sigma_alpha**2)))

            scores[t] = self.w_dist * s_d + self.w_vel * s_v + self.w_alpha * s_a

        # 2. Enforce Temporal Persistence Window [A3]:
        # S_catch(t) >= tau_catch_score for all frames in [t, t + k_persist]
        best_t = None
        best_score = 0.0

        for t in range(search_start, search_end - k_persist + 1):
            window_scores = scores[t:t + k_persist]
            min_window_score = float(np.min(window_scores))
            mean_window_score = float(np.mean(window_scores))

            if min_window_score >= self.tau_catch_score:
                # Verify foot-board continuity into landing (prevents false catches on bails where feet disconnect before touchdown)
                land_check_frame = min(n - 1, t_land_cand)
                land_dist = min_ankle_dist[land_check_frame] if not np.isnan(min_ankle_dist[land_check_frame]) else 0.0
                if land_dist > 80.0:
                    continue  # Feet disconnected before landing (bail)
                # First valid sustained contact window leading to touchdown
                return int(t), mean_window_score

            if mean_window_score > best_score:
                best_score = mean_window_score
                best_t = t

        # If strict persistence not met (e.g. bail where board is never caught), return None [A3]
        return None, best_score

