"""
Impact-Driven Landing Detector: Implements Amendment [A5]
(Causal kinematic sequence: Descent -> Ground Proximity -> Impact Shockwave -> Rollout)
to eliminate false landing triggers during airborne stalls.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional


class ImpactLandingDetector:
    """
    Detects t_land using a causal physical chain [A5]:
    1. Descent Phase (vel_z_deck < -epsilon)
    2. Full Ground Return (altitude descends to true landing baseline)
    3. Deceleration Shockwave (acc_z_deck impact peak or velocity zero-rebound)
    4. Post-Contact Rollout Stabilization
    """

    def __init__(
        self,
        descent_vel_threshold: float = -80.0,
        ground_margin_px: float = 25.0,
        shock_acc_threshold: float = 3000.0,
        max_descent_duration_sec: float = 0.85
    ):
        self.descent_vel_threshold = descent_vel_threshold
        self.ground_margin_px = ground_margin_px
        self.shock_acc_threshold = shock_acc_threshold
        self.max_descent_duration_sec = max_descent_duration_sec

    def detect_landing(
        self,
        df: pd.DataFrame,
        fps: float,
        t_apex: int
    ) -> int:
        """
        Locates t_land post-apex within the causal ballistic descent window [A5].
        Ensures the board has completed descent to the actual ground baseline before marking rollout.
        Returns: t_land_frame
        """
        n = len(df)
        max_descent_frames = int(round(self.max_descent_duration_sec * fps))
        search_start = min(n - 1, t_apex + 2)
        search_end = min(n - 1, t_apex + max_descent_frames)

        if search_start >= search_end:
            return min(n - 1, t_apex + max(2, int(round(0.20 * fps))))

        deck_alt = df['deck_altitude'].values
        vel_z = df['vel_z_deck'].values
        acc_deck = df['acc_z_deck'].values
        acc_hip = df['acc_z_mid_hip'].values

        # 1. Establish the minimum altitude reached during the post-apex descent
        sub_alt = deck_alt[search_start:search_end]
        valid_sub = sub_alt[~np.isnan(sub_alt)]
        if len(valid_sub) == 0:
            return search_start + (search_end - search_start) // 2

        min_post_apex_alt = float(np.min(valid_sub))
        # Ground touchdown band: within ground_margin_px of the lowest post-apex point
        ground_touchdown_band = min_post_apex_alt + self.ground_margin_px

        # 2. Track descent and look for the ground touchdown impact
        has_descended = False
        best_land = None

        for t in range(search_start, search_end):
            v = vel_z[t] if not np.isnan(vel_z[t]) else 0.0
            alt = deck_alt[t] if not np.isnan(deck_alt[t]) else 0.0
            a_d = acc_deck[t] if not np.isnan(acc_deck[t]) else 0.0
            a_h = acc_hip[t] if not np.isnan(acc_hip[t]) else 0.0

            if v < self.descent_vel_threshold:
                has_descended = True

            # Genuine landing requires having descended AND being physically in the ground band
            if has_descended and alt <= ground_touchdown_band:
                shockwave = max(0.0, a_d) + max(0.0, a_h)
                # Landing impact occurs when velocity rebounds from descent (v >= -150) or shockwave spike fires
                if (v >= -150.0) or (shockwave >= self.shock_acc_threshold):
                    best_land = t
                    break

        if best_land is not None:
            return int(best_land)

        # Fallback: frame of minimum altitude within search range
        min_idx = search_start + int(np.nanargmin(sub_alt))
        return int(min_idx)


