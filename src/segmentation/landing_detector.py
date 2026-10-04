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
    2. Ground Proximity (deck_altitude descends towards ground baseline)
    3. Deceleration Shockwave (acc_z_deck or acc_z_mid_hip positive impact peak)
    4. Post-Contact Rollout Stabilization
    """

    def __init__(
        self,
        descent_vel_threshold: float = -80.0,
        ground_proximity_threshold: float = 35.0,
        shock_acc_threshold: float = 2000.0,
        max_descent_duration_sec: float = 0.85
    ):
        self.descent_vel_threshold = descent_vel_threshold
        self.ground_proximity_threshold = ground_proximity_threshold
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

        # Track descent and find earliest touchdown impact
        has_descended = False
        best_land = search_start
        min_descending_alt = 1e9

        for t in range(search_start, search_end):
            v = vel_z[t] if not np.isnan(vel_z[t]) else 0.0
            alt = deck_alt[t] if not np.isnan(deck_alt[t]) else 0.0
            a_d = acc_deck[t] if not np.isnan(acc_deck[t]) else 0.0
            a_h = acc_hip[t] if not np.isnan(acc_hip[t]) else 0.0

            if v < self.descent_vel_threshold:
                has_descended = True

            if has_descended:
                if alt < min_descending_alt:
                    min_descending_alt = alt
                    best_land = t

                # Check for impact shockwave / velocity leveling:
                # If board has descended to near baseline and either rebounds or experiences shock deceleration
                shockwave = max(0.0, a_d) + max(0.0, a_h)
                if best_land is not None and t >= best_land + 1:
                    is_near_baseline = alt <= min_descending_alt + self.ground_proximity_threshold
                    is_impact_or_rebound = (v >= -50.0) or (shockwave >= self.shock_acc_threshold)
                    if is_near_baseline and is_impact_or_rebound:
                        return int(t)

        return int(best_land)

