"""
Pop & Apex Event Boundary Detector: Implements Multi-Cue Pop Formulation [A1]
and Canonical Z-Up Parabolic Apex Zero-Crossing Localization [A2].
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional


class PopApexDetector:
    """
    Detects Pop (t_pop) using multi-cue fusion (tail proximity, ankle impulse, board ascent)
    and Apex (t_apex) using canonical vertical velocity zero-crossing at maximum altitude.
    """

    def __init__(
        self,
        w_tail: float = 0.35,
        w_impulse: float = 0.30,
        w_ascent: float = 0.35,
        sigma_tail: float = 18.0,
        min_flight_duration_sec: float = 0.15
    ):
        self.w_tail = w_tail
        self.w_impulse = w_impulse
        self.w_ascent = w_ascent
        self.sigma_tail = sigma_tail
        self.min_flight_duration_sec = min_flight_duration_sec

    def detect_apex(self, df: pd.DataFrame, fps: float) -> int:
        """
        Locates t_apex as the global maximum of z_deck during flight where
        vertical velocity crosses zero from positive (ascent) to negative (descent) [A2].
        """
        z_deck = df['z_deck'].values
        vel_z = df['vel_z_deck'].values
        n = len(df)

        # Ignore boundary margins (first 10% and last 10% of clip)
        margin = max(5, int(0.10 * n))
        search_range = range(margin, n - margin)

        # 1. Primary candidate: Global altitude peak in search range
        best_apex = margin
        max_z = -1e9
        for i in search_range:
            if not np.isnan(z_deck[i]) and z_deck[i] > max_z:
                max_z = z_deck[i]
                best_apex = i

        # 2. Refine candidate around velocity zero-crossing
        # Look within +/- 10 frames of max altitude for vel_z zero crossing
        refine_window = range(max(margin, best_apex - 10), min(n - margin, best_apex + 10))
        zero_cross_candidates = []
        for i in refine_window:
            if i > 0 and not np.isnan(vel_z[i-1]) and not np.isnan(vel_z[i]):
                if vel_z[i-1] >= 0 and vel_z[i] < 0:
                    zero_cross_candidates.append(i)

        if len(zero_cross_candidates) > 0:
            # Pick zero crossing closest to max altitude peak
            best_apex = min(zero_cross_candidates, key=lambda idx: abs(idx - best_apex))

        return int(best_apex)

    def detect_pop(self, df: pd.DataFrame, fps: float, t_apex: int) -> int:
        """
        Locates t_pop using Multi-Cue Ensemble Formulation [A1]:
        S_pop(t) = w_tail * S_tail(t) + w_impulse * S_impulse(t) + w_ascent * S_ascent(t)
        Searched within the pre-apex window: [margin, t_apex - min_flight_frames].
        """
        n = len(df)
        min_flight_frames = max(3, int(round(self.min_flight_duration_sec * fps)))
        pop_search_end = max(5, t_apex - min_flight_frames)
        margin = max(3, int(0.05 * n))

        if pop_search_end <= margin:
            return max(0, t_apex - min_flight_frames)

        search_indices = np.arange(margin, pop_search_end)

        tail_alt = df['tail_altitude'].values
        nose_alt = df['nose_altitude'].values
        vel_z_deck = df['vel_z_deck'].values
        acc_la = df['acc_z_left_ankle'].values
        acc_ra = df['acc_z_right_ankle'].values

        scores = np.zeros(len(search_indices))

        for idx_pos, t in enumerate(search_indices):
            # 1. Cue 1: Tail or lowest board tip proximity to ground [0, 1]
            t_alt = tail_alt[t] if not np.isnan(tail_alt[t]) else 0.0
            n_alt = nose_alt[t] if not np.isnan(nose_alt[t]) else 0.0
            min_tip_alt = min(t_alt, n_alt)
            # Maximum score when tail is snapped against the ground
            s_tail = np.exp(-max(0.0, min_tip_alt)**2 / (2.0 * (self.sigma_tail**2)))

            # 2. Cue 2: Ankle downward impulse spike [0, 1]
            # When popping, rear foot drives down sharply (negative acc_z)
            a_la = acc_la[t] if not np.isnan(acc_la[t]) else 0.0
            a_ra = acc_ra[t] if not np.isnan(acc_ra[t]) else 0.0
            max_downward_acc = max(0.0, -min(a_la, a_ra))
            s_impulse = np.tanh(max_downward_acc / 1500.0)

            # 3. Cue 3: Board ascent initiation [0, 1]
            # Vertical velocity switches to positive ascent towards apex
            v_deck = vel_z_deck[t] if not np.isnan(vel_z_deck[t]) else 0.0
            s_ascent = np.tanh(max(0.0, v_deck) / 300.0)

            # Fused ensemble score [A1]
            scores[idx_pos] = (
                self.w_tail * s_tail +
                self.w_impulse * s_impulse +
                self.w_ascent * s_ascent
            )

        best_idx_pos = int(np.argmax(scores))
        t_pop = int(search_indices[best_idx_pos])
        return t_pop
