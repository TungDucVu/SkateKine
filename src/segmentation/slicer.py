"""
Continuous Window Slice Generator: Implements Amendment [A6]
(Time-scaled temporal windows defined in seconds tau, converted dynamically by FPS)
to prevent window distortion across mixed frame rates (60 FPS vs 120 FPS).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PhaseSlices:
    approach_slice: pd.DataFrame
    flight_slice: pd.DataFrame
    landing_slice: pd.DataFrame
    approach_indices: Tuple[int, int]
    flight_indices: Tuple[int, int]
    landing_indices: Tuple[int, int]


class ContinuousWindowSlicer:
    """
    Slices conditioned kinematic trajectories into canonical phase windows
    based on physical continuous seconds (tau), invariant to clip frame rates [A6].
    """

    def __init__(
        self,
        tau_approach_sec: float = 0.25,  # 250ms approach window pre-pop
        tau_rollout_sec: float = 0.35    # 350ms rollout window post-landing
    ):
        self.tau_approach_sec = tau_approach_sec
        self.tau_rollout_sec = tau_rollout_sec

    def slice_phases(
        self,
        df: pd.DataFrame,
        fps: float,
        t_pop: int,
        t_apex: int,
        t_catch: Optional[int],
        t_land: int
    ) -> PhaseSlices:
        n = len(df)
        n_approach = int(round(self.tau_approach_sec * fps))
        n_rollout = int(round(self.tau_rollout_sec * fps))

        # 1. Approach Phase: [max(0, t_pop - n_approach), t_pop]
        app_start = max(0, t_pop - n_approach)
        app_end = t_pop
        approach_slice = df.iloc[app_start:app_end].copy()

        # 2. Flight Phase: [t_pop, t_catch or t_land]
        flight_end = t_catch if t_catch is not None and t_catch > t_pop else t_land
        flight_slice = df.iloc[t_pop:flight_end].copy()

        # 3. Landing & Rollout Phase: [t_land, min(n, t_land + n_rollout)]
        rollout_end = min(n, t_land + n_rollout)
        landing_slice = df.iloc[t_land:rollout_end].copy()

        return PhaseSlices(
            approach_slice=approach_slice,
            flight_slice=flight_slice,
            landing_slice=landing_slice,
            approach_indices=(app_start, app_end),
            flight_indices=(t_pop, flight_end),
            landing_indices=(t_land, rollout_end)
        )
