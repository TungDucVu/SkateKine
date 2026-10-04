"""
Phase 2 Integrated Temporal Segmentation Pipeline: Integrates signal conditioning,
multi-cue pop detection [A1], canonical z-up apex detection [A2], persistent catch detection [A3],
impact-driven landing detection [A5], sequence validation, and continuous window slicing [A6].
"""

import os
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

from src.segmentation.signal_processor import KinematicSignalProcessor
from src.segmentation.event_detector import PopApexDetector
from src.segmentation.catch_detector import PersistentCatchDetector
from src.segmentation.landing_detector import ImpactLandingDetector
from src.segmentation.validator import TemporalSequenceValidator, TemporalEventResult
from src.segmentation.slicer import ContinuousWindowSlicer, PhaseSlices


@dataclass
class Phase2SegmentationOutput:
    clip_id: str
    events: TemporalEventResult
    slices: PhaseSlices
    conditioned_df: pd.DataFrame


class Phase2Pipeline:
    """
    End-to-end Phase 2 Temporal Event Localization and Phase Segmentation Engine.
    """

    def __init__(
        self,
        filter_window_sec: float = 0.075,
        tau_persist_sec: float = 0.04,
        tau_approach_sec: float = 0.25,
        tau_rollout_sec: float = 0.35
    ):
        self.signal_processor = KinematicSignalProcessor(filter_window_sec=filter_window_sec)
        self.event_detector = PopApexDetector()
        self.catch_detector = PersistentCatchDetector(tau_persist_sec=tau_persist_sec)
        self.landing_detector = ImpactLandingDetector()
        self.validator = TemporalSequenceValidator()
        self.slicer = ContinuousWindowSlicer(tau_approach_sec=tau_approach_sec, tau_rollout_sec=tau_rollout_sec)

    def process_trajectory(
        self,
        df_traj: pd.DataFrame,
        fps: float,
        clip_id: str = "unknown"
    ) -> Phase2SegmentationOutput:
        """
        Processes a Phase 1 trajectory DataFrame and extracts physical temporal events and phase slices.
        """
        # 1. Kinematic Signal Conditioning & Canonical Z-Up Framing [A2]
        df_cond = self.signal_processor.process(df_traj, fps=fps)

        # 2. Apex Localization [A2]
        t_apex = self.event_detector.detect_apex(df_cond, fps=fps)

        # 3. Multi-Cue Pop Localization [A1]
        t_pop = self.event_detector.detect_pop(df_cond, fps=fps, t_apex=t_apex)

        # 4. Impact-Driven Landing Localization [A5]
        t_land = self.landing_detector.detect_landing(df_cond, fps=fps, t_apex=t_apex)

        # 5. Temporally Persistent Catch Localization [A3]
        t_catch, catch_conf = self.catch_detector.detect_catch(
            df_cond, fps=fps, t_apex=t_apex, t_land_cand=t_land
        )

        # 6. Physical Sequence Validation
        events = self.validator.validate(
            clip_id=clip_id,
            t_pop=t_pop,
            t_apex=t_apex,
            t_catch=t_catch,
            t_land=t_land,
            catch_confidence=catch_conf,
            fps=fps
        )

        # 7. Time-Scaled Window Slicing [A6]
        slices = self.slicer.slice_phases(
            df=df_cond,
            fps=fps,
            t_pop=t_pop,
            t_apex=t_apex,
            t_catch=t_catch,
            t_land=t_land
        )

        return Phase2SegmentationOutput(
            clip_id=clip_id,
            events=events,
            slices=slices,
            conditioned_df=df_cond
        )

    def process_clip_by_id(
        self,
        clip_id: str,
        features_dir: str = "features/v1_trajectories",
        manifest_path: str = "data/metadata/video_manifest.csv"
    ) -> Phase2SegmentationOutput:
        """Helper to run directly from clip_id and parquet feature store."""
        parquet_path = os.path.join(features_dir, f"{clip_id}.parquet")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Trajectory file not found: {parquet_path}")

        df_manifest = pd.read_csv(manifest_path)
        row = df_manifest[df_manifest['clip_id'] == clip_id].iloc[0]
        fps = float(row['fps'])

        df_traj = pd.read_parquet(parquet_path)
        return self.process_trajectory(df_traj, fps=fps, clip_id=clip_id)
