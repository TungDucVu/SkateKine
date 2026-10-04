"""
Sequence Constraint Validator: Validates physical progression of temporal events
(t_pop < t_apex < t_catch <= t_land) and detects physical anomalies or temporal inversions.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TemporalEventResult:
    clip_id: str
    t_pop: int
    t_apex: int
    t_catch: Optional[int]
    t_land: int
    fps: float
    flight_duration_frames: int
    flight_duration_ms: float
    pop_to_apex_ms: float
    apex_to_land_ms: float
    catch_confidence: float
    is_valid_monotonic: bool
    error_flag: Optional[str]


class TemporalSequenceValidator:
    """
    Enforces strict physical temporal ordering:
    t_pop < t_apex < t_catch <= t_land.
    Flags temporal inversions, physically impossible flight durations, or anomalies.
    """

    def __init__(
        self,
        min_flight_duration_sec: float = 0.15,  # Min flight: 150ms
        max_flight_duration_sec: float = 1.20   # Max flight: 1200ms
    ):
        self.min_flight_duration_sec = min_flight_duration_sec
        self.max_flight_duration_sec = max_flight_duration_sec

    def validate(
        self,
        clip_id: str,
        t_pop: int,
        t_apex: int,
        t_catch: Optional[int],
        t_land: int,
        catch_confidence: float,
        fps: float
    ) -> TemporalEventResult:
        error_flag = None
        is_valid = True

        # 1. Monotonic ordering checks
        if not (t_pop < t_apex):
            is_valid = False
            error_flag = "POP_AFTER_APEX_INVERSION"
        elif not (t_apex < t_land):
            is_valid = False
            error_flag = "APEX_AFTER_LAND_INVERSION"
        elif t_catch is not None and not (t_pop < t_catch <= t_land):
            is_valid = False
            error_flag = "CATCH_OUT_OF_BOUNDS"

        # 2. Flight duration physical sanity checks
        flight_frames = max(0, t_land - t_pop)
        flight_sec = flight_frames / fps if fps > 0 else 0.0
        flight_ms = flight_sec * 1000.0

        pop_to_apex_ms = (t_apex - t_pop) / fps * 1000.0 if fps > 0 else 0.0
        apex_to_land_ms = (t_land - t_apex) / fps * 1000.0 if fps > 0 else 0.0

        if flight_sec < self.min_flight_duration_sec:
            is_valid = False
            error_flag = f"FLIGHT_TOO_SHORT_{flight_ms:.0f}ms"
        elif flight_sec > self.max_flight_duration_sec:
            is_valid = False
            error_flag = f"FLIGHT_TOO_LONG_{flight_ms:.0f}ms"

        return TemporalEventResult(
            clip_id=clip_id,
            t_pop=t_pop,
            t_apex=t_apex,
            t_catch=t_catch,
            t_land=t_land,
            fps=fps,
            flight_duration_frames=flight_frames,
            flight_duration_ms=flight_ms,
            pop_to_apex_ms=pop_to_apex_ms,
            apex_to_land_ms=apex_to_land_ms,
            catch_confidence=catch_confidence,
            is_valid_monotonic=is_valid,
            error_flag=error_flag
        )
