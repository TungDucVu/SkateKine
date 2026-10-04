"""
Phase 3 Step 3.4: Model A — Calibrated Kinematic Rule-Based Baseline.
Deterministic, physically grounded decision tree with frozen calibrated thresholds.
"""

from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import numpy as np


@dataclass
class RuleClassifierResult:
    predicted_class: str
    confidence: float
    decision_path: List[str]
    rule_features: Dict[str, float]


class CalibratedRuleClassifier:
    """
    Model A: Physical Kinematic Decision Tree.
    Evaluates net in-plane rotation, flip cycle count, flick direction, and body yaw.
    """

    def __init__(
        self,
        yaw_180_min: float = 110.0,
        yaw_180_max: float = 250.0,
        yaw_360_min: float = 290.0,
        flip_aspect_thresh: float = 0.42,
        body_yaw_thresh: float = 110.0
    ):
        self.yaw_180_min = yaw_180_min
        self.yaw_180_max = yaw_180_max
        self.yaw_360_min = yaw_360_min
        self.flip_aspect_thresh = flip_aspect_thresh
        self.body_yaw_thresh = body_yaw_thresh

    def predict_one(self, feat: Dict[str, Any]) -> RuleClassifierResult:
        """
        Classifies a single attempt using calibrated rule thresholds.
        """
        path = []

        # 1. Extract physical discriminators
        # Use canonical rotation (normalized by stance) [P3-A2, P3-A3]
        net_yaw = feat.get('canonical_delta_theta_net', feat.get('delta_theta_net', 0.0))
        abs_yaw = abs(net_yaw)
        theta_abs = feat.get('theta_abs', 0.0)
        flip_cycles = feat.get('flip_cycle_count', 0.0)
        min_aspect = feat.get('min_aspect', 1.0)
        flick_dir_x = feat.get('flick_direction_x', 0.0)
        flick_dx = feat.get('flick_dx', 0.0)
        body_yaw = feat.get('skater_torso_yaw', 0.0)
        apex_disp = feat.get('apex_displacement_px', 0.0)

        # 2. Flip detection
        has_flip = (flip_cycles >= 1.0) or (min_aspect <= self.flip_aspect_thresh)
        if has_flip:
            path.append(f"FlipDetected(cycles={flip_cycles}, min_ar={min_aspect:.2f})")
        else:
            path.append(f"NoFlip(min_ar={min_aspect:.2f})")

        # 3. Body 180 rotation check (Body yaw)
        has_body_180 = (abs(body_yaw) >= self.body_yaw_thresh)

        # 4. Decision Logic Matrix across 9 Classes
        # Case 1: 360 Rotation
        if abs_yaw >= self.yaw_360_min or theta_abs >= 320.0:
            path.append(f"Rotation360(abs_yaw={abs_yaw:.1f})")
            if has_flip:
                pred = "360 Flip"
                conf = 0.88
                path.append("Classified: 360 Flip (Tre Flip)")
            else:
                pred = "Pop Shove-it" # Fallback if no flip detected on high shuvit
                conf = 0.70
                path.append("Classified: Pop Shove-it (High rotation)")

        # Case 2: 180 Rotation
        elif self.yaw_180_min <= abs_yaw <= self.yaw_180_max or (120.0 <= theta_abs <= 260.0 and abs_yaw >= 90.0):
            path.append(f"Rotation180(abs_yaw={abs_yaw:.1f}, net={net_yaw:.1f})")
            if has_body_180:
                if net_yaw > 0:
                    pred = "Frontside 180"
                    conf = 0.85
                    path.append("Classified: Frontside 180 (Synchronized FS Body+Board)")
                else:
                    pred = "Backside 180"
                    conf = 0.85
                    path.append("Classified: Backside 180 (Synchronized BS Body+Board)")
            elif has_flip:
                pred = "Varial / Hardflip"
                conf = 0.82
                path.append("Classified: Varial / Hardflip (Coupled 180 Yaw + 360 Roll)")
            else:
                if net_yaw > 0:
                    pred = "Frontside Shove-it"
                    conf = 0.85
                    path.append("Classified: Frontside Shove-it (Board +180 FS yaw)")
                else:
                    pred = "Pop Shove-it"
                    conf = 0.85
                    path.append("Classified: Pop Shove-it (Board -180 BS yaw)")

        # Case 3: Zero or Minimal Board Yaw (< 100 deg)
        else:
            path.append(f"MinimalYaw(abs_yaw={abs_yaw:.1f})")
            if has_flip:
                # Distinguish Kickflip vs Heelflip using flick kinematics [P3-A3]
                if flick_dir_x > 0 or flick_dx > 0:
                    pred = "Kickflip"
                    conf = 0.86
                    path.append(f"Classified: Kickflip (Toe-side flick dx={flick_dx:.1f})")
                else:
                    pred = "Heelflip"
                    conf = 0.84
                    path.append(f"Classified: Heelflip (Heel-side flick dx={flick_dx:.1f})")
            else:
                pred = "Ollie"
                conf = 0.90
                path.append("Classified: Ollie (Ballistic pop, 0 yaw, 0 flip)")

        return RuleClassifierResult(
            predicted_class=pred,
            confidence=conf,
            decision_path=path,
            rule_features={
                'net_yaw': net_yaw,
                'abs_yaw': abs_yaw,
                'theta_abs': theta_abs,
                'min_aspect': min_aspect,
                'flick_dx': flick_dx
            }
        )

    def predict(self, feature_list: List[Dict[str, Any]]) -> List[RuleClassifierResult]:
        return [self.predict_one(f) for f in feature_list]
