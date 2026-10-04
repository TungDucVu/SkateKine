"""
Semantic 8-Point Skateboard Tracker (Phase 1, Step 1.3)
Guardrails implemented:
- [G1] Semantic 8-point topology: Nose (N), Tail (T), FL, FR, RL, RR, Front Truck (FT), Rear Truck (RT).
  Includes temporal identity tracking between Nose and Tail to prevent 180 flip swaps.
- [G5] Provenance state tracking:
  0: DETECTED, 1: OPTICAL_FLOW, 2: KALMAN_PREDICTED, 3: LINEAR_INTERPOLATED.
"""

from dataclasses import dataclass
from enum import IntEnum
import cv2
import numpy as np
from ultralytics import YOLO


class TrackingProvenance(IntEnum):
    DETECTED = 0
    OPTICAL_FLOW = 1
    KALMAN_PREDICTED = 2
    LINEAR_INTERPOLATED = 3
    LOST = -1


BOARD_KEYPOINT_NAMES = [
    'nose', 'tail', 'front_left', 'front_right',
    'rear_left', 'rear_right', 'front_truck', 'rear_truck'
]


@dataclass
class BoardStateFrame:
    frame_idx: int
    detected: bool
    keypoints: np.ndarray         # Shape: (8, 2) [x, y] in image coords
    confidence: float             # Overall detection confidence [0, 1]
    provenance: TrackingProvenance
    apparent_length: float        # Distance ||N - T|| in pixels
    apparent_width: float         # Width across perpendicular axis
    aspect_ratio: float           # width / length (proxy for vertical flip state)
    in_plane_angle: float         # atan2(yN - yT, xN - xT) in radians
    centroid: np.ndarray          # (2,) [x, y]


class SemanticBoardTracker:
    def __init__(self, model_weights: str = 'yolov8n.pt', conf_thresh: float = 0.25):
        self.model = YOLO(model_weights)
        self.conf_thresh = conf_thresh
        
        # State tracking
        self.prev_gray = None
        self.prev_keypoints = None
        self.prev_angle = None
        self.forward_direction = None # Velocity vector for initial nose/tail assignment
        
        # Kalman filter for (xN, yN, xT, yT, vxN, vyN, vxT, vyT)
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = 1.0 # dx/dt
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        cv2.setIdentity(self.kf.processNoiseCov, 1e-2)
        cv2.setIdentity(self.kf.measurementNoiseCov, 1e-1)
        cv2.setIdentity(self.kf.errorCovPost, 1.0)
        self.kf_initialized = False

    def _extract_8_points_from_box(self, frame_bgr: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, float, float]:
        """
        Extracts 8 semantic keypoints from the detected skateboard region using
        color segmentation and principal component contour analysis.
        """
        x1, y1, x2, y2 = map(int, box)
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            hw, hh = (x2 - x1) / 2.0, (y2 - y1) / 2.0
            p1 = np.array([cx - hw, cy])
            p2 = np.array([cx + hw, cy])
            kps = self._derive_8_points(p1, p2, width=hh * 0.5)
            return kps, float(2 * hw), float(hh * 0.5)

        # Grayscale and edge/threshold gradient
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_crop, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        # Minimum area rotated rectangle
        pts = cv2.findNonZero(edges)
        if pts is not None and len(pts) >= 6:
            rot_rect = cv2.minAreaRect(pts)
        else:
            rot_rect = ((crop.shape[1] / 2.0, crop.shape[0] / 2.0), (crop.shape[1], crop.shape[0]), 0)

        (rcx, rcy), (rw, rh), angle = rot_rect
        center_global = np.array([rcx + x1, rcy + y1])

        # Principal longitudinal axis
        if rw < rh:
            length, width = rh, max(4.0, rw)
            rad = np.deg2rad(angle + 90)
        else:
            length, width = rw, max(4.0, rh)
            rad = np.deg2rad(angle)

        unit_long = np.array([np.cos(rad), np.sin(rad)])
        tip_a = center_global + unit_long * (length / 2.0)
        tip_b = center_global - unit_long * (length / 2.0)

        # [G1] Temporal Nose/Tail Identity Locking
        tip_1, tip_2 = self._assign_nose_and_tail(tip_a, tip_b)
        kps = self._derive_8_points(tip_1, tip_2, width)
        return kps, float(length), float(width)

    def _assign_nose_and_tail(self, tip_a: np.ndarray, tip_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Maintains temporal identity of Nose vs Tail across flips [G1].
        """
        if self.prev_keypoints is None:
            # First frame: tip with greater x (forward travel) or motion is Nose
            if tip_a[0] >= tip_b[0]:
                return tip_a, tip_b
            return tip_b, tip_a

        prev_nose = self.prev_keypoints[0]
        prev_tail = self.prev_keypoints[1]

        # Cost matrix comparing (tip_a, tip_b) to (prev_nose, prev_tail)
        d_a_nose = np.linalg.norm(tip_a - prev_nose)
        d_b_tail = np.linalg.norm(tip_b - prev_tail)
        d_normal = d_a_nose + d_b_tail

        d_b_nose = np.linalg.norm(tip_b - prev_nose)
        d_a_tail = np.linalg.norm(tip_a - prev_tail)
        d_swapped = d_b_nose + d_a_tail

        if d_normal <= d_swapped:
            return tip_a, tip_b
        return tip_b, tip_a

    def _derive_8_points(self, nose: np.ndarray, tail: np.ndarray, width: float) -> np.ndarray:
        """
        Derives full 8-point semantic geometry:
        Nose, Tail, FL, FR, RL, RR, Front Truck (22.5% inset), Rear Truck (22.5% inset).
        """
        long_vec = nose - tail
        length = np.linalg.norm(long_vec)
        if length < 1e-3:
            u_long = np.array([1.0, 0.0])
            u_perp = np.array([0.0, 1.0])
        else:
            u_long = long_vec / length
            u_perp = np.array([-u_long[1], u_long[0]])

        hw = width / 2.0
        
        # 6 Outer contour points
        fl = nose - u_long * (length * 0.15) + u_perp * hw
        fr = nose - u_long * (length * 0.15) - u_perp * hw
        rl = tail + u_long * (length * 0.15) + u_perp * hw
        rr = tail + u_long * (length * 0.15) - u_perp * hw

        # 2 Truck pivot anchors (conforming to nominal 36cm wheelbase on 80cm deck => 22.5% inset)
        ft = nose - u_long * (length * 0.225)
        rt = tail + u_long * (length * 0.225)

        return np.array([nose, tail, fl, fr, rl, rr, ft, rt], dtype=np.float32)

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int, skater_box: np.ndarray = None) -> BoardStateFrame:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Run YOLO detection for skateboard (COCO class 36)
        results = self.model(frame_bgr, verbose=False, conf=self.conf_thresh, classes=[36], imgsz=480)
        
        best_box = None
        best_conf = 0.0
        
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                # Check spatial plausibility relative to skater feet if skater_box is given
                if skater_box is not None and not np.isnan(skater_box).any():
                    # Skateboard should be in the lower half of or below skater bounding box
                    sk_bottom = skater_box[3]
                    board_cy = (xyxy[1] + xyxy[3]) / 2.0
                    if board_cy < (skater_box[1] + (sk_bottom - skater_box[1]) * 0.4):
                        continue # Skip boards above skater waist (false detections on background)
                
                if conf > best_conf:
                    best_conf = conf
                    best_box = xyxy

        # Case 1: Direct YOLO Detection
        if best_box is not None:
            kps, length, width = self._extract_8_points_from_box(frame_bgr, best_box)
            provenance = TrackingProvenance.DETECTED
            
            # Update Kalman Filter with Nose and Tail coordinates
            measurement = np.array([[kps[0, 0]], [kps[0, 1]], [kps[1, 0]], [kps[1, 1]]], dtype=np.float32)
            if not self.kf_initialized:
                self.kf.statePost = np.array([[kps[0, 0]], [kps[0, 1]], [kps[1, 0]], [kps[1, 1]], [0], [0], [0], [0]], dtype=np.float32)
                self.kf_initialized = True
            else:
                self.kf.correct(measurement)

        # Case 2: Optical Flow Fallback [G5]
        elif self.prev_keypoints is not None and self.prev_gray is not None:
            p0 = self.prev_keypoints[:2].reshape(-1, 1, 2).astype(np.float32)
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, winSize=(15, 15), maxLevel=2)
            
            if st is not None and st.sum() == 2:
                nose_flow = p1[0, 0]
                tail_flow = p1[1, 0]
                length = float(np.linalg.norm(nose_flow - tail_flow))
                width = float(length * 0.25)
                kps = self._derive_8_points(nose_flow, tail_flow, width)
                provenance = TrackingProvenance.OPTICAL_FLOW
                best_conf = 0.50
                # Correct Kalman filter with optical flow measurement
                meas = np.array([[nose_flow[0]], [nose_flow[1]], [tail_flow[0]], [tail_flow[1]]], dtype=np.float32)
                if self.kf_initialized:
                    self.kf.correct(meas)
            
            # Case 3: Kalman Prediction Fallback [G5]
            elif self.kf_initialized:
                pred = self.kf.predict()
                nose_pred = np.array([pred[0, 0], pred[1, 0]])
                tail_pred = np.array([pred[2, 0], pred[3, 0]])
                length = float(np.linalg.norm(nose_pred - tail_pred))
                width = float(length * 0.25)
                kps = self._derive_8_points(nose_pred, tail_pred, width)
                provenance = TrackingProvenance.KALMAN_PREDICTED
                best_conf = 0.30
            else:
                # Lost
                kps = np.full((8, 2), np.nan)
                length, width = np.nan, np.nan
                provenance = TrackingProvenance.LOST
                best_conf = 0.0

        else:
            kps = np.full((8, 2), np.nan)
            length, width = np.nan, np.nan
            provenance = TrackingProvenance.LOST
            best_conf = 0.0

        self.prev_gray = gray
        if not np.isnan(kps).any():
            self.prev_keypoints = kps.copy()
            centroid = (kps[0] + kps[1]) / 2.0
            angle = float(np.arctan2(kps[0, 1] - kps[1, 1], kps[0, 0] - kps[1, 0]))
            aspect = float(width / length) if length > 1e-3 else 0.0
            detected = True
        else:
            centroid = np.full(2, np.nan)
            angle = np.nan
            aspect = np.nan
            detected = False

        return BoardStateFrame(
            frame_idx=frame_idx,
            detected=detected,
            keypoints=kps,
            confidence=best_conf,
            provenance=provenance,
            apparent_length=length,
            apparent_width=width,
            aspect_ratio=aspect,
            in_plane_angle=angle,
            centroid=centroid
        )

    def reset(self):
        self.prev_gray = None
        self.prev_keypoints = None
        self.kf_initialized = False
