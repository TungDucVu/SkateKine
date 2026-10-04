"""
Skater Pose Tracker (Phase 1, Step 1.2)
Extracts 17 COCO skeletal landmarks for the primary skater using YOLO-Pose.
Includes tracklet association and subject lock to suppress background spectators.
"""

from dataclasses import dataclass
import numpy as np
from ultralytics import YOLO


COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


@dataclass
class SkaterPoseFrame:
    frame_idx: int
    detected: bool
    keypoints: np.ndarray        # Shape: (17, 2) in image pixel coords (x, y)
    confidence: np.ndarray       # Shape: (17,) confidence scalars in [0, 1]
    bbox: np.ndarray             # Shape: (4,) [x1, y1, x2, y2]
    mid_hip: np.ndarray          # Shape: (2,) [x, y]
    mid_shoulder: np.ndarray     # Shape: (2,) [x, y]
    torso_length: float          # Torso scale normalizer in pixels


class SkaterPoseTracker:
    def __init__(self, model_weights: str = 'yolov8n-pose.pt', conf_thresh: float = 0.35):
        self.model = YOLO(model_weights)
        self.conf_thresh = conf_thresh
        self.prev_bbox = None

    def _select_primary_skater(self, boxes, keypoints_list) -> int:
        """
        Locks onto the primary skater using bounding box size,
        confidence, and temporal proximity to previous position.
        """
        if len(boxes) == 1:
            return 0

        scores = []
        for i, (box, kp) in enumerate(zip(boxes, keypoints_list)):
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            
            # Lower limb confidence score (hips, knees, ankles)
            lower_limb_confs = kp[11:, 2].mean()

            # Temporal IoU/proximity score if previous box exists
            if self.prev_bbox is not None:
                # Intersection over Union
                ix1 = max(xyxy[0], self.prev_bbox[0])
                iy1 = max(xyxy[1], self.prev_bbox[1])
                ix2 = min(xyxy[2], self.prev_bbox[2])
                iy2 = min(xyxy[3], self.prev_bbox[3])
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter = iw * ih
                union = area + ((self.prev_bbox[2] - self.prev_bbox[0]) * (self.prev_bbox[3] - self.prev_bbox[1])) - inter
                iou = inter / union if union > 0 else 0
                score = iou * 0.5 + conf * 0.25 + (area / 500000.0) * 0.15 + lower_limb_confs * 0.1
            else:
                # First frame: prefer largest human with high limb visibility
                score = conf * 0.4 + (area / 500000.0) * 0.3 + lower_limb_confs * 0.3
            scores.append(score)

        return int(np.argmax(scores))

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> SkaterPoseFrame:
        results = self.model(frame_bgr, verbose=False, conf=self.conf_thresh, imgsz=480)
        
        if not results or len(results[0].boxes) == 0 or results[0].keypoints is None or len(results[0].keypoints.data) == 0:
            return SkaterPoseFrame(
                frame_idx=frame_idx,
                detected=False,
                keypoints=np.full((17, 2), np.nan),
                confidence=np.zeros(17),
                bbox=np.full(4, np.nan),
                mid_hip=np.full(2, np.nan),
                mid_shoulder=np.full(2, np.nan),
                torso_length=np.nan
            )

        boxes = results[0].boxes
        kps_tensor = results[0].keypoints.data.cpu().numpy() # Shape: (N, 17, 3)

        idx = self._select_primary_skater(boxes, kps_tensor)
        best_box = boxes[idx].xyxy[0].cpu().numpy()
        best_kp = kps_tensor[idx] # (17, 3): x, y, conf

        self.prev_bbox = best_box

        coords = best_kp[:, :2]
        confs = best_kp[:, 2]

        # Compute mid-hip (indices 11: left hip, 12: right hip)
        l_hip, r_hip = coords[11], coords[12]
        mid_hip = (l_hip + r_hip) / 2.0 if (confs[11] > 0.2 and confs[12] > 0.2) else l_hip

        # Compute mid-shoulder (indices 5: left shoulder, 6: right shoulder)
        l_sh, r_sh = coords[5], coords[6]
        mid_shoulder = (l_sh + r_sh) / 2.0 if (confs[5] > 0.2 and confs[6] > 0.2) else l_sh

        # Torso length scale factor
        torso_len = float(np.linalg.norm(mid_hip - mid_shoulder))
        if torso_len <= 1e-3 or np.isnan(torso_len):
            torso_len = float(np.linalg.norm(best_box[3] - best_box[1]) * 0.35)

        return SkaterPoseFrame(
            frame_idx=frame_idx,
            detected=True,
            keypoints=coords,
            confidence=confs,
            bbox=best_box,
            mid_hip=mid_hip,
            mid_shoulder=mid_shoulder,
            torso_length=torso_len
        )

    def reset(self):
        self.prev_bbox = None
