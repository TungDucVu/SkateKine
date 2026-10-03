# Phase 1 Implementation Plan: Video Ingestion, Spatial Tracking & Canonical Feature Store

---

## 1. Overview & Scope

Phase 1 establishes the perceptual foundation of the SkateKine engine. Its primary objective is to ingest raw monocular video, reliably isolate the primary skater and skateboard, track 2D human skeletal landmarks and 8-point skateboard keypoints across aggressive rotations and occlusions, normalize coordinates into scale-invariant frames, and persist structured kinematic states into an Apache Parquet feature store.

### Key Deliverables of Phase 1:
1. **Video Ingestion & Metadata Pipeline:** Frame extraction, FPS/duration verification, and video manifest generator with skater-wise train/val/test splits.
2. **Skater Pose Estimation Module:** Multi-person suppression and continuous 2D skeleton extraction (17 COCO / 33 MediaPipe joints) using RTMPose / MediaPipe.
3. **Skateboard Oriented Bounding Box & 8-Keypoint Tracker:** YOLO-OBB model tracking Nose ($N$), Tail ($T$), 4 outer edges ($FL, FR, RL, RR$), and 2 truck baseplates ($FT, RT$) with optical flow / Kalman fallback.
4. **Canonical Coordinate Normalization:** Transformation of raw pixel coordinates into mid-hip centered, torso-length normalized, and board-local reference frames.
5. **Kinematic Feature Store:** Parquet schema persistence capturing continuous $(x, y, \dot{x}, \dot{y}, \ddot{x}, \ddot{y}, \omega)$ signals and frame-wise confidence scores.

---

## 2. Core Assumptions & Boundary Constraints

All design decisions in this phase rest upon the following explicit assumptions:

1. **Camera Geometry Assumption:** Video clips are recorded from near-side-profile viewpoints ($\pm 15^\circ$ deviation from perpendicular to the skater's travel line) with minimal perspective pitch. 
2. **Physical Deck Uniformity:** Skateboard decks conform to standard street dimensions ($L_{\text{board}} \approx 80.0\text{ cm}$, wheelbase $W_{\text{truck}} \approx 36.0\text{ cm}$). Centimeter scaling is derived from the apparent pixel distance between nose and tail when parallel to the camera.
3. **Dominant Subject Assumption:** The active skater is the largest detected human bounding box intersecting or in immediate proximity to the detected skateboard prior to the pop phase.
4. **Anatomical Keypoint Offsets:** Ankle landmarks do not touch the deck surface due to physical footwear thickness and sensor blur; foot-to-deck proximity must be evaluated probabilistically rather than expecting zero distance.
5. **Frame Rate Lower Bound:** Input video is at least 60 FPS. 30 FPS footage is rejected at ingestion due to severe motion blur during board flips.

---

## 3. Step-by-Step Implementation Sequence

### Step 1.1: Video Ingestion & Manifest Management (`src/tracking/ingestion.py`)
* Implement video reader supporting MP4/MOV via OpenCV / PyAV.
* Extract metadata: resolution, exact FPS, frame count, duration, and color space.
* Validate videos against operational minimum: $\ge 60\text{ FPS}$, $\ge 720\text{p}$, duration 2.0s–10.0s.
* Generate and validate `data/metadata/video_manifest.csv` enforcing leak-proof skater splits (70% train, 15% val, 15% test).

### Step 1.2: Human Pose Estimation & Subject Isolation (`src/tracking/pose_tracker.py`)
* Integrate RTMPose / MediaPipe Pose tracker.
* Implement tracklet association (SORT / ByteTrack logic) to lock onto the primary skater across frame drops and background spectators.
* Extract joint coordinates $(x, y)$ and confidence $c \in [0, 1]$ for 17 keypoints, prioritizing hips, knees, ankles, heels, and toes.

### Step 1.3: Skateboard OBB & 8-Keypoint Detection (`src/tracking/board_tracker.py`)
* Prepare dataset parser for 8-point board annotations (Nose, Tail, FL, FR, RL, RR, Front Truck, Rear Truck).
* Fine-tune YOLOv8-OBB / YOLOv11-OBB or deploy pre-trained detector with keypoint head.
* Implement Kalman Filter / Optical Flow (Lucas-Kanade) interpolation to maintain board trajectory during rapid vertical edge-on flips where surface grip-tape is occluded.

### Step 1.4: Canonical & Board-Local Coordinate Transformation (`src/coordinates/transformer.py`)
* **Torso Normalization:** Compute skater mid-hip position $\mathbf{C}_{\text{hip}}(t) = \frac{\mathbf{p}_{\text{L\_hip}} + \mathbf{p}_{\text{R\_hip}}}{2}$ and torso scale factor $L_{\text{torso}}(t) = \Vert\mathbf{C}_{\text{hip}}(t) - \mathbf{C}_{\text{shoulder}}(t)\Vert_2$.
* Normalize all body keypoints:
  $$\mathbf{p}_{\text{norm}}(t) = \frac{\mathbf{p}(t) - \mathbf{C}_{\text{hip}}(t)}{L_{\text{torso}}(t)}$$
* **Board Orientation Angle:**
  $$\theta_{\text{board}}(t) = \text{atan2}(y_N(t) - y_T(t),\, x_N(t) - x_T(t))$$
* **Board-Local Foot Transformation:** Project foot coordinates into the moving board frame:
  $$\mathbf{p}_{\text{foot}}^{\text{board}}(t) = \mathbf{R}(-\theta_{\text{board}}(t))\left(\mathbf{p}_{\text{foot}}(t) - \mathbf{C}_{\text{board}}(t)\right)$$

### Step 1.5: Kinematic Smoothing & Parquet Storage (`src/coordinates/feature_store.py`)
* Apply Savitzky-Golay filtering (window length 5–7 frames, polynomial order 2) to eliminate high-frequency tracking jitter.
* Compute analytical first derivatives $(\dot{x}, \dot{y})$ and second derivatives $(\ddot{x}, \ddot{y})$.
* Compute angular velocity $\omega_{\text{board}}(t) = \frac{d\theta_{\text{board}}}{dt}$.
* Serialize full temporal trace into Apache Parquet format at `features/v1_trajectories/{clip_id}.parquet`.

---

## 4. Post-Phase Checkpoint: QC, Evaluation & Acceptance Criteria

At the conclusion of Phase 1, the pipeline must be formally tested against the following gate criteria:

### Quality Control (QC) Gates

| Metric / Checkpoint | Hard Operational Gate | Aspirational Target | Verification Method |
| :--- | :--- | :--- | :--- |
| **Tracking Continuity Drop Rate** | $< 3.0\%$ lost frames | $< 1.0\%$ lost frames | Count frames where skater or board confidence drops below 0.3 on test clips. |
| **Board Keypoint Accuracy ($\text{PCK}@0.10$)** | $> 0.92$ | $> 0.96$ | Percentage of Correct Keypoints within $0.10 \times L_{\text{board}}$ on labeled ground truth. |
| **Board Precision ($\text{PCK}@0.05$)** | $> 0.85$ | $> 0.90$ | Percentage of Correct Keypoints within $0.05 \times L_{\text{board}}$. |
| **Pose Jitter (Jerk Index)** | $< 15.0\text{ px/frame}^3$ | $< 8.0\text{ px/frame}^3$ | Mean magnitude of 3rd derivative of hip and ankle trajectories before and after smoothing. |
| **Skater Split Integrity** | 0% skater identity overlap | 0% overlap | Automated unit test verifying train/val/test split disjointness. |

---

## 5. Phase Documentation & Reporting Requirements

Upon completing Phase 1, create `docs/reports/PHASE_1_REPORT.md` using the project's standard reporting format:
1. **Executive Summary:** Overview of models trained, scripts created, and dataset clips processed.
2. **Work Completed:** Granular breakdown of completed tasks (ingestion pipeline, pose tracker, board OBB tracker, coordinate transform, Parquet exporter).
3. **Assumptions Made:** Formal list of assumptions applied during Phase 1 (viewpoint limits, standard deck length, subject detection threshold).
4. **Artifacts Built on Assumptions:** Description of coordinate transforms and scale factors engineered using those assumptions.
5. **QC Results vs. Expected Targets:** Table comparing actual achieved metrics (PCK@0.10, PCK@0.05, frame drop rate, FPS throughput) against Gate 1 criteria.
6. **Failure Analysis & Edge Cases:** Identification of tracking dropouts (e.g. edge-on flip occlusion, loose pants) and mitigating fallbacks.
7. **Sign-off Decision:** Clear GO / NO-GO recommendation for advancing to Phase 2.
