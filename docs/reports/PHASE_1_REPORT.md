# Phase 1 Execution & Quality Control Report

**Phase:** Phase 1 — Video Ingestion, Spatial Tracking & Canonical Feature Store  
**Status:** In Progress / Pending Execution  
**Target Gate:** Gate 1 (Tracking Stability & PCK@0.10)  

---

## 1. Executive Summary & Work Completed
* **Objective:** Establish the perceptual backbone by extracting high-fidelity 2D pose and 8-point skateboard trajectories from monocular video, normalizing into scale-invariant frames, and persisting structured Parquet feature stores.
* **Work Completed:**
  - [ ] Implement video ingestion pipeline and frame-rate validator (`src/tracking/ingestion.py`).
  - [ ] Generate skater-wise train/val/test splits in `data/metadata/video_manifest.csv`.
  - [ ] Deploy primary skater isolation and pose tracking (`src/tracking/pose_tracker.py`).
  - [ ] Train/fine-tune 8-point skateboard OBB detector (`src/tracking/board_tracker.py`).
  - [ ] Construct canonical hip-centered, torso-normalized, and board-local transformation engine (`src/coordinates/transformer.py`).
  - [ ] Implement Savitzky-Golay filtering and analytical derivative computation (`src/coordinates/feature_store.py`).
* **Deliverable Artifacts:**
  - Video Manifest: `data/metadata/video_manifest.csv`
  - Feature Store: `features/v1_trajectories/*.parquet`
  - Unit Tests: `tests/test_tracking_continuity.py`, `tests/test_coordinate_transforms.py`

---

## 2. Explicit Assumptions Made
1. **Camera Angle:** Video input is restricted to near-side-profile viewpoints within $\pm 15^\circ$ of perpendicular to the travel vector.
2. **Standard Deck Dimension:** Deck length is assumed to be $80.0\text{ cm}$ nominal, providing the metric scale factor $s(t)$ when parallel to the camera sensor.
3. **Primary Skater Isolation:** The target skater is the human bounding box exhibiting highest overlap with the skateboard bounding polygon prior to $t_{\text{pop}}$.
4. **Anatomical Offset:** Foot landmarks (ankles) reside above the deck contact plane by approximately $4\text{--}8\text{ cm}$ due to shoe soles and joint positioning.
5. **Temporal Continuity:** Board occlusions during rapid edge-on flips persist for $\le 4$ consecutive frames, allowing Kalman and optical flow interpolation.

---

## 3. Systems & Artifacts Built on Assumptions
* **Pixel-to-Centimeter Converter:** $s(t) = \frac{80.0}{\Vert\mathbf{p}_N(t) - \mathbf{p}_T(t)\Vert_2}$ enabled by the physical deck length assumption.
* **Board-Centric Coordinate Frame:** Foot projection $\mathbf{p}_{\text{foot}}^{\text{board}}(t) = \mathbf{R}(-\theta_{\text{board}})\left(\mathbf{p}_{\text{foot}} - \mathbf{C}_{\text{board}}\right)$ built on the rigid 2D planar deck assumption.
* **Kalman Edge-on Compensator:** Predictor bridge activating when board aspect ratio drops below $0.15$ during rapid flip rotations.

---

## 4. Quality Control (QC) & Evaluation Results

| Metric Name | Operational Gate Target | Aspirational Target | Expected Result | Actual Achieved Result | QC Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tracking Frame Drop Rate** | $< 3.0\%$ | $< 1.0\%$ | $< 2.0\%$ | *Pending test run* | PENDING |
| **Board Keypoint $\text{PCK}@0.10$** | $> 0.92$ | $> 0.96$ | $> 0.94$ | *Pending evaluation* | PENDING |
| **Board Keypoint $\text{PCK}@0.05$** | $> 0.85$ | $> 0.90$ | $> 0.87$ | *Pending evaluation* | PENDING |
| **Pose Jitter (Jerk Index)** | $< 15.0\text{ px/frame}^3$ | $< 8.0\text{ px/frame}^3$ | $< 10.0\text{ px/frame}^3$ | *Pending evaluation* | PENDING |
| **Skater Split Leakage** | $0.0\%$ | $0.0\%$ | $0.0\%$ | *Pending test run* | PENDING |

---

## 5. Failure Analysis, Edge Cases & Mitigation
* **Optical Failure:** Motion blur during fast flips mitigated via directional optical flow interpolation.
* **Tracking Failure:** Nose/tail keypoint flip-inversion mitigated via temporal vector continuity check against velocity vector.
* **Occlusion Failure:** Ankle landmark drop during catch handled via Kalman filtering and confidence gating.

---

## 6. Phase Gate Sign-off
* **Gate Decision:** PENDING (Awaiting test set execution and metric verification)
* **Approver:** Lead Research Engineer
* **Next Step:** Advance to Phase 2 upon achieving $\text{PCK}@0.10 > 0.92$ and Drop Rate $< 3.0\%$.
