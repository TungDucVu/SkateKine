# SkateKine: Skateboard Trick Recognition & Kinematic Cleanliness Scoring Engine

An interpretable computer vision and kinematic modeling engine designed to recognize skateboard tricks and evaluate execution cleanliness from monocular video.

Instead of end-to-end black-box video classification, SkateKine maps raw video into an inspectable spatio-temporal state space:
```
Raw Video ──> Pose + Board Tracking ──> Canonical Coordinates ──> Temporal Event Segmentation ──> Trick Classification + Land/Bail Gate ──> Kinematic Features ──> Cleanliness Scoring Models ──> Telemetry HUD
```

---

## 📖 Specifications & Architecture Plan

The complete engineering, research, and phase implementation specification is documented in:
👉 **[MASTER_FINAL.md](./MASTER_FINAL.md)**

### Key Modules
1. **Pose & Board Tracking Pipeline**: Multi-person suppression, 2D human pose estimation (ViTPose/RTMPose), oriented bounding box deck tracker (YOLOv8-OBB/RT-DETR).
2. **Canonical Coordinate Normalization**: Viewpoint invariance via homography and board-centric coordinate projections.
3. **Temporal Phase Segmentation**: Pop, Apex, Catch, and Landing event boundary detection via kinematic derivates ($\dot{y}, \ddot{y}, \omega$).
4. **Trick Classification Engine**: Hierarchical multi-branch classifier (Pop vs No-pop, Board Flips, Shuvit/Rotations) evaluated against human expert standards.
5. **Cleanliness & Execution Scoring (Regression)**: Quantifying pop height, catch timing, truck proximity, angular deviation, and landing stability.
6. **Telemetry & Visual HUD**: Augmented video renderer displaying real-time metrics, phase timelines, and overlay stats.

---

## 🛠️ Repository Status

- [x] Master Research & Implementation Specification (v3.0)
- [ ] Phase 1: Video Ingestion & Synchronized Tracking
- [ ] Phase 2: Canonical Normalization & Phase Segmentation
- [ ] Phase 3: Trick Classification & Land/Bail Gating
- [ ] Phase 4: Kinematic Cleanliness Regression & Calibration
- [ ] Phase 5: Telemetry HUD & CLI / Web Interface
