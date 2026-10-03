# SkateKine: Skateboard Trick Recognition & Kinematic Cleanliness Scoring Engine

An interpretable computer vision and kinematic modeling engine designed to recognize skateboard tricks and evaluate execution cleanliness from monocular video.

Instead of end-to-end black-box video classification, SkateKine maps raw video into an inspectable spatio-temporal state space:
```
Raw Video ──> Pose + Board Tracking ──> Canonical Coordinates ──> Temporal Event Segmentation ──> Trick Classification + Land/Bail Gate ──> Kinematic Features ──> Cleanliness Scoring Models ──> Telemetry HUD
```

---

## 📖 Project Documentation & Architecture Plans

* **Master Specification:** 👉 **[MASTER_FINAL.md](./MASTER_FINAL.md)** (v3.0 Production & Research Ready)
* **Resource & Data Requirements:** 📋 **[REQUIREMENT.md](./REQUIREMENT.md)** (Datasets, hardware, annotations, libraries, and schemas)
* **Reporting Standard:** 📝 **[docs/PHASE_REPORT_GUIDELINE.md](./docs/PHASE_REPORT_GUIDELINE.md)**

### Phase-by-Phase Implementation Plans
1. **Phase 1: Video Ingestion, Spatial Tracking & Canonical Feature Store** ── [Plan](./docs/PHASE_1_PLAN.md) | [Report Template](./docs/reports/PHASE_1_REPORT.md)
2. **Phase 2: Temporal Event Localization & Phase Segmentation** ── [Plan](./docs/PHASE_2_PLAN.md) | [Report Template](./docs/reports/PHASE_2_REPORT.md)
3. **Phase 3: Trick Classification Engine & Land/Bail Verification** ── [Plan](./docs/PHASE_3_PLAN.md) | [Report Template](./docs/reports/PHASE_3_REPORT.md)
4. **Phase 4: Kinematic Cleanliness Regression & Expert Calibration** ── [Plan](./docs/PHASE_4_PLAN.md) | [Report Template](./docs/reports/PHASE_4_REPORT.md)
5. **Phase 5: Telemetry HUD Overlay & Interactive Dashboard** ── [Plan](./docs/PHASE_5_PLAN.md) | [Report Template](./docs/reports/PHASE_5_REPORT.md)
6. **Phase 6: End-to-End Evaluation, Error Tagging & Release** ── [Plan](./docs/PHASE_6_PLAN.md) | [Report Template](./docs/reports/PHASE_6_REPORT.md)

---

## 🛠️ Repository Status

- [x] Master Research & Implementation Specification (`MASTER_FINAL.md`)
- [x] Project Requirements & Resource Tracking Registry (`REQUIREMENT.md`)
- [x] Phase Implementation Plans & QC Reporting Framework (`docs/`)
- [ ] Phase 1: Video Ingestion & Spatial Tracking
- [ ] Phase 2: Temporal Event Localization & Phase Segmentation
- [ ] Phase 3: Trick Classification & Land/Bail Gating
- [ ] Phase 4: Kinematic Cleanliness Regression & Calibration
- [ ] Phase 5: Telemetry HUD Overlay & Streamlit App
- [ ] Phase 6: End-to-End Evaluation & Release
