# SkateKine — Project Requirements & Resource Tracking Specification

This document serves as the master registry of all datasets, hardware/compute infrastructure, software environments, annotation schemas, physical calibration references, and storage architectures required for the **SkateKine: Skateboard Trick Recognition & Kinematic Cleanliness Scoring Engine**.

---

## 1. Video Data Requirements & Ingestion Standards

SkateKine requires structured video data across three hierarchical tiers to support tracking sanity, model training, and human-expert alignment without data leakage.

### 1.1 Dataset Tiers & Volume Targets

| Tier | Entity Unit | Target Volume | Composition & Criteria | Primary Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Prototype Set** | Video Clips (trimmed) | 300–500 clips (3–8s each) | 1 trick attempt per clip across $\ge 10$ skaters. Profile & semi-profile ($\pm 15^\circ$). | Baseline tracking, pipeline smoke testing, coordinate sanity. |
| **Tier 2: Research Dataset** | Annotated Trick Attempts | 900–1,800 distinct attempts | Multiple continuous attempts, cruising, flatground bails, diverse lighting across $\ge 15$ skaters. | Full classifier training, sequence models (TCN/ST-GCN), land/bail gating. |
| **Tier 3: Expert Calibration Set** | Evaluated Attempts | 80–120 verified landed attempts | 3+ independent expert skater annotations per attempt; 9 primary trick classes. | Calibrating regression models (ElasticNet/Ordinal), validating metric alignments. |

### 1.2 Video Capture & Encoding Specifications

* **Frame Rate (FPS):**
  * *Operational Minimum:* 60 FPS (required for temporal event boundary resolution within $\le 5$ frames).
  * *Aspirational / High-Speed:* 120 FPS or 240 FPS (ideal for resolving rapid flip rotations and precise foot-deck contact catch).
* **Resolution:** 1080p ($1920 \times 1080$) preferred; 720p ($1280 \times 720$) acceptable minimum. Raw 4K is downscaled to 1080p to maintain pipeline throughput.
* **Camera Viewpoint:**
  * Near-side-profile ($\pm 15^\circ$ from perpendicular to skater's travel line) is mandatory for Phase 1–3 rotational baselines to avoid uncalibrated 3D foreshortening.
  * Static camera or smooth tripod pans preferred over shaky handheld or extreme fisheye.
* **Lighting & Environment:** Outdoor daylight or evenly lit indoor skateparks; avoid heavy backlighting and low-light motion blur.
* **Container & Codec:** MP4 / MOV containers encoded in H.264 / HEVC (AVC1 / HVC1) with fast-start enabled.

### 1.3 Target Trick Taxonomy (9 Primary Classes)

The system targets flatground street skateboard tricks categorized by rotation and flip axes:
1. **Ollie** (baseline pop, no flip, no board rotation)
2. **Kickflip** (longitudinal roll axis $-360^\circ$ for regular stance)
3. **Heelflip** (longitudinal roll axis $+360^\circ$ for regular stance)
4. **Pop Shuvit** (board vertical yaw axis $+180^\circ$, no flip)
5. **Frontside Pop Shuvit** (board vertical yaw axis $-180^\circ$, no flip)
6. **Varial Kickflip** (combined $180^\circ$ Shuvit + $360^\circ$ Kickflip)
7. **Varial Heelflip** (combined $180^\circ$ FS Shuvit + $360^\circ$ Heelflip)
8. **360 Flip (Tre Flip)** ($360^\circ$ Shuvit + $360^\circ$ Kickflip)
9. **Frontside 180 Ollie / Backside 180 Ollie** (coupled body + board rotation)

---

## 2. Skater-Wise Partitioning & Anti-Leakage Constraints

To guarantee that models generalize to unseen skaters and environments, strict partitioning rules are enforced:

1. **Skater Identity Isolation:** All attempts from a unique skater $S_i$ belong exclusively to one split ($\text{Train}$, $\text{Val}$, or $\text{Test}$). No skater may appear in both training and test sets.
2. **Camera / Location Holdout:** The test set must contain footage from at least two skate spots / camera angles not present in the training set.
3. **Target Split Ratio:**
   * **Train:** 70% of skaters (~11–14 skaters, ~700–1,200 attempts).
   * **Validation:** 15% of skaters (~2–3 skaters, ~150–300 attempts).
   * **Test:** 15% of skaters (~2–3 skaters, ~150–300 attempts).

---

## 3. Physical Calibration & Metric Ground Truth

Monocular video lacks direct metric depth. Real-world centimeter conversion relies on standardized physical skateboard hardware dimensions.

| Physical Parameter | Nominal Dimension | Standard Tolerance | Usage in Engine |
| :--- | :--- | :--- | :--- |
| **Standard Deck Length ($L_{\text{board}}$)** | $80.0\text{ cm}$ (31.5 inches) | $\pm 1.5\text{ cm}$ | Dynamic pixel-to-cm scale factor $s(t) = \frac{L_{\text{board}}}{\Vert\mathbf{p}_N(t) - \mathbf{p}_T(t)\Vert_2}$. |
| **Standard Wheelbase ($W_{\text{truck}}$)** | $36.0\text{ cm}$ (14.2 inches) | $\pm 1.0\text{ cm}$ | Reference for truck baseplate localization and foot landing offset. |
| **Standard Deck Width ($W_{\text{deck}}$)** | $20.3\text{ cm}$ (8.0 inches) | $\pm 0.6\text{ cm}$ | Flip completeness proxy via aspect-ratio transition. |
| **Skater Torso Normalizer ($L_{\text{torso}}$)** | Mid-Hip to Mid-Shoulder | Subject-specific | Scale invariance normalizer for human body kinematics. |

---

## 4. Annotation Schema & Tooling Resources

### 4.1 Annotation Tooling
* **CVAT (Computer Vision Annotation Tool)** or **Roboflow** for frame-level keypoints and bounding polygons.
* **Label Studio** for temporal event segmentation and multi-rater subjective scoring.

### 4.2 Spatial Keypoint Definitions
1. **Board Keypoints (8 Points):**
   * **Outer Contour (6 points):** Nose Tip ($N$), Tail Tip ($T$), Front-Left Edge ($FL$), Front-Right Edge ($FR$), Rear-Left Edge ($RL$), Rear-Right Edge ($RR$).
   * **Truck Anchors (2 points):** Front Truck Pivot ($FT$), Rear Truck Pivot ($RT$).
2. **Human Body Keypoints (17/33 Points):**
   * Lower-body priority: Left/Right Hip, Left/Right Knee, Left/Right Ankle, Left/Right Heel, Left/Right Big Toe.
   * Upper-body posture: Shoulders, Elbows, Wrists, Nose.

### 4.3 Temporal Phase Labels
Every trick attempt is labeled with precise frame indices:
* $t_{\text{pop}}$: Tail makes contact with the ground plane / rear ankle reaches peak downward acceleration.
* $t_{\text{apex}}$: Skateboard centroid reaches maximum height ($\dot{y} = 0$).
* $t_{\text{catch}}$: Skater's feet make initial decelerating contact with the deck.
* $t_{\text{land}}$: Skateboard wheels touch down and ground-reaction force stabilizes.
* $t_{\text{bail}}$: Frame where attempt fails (feet touch ground before deck, board shoots away, skater falls).

### 4.4 Expert Cleanliness Rating Protocol
* **Panel:** Minimum of 3 experienced skateboarders (5+ years active skateboarding experience).
* **Metrics:**
  1. *Overall Cleanliness Score:* Continuous scale 1–10 (mapped to 0–100).
  2. *Catch Elevation Sub-Score (1–10):* Catching the board high in the air vs. stomping on the ground.
  3. *Foot Placement / Bolts Sub-Score (1–10):* Feet landing directly over truck hardware.
  4. *Landing Stability Sub-Score (1–10):* Rolling away clean without wheel bite, toe drag, tick-tack, or loss of balance.
* **Inter-Rater Reliability Threshold:** Intraclass Correlation Coefficient $\text{ICC}(2, k) \ge 0.75$. Outlier ratings differing by $> 2.5$ points trigger consensus review.

---

## 5. Compute, Hardware & System Infrastructure

### 5.1 Development & Training Environment
* **OS:** Windows 11 (Host) / Ubuntu 22.04 LTS (WSL2 / Dedicated Cloud VM).
* **GPU Compute:**
  * *Local Development:* NVIDIA GeForce RTX 4070 (12GB GDDR6X VRAM) or equivalent.
  * *Training Cloud (Optional):* NVIDIA A10G / RTX 3090 / A100 for heavy ST-GCN / OBB fine-tuning runs.
* **Host CPU & Memory:** 8-core CPU, 32GB System RAM.
* **Storage:** Fast NVMe SSD ($\ge 250\text{ GB}$ dedicated workspace) for multi-threaded video decoding and Parquet I/O.

### 5.2 Software Stack & Core Libraries

| Category | Package / Library | Version Pin | Usage |
| :--- | :--- | :--- | :--- |
| **Language Runtime** | Python | `3.10.x` or `3.11.x` | Core application and ML logic. |
| **Deep Learning** | PyTorch / Torchvision | `^2.2.0` (CUDA 12.1/12.2) | Neural network training and tensor acceleration. |
| **Object Tracking** | Ultralytics (YOLO) | `^8.2.0` / `v11` | YOLO-OBB board detection and contour tracking. |
| **Pose Estimation** | MMPose / RTMPose / MediaPipe | `rtmpose-m` or MediaPipe `^0.10.0` | 2D Skater skeletal landmark tracking. |
| **Classical Vision** | OpenCV (`opencv-python`) | `^4.9.0` | Video I/O, homography, optical flow, HUD rendering. |
| **Signal Processing** | SciPy, NumPy | `^1.12.0`, `^1.26.0` | Savitzky-Golay filtering, derivatives, peak detection. |
| **Tabular & Feature Store** | Pandas, PyArrow, Fastparquet | `^2.2.0`, `^15.0.0` | Structured `.parquet` storage of kinematic states. |
| **Classical ML & Scoring**| XGBoost, LightGBM, Scikit-Learn | `^2.0.0`, `^4.3.0`, `^1.4.0` | Tree classifiers, ElasticNet, Ordinal Ridge, metrics. |
| **UI & Visualization** | Streamlit, Matplotlib, Plotly | `^1.32.0`, `^3.8.0` | Interactive Telemetry HUD, signal plots, web inspection. |
| **Code Quality & CI** | Ruff, Pytest, Black | `^0.3.0`, `^8.0.0` | Linting, unit tests, kinematic regression test suite. |

---

## 6. Directory Layout & Storage Architecture

```text
SkateKine/
├── .gitignore
├── README.md
├── MASTER_FINAL.md
├── REQUIREMENT.md
├── docs/
│   ├── PHASE_1_PLAN.md
│   ├── PHASE_2_PLAN.md
│   ├── PHASE_3_PLAN.md
│   ├── PHASE_4_PLAN.md
│   ├── PHASE_5_PLAN.md
│   ├── PHASE_6_PLAN.md
│   └── reports/                       # Formal markdown execution reports per phase
│       ├── PHASE_1_REPORT.md
│       ├── PHASE_2_REPORT.md
│       └── ...
├── data/                              # Excluded from git via .gitignore
│   ├── raw_videos/                    # Raw uncompressed MP4/MOV files
│   ├── metadata/                      # video_manifest.csv (skater_id, trick, split, fps)
│   ├── annotations/                   # CVAT / COCO / JSON keypoint labels
│   │   ├── board_keypoints/
│   │   ├── temporal_events/
│   │   └── expert_ratings/
│   └── processed_frames/
├── features/                          # Versioned Apache Parquet feature stores
│   ├── v1_trajectories/
│   └── v2_kinematics/
├── models/                            # Trained weights & ONNX exports (gitignored)
│   ├── board_detector/
│   ├── trick_classifier/
│   └── cleanliness_regressor/
├── src/                               # Core Python engine packages
│   ├── tracking/                      # Board and pose extraction modules
│   ├── coordinates/                   # Canonical normalization & homography
│   ├── segmentation/                  # Event localization (Pop, Apex, Catch, Land)
│   ├── classification/                # Rule-based, XGBoost, and Graph models
│   ├── cleanliness/                   # Kinematic feature extractors & scoring heads
│   ├── visualization/                 # HUD rendering & video generator
│   └── app/                           # Streamlit dashboard entrypoints
└── tests/                             # Pytest test suite for kinematic sanity
```

---

## 7. Resource Sign-Off & Verification Checklist

Before advancing through implementation phases, the following prerequisite checklist must be satisfied:

- [x] Master specification defined (`MASTER_FINAL.md`).
- [x] Project resource and data requirements registry established (`REQUIREMENT.md`).
- [ ] Phase-by-phase implementation plans generated in `docs/`.
- [ ] Video manifest format established (`data/metadata/video_manifest.csv`).
- [ ] Conda / venv environment created with pinned requirements.
- [ ] GPU compute availability and CUDA driver validation confirmed.
