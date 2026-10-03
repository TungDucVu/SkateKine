# Skateboard Trick Recognition & Kinematic Cleanliness Scoring Engine

### Master Research & Implementation Specification (v3.0 — Production & Research Ready)

---

## 0. Executive Vision & Scientific Framing

The system rejects end-to-end black-box video classification (e.g., feeding raw RGB pixels into spatio-temporal video transformers). Instead, it translates video into an interpretable, physically inspectable state space:

$$\text{Raw Video} \longrightarrow \begin{matrix} \text{Pose + Board} \\ \text{Tracking} \end{matrix} \longrightarrow \begin{matrix} \text{Canonical 2D / Board} \\ \text{Local Coordinates} \end{matrix} \longrightarrow \begin{matrix} \text{Temporal Event} \\ \text{Segmentation} \end{matrix} \longrightarrow \begin{matrix} \text{Trick Classification} \\ + \text{ Land/Bail Gate} \end{matrix} \longrightarrow \begin{matrix} \text{Candidate Kinematic} \\ \text{Features} \end{matrix} \longrightarrow \begin{matrix} \text{Expert-Calibrated} \\ \text{Cleanliness Models} \end{matrix} \longrightarrow \text{Telemetry HUD}$$

### Core Scientific Thesis

*"Rather than estimating an opaque aesthetic score, skateboard tricks can be decomposed into an interpretable spatio-temporal framework (board trajectory, body-deck kinematics, temporal phase windows) to rigorously investigate how measurable kinematic and geometric parameters correlate with expert-adjudicated execution quality."*

---

## 1. Guiding Principles & Methodological Guardrails

1. **Structured Representation over Pixels:** All downstream recognition and scoring models operate exclusively on joint/board keypoints, image-plane orientations, relative distances, and their numerical derivatives ($\dot{x}, \ddot{x}, \omega$), never on raw pixel tensors.
2. **Decouple Recognition from Quality:** *"What trick was performed?"* (Categorical multi-class classification) and *"How cleanly was it executed?"* (Continuous regression) are formulated and evaluated as independent tasks.
3. **No Uncalibrated Heuristic Scores:** Physical metrics (catch timing, truck proximity, angular residuals) are treated strictly as candidate explanatory predictors. Their weights and non-linear penalties must be learned from and calibrated against human expert agreement.
4. **Leak-Proof Skater-Wise Splits:** Splitting individual frames or consecutive attempts from the same skater across train and test sets is forbidden. Generalization is measured on **unseen skaters** and **held-out camera viewpoints**.
5. **Stratified, Real-World Phase Gates:** Success criteria use tiered targets (aspirational targets vs. hard operational minimums) rather than fragile all-or-nothing thresholds.
6. **Baselines Precede Deep Architectures:**

$$\text{Rule-based Kinematics} \longrightarrow \text{XGBoost / LightGBM} \longrightarrow \text{TCN} \longrightarrow \text{ST-GCN}$$



Deep graph architectures are justified only if they outperform tree ensembles on unseen-skater splits.

---

## 2. Theoretical Clarifications & Sensor Boundaries

### 2.1 2D Image-Plane Rotation vs. True 3D Angles

A single monocular camera cannot independently decouple true 3D Euler angles (roll $\Delta\phi$, pitch $\Delta\theta$, yaw $\Delta\psi$) across unconstrained viewpoints.

* **Formal Definition:**

$$\theta_{\text{board-in-plane}}(t) = \text{atan2}(y_N(t) - y_T(t),\, x_N(t) - x_T(t))$$



measures only the projected 2D orientation of the longitudinal axis in the camera plane.
* **Scope Constraint:** The initial system operates on **projected 2D apparent rotation** ($\Delta\theta_{\text{image-plane}}$) and **projected aspect-ratio transitions** (deck width-to-length ratio as a proxy for flip execution).
* **Evaluation Condition:** Evaluation of rotation completion is restricted to near-side-profile viewpoints ($\pm 15^\circ$ from perpendicular) until a multi-view rig or formal 3D shape-prior mesh fitter is deployed. True 3D kinematic rotation is not claimed.

### 2.2 Metric Scale via Known Board Geometry

Monocular pixel coordinates cannot provide metric dimensions (e.g., centimeters) without an explicit physical scale reference.

* **Board as Metric Calibrator:** Standard street skateboard decks have known physical bounds: nominal length $L_{\text{board}} \approx 80.0\text{ cm}$ (nose to tail) and nominal wheelbase $W_{\text{truck}} \approx 36.0\text{ cm}$ (front truck to rear truck).
* **Dynamic Pixel-to-Metric Factor:** For each frame $t$:

$$s(t) = \frac{L_{\text{board}}}{\Vert{}\mathbf{p}_N(t) - \mathbf{p}_T(t)\Vert{}_2} \quad \left[\frac{\text{cm}}{\text{pixel}}\right]$$



Metric landing offsets are computed by scaling board-plane pixel distances by $s(t)$ when the deck is parallel to the camera sensor. When perspective foreshortening is detected, measurements fall back to dimensionless ratios normalized by the apparent deck length:

$$d_{\text{norm}} = \frac{d_{\text{pixels}}}{\Vert{}\mathbf{p}_N - \mathbf{p}_T\Vert{}_2}$$



### 2.3 Probabilistic Catch Definition (Non-Zero Thresholds)

Due to anatomical keypoint tracking offsets (the ankle landmark resides above the shoe sole) and optical blur, feet never register a Euclidean distance of zero to the deck.

* **Formulation:** Catch event $t_{\text{catch}}$ is localized as the earliest frame within the descent window ($t_{\text{apex}} < t < t_{\text{land}}$) satisfying three simultaneous conditions:

$$\begin{cases}    \Vert{}\mathbf{p}_{\text{foot}}(t) - \mathbf{p}_{\text{deck}}(t)\Vert{}_2 < \tau_d \\   \left\Vert{}\dot{\mathbf{p}}_{\text{foot}}(t) - \dot{\mathbf{p}}_{\text{deck}}(t)\right\Vert{}_2 < \tau_v \\   \left\vert{}\frac{d\omega_{\text{board}}}{dt}\right\vert{} < \tau_\alpha   \end{cases}$$


* Thresholds $\tau_d, \tau_v, \tau_\alpha$ are empirically calibrated from the inter-annotator catch distribution, not hand-tuned.

### 2.4 Critical-Moment Confidence Gating

If tracking confidence degrades during high-speed rotation or foot-deck occlusion, the engine must not fabricate a score:


$$\text{If } \min_{t \in [t_{\text{catch}}-\delta, t_{\text{catch}}+\delta]} \left(c_{\text{board}}(t), c_{\text{ankles}}(t)\right) < \tau_{\text{conf}}:$$

$$\text{Return: } \left\{\text{Cleanliness: Incomplete}, \, \text{Reliability: LOW}, \, \text{Reason: Occlusion at catch}\right\}$$

---

## 3. Dataset Architecture & Annotation Schema

### 3.1 Dataset Hierarchy & Sizing

To eliminate ambiguity between video clips and trick counts:

| Tier | Entity Unit | Target Volume | Composition |
| --- | --- | --- | --- |
| **Prototype Set** | Video Clips (trimmed) | 300–500 clips (3–8 s each) | 1 trick attempt per clip across $\ge 10$ skaters. Focus on tracking sanity and baseline validation. |
| **Research Dataset** | Annotated Trick Attempts | 900–1,800 distinct attempts | Contains multiple continuous attempts per long clip, cruising, and bails across $\ge 15$ skaters. |
| **Expert Calibration Set** | Held-Out Evaluated Attempts | 80–120 verified landed attempts | Evaluated by 3+ independent expert skaters for model alignment. |

### 3.2 Annotation Layers

1. **Board Primitives (8 Points Total):**
* 6 Outer Contour Points: Nose ($N$), Tail ($T$), Front-Left ($FL$), Front-Right ($FR$), Rear-Left ($RL$), Rear-Right ($RR$).
* 2 Derived/Labeled Truck Anchors: Midpoints of front and rear truck baseplates.


2. **Skater Pose:** 17 COCO / 33 MediaPipe skeletal landmarks, prioritizing lower-limb kinematics (hips, knees, ankles, heels, big toes).
3. **Temporal Boundaries:** Frame-level integer timestamps for $t_{\text{pop}}, t_{\text{apex}}, t_{\text{catch}}, t_{\text{land}}$, and $t_{\text{bail}}$.
4. **Dual-Tier Expert Quality Labels:**
* **Overall Execution Quality:** Continuous scale $1\text{--}10$.
* **Component Sub-Scores:** Independent $1\text{--}10$ ratings for *Catch Elevation*, *Foot Placement (Bolts)*, *Landing Stability*, and *Rotational Completeness*.



---

## 4. End-to-End System Pipeline

```text
VIDEO STREAM (60 / 120 FPS)
   │
   ├──▶ Person Pose Tracking (RTMPose) ─────────────┐
   │                                                ▼
   └──▶ Board OBB + 8 Keypoints (YOLOv11-OBB) ─▶ Coordinate Normalization & Savitzky-Golay
                                                        │
                                                        ▼
                                           Temporal Event Localization
                                        (POP ──▶ APEX ──▶ CATCH ──▶ LAND)
                                                        │
                                                        ▼
                                            Action Recognition Engine
                                        (Kinematic Baseline vs XGBoost vs ST-GCN)
                                                        │
                                                        ▼
                                           Land vs. Bail Verification
                                        (Gated: Target F1 ≥ 0.90, Min ≥ 0.80)
                                                        │
                                                        ▼
                                           Candidate Kinematic Features
                                        (Catch Ratio, Bolts Offset, Stability)
                                                        │
                                                        ▼
                                            Dual-Head Expert Scoring
                                          ┌─────────────┴─────────────┐
                                          ▼                           ▼
                                    Sub-Score Heads          Overall Quality Head
                                    (Bolts, Catch)             (Calibrated 0-100)
                                          │                           │
                                          └─────────────┬─────────────┘
                                                        ▼
                                             Confidence Gate Check
                                                        │
                                                        ▼
                                          Telemetry Render & Streamlit HUD

```

### Phase 1: Spatial Tracking & Canonical Feature Store

1. **Pose & Board Tracking:** RTMPose for skater joints + YOLOv11-OBB fine-tuned on board keypoints. Kalman filter + optical flow handles edge-on board occlusions.
2. **Canonical Transformation:** Normalize all 2D pixel coordinates relative to the skater's mid-hip coordinate and scale by torso length $L_{\text{torso}}$.
3. **Board-Local Mapping:** Project foot coordinates into the board's moving reference frame:

$$\mathbf{p}_{\text{foot}}^{\text{board}}(t) = \mathbf{R}(-\theta_{\text{board-in-plane}}(t))\left(\mathbf{p}_{\text{foot}}(t) - \mathbf{C}_{\text{board}}(t)\right)$$


4. **Feature Store:** Persist as versioned Apache Parquet files containing coordinates, first derivatives $(\dot{x}, \dot{y})$, second derivatives $(\ddot{x}, \ddot{y})$, apparent angular velocity $\omega$, and tracking confidence scalars per frame.

### Phase 2: Temporal Event Localization & Action Recognition

1. **Temporal Boundaries:**
* Pop ($t_{\text{pop}}$): Tail-ground distance minimum + rear ankle acceleration peak.
* Apex ($t_{\text{apex}}$): Board vertical velocity zero-crossing $\dot{y}_{\text{board}} \approx 0$ at maximum altitude.
* Catch ($t_{\text{catch}}$): Velocity convergence $\Delta v < \tau_v$ and distance threshold $\Delta d < \tau_d$.
* Land ($t_{\text{land}}$): Board vertical displacement stabilizes at the ground-plane line.


2. **Trick Classification Hierarchy:**
* *Model A (Kinematic Rules):* Flight duration, flip-direction sign, and integrated 2D in-plane rotation.
* *Model B (Interpretable Tree):* XGBoost trained on aggregated kinematic summary statistics.
* *Model C (Spatio-Temporal Graph):* ST-GCN connecting joint nodes and board anchors.


3. **Bail Gatekeeper:**
* Evaluates foot-board contact maintenance, post-landing board variance, and CoM-proxy trajectory over $[t_{\text{land}}, t_{\text{land}} + 30\text{ frames}]$.
* Clips classified as `BAILED` skip the cleanliness scoring pipeline entirely.



### Phase 3: Kinematic Feature Engine & Expert-Calibrated Scoring

Extract candidate explanatory features:

* **Catch Elevation Ratio:**

$$r_{\text{catch}} = \frac{y_{\text{deck}}(t_{\text{catch}}) - y_{\text{ground}}}{y_{\text{deck}}(t_{\text{apex}}) - y_{\text{ground}}}$$


* **Truck Placement Offset (Metric-scaled or Normalized):**

$$d_{\text{bolts}} = s(t_{\text{catch}}) \left( \Vert{}\mathbf{p}_{\text{front\_foot}} - \mathbf{p}_{\text{front\_truck}}\Vert{} + \Vert{}\mathbf{p}_{\text{rear\_foot}} - \mathbf{p}_{\text{rear\_truck}}\Vert{} \right)$$


* **Landing Stability / Drift:** Variance of board image-plane orientation and lateral displacement of the skater's CoM proxy over 30 frames post-impact.
* **Dual-Model Cleanliness Calibration:**
1. *Sub-score Models:* Regress specific physical features directly against human sub-scores (e.g., $d_{\text{bolts}} \to \text{Foot Placement Rating}$).
2. *Overall Score Model:* Train ElasticNet / Ordinal Ridge regression to map the composite feature space to the human Overall Quality score (0–100).



---

## 5. Experimental Ablations & Failure Modes

### 5.1 Ablation Study Matrix

Every model variant is evaluated against identical skater-wise train/val/test splits:

| Run ID | Feature Set Input | Research Hypothesis Addressed |
| --- | --- | --- |
| **A1** | Pose Landmarks Only | Can skater kinematics alone imply the trick without board data? |
| **A2** | Board Tracking Only | Can board motion alone identify tricks without skater kinematics? |
| **A3** | Pose + Board Coordinates | What is the raw spatial configuration baseline? |
| **A4** | Coordinates + Velocities ($\mathbf{x}, \mathbf{v}$) | Does rate-of-change resolve rotational ambiguities? |
| **A5** | Full Dynamics ($\mathbf{x}, \mathbf{v}, \mathbf{a}, \omega, \mathbf{p}^{\text{board}}$) | Maximum discriminative capacity of the engineered feature space. |

### 5.2 Failure Mode Taxonomy

Every inference failure is systematically tagged in the error report:

1. **Optical Failure:** Frame drop, motion blur across key rotation frames, severe backlight.
2. **Tracking Failure:** Board keypoint swap (nose/tail inversion during flip), ankle drop during catch occlusion.
3. **Kinematic Ambiguity:** Subtle variation misclassification (e.g., slow-flick Kickflip vs. late Heelflip).
4. **Boundary Localization Error:** Catch frame placed too early or late due to loose clothing obscuring the ankles.

---

## 6. Stratified Phase Gates & Exit Criteria

```text
┌─────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│   Gate 1: Tracking      │      │  Gate 2: Segmentation   │      │  Gate 3: Recognition    │
│ • Drop rate: < 3%       │ ───▶ │ • Pop/Apex/Land: ≤ 3-5f │ ───▶ │ • Macro F1 ≥ 85%        │
│ • PCK@0.10 > 0.92       │      │ • Catch: ≤ 5-7 frames   │      │ • Unseen skater split   │
│ • PCK@0.05 > 0.85 (asp.)│      │ • Validated temporal seq│      │ • Confusion matrix clear│
└─────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
                                                                               │
┌─────────────────────────┐      ┌─────────────────────────┐                   │
│   Gate 6: Demo & HUD    │      │  Gate 5: Cleanliness    │                   ▼
│ • Latency ≤ 1.5x RT     │ ◀─── │ • Spearman ρ ≥ 0.78     │ ◀─── ┌─────────────────────────┐
│ • Zero unhandled crashes│      │ • Sub-score alignment   │      │  Gate 4: Land / Bail    │
│ • Auto confidence flags │      │ • Outperforms heuristic │      │ • Target F1 ≥ 0.90      │
│                         │      │                         │      │ • Min acceptable ≥ 0.80 │
└─────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘

```

1. **Gate 1 (Tracking Stability):**
* Low-confidence frame drop rate $< 3\%$ on controlled test footage.
* Board keypoint $\text{PCK}@0.10 > 0.92$ (operational gate); $\text{PCK}@0.05 > 0.85$ (aspirational target).


2. **Gate 2 (Temporal Event Localization at 60 fps):**
* Mean absolute error for $t_{\text{pop}}, t_{\text{apex}}, t_{\text{land}} \le 3\text{--}5\text{ frames}$ ($\approx 50\text{--}83\text{ ms}$).
* Mean absolute error for $t_{\text{catch}} \le 5\text{--}7\text{ frames}$ ($\approx 83\text{--}116\text{ ms}$).
* Temporal order constraint $t_{\text{pop}} < t_{\text{apex}} < t_{\text{catch}} \le t_{\text{land}}$ strictly satisfied in $\ge 98\%$ of landed attempts.


3. **Gate 3 (Trick Recognition):**
* Macro $\text{F1} \ge 85\%$ across the 9 primary classes evaluated strictly on unseen skaters.


4. **Gate 4 (Landing Verification):**
* Landed vs. Bailed classification: Target $\text{F1} \ge 0.90$; operational minimum $\text{F1} \ge 0.80$.


5. **Gate 5 (Cleanliness Model Alignment):**
* Spearman rank correlation $\rho \ge 0.78$ against consensus human ratings on the held-out evaluation set.
* Statistically significant improvement over arbitrary equal-weighted scoring heuristics ($p < 0.01$).


6. **Gate 6 (Production Demo & HUD):**
* End-to-end processing pipeline runs under $1.5\times$ clip real-time on an RTX 4070-class GPU.
* Zero unhandled exceptions across 50 consecutive real-world clips. Explicit `LOW RELIABILITY` warnings rendered when tracking confidence drops.



---

## 7. Telemetry HUD & Interactive Application

The interactive dashboard (Streamlit / Gradio) ingests raw video and renders a telemetry-augmented video alongside synchronized analytical inspectability panels:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  VIDEO TELEMETRY HUD OVERLAY                                                │
│                                                                             │
│               [ TRICK: KICKFLIP  |  CONFIDENCE: 94% ]                       │
│                                                                             │
│                                ● APEX (1.02s)                               │
│                             ↗                                               │
│             ● CATCH (1.28s)                                                 │
│          ↗                                                                  │
│  ● POP (0.71s)                                                              │
│                                                 ● LAND (1.42s) [LANDED]     │
│                                                                             │
│  CLEANLINESS: 88 / 100 [HIGH RELIABILITY]                                   │
│  ├── Catch Height Ratio:      82%  (Sub-Score: 8.5/10)                      │
│  ├── Projected Roll Error:    -9°  (Image-Plane Proxy)                      │
│  ├── Truck Landing Offset:    3.2 cm (Scale Ref: 80cm Deck)                 │
│  └── Rollout Stability:       91%  (Sub-Score: 9.0/10)                      │
└─────────────────────────────────────────────────────────────────────────────┘

```

### Dashboard Panels

1. **Video Inspection Window:** Synchronized playback showing human joint skeleton, board bounding polygon, trajectory trails for nose/tail, and event markers.
2. **Kinematic Signal Plots:**
* Board height curve over time with annotated Pop, Apex, Catch, and Landing events.
* Apparent angular velocity $\omega_{\text{board-in-plane}}(t)$ showing rotation acceleration and stabilization at catch.


3. **Score & Sub-Score Diagnostics:**
* Overall Cleanliness Index (0–100) alongside individual sub-score bars (Catch, Bolts, Stability).
* Feature attribution breakdown showing which parameters contributed positively vs. which triggered deductions.


4. **Data Export:** Download options for the annotated MP4 video, structured feature `.parquet`, and JSON scorecard.

---

## 8. Step-by-Step Implementation Sequence

```text
Step 01: Capture/curate initial 300-500 clips; establish skater-wise train/val/test splits.
   │
Step 02: Annotate 200 prototype frames (8-point board layout + pose) in CVAT/Roboflow.
   │
Step 03: Train YOLOv11-OBB board detector; benchmark RTMPose tracking continuity.
   │
Step 04: Build canonical coordinate normalization & Savitzky-Golay filtering pipeline.
   │
Step 05: Implement probabilistic event localization (Pop, Apex, Catch, Land) & validate tolerances.
   │
Step 06: Train baseline XGBoost classifier on engineered kinematics; run unseen-skater evaluation.
   │
Step 07: Implement TCN and ST-GCN sequence models; conduct feature ablation matrix (A1-A5).
   │
Step 08: Train Land vs. Bail gating classifier; enforce operational threshold (F1 ≥ 0.80).
   │
Step 09: Collect dual-tier expert ratings (overall + sub-scores) on 80-120 held-out clips; check ICC.
   │
Step 10: Fit regression models (ElasticNet / Ordinal) mapping kinematics to expert scores.
   │
Step 11: Implement scale-inferred metric conversions and catch-window confidence gating.
   │
Step 12: Build OpenCV telemetry overlay generator and deploy interactive Streamlit dashboard.

```