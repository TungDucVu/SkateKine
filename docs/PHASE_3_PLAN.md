# Phase 3 Implementation Plan: Trick Classification Engine & Post-Impact Land/Bail Verification

---

## 1. Overview & Architectural Scope

Phase 3 implements the core action recognition and execution verification intelligence of SkateKine. Following the core architectural principle that **recognition and quality must be decoupled**, this phase focuses exclusively on two discrete classification tasks:

1. **Trick Categorization:** Classifying which specific trick occurred across the **9 primary flatground trick classes** using structured kinematic features on unseen skaters.
2. **Post-Impact Land/Bail Verification Gate:** Determining whether the attempt was successfully landed or bailed over a post-impact verification window $[t_{\text{land}}, t_{\text{land}} + 30]$, serving as a critical upstream gatekeeper before cleanliness scoring in Phase 4.

In accordance with scientific rigor, deep graph neural networks (ST-GCN) are evaluated strictly as candidate improvements over interpretable tree ensembles (XGBoost/LightGBM) and calibrated rule-based baselines. Furthermore, downstream models operate **strictly on structured kinematic coordinate trajectories and graph representations—raw RGB imagery is completely prohibited downstream of Phase 1**.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SKATEKINE PHASE 3 PIPELINE                         │
│                                                                             │
│  Phase 1 Trajectories (Pose + Board 8-pt)                                   │
│  Phase 2 Predicted Temporal Events [t_pop, t_apex, t_catch, t_land]         │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────┐                                       │
│  │ Viewpoint / Stance Normalization│  (Canonical Coordinate Transform)      │
│  └────────────────┬────────────────┘                                       │
│                   │                                                         │
│         ┌─────────┴──────────────────────┐                                  │
│         ▼                                ▼                                  │
│  ┌───────────────────────────────┐  ┌────────────────────────────────────┐  │
│  │   Trick Recognition Engine    │  │ Post-Impact Land/Bail Verification │  │
│  │   • Model A: Rule Baseline    │  │ • Input: [t_land, t_land + 30]     │  │
│  │   • Model B: GBDT (XGBoost)   │  │ • Tri-State: LANDED/BAILED/UNCERTAIN│ │
│  │   • Model C: ST-GCN Graph     │  │ • Cleanliness-independent          │  │
│  │   • Dual: End-to-End & Oracle │  └──────────────────┬─────────────────┘  │
│  └──────────────┬────────────────┘                     │                    │
│                 │                                      ▼                    │
│                 │                             Is attempt LANDED?            │
│                 │                             ├── YES: Pass to Phase 4      │
│                 │                             └── NO/UNCERTAIN: Terminate   │
│                 ▼                                                           │
│    Predicted Trick Label (1 of 9)                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Methodological Amendments (P3-A1 through P3-A10)

This plan integrates the 10 mandatory methodological amendments established during technical review:

| Amendment ID | Focus Area | Mandate & Methodological Resolution |
| :--- | :--- | :--- |
| **P3-A1** | **Event Provenance** | Every experiment records `event_source = predicted \| ground_truth`. Phase 3 features must be extracted from Phase 2 *predicted* event timestamps for the primary benchmark to avoid artificial temporal perfection. |
| **P3-A2** | **Rotation Features** | In-plane deck rotation is decoupled into: net rotation $\Delta\theta$, total angular travel $\Theta_{\text{abs}}$, rotation consistency ratio $C_\theta$, and peak angular velocity $\omega_{\text{peak}}$. |
| **P3-A3** | **Stance/Viewpoint Invariance** | Projected rotation signs cannot be mapped directly to physical tricks (Kickflip vs. Heelflip). A camera azimuth and stance conditioning layer canonicalizes coordinates before semantic mapping. |
| **P3-A4** | **Formal 9-Class Taxonomy** | All 9 flatground trick classes are explicitly enumerated with physics-based criteria, yaw/roll boundaries, and stance conventions. |
| **P3-A5** | **Tri-State Land/Bail** | The verification gate outputs `LANDED`, `BAILED`, or `UNCERTAIN`. Ambiguous or occluded attempts are quarantined to prevent training on label noise. |
| **P3-A6** | **Post-Impact Terminology** | Formalized as **Post-impact Land/Bail Verification Gate** (operating on $[t_{\text{land}}, t_{\text{land}} + 30]$), strictly decoupled from execution cleanliness (e.g., sketchy style $\neq$ bailed). |
| **P3-A7** | **Nested GroupKFold** | Skater is the fundamental unit of statistical independence. Hyperparameters are selected via an inner GroupKFold; outer test skaters remain strictly untouched. |
| **P3-A8** | **Pre-Training Corpus Audit** | A mandatory data audit generates `class × skater × viewpoint × FPS × bail` matrices before training to prevent overclaiming on small-sample classes. |
| **P3-A9** | **Dual Evaluation Schema** | Report both: (A) Oracle-event evaluation (upper bound on GT events) and (B) End-to-end evaluation (Phase 2 predicted events = primary scientific result). |
| **P3-A10** | **Standardized Latency** | Runtime must be benchmarked as **ms/frame** and **Real-Time Factor (RTF)** with explicit hardware and preprocessing bounds, not ambiguous "ms/clip". |

---

## 3. Formal 9-Class Flatground Trick Taxonomy (P3-A4)

Phase 3 targets the 9 core flatground skateboarding tricks. To avoid label ambiguity, each trick is defined by its mechanical state transitions over the flight interval $[t_{\text{pop}}, t_{\text{catch}}]$:

| # | Trick Class | Board Yaw ($\Delta\theta_{\text{yaw}}$) | Board Roll ($\Delta\phi_{\text{roll}}$) | Skater Torso Yaw ($\Delta\theta_{\text{body}}$) | Primary Mechanical Discriminator |
| :-: | :--- | :-: | :-: | :-: | :--- |
| **1** | **Ollie** | $\approx 0^\circ$ ($< 45^\circ$) | $\approx 0^\circ$ ($< 45^\circ$) | $\approx 0^\circ$ ($< 45^\circ$) | Ballistic pop and level catch; no axial rotation. |
| **2** | **Frontside 180** | $\approx +180^\circ$ ($[135^\circ, 225^\circ]$) | $\approx 0^\circ$ ($< 45^\circ$) | $\approx +180^\circ$ (chest turns forward) | Synchronized board + body frontside yaw rotation. |
| **3** | **Backside 180** | $\approx -180^\circ$ ($[-225^\circ, -135^\circ]$) | $\approx 0^\circ$ ($< 45^\circ$) | $\approx -180^\circ$ (back turns forward) | Synchronized board + body backside yaw rotation. |
| **4** | **Pop Shove-it** | $\approx -180^\circ$ (Backside) | $\approx 0^\circ$ ($< 45^\circ$) | $\approx 0^\circ$ ($< 45^\circ$) | Board rotates $180^\circ$ BS beneath stationary torso. |
| **5** | **Frontside Shove-it** | $\approx +180^\circ$ (Frontside) | $\approx 0^\circ$ ($< 45^\circ$) | $\approx 0^\circ$ ($< 45^\circ$) | Board rotates $180^\circ$ FS beneath stationary torso. |
| **6** | **Kickflip** | $\approx 0^\circ$ ($< 45^\circ$) | $\approx 360^\circ$ (Toe-side flick) | $\approx 0^\circ$ ($< 45^\circ$) | Board completes 1 full roll rotation toward toe side. |
| **7** | **Heelflip** | $\approx 0^\circ$ ($< 45^\circ$) | $\approx 360^\circ$ (Heel-side flick) | $\approx 0^\circ$ ($< 45^\circ$) | Board completes 1 full roll rotation toward heel side. |
| **8** | **Varial Kickflip** | $\approx -180^\circ$ (Backside) | $\approx 360^\circ$ (Toe-side flick) | $\approx 0^\circ$ ($< 45^\circ$) | Combined $180^\circ$ BS board yaw + $360^\circ$ kickflip roll. |
| **9** | **360 Flip (Tre Flip)**| $\approx -360^\circ$ (Backside) | $\approx 360^\circ$ (Toe-side flick) | $\approx 0^\circ$ ($< 45^\circ$) | Combined $360^\circ$ BS board yaw + $360^\circ$ kickflip roll. |

### Stance & Directionality Conventions:
* **Stance Normalization:** All angular conventions are normalized to **Regular Stance** (left foot forward). For Goofy skaters (right foot forward), lateral axes and yaw directions are mirrored across the sagittal plane prior to classification.
* **Scope Exclusion:** Switch, Nollie, and Fakie variations are cataloged during data auditing. For initial Phase 3 training, they are either canonicalized by stance metadata or isolated into specific sub-evaluations to prevent label pollution.

---

## 4. Kinematic Feature Design & Mathematical Formulation

Features are extracted primarily over the active flight window $[t_{\text{pop}}, t_{\text{catch}}]$ (or $[t_{\text{pop}}, t_{\text{land}}]$ if catch is unobserved / bailed).

### 4.1 In-Plane Deck Rotation Dynamics (P3-A2)
Let $\theta(t) = \text{unwrap}(\text{atan2}(y_{\text{nose}}(t) - y_{\text{tail}}(t), x_{\text{nose}}(t) - x_{\text{tail}}(t)))$ be the continuous image-plane deck orientation angle:

1. **Net Board Rotation:**
   $$\Delta\theta_{\text{net}} = \theta(t_{\text{catch}}) - \theta(t_{\text{pop}})$$
   *(Distinguishes true $0^\circ$, $180^\circ$, and $360^\circ$ displacements).*
2. **Total Angular Travel:**
   $$\Theta_{\text{abs}} = \int_{t_{\text{pop}}}^{t_{\text{catch}}} |\dot{\theta}(t)| \, dt$$
   *(Captures reversals, under-rotations, and erratic wobbles).*
3. **Rotational Consistency Ratio:**
   $$C_\theta = \frac{|\Delta\theta_{\text{net}}|}{\Theta_{\text{abs}} + \epsilon} \in [0, 1]$$
   *($C_\theta \approx 1$ indicates monotonic clean rotation; $C_\theta \ll 1$ indicates oscillation or reversal).*
4. **Peak Angular Velocity:**
   $$\omega_{\text{peak}} = \max_{t \in [t_{\text{pop}}, t_{\text{catch}}]} |\dot{\theta}(t)|$$

### 4.2 Temporal Flip & Aspect-Ratio Trajectory (P3-A3)
Rather than relying on a fragile scalar derivative integral, flip dynamics are captured via the **temporal aspect-ratio trajectory** $W_{\text{aspect}}(t) = \frac{w_{\text{proj}}(t)}{L_{\text{proj}}(t)}$:
* **Baseline-Normalized Aspect Curve:** $\tilde{W}_{\text{aspect}}(t) = \frac{W_{\text{aspect}}(t)}{W_{\text{aspect}}(t_{\text{pop}})}$.
* **Cycle Count & Peak Amplitude:** Number of local minima $\tilde{W}_{\text{aspect}} < 0.25$ during flight, reflecting vertical deck edge-on transitions.
* **Front-Foot Flick Kinematics:** 2D displacement and velocity vector of the front ankle relative to the board nose during the flick window $[t_{\text{pop}}, t_{\text{pop}} + 0.15\text{ s}]$:
  $$\mathbf{v}_{\text{flick}} = \frac{\mathbf{p}_{\text{ankle}}(t_{\text{pop}} + \delta) - \mathbf{p}_{\text{nose}}(t_{\text{pop}})}{\delta}$$
* **Stance/Viewpoint Canonicalization:** Projected roll direction is signed relative to the skater's facing normal vector $\mathbf{n}_{\text{skater}} = \mathbf{p}_{\text{chest}} \times \mathbf{p}_{\text{up}}$. This ensures toe-side flicks are distinguished from heel-side flicks across frontside vs. backside camera perspectives.

### 4.3 Torso & Body Kinematics
* **Skater Torso Yaw:** Net rotation of the bi-acromial (shoulder) and bi-iliac (hip) axes between $t_{\text{pop}}$ and $t_{\text{land}}$:
  $$\Delta\theta_{\text{body}} = \text{unwrap}(\theta_{\text{shoulder}}(t_{\text{land}})) - \text{unwrap}(\theta_{\text{shoulder}}(t_{\text{pop}}))$$
* **Center of Mass Trajectory:** Vertical displacement $h_{\text{apex}} = \max_t(z_{\text{com}}(t)) - z_{\text{com}}(t_{\text{pop}})$ and airtime $\Delta t_{\text{air}} = t_{\text{land}} - t_{\text{pop}}$.

---

## 5. Model Architectures & Hierarchy

We implement a rigorous 3-tiered model hierarchy from transparent baselines to graph architectures:

### 5.1 Model A: Calibrated Rule-Based Baseline (`src/classification/rule_classifier.py`)
A transparent, fully deterministic decision tree. Crucially, its threshold parameters are **calibrated and frozen on the training split**, not guessed:
* $t_{\text{pop}}$ and $t_{\text{land}}$ validated via airtime $\Delta t \ge 0.30\text{ s}$.
* Yaw partition: $|\Delta\theta_{\text{net}}| \in [0^\circ, 45^\circ] \to \text{No Yaw}$; $[135^\circ, 225^\circ] \to 180^\circ \text{ Yaw}$; $[315^\circ, 405^\circ] \to 360^\circ \text{ Yaw}$.
* Roll partition: Aspect ratio cycle count $N_{\text{flip}} = 0 \to \text{No Flip}$; $N_{\text{flip}} = 1 \to \text{Single Flip}$.
* Direction: Conditioned on $\mathbf{v}_{\text{flick}} \cdot \mathbf{n}_{\text{toe}} > 0 \to \text{Kickflip}$; $< 0 \to \text{Heelflip}$.
* Body rotation: $|\Delta\theta_{\text{body}}| \ge 135^\circ \to \text{180 Ollie / Body Varial}$.

### 5.2 Model B: Gradient Boosted Decision Trees (`src/classification/tree_classifier.py`)
* **Algorithm:** LightGBM / XGBoost multi-class classifier operating on the 48-dimensional tabular kinematic feature vector.
* **Optimization:** Hyperparameters (depth: 3–6, learning rate: 0.01–0.1, subsample: 0.8) tuned via nested GroupKFold by `skater_id`.
* **Interpretability:** Compute SHAP (SHapley Additive exPlanations) values to verify that predictions rely on valid physical cues (e.g. board yaw for shove-its, flick vector for kickflips) rather than dataset artifacts.

### 5.3 Model C: Spatio-Temporal Graph Convolutional Network (`src/classification/graph_classifier.py`)
* **Input Graph Topology:**
  * **Nodes (25 total):** 17 COCO human pose joints + 8 semantic board contour keypoints.
  * **Node Features:** Normalized coordinate tuple $(x, y, v_x, v_y, c)$ where $c$ is detection confidence. Low-confidence joints ($c < 0.3$) have coordinates interpolated from adjacent frames with $c=0$.
  * **Spatial Edges ($E_{\text{spatial}}$):** Human skeletal adjacency (16 edges) + Board perimeter loop (8 edges) + Dynamic foot-to-board contact edges (left ankle $\leftrightarrow$ tail/middle, right ankle $\leftrightarrow$ nose/middle).
  * **Temporal Edges ($E_{\text{temporal}}$):** Trajectory links connecting joint $i$ at frame $t$ to joint $i$ at frame $t+1$.
* **Strict Privacy Rule:** **ZERO raw RGB imagery.** The ST-GCN operates strictly on the $(25, T, 5)$ graph tensor over normalized time slices $T=64$.

---

## 6. Post-Impact Land/Bail Verification Gate (P3-A5, P3-A6)

The verification gate operates strictly on the post-touchdown interval $[t_{\text{land}}, t_{\text{land}} + 30]$ (approx. 0.5–1.0 s post-impact) to verify attempt completion.

```text
               Touchdown (t_land)
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│             POST-IMPACT VERIFICATION WINDOW                 │
│                 [t_land,  t_land + 30]                      │
│                                                             │
│  Stability Cues:                                            │
│  • Both feet maintain contact with deck (distance < tol)    │
│  • Skater CoM velocity matches board velocity (no slip/fall)│
│  • Board maintains upright roll (no pitch runaway)          │
└──────────────────────────────┬──────────────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
       [ LANDED ]         [ BAILED ]        [ UNCERTAIN ]
    (Rollout verified)  (Step-off/fall)  (Severe occlusion/loss)
            │                  │                  │
            ▼                  ▼                  ▼
     Pass to Phase 4    Bypass Phase 4     Quarantine attempt
      (Cleanliness)     (Log Bail Telemetry) (Exclude from eval)
```

### 6.1 Tri-State Label Definitions
* **`LANDED`:** Both feet make contact with the board near the bolts, the board maintains forward momentum without overturning, and the skater rides away for $\ge 15$ frames post-impact.
* **`BAILED`:** Attempt failed: skater steps off onto the ground, board flips away or rolls away unridden, skater falls to the ground, or one/both feet touch the pavement before the board settles.
* **`UNCERTAIN`:** The camera cuts off immediately at $t_{\text{land}}$, severe occlusion blocks the board, or the attempt exits the camera frame before rollout stability can be verified. **Excluded from supervised binary training**.

### 6.2 Decoupling from Cleanliness
The gate evaluates **attempt completion only**. A trick landed with toe-drag, hand-drag, sketchy tick-tack, or off-center foot placement is labeled **`LANDED`** (and will be penalized for cleanliness in Phase 4). It is **not** a bail.

---

## 7. Systematic Feature Ablation Matrix (A0–A5)

To disentangle representation modality from parameter capacity explosion, we establish a standardized ablation matrix evaluated across identical nested GroupKFold skater splits:

| Experiment | Modality / Feature Set | Dimensionality | Research Question Addressed |
| :---: | :--- | :---: | :--- |
| **A0** | **Baseline Raw Coordinates** | $25 \times 2$ (50) | Simple common baseline: Can raw positional endpoints alone classify tricks? |
| **A1** | **Pose Landmarks Only** | 34 | Can skater body kinematics alone predict the trick without board coordinates? |
| **A2** | **Board Tracking Only** | 16 | Can isolated board motion identify the trick without human context? |
| **A3** | **Pose + Board Coordinates** | 50 | Does fusing human and object geometry outperform isolated tracking? |
| **A4** | **Coordinates + Velocities** | 100 | What is the marginal information gain of 1st-order temporal derivatives? |
| **A5** | **Full Engineered Dynamics** | 48 (summary) / 150 (seq) | Contribution of physical invariants ($\Theta_{\text{abs}}, \Delta\theta_{\text{net}}, C_\theta, \mathbf{v}_{\text{flick}}$). |

---

## 8. Experimental & Evaluation Methodology

### 8.1 Pre-Training Corpus Audit (P3-A8)
Prior to training, generate a complete corpus audit across all available clips in `video_manifest.csv`:
* Class balance: Sample count per trick class $k \in \{1 \dots 9\}$.
* Skater distribution: Unique skater IDs per class (ensuring no single-skater class bias).
* Viewpoint distribution: Camera azimuth bins (Front / 45° Front-Oblique / Side / 45° Back-Oblique / Back).
* FPS & Resolution profile: Distribution of 30, 60, and 120 FPS sources.
* Land / Bail / Uncertain ratio.

### 8.2 Nested GroupKFold Cross-Validation (P3-A7)
* **Unit of Independence:** `skater_id`.
* **Outer Evaluation:** 5-fold GroupKFold. In each outer fold, test skaters are completely withheld.
* **Inner Hyperparameter Selection:** 4-fold GroupKFold nested strictly within the training skaters of that outer fold.
* **Viewpoint Stratification:** Ensure test folds contain representation across camera azimuth bins to formally evaluate viewpoint invariance.

```text
Full Dataset (All Skaters)
   │
   ├── Outer Fold i: Test Skaters (Strictly Held Out)
   │
   └── Outer Fold i: Train Skaters
             │
             ├── Inner Fold 1 (Validation) ── Hyperparameter Tuning
             ├── Inner Fold 2 (Validation) ── (Grid / Bayesian Search)
             └── Inner Fold 3 (Validation) ── Select Best Config θ*
                         │
                         ▼
             Train on ALL Outer Train Skaters with θ*
                         │
                         ▼
             Evaluate on Outer Test Skaters (Unseen)
```

### 8.3 Dual Evaluation Schema (P3-A1, P3-A9)
For every model (A, B, C), we report two evaluation benchmarks:
1. **Benchmark A (Oracle-Event Segmentation):** Features extracted using human consensus Ground-Truth event timestamps ($t_{\text{pop}}^{\text{GT}}, t_{\text{catch}}^{\text{GT}}, t_{\text{land}}^{\text{GT}}$). Demonstrates the theoretical upper bound of trick classification given perfect temporal segmentation.
2. **Benchmark B (End-to-End Pipeline — Primary Result):** Features extracted using Phase 2 *predicted* event timestamps ($t_{\text{pop}}^{\text{pred}}, t_{\text{catch}}^{\text{pred}}, t_{\text{land}}^{\text{pred}}$). Demonstrates real-world operational performance of the integrated SkateKine engine.

### 8.4 Standardized Runtime Latency Metric (P3-A10)
Latency will be measured under controlled conditions:
* Hardware specification: Standard single-GPU (e.g. RTX 3060/4090) and single-thread CPU (x86_64).
* Metrics:
  * **Average Latency:** $\text{ms / frame}$ (averaged across all frames in clip).
  * **Real-Time Factor (RTF):**
    $$\text{RTF} = \frac{\text{Wall-Clock Processing Time (s)}}{\text{Video Duration (s)}}$$
    *($\text{RTF} < 1.0$ indicates faster-than-real-time throughput).*

---

## 9. Post-Phase Quality Control (QC) & Acceptance Criteria

| Checkpoint / Metric | Hard Operational Gate | Aspirational Research Target | Evaluation Method |
| :--- | :---: | :---: | :--- |
| **End-to-End Macro F1** | $\ge 82.0\%$ | $\ge 88.0\%$ | Macro-averaged F1 across 9 trick classes on unseen skaters with Phase 2 predicted events. |
| **Oracle Macro F1** | $\ge 85.0\%$ | $\ge 92.0\%$ | Macro-averaged F1 using GT event timestamps (upper-bound reference). |
| **Minimum Class F1** | $\ge 70.0\%$ | $\ge 80.0\%$ | Lowest per-class F1 score across all 9 classes (guarantees no class collapse). |
| **Top-2 Classification Accuracy** | $\ge 92.0\%$ | $\ge 96.0\%$ | Prediction is correct if true trick is within top-2 model probabilities. |
| **Post-Impact Bail F1** | $\ge 0.80$ | $\ge 0.90$ | Binary F1 for Landed vs Bailed verification on held-out test skaters. |
| **Bail False Positive Rate** | $< 8.0\%$ | $< 3.0\%$ | Percentage of clean landed attempts falsely flagged as bails. |
| **Pipeline Latency (RTF)** | $\text{RTF} \le 0.50$ | $\text{RTF} \le 0.20$ | End-to-end Phase 3 inference throughput on target GPU. |

---

## 10. Execution Sequence & Work Breakdown

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 3 EXECUTION PHASES                             │
│                                                                             │
│  [Step 3.1] Dataset Audit & Stratified Manifest Preparation                 │
│             • Script: src/classification/dataset_audit.py                   │
│             • Matrix: class × skater × viewpoint × FPS × bail               │
│                                                                             │
│  [Step 3.2] Feature Extractor with Stance/Viewpoint Normalization           │
│             • Script: src/classification/feature_extractor.py               │
│             • Extract Δθ_net, Θ_abs, C_θ, ω_peak, aspect curve, flick vec   │
│             • Dual modes: event_source in ['ground_truth', 'predicted']     │
│                                                                             │
│  [Step 3.3] Post-Impact Land/Bail Verification Gate                         │
│             • Script: src/classification/bail_gatekeeper.py                 │
│             • Tri-state: LANDED / BAILED / UNCERTAIN                        │
│             • Persistence analysis on [t_land, t_land + 30]                 │
│                                                                             │
│  [Step 3.4] Model A — Calibrated Rule-Based Baseline                        │
│             • Script: src/classification/rule_classifier.py                 │
│             • Calibrate thresholds on train split; freeze on test           │
│                                                                             │
│  [Step 3.5] Model B — Gradient Boosted Decision Trees (XGBoost/LightGBM)   │
│             • Script: src/classification/tree_classifier.py                 │
│             • Nested GroupKFold cross-validation + SHAP importances         │
│                                                                             │
│  [Step 3.6] Model C — Spatio-Temporal Graph Convolutional Network (ST-GCN)  │
│             • Script: src/classification/graph_classifier.py                │
│             • 25-node pose+board graph tensor (zero RGB downstream)         │
│                                                                             │
│  [Step 3.7] Systematic Ablation Runner (A0–A5)                              │
│             • Script: src/classification/ablation_runner.py                 │
│             • Evaluate A0 (baseline), A1, A2, A3, A4, A5                    │
│                                                                             │
│  [Step 3.8] Benchmark Synthesis & Formal Report Generation                  │
│             • Output: docs/reports/PHASE_3_REPORT.md                        │
│             • QC verification against Gate criteria                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Deliverables & Documentation Checklist

Upon completion of Phase 3, the following artifacts must be created and verified:
1. `src/classification/dataset_audit.py`: Pre-training audit script and generated JSON summary.
2. `src/classification/feature_extractor.py`: Tabular and graph feature extraction engine with viewpoint/stance normalization.
3. `src/classification/bail_gatekeeper.py`: Post-impact tri-state Land/Bail verification model.
4. `src/classification/rule_classifier.py`: Calibrated Model A rule baseline.
5. `src/classification/tree_classifier.py`: Model B XGBoost/LightGBM training pipeline with nested GroupKFold and SHAP analysis.
6. `src/classification/graph_classifier.py`: Model C ST-GCN graph classification pipeline.
7. `src/classification/ablation_runner.py`: A0–A5 systematic ablation test harness.
8. `docs/reports/phase3_benchmark_results.json`: Machine-readable results containing both Oracle and End-to-End performance metrics.
9. `docs/reports/PHASE_3_REPORT.md`: Comprehensive engineering report detailing methodology, audit findings, model comparisons, confusion matrices, SHAP explanations, failure modes, and Phase 4 readiness certification.
