# Phase 3 Implementation Plan: Trick Classification Engine & Post-Impact Land/Bail Verification

---

## 1. Overview & Architectural Scope

Phase 3 implements the core action recognition and execution verification intelligence of SkateKine. Following the foundational architectural principle that **trick recognition and execution cleanliness must be strictly decoupled**, this phase focuses exclusively on two discrete classification tasks:

1. **Trick Categorization:** Classifying which specific trick occurred across the **9 primary flatground trick classes** using structured kinematic features on unseen skaters.
2. **Post-Impact Land/Bail Verification Gate:** Determining whether the attempt was successfully landed or bailed over a post-impact verification window $[t_{\text{land}}, t_{\text{land}} + 30]$, serving as a critical upstream gatekeeper before cleanliness scoring in Phase 4.

In accordance with scientific rigor, deep graph neural networks (ST-GCN) are evaluated strictly as candidate improvements over interpretable tree ensembles (XGBoost/LightGBM) and calibrated rule-based baselines. Furthermore, downstream models operate **strictly on structured kinematic coordinate trajectories and graph representations—raw RGB imagery is completely prohibited downstream of Phase 1**.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SKATEKINE PHASE 3 PIPELINE                         │
│                                                                             │
│  Phase 1 Trajectories (Pose + Board 8-pt, Parquet Feature Store)            │
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

## 2. Core Methodological Amendments & Governance (P3-A1 through P3-A10)

This master plan integrates all 10 non-negotiable methodological amendments established during technical review:

| Amendment ID | Focus Area | Mandate & Methodological Resolution |
| :--- | :--- | :--- |
| **P3-A1** | **Event Provenance** | Every experiment records `event_source = predicted \| ground_truth`. Phase 3 features must be extracted from Phase 2 *predicted* event timestamps for the primary benchmark to avoid artificial temporal perfection. |
| **P3-A2** | **Rotation Features** | In-plane deck rotation is decoupled into: net rotation $\Delta\theta_{\text{net}}$, total angular travel $\Theta_{\text{abs}} = \int |\dot{\theta}| dt$, rotational consistency ratio $C_\theta$, and peak angular velocity $\omega_{\text{peak}}$. |
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

Phase 3 targets the 9 core flatground skateboarding tricks. Each trick is defined by its mechanical state transitions over the flight interval $[t_{\text{pop}}, t_{\text{catch}}]$:

| # | Trick Class | Board Yaw ($\Delta\theta_{\text{yaw}}$) | Board Roll ($\Delta\phi_{\text{roll}}$) | Skater Torso Yaw ($\Delta\theta_{\text{body}}$) | Primary Mechanical Discriminator |
| :-: | :--- | :-: | :-: | :-: | :--- |
| **1** | **Ollie** | $\approx 0^\circ$ ($< 45^\circ$) | $\approx 0^\circ$ ($< 45^\circ$) | $\approx 0^\circ$ ($< 45^\circ$) | Ballistic pop and level catch; no axial rotation. |
| **2** | **Frontside 180** | $\approx +180^\circ$ ($[135^\circ, 225^\circ]$) | $\approx 0^\circ$ ($< 45^\circ$) | $\approx +180^\circ$ (chest turns forward) | Synchronized board + body frontside yaw rotation. |
| **3** | **Backside 180** | $\approx -180^\circ$ ($[-225^\circ, -135^\circ]$) | $\approx 0^\circ$ ($< 45^\circ$) | $\approx -180^\circ$ (back turns forward) | Synchronized board + body backside yaw rotation. |
| **4** | **Pop Shove-it** | $\approx -180^\circ$ (Backside) | $\approx 0^\circ$ ($< 45^\circ$) | $\approx 0^\circ$ ($< 45^\circ$) | Board rotates $180^\circ$ BS beneath stationary torso. |
| **5** | **Frontside Shove-it** | $\approx +180^\circ$ (Frontside) | $\approx 0^\circ$ ($< 45^\circ$) | $\approx 0^\circ$ ($< 45^\circ$) | Board rotates $180^\circ$ FS beneath stationary torso. |
| **6** | **Kickflip** | $\approx 0^\circ$ ($< 45^\circ$) | $\approx 360^\circ$ (Toe-side flick) | $\approx 0^\circ$ ($< 45^\circ$) | Board completes 1 full roll rotation toward toe side. |
| **7** | **Heelflip** | $\approx 0^\circ$ ($< 45^\circ$) | $\approx 360^\circ$ (Heel-side flick) | $\approx 0^\circ$ ($< 45^\circ$) | Board completes 1 full roll rotation toward heel side. |
| **8** | **Varial / Hardflip** | $\approx \pm 180^\circ$ (BS / FS) | $\approx 360^\circ$ (Toe-side flick) | $\approx 0^\circ$ ($< 45^\circ$) | Combined $180^\circ$ board yaw + $360^\circ$ kickflip roll. |
| **9** | **360 Flip (Tre Flip)**| $\approx -360^\circ$ (Backside) | $\approx 360^\circ$ (Toe-side flick) | $\approx 0^\circ$ ($< 45^\circ$) | Combined $360^\circ$ BS board yaw + $360^\circ$ kickflip roll. |

### Stance & Directionality Conventions:
* **Stance Normalization:** All angular conventions are normalized to **Regular Stance** (left foot forward). For Goofy skaters (right foot forward), lateral axes and yaw directions are mirrored across the sagittal plane prior to classification.
* **Scope Exclusion:** Switch, Nollie, and Fakie variations are cataloged during data auditing and canonicalized by stance metadata.

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
Flip dynamics are captured via the **temporal aspect-ratio trajectory** $W_{\text{aspect}}(t) = \frac{w_{\text{proj}}(t)}{L_{\text{proj}}(t)}$:
* **Baseline-Normalized Aspect Curve:** $\tilde{W}_{\text{aspect}}(t) = \frac{W_{\text{aspect}}(t)}{W_{\text{aspect}}(t_{\text{pop}})}$.
* **Cycle Count & Peak Amplitude:** Number of local minima $\tilde{W}_{\text{aspect}} < 0.40$ during flight, reflecting vertical deck edge-on transitions.
* **Front-Foot Flick Kinematics:** 2D displacement and velocity vector of front ankle relative to board nose during flick window $[t_{\text{pop}}, t_{\text{pop}} + 0.15\text{ s}]$:
  $$\mathbf{v}_{\text{flick}} = \frac{\mathbf{p}_{\text{ankle}}(t_{\text{pop}} + \delta) - \mathbf{p}_{\text{nose}}(t_{\text{pop}})}{\delta}$$
* **Stance/Viewpoint Canonicalization:** Projected roll direction is signed relative to the skater's facing normal vector $\mathbf{n}_{\text{skater}} = \mathbf{p}_{\text{chest}} \times \mathbf{p}_{\text{up}}$. This ensures toe-side flicks are distinguished from heel-side flicks across frontside vs. backside camera perspectives.

### 4.3 Torso & Body Kinematics
* **Skater Torso Yaw:** Net rotation of bi-acromial and bi-iliac axes between $t_{\text{pop}}$ and $t_{\text{land}}$:
  $$\Delta\theta_{\text{body}} = \text{unwrap}(\theta_{\text{shoulder}}(t_{\text{land}})) - \text{unwrap}(\theta_{\text{shoulder}}(t_{\text{pop}}))$$
* **Center of Mass Trajectory:** Vertical displacement $h_{\text{apex}} = \max_t(z_{\text{com}}(t)) - z_{\text{com}}(t_{\text{pop}})$ and airtime $\Delta t_{\text{air}} = t_{\text{land}} - t_{\text{pop}}$.

---

## 5. Model Architectures & Hierarchy

We implement a rigorous 3-tiered model hierarchy from transparent baselines to graph architectures:

### 5.1 Model A: Calibrated Rule-Based Baseline (`src/classification/rule_classifier.py`)
A transparent, fully deterministic decision tree with calibrated frozen thresholds:
* $t_{\text{pop}}$ and $t_{\text{land}}$ validated via airtime $\Delta t \ge 0.25\text{ s}$.
* Yaw partition: $|\Delta\theta_{\text{net}}| \in [0^\circ, 45^\circ] \to \text{No Yaw}$; $[120^\circ, 240^\circ] \to 180^\circ \text{ Yaw}$; $\ge 290^\circ \to 360^\circ \text{ Yaw}$.
* Roll partition: Aspect ratio cycle count $N_{\text{flip}} = 0 \to \text{No Flip}$; $N_{\text{flip}} \ge 1 \to \text{Flip}$.
* Direction: Conditioned on $\mathbf{v}_{\text{flick}} \cdot \mathbf{n}_{\text{toe}} > 0 \to \text{Kickflip}$; $< 0 \to \text{Heelflip}$.
* Body rotation: $|\Delta\theta_{\text{body}}| \ge 110^\circ \to \text{180 Ollie / Body Varial}$.

### 5.2 Model B: Gradient Boosted Decision Trees (`src/classification/tree_classifier.py`)
* **Algorithm:** Multi-class LightGBM / XGBoost operating on the 48-dimensional tabular kinematic feature vector.
* **Optimization:** Hyperparameters tuned via nested GroupKFold by `skater_id` (outer test skaters completely withheld).
* **Interpretability:** Compute SHAP values to verify biomechanical plausibility of top predictors.

### 5.3 Model C: Spatio-Temporal Graph Convolutional Network (`src/classification/graph_classifier.py`)
* **Input Graph Topology (25 Nodes):** 17 COCO human pose joints + 8 semantic board contour keypoints.
* **Node Features:** Normalized coordinate tuple $(x, y, v_x, v_y, c)$ over standardized sequence length $T=64$.
* **Edges:** Spatial human skeleton (16 edges) + Board perimeter loop (8 edges) + Dynamic foot-board contact edges + Inter-frame temporal edges.
* **Strict Project Rule:** **ZERO raw RGB downstream of Phase 1.** Operates exclusively on $(25, T=64, 5)$ graph tensors.

---

## 6. Post-Impact Land/Bail Verification Gate (P3-A5, P3-A6)

The verification gate operates strictly on $[t_{\text{land}}, t_{\text{land}} + 30]$ to evaluate attempt completion.

```text
               Touchdown (t_land)
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│             POST-IMPACT VERIFICATION WINDOW                 │
│                 [t_land,  t_land + 30]                      │
│                                                             │
│  Stability Cues:                                            │
│  • Dimensionless foot-deck proximity: Dist_norm < 0.55      │
│  • Velocity coherence: |v_skater_x - v_board_x| < tol       │
│  • Board upright roll persistence (no pitch runaway)        │
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
* **`LANDED`:** Both feet maintain contact with the deck ($\text{Dist}_{\text{norm}} \le 0.55$), the board maintains forward momentum without overturning, and the skater rides away for $\ge 15$ frames post-impact.
* **`BAILED`:** Attempt failed: skater steps off onto the ground ($\text{Dist}_{\text{norm}} > 1.0$), board rolls away unridden, skater falls to the ground, or one/both feet hit the pavement before the board settles.
* **`UNCERTAIN`:** The camera cuts off immediately at $t_{\text{land}}$, severe occlusion blocks the board, or the attempt exits the camera frame ($< 8$ post-impact frames). **Excluded from supervised binary training**.

### 6.2 Decoupling from Cleanliness
The gate evaluates **attempt completion only**. A trick landed with toe-drag, hand-drag, sketchy tick-tack, or off-center foot placement is labeled **`LANDED`** (and will be penalized for cleanliness in Phase 4). It is **not** a bail.

---

## 7. High-Throughput Batch Expansion & Stratification Architecture

To resolve the Trajectory Store Starvation identified in the Phase 3 audit, Phase 3 integrates an automated batch ingestion subsystem:

### 7.1 Batch Trajectory Processor (`src/tracking/batch_processor.py`)
* Automatically ingests tracking-eligible clips from `data/metadata/video_manifest.csv`.
* Employs non-destructive checkpointing: checks for existing Parquet trajectory files in `features/v1_trajectories/` and skips reprocessing.
* Balances class representation while maximizing skater diversity across professional athletes (`sean_malto`, `jack_colbourn`, `sewa_kroetkov`, `chris_joslin`, `cody_cepeda`, `shane_oneill`, `luan_oliveira`, `chris_cole`).

### 7.2 Skater-Confounding Control & Anti-Leakage Protocol
To eliminate skater-identity leakage, dataset auditing and nested cross-validation protocols enforce:
* Outer 5-fold GroupKFold by `skater_id` ensuring test skaters are strictly withheld from training and hyperparameter selection.
* Contrastive benchmarking: evaluating both Stratified Cross-Validation (with skater overlap) and Nested GroupKFold to quantify the exact **skater-confounding margin** ($\Delta_{\text{leakage}}$).

---

## 8. Systematic Feature Ablation Matrix (A0–A5)

To disentangle representation modality from parameter capacity explosion, we establish a standardized ablation matrix evaluated across identical nested GroupKFold skater splits:

| Experiment | Modality / Feature Set | Dimensionality ($D$) | Research Question Addressed |
| :---: | :--- | :---: | :--- |
| **A0** | **Sparse Phase-Summary Positional Baseline** | 7 | Simple common baseline: Can discrete keypoints at Pop, Apex, and Land classify tricks? |
| **A1** | **Pose Landmarks Only** | 10 | Can skater body kinematics alone predict the trick without board coordinates? |
| **A2** | **Board Tracking Only** | 18 | Can isolated board motion identify the trick without human context? |
| **A3** | **Pose + Board Coordinates** | 11 | Does fusing human and object geometry outperform isolated tracking? |
| **A4** | **Coordinates + Velocities** | 16 | What is the marginal information gain of 1st-order temporal derivatives? |
| **A5** | **Full Engineered Dynamics** | 39 | Contribution of physical invariants ($\Theta_{\text{abs}}, \Delta\theta_{\text{net}}, C_\theta, \mathbf{v}_{\text{flick}}$). |

---

## 9. Quality Control (QC) Gates & Acceptance Criteria

| Checkpoint / Metric | Hard Operational Gate | Aspirational Research Target | Evaluation Method |
| :--- | :---: | :---: | :--- |
| **Post-Impact Bail F1 (Gate 4)** | $\ge 0.80$ | $\ge 0.90$ | Binary F1 for Landed vs Bailed verification on held-out test skaters. |
| **Bail False Positive Rate (FPR)** | $< 8.0\%$ | $< 3.0\%$ | Percentage of clean landed attempts falsely flagged as bails. |
| **End-to-End Macro F1 (Gate 3)** | $\ge 70.0\%$ | $\ge 85.0\%$ | Macro-averaged F1 across 9 trick classes on unseen skaters with Phase 2 predicted events. |
| **Top-2 Classification Accuracy** | $\ge 85.0\%$ | $\ge 95.0\%$ | Prediction is correct if true trick is within top-2 model probabilities. |
| **Pipeline Latency (RTF)** | $\text{RTF} \le 0.50$ | $\text{RTF} \le 0.10$ | Real-Time Factor on target hardware ($\text{RTF} = t_{\text{proc}} / t_{\text{video}}$). |

---

## 10. Execution Sequence & Work Breakdown

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 3 COMPLETE EXECUTION PHASES                    │
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
│  [Step 3.8] Batch Trajectory Expansion & Stratification Recovery            │
│             • Script: src/tracking/batch_processor.py                       │
│             • Expand store to >= 70-80 canonical clips across 14 skaters    │
│                                                                             │
│  [Step 3.9] Benchmark Synthesis & Formal Report Generation                  │
│             • Output: docs/reports/PHASE_3_REPORT.md                        │
│             • Final QC Gate verification and Phase 3.5 recovery transition  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Master Deliverables Checklist

The unified Phase 3 implementation provides the following deliverables:
1. `src/classification/dataset_audit.py`: Pre-training audit script and generated JSON summary.
2. `src/classification/feature_extractor.py`: Tabular and graph feature extraction engine with viewpoint/stance normalization.
3. `src/classification/bail_gatekeeper.py`: Post-impact tri-state Land/Bail verification model.
4. `src/classification/rule_classifier.py`: Calibrated Model A rule baseline.
5. `src/classification/tree_classifier.py`: Model B XGBoost/LightGBM training pipeline with nested GroupKFold and SHAP analysis.
6. `src/classification/graph_classifier.py`: Model C ST-GCN graph classification pipeline.
7. `src/classification/ablation_runner.py`: A0–A5 systematic ablation test harness.
8. `src/tracking/batch_processor.py`: Automated batch trajectory expansion engine.
9. `src/classification/phase3_pipeline.py`: Master Phase 3 orchestrator and dual evaluation benchmark runner.
10. `docs/reports/phase3_benchmark_results.json`: Complete machine-readable benchmark results.
11. `docs/reports/PHASE_3_REPORT.md`: Comprehensive engineering report detailing methodology, audit findings, model comparisons, confusion matrices, SHAP explanations, failure modes, Gate 3 failure diagnosis, Gate 4 pilot certification, and Phase 3.5 recovery mandate.

---

## 12. Phase 3.5.7 Plan: Physics-Informed Kinematic Engine & Multi-Task Factorization

### 12.1 Core Architectural Thesis
Monocular video creates severe geometric ambiguity for planar 2D rotations. Skateboarding tricks are not arbitrary visual patterns—they are physically governed rigid-body kinematic systems. Rather than forcing a flat or naive tree to classify 9 classes directly from noisy pixel coordinates:
$$\text{Video} \longrightarrow \text{Kinematic State} \longrightarrow \text{Physics Invariants} \longrightarrow \text{Physical Component Heads} \longrightarrow \text{ML Ambiguity Resolver}$$

By decomposing tricks into physical components based on the reference taxonomy (`Skateboarding_Kinematic_Dataset.csv`), we achieve multi-task sample efficiency:
- *Kickflip*, *Heelflip*, *Varial*, and *360 Flip* train the **Flip Head** ($N = 50$).
- *Pop Shove-it*, *FS Shove-it*, *Varial*, *360 Flip*, *BS 180*, and *FS 180* train the **Board Shuv Head** ($N = 80$).
- *BS 180* and *FS 180* train the **Body Spin Head** ($N = 26$).

### 12.2 Implementation Tiers

#### Tier 1 — Physics Invariant Feature Suite
1. **Deck Inversion Invariant ($C_{\text{inv}}$):**
   $$\vec v_{\text{board}}(t) = P_{\text{nose}}(t) - P_{\text{tail}}(t), \quad C_{\text{inv}} = \frac{\vec v_{\text{pop}} \cdot \vec v_{\text{land}}}{\|\vec v_{\text{pop}}\| \|\vec v_{\text{land}}\|}$$
   - $C_{\text{inv}} \approx +1$: Nose/tail orientation preserved (Ollie, pure Flips).
   - $C_{\text{inv}} \approx -1$: Board physically rotated $180^\circ$ (Pop Shove-it, FS Shove-it, 180s).
2. **Temporal Foreshortening Curve ($L(t) = \|P_{\text{nose}}(t) - P_{\text{tail}}(t)\|$):**
   - Trough depth: $1 - \frac{\min L}{L_0}$
   - Trough symmetry: $(t_{\text{trough}} - t_{\text{pop}}) / (t_{\text{land}} - t_{\text{pop}})$
   - Trough duration & recovery slope
   - Pre/post length ratio: $L_{\text{land}} / L_{\text{pop}}$
3. **Body vs. Board Rotational Decoupling & Coupling:**
   - $\theta_{\text{body}}(t)$ from ankle/hip vector, $\theta_{\text{board}}(t)$ from deck vector
   - Relative yaw: $\Delta\theta_{\text{rel}} = \Delta\theta_{\text{board}} - \Delta\theta_{\text{body}}$
   - Angular velocity coupling:
     $$C_{\text{body,board}} = \frac{\sum \dot\theta_b \dot\theta_s}{\sqrt{\sum \dot\theta_b^2 \sum \dot\theta_s^2 + \epsilon}} \in [-1, 1]$$
4. **Ballistic Flight Residuals:**
   - Parabolic fit $y(t) = at^2 + bt + c$ over $[t_{\text{pop}}, t_{\text{land}}]$: record $R^2$, parabolic RMSE, and effective acceleration $a_{\text{eff}} = 2a$.
   - CoM-to-board vertical clearance profile.
5. **Pop & Flick Impulse Mechanics:**
   - Pop impulse proxy $\Delta v_{\text{pop}} = v_y(t_{\text{pop}}^+) - v_y(t_{\text{pop}}^-)$.
   - Transverse flick velocity, peak acceleration, flick onset latency, and clearance.

#### Tier 2 — Multi-Task Physical Component Heads
- **Head 1 (Flip Type):** None (0) vs. Kickflip (1) vs. Heelflip (2)
- **Head 2 (Body Spin):** 0° (0) vs. 180° Frontside (1) vs. 180° Backside (2)
- **Head 3 (Board Shuv):** 0° (0) vs. 180° Pop Shov (1) vs. 180° FS Shov (2) vs. 360° Shuv (3)

#### Tier 3 — Physical Consistency Constraints
Reject physically invalid state combinations via Bayesian prior masking:
- $(\text{Body}=0^\circ, \text{Shuv}=0^\circ, \text{Flip}=\text{None}) \implies \text{Ollie}$
- $(\text{Body}=0^\circ, \text{Shuv}=180^\circ\text{ BS}, \text{Flip}=\text{None}) \implies \text{Pop Shove-it}$
- $(\text{Body}=180^\circ\text{ BS}, \text{Shuv}=180^\circ\text{ BS}, \text{Flip}=\text{None}) \implies \text{Backside 180}$
- $(\text{Body}=0^\circ, \text{Shuv}=180^\circ\text{ BS}, \text{Flip}=\text{Kick}) \implies \text{Varial / Hardflip}$
- $(\text{Body}=0^\circ, \text{Shuv}=360^\circ\text{ BS}, \text{Flip}=\text{Kick}) \implies \text{360 Flip}$

#### Tier 4 — ML Ambiguity Resolution & Experimental Matrix
Evaluate structured configurations under strict 4-split nested GroupKFold on the 110 clean pro attempts:
1. **Config 0:** Current B2 Hierarchical Baseline (28.04% Macro F1)
2. **Config 1:** B2 + Tier 1 Physics Invariant Features
3. **Config 2:** Multi-Task Factorized Component Heads
4. **Config 3:** Multi-Task Heads + Physical Consistency Constraints
5. **Config 4:** Full Physics-Informed Kinematic Engine

---

## 13. Phase 3.5.8 — Physics-First Temporal Recognition Engine

### 13.1 Motivation & Architectural Paradigm Shift
Empirical findings in Phase 3.5.7 revealed that adding more features to an 72D flat vector or fusing probabilistic heads with strict Bayesian multiplication yields diminishing returns (Flat XGBoost 13.35%, B3 Product Bayes 17.44%), while physical decomposition in B2 proved superior (28.04% Macro F1, 52.73% Top-2). However, planar board rotation (Pop Shove-it vs. FS Shove-it) remained unresolved (0% F1 in B2) because monocular projection creates 2D directional ambiguity.

Phase 3.5.8 replaces high-dimensional direct classification with a **physics-first, phase-aligned temporal recognition architecture**:

$$\boxed{
\text{Trajectory}
\longrightarrow
\text{Physical State}
\longrightarrow
\text{Physics Candidate Routing}
\longrightarrow
\text{Temporal Signature Matching (DTW)}
\longrightarrow
\text{Small ML Resolver}
}$$

- **Physics determines what motions are physically plausible.**
- **Temporal trajectory matching determines which possible trick the attempt's kinematic shape most closely follows.**
- **Lightweight ML resolves residual ambiguities within the routed candidate set.**

---

### 13.2 System Layers

#### Layer A — Compact Deterministic Physical State Consensus
Derive a compact physical state from multi-signal evidence rather than a single brittle threshold:
1. **Body State ($S_{\text{body}} \in \{-1, 0, +1\}$):**
   - Detect $180^\circ$ body rotation using ankle/hip/shoulder spatial inversion and integrated body yaw.
   - `-1`: Backside rotation; `0`: No body spin; `+1`: Frontside rotation.
2. **Board State ($S_{\text{board}} \in \{-180, 0, +180, \pm 360\}$):**
   - Multi-signal consensus combining:
     - Nose-tail inversion invariant $C_{\text{inv}}$
     - Temporal foreshortening trough depth and symmetry
     - Accumulated board yaw and rotational velocity
     - Board-axis recovery profile
3. **Flip State ($S_{\text{flip}} \in \{-1, 0, +1\}$):**
   - Transverse foot flick velocity $\dot y_{\text{front}}$, board aspect-ratio oscillation, foot-board relative clearance, and flip onset timing.
   - `-1`: Kickflip flick; `0`: No flip; `+1`: Heelflip flick.

$$\text{Physical State Vector} = [S_{\text{body}}, S_{\text{board}}, S_{\text{flip}}]$$

```text
Ollie       = [0,    0,   0]
BS180       = [-1, -180,  0]
FS180       = [+1, +180,  0]
Pop Shove   = [0,  -180,  0]
FS Shove    = [0,  +180,  0]
Kickflip    = [0,    0,  -1]
Heelflip    = [0,    0,  +1]
Varial      = [0,  -180, -1]
360 Flip    = [0,  -360, -1]
```

#### Layer B — Signed Scoop Sweep Momentum ($\tau_{\text{scoop}}$)
To definitively disambiguate **Pop Shove-it vs. FS Shove-it**, compute the signed planar sweep momentum of the tail relative to the board centroid / body coordinate frame:

$$\tau_{\text{scoop}} = \sum_{t=t_{\text{pop}}}^{t_{\text{apex}}} \left( \vec r_{\text{tail}}(t) \times \dot{\vec r}_{\text{tail}}(t) \right)_z$$

where $\vec r_{\text{tail}}(t) = P_{\text{tail}}(t) - P_{\text{centroid}}(t)$.
- $\tau_{\text{scoop}} > 0$: Clockwise / Backside sweep (Pop Shove-it in regular stance).
- $\tau_{\text{scoop}} < 0$: Counter-clockwise / Frontside sweep (FS Shove-it in regular stance).
- Stance calibration: Sign convention is calibrated per regular vs. goofy skater stance to maintain rotational invariance.

#### Layer C — Phase-Aligned Temporal Trajectory Normalization
Preserve the full dynamic sequence across the flight phase without lossy aggregation:
- Every trick flight $[t_{\text{pop}}, t_{\text{land}}]$ is resampled to a fixed temporal grid: $X(t) \in \mathbb{R}^{64 \times D}$, where $t \in [0, 63]$ ($0\% = \text{POP}$, $50\% = \text{APEX}$, $100\% = \text{LAND}$).
- $D$ normalized physical kinematic channels:
  1. Board yaw angle $\theta_{\text{board}}(t)$
  2. Board angular velocity $\dot\theta_{\text{board}}(t)$
  3. Board apparent foreshortened length $L_{\text{board}}(t) / L_0$
  4. Body yaw angle $\theta_{\text{body}}(t)$
  5. Body angular velocity $\dot\theta_{\text{body}}(t)$
  6. Relative body-board yaw $\theta_{\text{rel}}(t) = \theta_{\text{board}}(t) - \theta_{\text{body}}(t)$
  7. Front foot-to-board distance $d_{\text{front,board}}(t)$
  8. Back foot-to-board distance $d_{\text{back,board}}(t)$
  9. Vertical board flight trajectory $y_{\text{board}}(t) - y_{\text{pop}}$
  10. Skater CoM to board vertical clearance $y_{\text{CoM}}(t) - y_{\text{board}}(t)$

#### Layer D — Physics-Guided Temporal Prototypes & DTW
- **Training-Fold Only Prototypes:** For each canonical trick $T \in \{1 \dots 9\}$, compute a class physical prototype trajectory $P_T \in \mathbb{R}^{64 \times D}$ exclusively from training-fold skaters:
  $$P_T(t) = \frac{1}{|S_T|} \sum_{i \in S_T} X_i(t)$$
- Measure the temporal alignment cost between an unseen test attempt $X$ and each class prototype $P_T$ using Dynamic Time Warping (DTW) with a Sakoe-Chiba band:
  $$D_T = \text{DTW}(X, P_T)$$

#### Layer E — Continuous Soft Physics Contradiction Cost
Rather than hard, brittle Bayesian masking, compute a smooth contradiction penalty $E(T \mid X) \ge 0$:
- Evaluates kinematic violations (e.g. candidate is Ollie but $|\Delta\theta_{\text{board}}| > 120^\circ$ or $C_{\text{inv}} < -0.3$).
- Pop Shove-it with large body rotation incurs high $E(T \mid X)$, whereas BS180 does not.
- Prevents tracking glitches on single frames from permanently zeroing out the true class.

#### Layer F — Hybrid Scoring & Candidate Routing
1. **Candidate Routing:** Compact physical state $S$ identifies a small candidate set $\mathcal{C} \subset \{1 \dots 9\}$ (typically 2 to 4 tricks) based on physical plausibility.
2. **Hybrid Trick Scoring:** For candidate $T \in \mathcal{C}$:
   $$\text{Score}(T) = w_1 D_T^{\text{temporal}} + w_2 E(T \mid X)^{\text{physics}} + w_3 D_T^{\text{rotation}} + w_4 D_T^{\text{foot}} - w_5 \log P_{\text{ML}}(T)$$
   The candidate with the lowest cost wins: $\hat T = \arg\min_{T \in \mathcal{C}} \text{Score}(T)$.

---

### 13.3 Experimental Matrix (Phase 3.5.8)

All configurations evaluated under the identical strict 4-split nested GroupKFold by skater across 110 clean pro attempts:

| Config | Model System | Research Question Addressed |
|---|---|---|
| **C0** | Current B2 Baseline | Reference performance (Macro F1 = 28.04%, Top-2 = 52.73%) |
| **C1** | Compact Physics State Rule | Are deterministic physical invariants alone sufficient? |
| **C2** | Physics Routing + Signed Scoop | Can signed scoop sweep $\tau_{\text{scoop}}$ recover Pop Shove & FS Shove? |
| **C3** | Temporal Prototypes + DTW | Does phase-aligned temporal motion shape generalize across unseen skaters? |
| **C4** | Physics + Temporal Prototypes | Does combining physical invariants with DTW matching beat pure DTW? |
| **C5** | Full Hybrid (Physics + DTW + Small ML) | Does lightweight ML ambiguity resolution maximize Macro F1 & MinClass F1? |

### 13.4 Strict Success Criteria
1. **Shove-it Recovery:** Pop Shove-it and FS Shove-it F1 must both become non-zero without degrading Varial / Hardflip (53.7%) and 360 Flip (50.0%).
2. **Macro F1 Improvement:** Statistically superior or competitive with B2 baseline across unseen skaters.
3. **MinClass F1:** Exceed 0.00% across all 9 canonical trick classes.

---

## 14. Phase 3.5.9 — Cross-Skater Generalization + Kinematic State Routing

### 14.1 Objective & Paradigm Pivot
Shift primary research focus from:
> **"Can we squeeze more F1 out of B4?"**

to:
> **"Can we build a recognition system that actually generalizes to unseen skaters?"**

The empirical Phase 3.5.8 benchmark established the **frozen B4 baseline**:
- **Macro F1:** 28.08%
- **Top-1 Accuracy:** 30.00%
- **Top-2 Accuracy:** 46.36%
- **Min-Class F1:** 9.09% (All 9 canonical classes non-zero; Pop Shove recovered to 12.5%, FS Shove recovered to 26.7%)

---

### 14.2 Phase 3.5.9A — Fix the Data Bottleneck First (Highest Priority)
Cross-skater generalization is structurally bounded by extreme skater concentration in canonical holdouts:

| Canonical Trick | Current Dataset Status | Target Representation |
|---|---|---|
| **Ollie** | 2 skaters (monopolized) | $\ge 5\text{--}6$ independent skaters |
| **Backside 180** | 3 skaters | $\ge 5\text{--}6$ independent skaters |
| **Frontside 180** | 4 skaters | $\ge 5\text{--}6$ independent skaters |
| **Pop Shove-it** | 3 skaters | $\ge 5\text{--}6$ independent skaters |
| **Frontside Shove-it** | 4 skaters (limited diversity) | $\ge 5\text{--}6$ independent skaters |

#### Ingestion Protocol & Quality Gate
- Ingest approximately **15–20 visually verified clips**, strictly prioritizing **new unseen skaters** (zero additional clips from already over-represented skaters like Player A).
- **Mandatory Clip Verification Checklist:**
  1. Manual ground-truth trick verification.
  2. Frame-accurate boundary marking: `gt_pop`, `gt_apex`, `gt_catch`, `gt_land`.
  3. Camera viewpoint characterization (lateral / oblique / frontal).
  4. Outcome verification (clean landed vs. bail rollout).
  5. Deterministic ingestion through Phase 1 tracking and Phase 2 event segmentation pipelines.

---

### 14.3 Phase 3.5.9B — Establish the Ollie / Shove-it Physical Boundary
Evaluate a crisp physical foreshortening gate before complex downstream matching:

```text
if min(L_board(t) / L_0) > 0.65:
    prohibit Shove-it candidates (enforce non-rotating planar candidate set)
```

**Interpretation:** If there is insufficient deck foreshortening or yaw inversion evidence during flight, the trick cannot physically be a Pop Shove-it or FS Shove-it.
- **Evaluation Criteria:** Monitor Ollie F1, Pop Shove-it F1, FS Shove-it F1, Macro F1, Top-2 accuracy, and confusion matrix.
- If the hard constraint causes brittle clipping under oblique camera angles, soften to an adaptive continuous penalty.

---

### 14.4 Phase 3.5.9C — Build the Kinematic Tokenizer
Convert noisy, high-dimensional continuous trajectories into a compact, robust physical state token representation with continuous confidence scores:

```text
SNAP
├── NEUTRAL (conf)
├── BS_SCOOP (conf)
└── FS_SWEEP (conf)

FLICK
├── NONE (conf)
├── KICK_FLICK (conf)
└── HEEL_FLICK (conf)

BOARD
├── PRESERVED (conf)
├── INVERTED (conf)
└── FULL_SPIN (conf)

BODY
├── STATIONARY (conf)
├── BS180 (conf)
└── FS180 (conf)

FORESHORTENING
├── NONE (conf)
├── MODERATE (conf)
└── STRONG (conf)
```

*Example Token Output:*
```text
BS_SCOOP        : 0.84
FORESHORTENING  : 0.76
INVERSION       : 0.11
BS180           : 0.06
```

---

### 14.5 Phase 3.5.9D — Hierarchical Candidate Routing
Route attempts through physically decoupled candidate families before ML resolution:

```text
                    Kinematic State Vector
                              │
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
           Body Yaw       Board Flip      No Major Rotation
              │               │               │
            180s            Flips        Ollie / Shove
              │               │               │
         FS180 / BS180   Kickflip / Heel  Pop Shov / FS Shov / Ollie
```

The ML model operates strictly as a **fine-grained ambiguity resolver** within the routed candidate set, rather than learning global cross-family boundaries from scratch.

---

### 14.6 Phase 3.5.9E — Multi-Template Matching ($K=2$ Medoid Prototypes)
Address skater style variation by replacing the single mean prototype with $K=2$ exemplar medoid templates per class (computed strictly from training skaters per fold):

```text
Pop Shove-it
    ├── Prototype A: Compact snappy pop style
    └── Prototype B: Extended high-clearance pop style
```

Temporal distance scoring:
$$D(X, \text{Class}) = \min \left( \text{DTW}(X, P_{T, A}), \text{DTW}(X, P_{T, B}) \right)$$

---

### 14.7 Phase 3.5.9F — Experimental Comparison Progression

All systems evaluated under the exact 4-split nested GroupKFold by skater:
1. **B4 Baseline:** Reference frozen system (Macro F1 = 28.08%, Top-1 = 30.00%).
2. **B4 + Ollie / Shove Boundary:** Testing the foreshortening threshold rule.
3. **B4 + Kinematic Tokens:** Discrete confidence token representation.
4. **B4 + Multi-Template Prototypes:** $K=2$ medoid temporal templates.
5. **Full Phase 3.5.9 System:** Tokens + Hierarchical Routing + $K=2$ Prototypes + ML Ambiguity Resolver.

#### Target Metrics
- Strict unseen-skater Macro F1 (primary metric).
- Top-1 and Top-2 accuracy.
- Min-Class F1 across all 9 classes.
- Ollie / Shove confusion matrix reduction.
- Per-skater generalization variance.

---

### 14.8 End-to-End Phase 3.5.9 Architecture

```text
                        RAW VIDEO CLIP
                              │
                              ▼
                    Phase 1/2 Kinematics
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Kinematic Tokenizer  │
                   │                     │
                   │ • Snap Momentum     │
                   │ • Scoop Direction   │
                   │ • Flick Velocity    │
                   │ • Board Inversion   │
                   │ • Body Yaw Angle    │
                   │ • Foreshortening    │
                   └──────────┬──────────┘
                              │
                              ▼
                    Physical State Vector
                              │
                              ▼
                    Candidate Trick Family
                              │
                   ┌──────────┴──────────┐
                   ▼                     ▼
            K-Medoid Templates        B4 ML
                   │                     │
                   └──────────┬──────────┘
                              ▼
                      Ambiguity Resolver
                              │
                              ▼
                     Final Trick Decision
```

---

### 14.9 Execution Priority Order & Success Criteria

1. **Expand cross-skater data** (Highest priority — 15–20 verified clips across new skaters).
2. **Freeze B4 baseline** (28.08% Macro F1, 9.09% Min-Class F1).
3. **Test Ollie / Shove boundary** (Foreshortening constraint experiment).
4. **Implement kinematic tokenizer** (Continuous confidence physical tokens).
5. **Implement hierarchical candidate routing** (Family-level decoupling).
6. **Test $K=2$ exemplar prototypes** (Multi-medoid style alignment).
7. **Benchmark with strict nested GroupKFold** (Zero skater leakage).
8. **Decide on learned architecture requirements**.

#### Phase 3.5.9 Success Criterion
> **Demonstrate a meaningful improvement in unseen-skater generalization (moving Macro F1 materially above 28.08%, improving Min-Class F1, and reducing Ollie/Shove confusion) under strict skater separation.**

*Phase 4 Cleanliness Scoring remains paused until this cross-skater generalization criterion is met.*

---

### 14.10 Grounding in Academic Literature & External Data Catalog
Detailed analysis of 3 new peer-reviewed reference papers has been incorporated into [`docs/references/PAPER_ANALYSIS_PHASE_3_5_9.md`](file:///d:/Future%20Career/Skateboard%20Trick%20Recognition%20&%20Cleanliness%20Scoring%20Engine/docs/references/PAPER_ANALYSIS_PHASE_3_5_9.md):

1. **Shilaskar et al. (IEEE 2023) — Video Action Recognition (Ollie & Kickflip):**
   - Evaluated 2D CNN, ConvLSTM2D (79% acc), and MobileNet+BiLSTM (75% acc) on 222 clips (`LightningDrop/SkateboardML`).
   - Proved that end-to-end 2D vision saturates at ~75–79% even on binary Ollie vs. Kickflip classification, proving the necessity of SkateKine's explicit kinematic tracking.
   - **Data Integration:** All 222 clips have been organized into [`data/raw_videos/skateboard_ml/`](file:///d:/Future%20Career/Skateboard%20Trick%20Recognition%20&%20Cleanliness%20Scoring%20Engine/data/raw_videos/skateboard_ml/) and cataloged in [`data/metadata/skateboard_ml_manifest.csv`](file:///d:/Future%20Career/Skateboard%20Trick%20Recognition%20&%20Cleanliness%20Scoring%20Engine/data/metadata/skateboard_ml_manifest.csv) to alleviate the Phase 3.5.9A Ollie data bottleneck.

2. **Hollaus et al. (IEEE 2023) — Motion-Based Trick Classification (12 Skaters, IMU):**
   - **Ollie Fallback Black Hole:** Confirmed that Ollie is the baseline phase of all tricks; low-confidence rotational trick attempts default to Ollie. Directly motivates Phase 3.5.9B's strict physical gate $\min(L_{\text{board}}/L_0) > 0.65$ before allowing Ollie fallback.
   - **Stance Symmetry / Inversion:** Confirmed that Goofy BS180 mirrors Regular FS180 in absolute rotation coordinates. Phase 3.5.9C adopts explicit stance sign normalization: $\Delta \psi_{\text{norm}} = \text{sign}(\text{stance}) \cdot \Delta \psi$.

3. **Abdullah, Zakaria et al. (PeerJ 2021) — CWT Time-Frequency Scalograms & SVM:**
   - Evaluated Continuous Wavelet Transform (Morlet CWT) + transfer learning feature extractors + Linear SVM (100% test accuracy).
   - **Planar Scoop Frequency Energy:** Confirmed that Pop Shove-it and Nollie FS Shuvit produce distinct high-frequency energy bursts during the pop-to-apex scoop window, motivating the transient aspect-ratio velocity metric for Phase 3.5.9C.



