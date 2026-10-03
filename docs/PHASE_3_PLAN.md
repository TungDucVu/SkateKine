# Phase 3 Implementation Plan: Trick Classification Engine & Land/Bail Verification

---

## 1. Overview & Scope

Phase 3 implements the core action recognition and execution gating intelligence of SkateKine. Following the guiding principle that recognition and quality must be decoupled, this phase focuses exclusively on two discrete classification tasks:
1. **Trick Categorization:** Determining which specific trick was performed across the 9 primary flatground trick classes using structured kinematic features on unseen skaters.
2. **Land vs. Bail Verification Gating:** Determining whether the attempt was successfully landed or bailed, serving as a critical upstream gatekeeper before cleanliness scoring.

In accordance with scientific rigor, deep graph neural networks are evaluated strictly as candidate improvements over interpretable tree ensembles and rule-based kinematic baselines.

### Key Deliverables of Phase 3:
1. **Model A (Kinematic Rule-Based Baseline):** Analytical decision logic based on integrated rotation, flip-direction sign, and airtime.
2. **Model B (Gradient Boosted Decision Trees):** XGBoost / LightGBM classifier trained on extracted spatio-temporal kinematic summary statistics.
3. **Model C (Sequence & Graph Models):** Temporal Convolutional Network (TCN) and Spatio-Temporal Graph Convolutional Network (ST-GCN).
4. **Feature Ablation Suite (A1–A5):** Systematic empirical comparison verifying the contribution of each kinematic modality.
5. **Land vs. Bail Gatekeeper Classifier:** Binary classifier filtering out failed attempts over the post-landing interval $[t_{\text{land}}, t_{\text{land}} + 30]$.

---

## 2. Core Assumptions & Boundary Constraints

1. **Decoupled Formulation:** A poorly executed trick (e.g. sketchy landing, late catch) is still classified under its intended trick category, provided the rotation and flip mechanics were executed; aesthetic quality is reserved for Phase 4.
2. **Unseen-Skater Generalization Mandate:** All model evaluations are performed on test folds featuring skaters and camera viewpoints never seen during training. In-sample frame splits are forbidden.
3. **Projected 2D Rotation Bound:** Flips and rotations are evaluated via projected 2D rotation $\Delta\theta_{\text{image-plane}}$ and aspect-ratio transitions (deck width-to-length ratio) rather than unconstrained 3D Euler angles.
4. **Bail Filtration Invariance:** If an attempt is classified as `BAILED`, the cleanliness scoring pipeline must be completely bypassed to prevent generating spurious quality scores.

---

## 3. Step-by-Step Implementation Sequence

### Step 3.1: Kinematic Summary Feature Extractor (`src/classification/feature_extractor.py`)
* Extract temporal window metrics over the flight phase $[t_{\text{pop}}, t_{\text{catch}}]$:
  * Total airtime: $\Delta t_{\text{flight}} = t_{\text{land}} - t_{\text{pop}}$.
  * Maximum apex clearance: $h_{\text{apex}} = \max(y_{\text{ground}} - y_{\text{deck}}(t))$.
  * Total cumulative in-plane deck rotation: $\Theta = \int_{t_{\text{pop}}}^{t_{\text{catch}}} |\omega(t)| dt$.
  * Flip-direction sign: $\text{sgn}\left(\int_{t_{\text{pop}}}^{t_{\text{apex}}} \dot{W}_{\text{aspect}}(t) dt\right)$.
  * Front foot flick vector: relative displacement and velocity of front ankle landmark relative to board nose at $t_{\text{pop}} + \delta$.
  * Skater body rotation: hip and shoulder rotation vector $\Delta\theta_{\text{body}}$.
* Format extracted features into tabular datasets aligned with skater-wise splits.

### Step 3.2: Model A — Rule-Based Kinematic Baseline (`src/classification/rule_classifier.py`)
* Construct deterministic decision tree:
  * *Pop check:* Flight duration $\Delta t_{\text{flight}} \ge 0.35\text{ s}$ and apex height $> 15\text{ cm}$.
  * *Rotation check:* $|\Theta| \approx 0^\circ$ (Ollie/Flips), $\approx 180^\circ$ (Shuvit/Varial), $\approx 360^\circ$ (360 Shuvit/Tre Flip).
  * *Flip check:* Aspect ratio cycle count (0 for Ollie/Shuvit, 1 for Kickflip/Heelflip/Tre Flip).
  * *Flip direction:* Positive roll $\to$ Heelflip; Negative roll $\to$ Kickflip.
  * *Body rotation:* Skater torso yaw $\approx 180^\circ$ $\to$ 180 Ollie.
* Benchmark rule-based accuracy as baseline zero.

### Step 3.3: Model B — XGBoost / LightGBM Classifier (`src/classification/tree_classifier.py`)
* Train multi-class gradient boosted trees on engineered kinematic feature sets.
* Optimize hyperparameters (max depth, learning rate, regularization) via GroupKFold cross-validation grouped by `skater_id`.
* Compute feature importances (SHAP values) to verify biomechanical plausibility of predictors.

### Step 3.4: Model C — ST-GCN Sequence Model (`src/classification/graph_classifier.py`)
* Define spatio-temporal graph topology connecting 17 human joints and 8 board keypoints with inter-frame temporal edges.
* Train ST-GCN on normalized coordinate sequences.
* Compare validation and unseen-skater test performance against Model B.

### Step 3.5: Systematic Ablation Study Matrix (`src/classification/ablation_runner.py`)
Execute the five formal ablation experiments on identical skater splits:
* **A1:** Pose landmarks only (Can skater body kinematics alone predict the trick?).
* **A2:** Board tracking only (Can board motion alone identify the trick?).
* **A3:** Pose + Board raw coordinates.
* **A4:** Coordinates + Velocities $(\mathbf{x}, \mathbf{v})$.
* **A5:** Full engineered dynamics $(\mathbf{x}, \mathbf{v}, \mathbf{a}, \omega, \mathbf{p}_{\text{foot}}^{\text{board}})$.
* Generate confusion matrices and per-class F1 comparisons across A1–A5.

### Step 3.6: Land vs. Bail Gating Classifier (`src/classification/bail_gatekeeper.py`)
* Ingest 30 frames post-impact $[t_{\text{land}}, t_{\text{land}} + 30]$.
* Extract stability indicators:
  * Foot-deck distance persistence: $\sum_{t=t_{\text{land}}}^{t_{\text{land}}+30} \Vert\mathbf{p}_{\text{feet}}(t) - \mathbf{p}_{\text{board}}(t)\Vert$.
  * Board lateral roll variance and pitch divergence.
  * Skater Center-of-Mass (CoM) trajectory continuity (detecting sudden ground slips or falls).
* Train binary classifier (Target $\text{F1} \ge 0.90$, Operational min $\text{F1} \ge 0.80$).
* Implement gatekeeper wrapper: if `status == BAILED`, output bail telemetry and halt downstream scoring.

---

## 4. Post-Phase Checkpoint: QC, Evaluation & Acceptance Criteria

### Quality Control (QC) Gates

| Metric / Checkpoint | Hard Operational Gate | Aspirational Target | Verification Method |
| :--- | :--- | :--- | :--- |
| **Trick Recognition Macro F1** | $\ge 85.0\%$ | $\ge 90.0\%$ | Macro-averaged F1 across 9 trick classes on unseen-skater test set. |
| **Minimum Class F1** | $\ge 75.0\%$ for every class | $\ge 82.0\%$ for every class | Lowest single-class F1 score (preventing severe class imbalance collapse). |
| **Land vs. Bail Gating F1** | $\ge 0.80$ | $\ge 0.90$ | Binary F1 on held-out test attempts containing real-world flatground bails. |
| **Bail False Positive Rate** | $< 8.0\%$ | $< 3.0\%$ | Landed tricks incorrectly flagged as bails (clean attempts wrongly discarded). |
| **Inference Latency** | $< 40\text{ ms}$ per clip | $< 15\text{ ms}$ per clip | CPU/GPU execution time for classification and gating inference. |

---

## 5. Phase Documentation & Reporting Requirements

Upon completing Phase 3, generate `docs/reports/PHASE_3_REPORT.md` including:
1. **Executive Summary:** Overview of models trained, ablation results, and classification accuracy.
2. **Work Completed:** Details on feature extraction, Model A/B/C architectures, ablation runner, and the bail gatekeeper.
3. **Assumptions Made:** Formal documentation of classification assumptions (9-class flatground scope, 2D projected rotation bounds, bail threshold definition).
4. **Artifacts Built on Assumptions:** Mathematical definitions of kinematic features, graph topologies, and bail stability indices.
5. **QC Results vs. Expected Targets:** Full quantitative tables:
   * Per-class precision, recall, and F1 score for Model A, B, and C.
   * Complete Ablation Matrix results (A1–A5).
   * Confusion matrix heatmap and Land/Bail F1 scores vs. Gate 3 & Gate 4 criteria.
6. **Failure Analysis & Edge Cases:** Examination of confusion pairs (e.g. Heelflip vs Kickflip on switch stances, Varial vs Tre Flip under-rotation).
7. **Sign-off Decision:** Clear GO / NO-GO recommendation for advancing to Phase 4.
