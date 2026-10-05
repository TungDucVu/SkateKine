# Phase 3 Execution & Quality Control Report: Trick Classification Engine & Post-Impact Land/Bail Verification

**Execution Date:** 2026-10-05  
**Dataset Universe:** 1,035 Total Manifest Clips | 688 Tracking-Eligible Canonical Candidates | **125 Active Trajectory-Store Parquet Clips** (110 Canonical 9-Class Attempts across 14 Pro Skaters + 15 Non-Canonical/Bail Attempts)  
**Target Gates:** Gate 3 (Trick Recognition Engine) & Gate 4 (Post-Impact Land/Bail Verification)  
**Exit Gate Status:** 
* **GATE 3:** **FAILED / RECOVERY IN PROGRESS** (Model B2 Hierarchical: 28.04% Macro F1, 31.82% Top-1, 51.82% Top-2; Flat GBDT: 16.88% Unweighted / 17.65% Class-Weighted Macro F1 $\ll$ 82.0% Target; Min Class F1 = 0.0%)
* **GATE 4:** **PASS (PILOT CERTIFIED / PROVISIONAL)** (92.55% Landed F1, 70.0% Bail Recall [21/30 bails caught], 5.43% Bail FPR $\le$ 8.0% Target)
* **PHASE 4 STATUS:** **PAUSED** (Recognition recovery and dataset balancing mandated under Phase 3.5)

---

## 1. Executive Summary & Master Evaluation Scorecard

Phase 3 implements the core action recognition and execution verification intelligence of SkateKine. Following the foundational architectural mandate that **trick classification and execution cleanliness scoring must remain strictly decoupled** (with zero raw RGB permitted downstream of Phase 1), Phase 3 operates strictly downstream of Phase 1 spatial trajectories and Phase 2 temporal event boundaries to resolve two discrete physical questions:
1. **What trick was attempted?** (Multi-class categorization across the 9 primary flatground trick classes).
2. **Was the attempt landed or bailed?** (Post-impact rollout verification gatekeeper).

Under **Step 3.5.5: Literature-Informed Physics Refinement & Stance Augmentation** and **Step 3.5.6: Hierarchical Kinematic Factorization**, we incorporated key insights from prior academic literature (Juriga 2023, Chen 2023) and the 70-trick reference taxonomy (`Skateboarding_Kinematic_Dataset.csv`):
* **Differential Yaw Physics:** Implemented continuous normalized foot and board displacement ratios ($\Delta\theta_{\text{diff}} = \text{Board Yaw} - \text{Body Yaw}$) to separate flat-ground spins from Ollies.
* **Toe/Heel Flick Projection:** Added body-centric local Y foot displacement to decouple symmetrical flip confusion.
* **In-Fold Stance Mirroring Augmentation:** Applied left-to-right physical mirroring (Regular $\leftrightarrow$ Goofy stance swap) inside training splits to enforce stance invariance without data leakage.
* **Hierarchical Kinematic Factorization (Model B2):** Decomposed flat 9-class classification into a physically structured multi-stage tree (Stage 1: Flip vs Flat; Stage 2A: 180 vs Straight Pop; Stage 2B: Flip Sub-tree Specialist). This broke the "Ollie Black Hole" and lifted Macro F1 from 16.88% to 28.04% (+11.16% absolute lift, +66% relative gain!).

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PHASE 3 MASTER STATUS SCORECARD                    │
│                                                                             │
│  [1] Architectural & Pipeline Hygiene : 9.8 / 10 (EXEMPLARY)                │
│      • Zero raw RGB leakage, stance normalization, runtime = 0.43 ms/clip   │
│      • Real-Time Factor (RTF) = 0.00017 (5,800x faster than real-time)      │
│                                                                             │
│  [2] Gate 4 (Post-Impact Land/Bail)   : PASS (PILOT CERTIFIED / PROVISIONAL) │
│      • Confusion Matrix: [[21, 9], [5, 87]] + 3 Uncertain (125 clips total)  │
│      • F1-Score: 92.55% | Precision: 90.62% | Landed Recall: 94.57%         │
│      • Bail Detection Recall: 70.0% (21/30 caught, up from 25.0% in pilot)  │
│      • False Positive Rate on Landed: 5.43% (5/92 clean attempts rejected)  │
│                                                                             │
│  [3] Trajectory Store Representation : 110 Canonical Attempts (125 Parquet) │
│      • Universe: 1,035 manifest clips -> 688 eligible -> 125 in active store│
│      • Multi-Skater Roster           : 14 Professional Skaters              │
│      • 100% Saturation of Non-Player A Basics: Ollie (2 sk), BS180/PopShov(3)│
│                                                                             │
│  [4] Gate 3 (Trick Recognition Engine): FAILED / RECOVERY IN PROGRESS       │
│      • Model B2 (Hierarchical Tree)  : Macro F1 = 28.04% | Top-1: 31.82%    │
│      • Model B2 Top-2 Accuracy       : 51.82% (up from 33.64% baseline)     │
│      • Absolute Macro F1 Lift        : +11.16% (+66% relative improvement)  │
│      • Class Revival                 : BS180 (43.5%), Kickflip (33.3%),     │
│                                        FS180 (28.6%), Heel (21.1%),         │
│                                        360 Flip (50.0%), Varial (53.7%)     │
│      • Target Thresholds              : Macro F1 >= 82.0% | Min Class >= 70% │
│                                                                             │
│  [5] Model C (ST-GCN Graph Audit)    : AUDITED & REGULARIZED (7.81% F1)     │
│      • Audited root-relative, torso-normalized, drop-edge p=0.2 6-node GCN  │
│      • Fold 1 reaches 22.7% on unseen pro skaters                           │
│      • Diagnosis: Deep GCNs suffer acute sample starvation at N=110 clips   │
│                                                                             │
│  [6] Skater-Confounding Thesis Margin: +21.70% Performance Gap Quantified   │
│      • Stratified 5-Fold (With Skater Overlap) : 38.58% Macro F1 | 40.91% Acc│
│      • Nested GroupKFold (Strict Unseen Skater): 16.88% Macro F1 | 19.09% Acc│
│      • Confirms strong performance inflation under skater overlap           │
│                                                                             │
│  [7] Systematic Ablations (A0-A5)    : Physical Invariance Confirmed        │
│      • A2 (Board Dynamics Only)      : 15.1% Macro F1 | 17.3% Acc           │
│      • A5 (Full Dynamics)            : 14.6% Macro F1 | 17.3% Acc           │
│      • A0 (Sparse Positional)        : 13.9% Macro F1 | 15.5% Acc           │
│                                                                             │
│  [8] Phase 4 Cleanliness Status      : PAUSED PENDING PHASE 3.5 RECOVERY    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Key Research Insights & Empirical Diagnoses

### 2.1 Insights from Academic References & Literature
An analysis of published research in `data/references/` revealed key context for why skateboarding trick recognition on keypoints is an open academic challenge:
1. **Juriga (2023, MCI Innsbruck):** Evaluated multivariate LSTM/GRU architectures (MLSTM-FCN, MGRU-FCN) on 439 clips using YOLOv8 keypoints across 6 basic tricks. **The best academic model achieved only 36.36% accuracy**, identifying skater style differences and camera angle shifts as the primary limiting factors. He demonstrated that **horizontal flip mirroring and translation augmentation** was the single most effective intervention (+12% lift).
2. **Chen (2023, Harbin Institute of Technology):** Trained CNN-BiLSTM and I3D models on 750 raw video clips, reporting 84% accuracy. However, Chen evaluated on random train/test splits where the same skaters and backgrounds appeared in both sets, confirming our finding that random video splits inflate performance through background and identity memorization.
3. **Kinematic Decomposition (`Skateboarding_Kinematic_Dataset.csv`):** Formally decomposes all flatground skateboarding tricks into independent kinematic axes: Body Spin ($0^\circ, 180^\circ$), Board Shuv ($0^\circ, 180^\circ, 360^\circ$), and Board Flip ($0, 1, 2$). This confirms that multi-task/hierarchical factorized prediction is the structurally sound path for Phase 3.5.

### 2.2 Empirical Lift from Step 3.5.5 Refinements
Implementing bounded differential yaw ($\text{diff\_yaw\_norm}$), toe/heel local Y flick ($\text{flick\_local\_y\_delta}$), and in-fold stance mirroring augmentation on the clean 110-attempt dataset produced measurable gains:
* **Unweighted Macro F1:** Increased from **$15.63\%$ to $16.88\%$** ($+1.25\%$ absolute lift).
* **Class-Weighted Macro F1:** Increased from **$16.16\%$ to $17.65\%$** ($+1.49\%$ absolute lift).
* **360 Flip:** F1 rose from $26.67\%$ to **$46.15\%$** with **$75.0\%$ Precision**.
* **Frontside Shove-it:** Unfroze from **$0.0\%$ to $9.09\%$ F1**.
* **SHAP Feature Importance:** SHAP analysis confirmed that the newly engineered features immediately became the top predictive drivers of the model:
  1. `flick_local_y_delta` ($0.097$): Top overall physical feature.
  2. `diff_yaw_norm` ($0.069$): Top rotation feature.

### 2.3 The Manifest Skater Monopoly Bottleneck
While feature refinement produced solid gains, strict cross-skater holdouts remain constrained by the structural distribution of the 1,035-clip manifest:
* **Ollie:** 52 clips total $\rightarrow$ 48 `player_a`, 4 `player_b` (**2 skaters in existence**).
* **Backside 180:** 53 clips total $\rightarrow$ 51 `player_a`, 1 `jack_colbourn`, 1 `sean_malto` (**3 skaters in existence**).
* **Frontside 180:** 56 clips total $\rightarrow$ 53 `player_a`, 1 `shane_oneill`, 1 `jack_colbourn`, 1 `sean_malto` (**4 skaters in existence**).
* **Pop Shove-it:** 55 clips total $\rightarrow$ 53 `player_a`, 1 `luan_oliveira`, 1 `torey_pudwill` (**3 skaters in existence**).

In Fold 0 of nested GroupKFold, where `player_a` ($N=46$) is held out as the test set, the training set contains only 4 Ollies, 2 Backside 180s, 3 Frontside 180s, and 2 Pop Shove-its. No machine learning model can generalize across unseen styles from 2 training examples.

### 2.4 Hierarchical Kinematic Factorization (`Skateboarding_Kinematic_Dataset.csv`)
To overcome the "Ollie Black Hole" (where a flat 9-class classifier predicts 40% of all attempts as Ollie due to low penalty on the non-flipping, non-spinning majority class), we restructured the classification architecture using the physical decomposition in `Skateboarding_Kinematic_Dataset.csv`.

Every flatground trick decomposes along independent kinematic axes:
$$\text{Trick} = \text{Body Spin} \otimes \text{Board Shuv} \otimes \text{Flip Roll}$$

Evaluating independent binary/multi-class factorized heads on unseen pro skaters demonstrated:
* **Body Spin Head (0° vs 180°):** **$80.0\%$ Accuracy** (Torso/ankle yaw tracking provides robust separation).
* **Flip Roll Head (Flat vs Flip):** **$63.6\%$ Accuracy** (Transverse flick displacement separates ollie/spin from flip tricks).
* **Board Shuv Head (0° vs 180° vs 360°):** **$30.9\%$ Accuracy** (The isolated remaining physical bottleneck: monocular 2D deck foreshortening suffers from camera-angle planar ambiguity).

We implemented **Model B2: Hierarchical Kinematic Classifier**, which enforces a structured decision tree:
1. **Stage 1 (Flip Roll Detector):** Classifies the attempt into *Flip Family* (Kickflip, Heelflip, Varial/Hardflip, 360 Flip) vs. *Flat Family* (Ollie, BS 180, FS 180, Pop Shove-it, FS Shove-it).
2. **Stage 2A (Flat Sub-Tree):** Uses a Body 180 detector. If body spin is detected, routes to the *180 Specialist* (FS 180 vs BS 180). If straight pop is detected, routes to the *Straight Specialist* (Ollie vs Pop Shove-it vs FS Shove-it).
3. **Stage 2B (Flip Sub-Tree Specialist):** Specialized GBDT resolving Kickflip, Heelflip, Varial/Hardflip, and 360 Flip.

**Result:** Model B2 eliminates cross-family contamination. Kickflips and 180s can no longer collapse into Ollies. Under strict 4-split nested GroupKFold on the 110 clean attempts, Macro F1 jumped from **$16.88\%$ to $28.04\%$** ($+11.16\%$ absolute lift, $+66\%$ relative improvement) and Top-1 Accuracy rose from **$19.09\%$ to $31.82\%$**.

---

## 3. Post-Impact Land/Bail Verification Gate (Gate 4 Provisional PASS)

The Post-Impact Land/Bail Verification Gate evaluates post-touchdown physical persistence over $[t_{\text{land}}, t_{\text{land}} + 30]$.

### 3.1 Certified Performance Metrics (N = 125 Trajectories)

Across the 125 evaluated attempts in the store (96 predicted Landed, 26 predicted Bailed, 3 quarantined as UNCERTAIN due to truncated rollout):

```text
Confusion Matrix (Landed vs. Bailed):
                    Predicted Bailed    Predicted Landed
Actual Bailed:            21                    9
Actual Landed:             5                   87
```

| Metric | Target Threshold | Verified Empirical Result | Operational Gate Status |
| :--- | :---: | :---: | :---: |
| **Verification F1-Score** | $\ge 0.80$ | **$0.9255$ (92.6%)** | **PASS (Pilot Certified / Provisional)** |
| **Bail Detection Recall** | $\ge 70.0\%$ | **$70.0\%$ (21 / 30 actual bails caught)** | **PASSED (Up from 25.0% in pilot)** |
| **Bail False Positive Rate (FPR)** | $< 8.0\%$ | **$5.43\%$ (5 / 92 clean clips rejected)** | **PASSED** |
| **Landed Precision** | $\ge 80.0\%$ | **$90.62\%$ (87 / 96 predicted landed)** | **PASSED** |
| **Landed Recall** | $\ge 90.0\%$ | **$94.57\%$ (87 / 92 landed attempts)** | **PASSED** |

> [!NOTE]
> **Provisional Certification Status:** Gate 4 successfully achieved all quantitative operational thresholds: 92.55% F1, 70.0% bail recall, and 5.43% FPR. Because 9 bails were missed and 5 clean landings were rejected, Gate 4 is designated as **PASS (Pilot Certified / Provisional)**.

---

## 4. Trick Classification Engine Benchmark (Gate 3 FAILED)

Evaluated across the **110 canonical flatground attempts** in the active trajectory store spanning 14 distinct skaters:

### 4.1 Master Architecture Comparison

| Model Architecture | Input Representation | Macro F1 | Top-1 Accuracy | Top-2 Accuracy | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Model A: Calibrated Rules** | Kinematic Thresholds ($\Delta\theta, \Theta_{\text{abs}}, \tilde{W}_{\text{aspect}}, \mathbf{v}_{\text{flick}}$) | $8.13\%$ | $9.09\%$ | — | $< 0.10\text{ ms}$ |
| **Model B: XGBoost (Unweighted)** | 53-dim Kinematic Vector + Mirroring (Nested GroupKFold) | $16.88\%$ | $19.09\%$ | $33.64\%$ | $0.43\text{ ms}$ |
| **Model B: XGBoost (Class-Weighted)** | 53-dim Kinematic Vector ($w_c = \frac{N}{K \cdot N_c}$) | $17.65\%$ | $19.09\%$ | $36.36\%$ | $0.43\text{ ms}$ |
| **Model B2: Hierarchical Kinematic Tree** | Physically Factorized Multi-Stage GBDT (Flip/180/Straight) | **$28.04\%$** | **$31.82\%$** | **$51.82\%$** | $0.85\text{ ms}$ |
| **Model B: XGBoost (Oracle Events)** | 53-dim Kinematic Vector (Ground Truth Event Timestamps) | $16.88\%$ | $19.09\%$ | $33.64\%$ | $0.43\text{ ms}$ |
| **Model B: XGBoost (Stratified 5-Fold)** | 53-dim Kinematic Vector (Diagnostic Skater Overlap) | **$38.58\%$** | **$40.91\%$** | — | $0.43\text{ ms}$ |
| **Model C: Audited Compact ST-GCN** | 6-Node Graph $(C=4, T=64, V=6)$ (Zero Raw RGB) | **$7.81\%$** | **$11.82\%$** | **$27.27\%$** | $2.80\text{ ms}$ |

### 4.2 Per-Class Breakdown: Flat GBDT vs. Hierarchical Kinematic Tree

#### 4.2.1 Hierarchical Kinematic Classifier (Model B2 - Primary Architecture)
Evaluated under strict 4-split nested GroupKFold by skater on the 110 clean attempts:

```text
Confusion Matrix (Model B2 Hierarchical Tree on Strict Nested GroupKFold across 110 attempts):
Classes: [360 Flip, Backside 180, Frontside 180, Frontside Shove-it, Heelflip, Kickflip, Ollie, Pop Shove-it, Varial/Hardflip]
Pred ->   360   BS180  FS180  FSShov  Heel   Kick   Ollie  PopShov Varial
360 Flip    4     0      0      0       2      3      0      0       0
BS 180      0     5      3      0       1      0      4      0       0
FS 180      0     4      3      0       0      2      3      0       1
FS Shov     1     0      0      0       1      2      5      0       4
Heelflip    0     0      0      0       2      2      2      0       3
Kickflip    0     0      2      0       2      6      1      0       2
Ollie       0     0      0      1       2      1      4      0       0
Pop Shov    1     0      0      1       0      1      9      0       1
Varial      1     1      0      0       0      6      0      0      11
```

| Canonical Class | Store Samples ($N$) | Unique Skaters | Model B (Flat F1) | Model B2 (Hierarchical F1) | Absolute Delta | Precision | Recall | Generalization Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Varial / Hardflip** | 19 | 7 | $51.3\%$ | **$53.7\%$** | $+2.4\%$ | $50.0\%$ | $57.9\%$ | **Robust:** Strong compound roll+shuv signal; top performing class. |
| **360 Flip (Tre Flip)** | 9 | 7 | $46.2\%$ | **$50.0\%$** | $+3.8\%$ | $57.1\%$ | $44.4\%$ | **High Precision:** Correctly isolated 360 shuv + kickflip dynamics. |
| **Backside 180** | 13 | 3 | $11.8\%$ | **$43.5\%$** | **$+31.7\%$** | $50.0\%$ | $38.5\%$ | **Breakthrough:** Body yaw inversion decoupled from straight pop. |
| **Kickflip** | 13 | 10 | $7.1\%$ | **$33.3\%$** | **$+26.2\%$** | $26.1\%$ | $46.2\%$ | **Rescued from Ollie:** 6 correct landings (up from 1). |
| **Frontside 180** | 13 | 4 | $0.0\%$ | **$28.6\%$** | **$+28.6\%$** | $50.0\%$ | $23.1\%$ | **Unfrozen:** Recovered from 0.0% complete collapse. |
| **Ollie** | 8 | 2 | $15.4\%$ | **$22.2\%$** | $+6.8\%$ | $14.3\%$ | $50.0\%$ | **Ollie Trap Reduced:** Total predicted Ollies dropped 44 $\rightarrow$ 28. |
| **Heelflip** | 9 | 6 | $11.1\%$ | **$21.1\%$** | $+10.0\%$ | $20.0\%$ | $22.2\%$ | Toe flick feature helps resolve Kickflip/Heelflip axis. |
| **Frontside Shove-it** | 13 | 6 | $9.1\%$ | **$0.0\%$** | $-9.1\%$ | $0.0\%$ | $0.0\%$ | Ambiguous 2D rotation without body yaw collapses to Ollie/Varial. |
| **Pop Shove-it** | 13 | 3 | $0.0\%$ | **$0.0\%$** | $+0.0\%$ | $0.0\%$ | $0.0\%$ | **Physical Bottleneck:** Monocular 2D shuv without 3D pitch/yaw. |
| **MACRO AVERAGE** | **110** | **14** | **$16.88\%$** | **$28.04\%$** | **$+11.16\%$** | **$29.7\%$** | **$31.4\%$** | **+66% relative gain over flat baseline; Top-1: 31.8%, Top-2: 51.8%.** |

### 4.3 Feature Importance (SHAP Global Kinematic Drivers)
SHAP value analysis on Model B identified the dominant physical predictors:
1. `flick_local_y_delta` ($0.097$): Transverse toe/heel flick displacement relative to board centerline.
2. `board_length_pop` ($0.082$): Apparent deck geometry at pop.
3. `diff_yaw_norm` ($0.069$): Differential rotation ($\text{Board Yaw} - \text{Body Yaw}$).
4. `right_foot_dist` ($0.051$): Rear foot proximity during flight.
5. `min_norm_length` ($0.050$): Foreshortening trough depth during horizontal board spins.
6. `jerk_std` ($0.048$): Smoothness of aerial trajectory.
7. `theta_abs` ($0.045$): Total integrated board rotation magnitude.

---

## 5. Systematic Feature Ablation Matrix (A0–A5)

Evaluated under 4-fold GroupKFold by skater on the 110 canonical clips:

| Ablation ID | Feature Modality | Features ($D$) | Macro F1 | Accuracy | Empirical Finding |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **A0** | **Sparse Phase-Summary Positional Baseline** | 7 | $13.9\%$ | $15.5\%$ | Basic keypoint positions at Pop, Apex, Land provide minimal separation. |
| **A1** | **Pose Landmarks Only** | 10 | $11.1\%$ | $13.6\%$ | Skater body joints alone cannot distinguish board flip vs shove tricks. |
| **A2** | **Board Tracking Only** | 18 | **$15.1\%$** | **$17.3\%$** | **Strongest Sub-System:** Board rotational dynamics provide primary signal. |
| **A3** | **Pose + Board Raw Coordinates** | 11 | $11.7\%$ | $12.7\%$ | Un-engineered spatial coordinates suffer from camera perspective shifts. |
| **A4** | **Coordinates + Velocities** | 16 | $12.3\%$ | $14.5\%$ | 1st-order velocity terms provide slight lift over raw positions. |
| **A5** | **Full Engineered Dynamics** | 45 | **$14.6\%$** | **$17.3\%$** | **Top Multi-Modality Model:** Combines jerk, angular integrals, and foot flick vectors. |

---

## 6. Standardized Latency & Real-Time Performance (P3-A10)

Measured over 50 consecutive inference cycles on a standard single-threaded CPU:

| Metric | Target Threshold | Empirical Result | Status |
| :--- | :---: | :---: | :---: |
| **Inference Time per Clip** | $< 40.0\text{ ms}$ | **$0.49\text{ ms}$** | **PASSED** |
| **Latency per Frame** | $< 2.0\text{ ms / frame}$ | **$3.25\text{ }\mu\text{s / frame}$** | **PASSED** |
| **Real-Time Factor (RTF)** | $\text{RTF} < 0.50$ | **$\text{RTF} = 0.00019$** | **PASSED (5,200x faster than real-time)** |

---

## 7. Quality Control Gate Certification Verdict

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 3 FINAL CERTIFICATION VERDICT                     │
│                                                                             │
│  [1] Gate 3 (Trick Recognition Engine)   : FAILED / RECOVERY IN PROGRESS    │
│      • Target Threshold                  : Macro F1 >= 82.0% | Min Class >= 70%
│      • Model B2 Hierarchical Tree        : Macro F1 = 28.04% | Top-1 = 31.82%│
│      • Model B2 Top-2 Accuracy           : 51.82% (up from 33.64% baseline)  │
│      • Unweighted Flat Nested GroupKFold : Macro F1 = 16.88% | Top-1 = 19.09%│
│      • Stratified 5-Fold (Diagnostic)    : Macro F1 = 38.58%                │
│      • Skater-Overlap Confounding Delta  : +21.70%                          │
│      • Verdict                           : HARD FAILURE / PAUSES PHASE 4    │
│                                                                             │
│  [2] Gate 4 (Post-Impact Land/Bail Gate) : PASS (PILOT CERTIFIED / PROV.)   │
│      • Target Threshold                  : F1 >= 80.0% | Bail FPR < 8.0%    │
│      • Empirical Result                  : F1 = 92.55% | Bail FPR = 5.43%   │
│      • Bail Detection Recall             : 70.0% (21/30 bails caught)       │
│      • Confusion Matrix                  : [[21, 9], [5, 87]] (125 clips)   │
│      • Verdict                           : PROVISIONALLY CERTIFIED          │
│                                                                             │
│  [3] Gate 3 Latency / RTF Gate           : PASS                             │
│      • Empirical RTF                     : 0.00017 << 0.50 threshold        │
│                                                                             │
│  [4] Phase 4 Cleanliness Scoring Status  : PAUSED PENDING RECOVERY          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Path Forward (Phase 3.5 Next Steps)

Because **Gate 3 remains formally FAILED** (28.04% vs 82.0% operational target), Phase 4 cleanliness scoring remains strictly **PAUSED**. 

The immediate next priority roadmap:
1. **Resolve the Board Shuv Physical Bottleneck (Pop Shove-it & FS Shove-it):**
   * While Body Spin (80.0%) and Flip Roll (63.6%) heads perform reliably across unseen skaters, planar Board Shuv rotation without body spin currently sits at 30.9% accuracy due to 2D monocular foreshortening ambiguity.
   * Develop a 3D board normal estimator using wheel contact/shadow tracking or temporal graph attention over the $[t_{\text{pop}}, t_{\text{apex}}]$ window to distinguish planar deck spins from level ollie ascents.
2. **Curated Frame-Accurate Video Ingestion (Breaking the 2-Skater Monopoly):**
   * Acquire 15–20 competition clips of pro street skaters for Ollie (currently only 2 skaters), Backside 180 (3 skaters), Frontside 180 (4 skaters), and Pop Shove-it (3 skaters).
   * **Mandatory Quality Rule:** Every clip must have its exact `[t_pop, t_apex, t_land]` frames visually verified; zero unverified automated web scrapers.
3. **Preserve Anti-Leakage Protocol:**
   * Maintain the strict nested GroupKFold evaluation protocol without compromise. Gate 3 will not be marked as passed until empirical cross-skater Macro F1 satisfies the required operational target ($\ge 82.0\%$).
