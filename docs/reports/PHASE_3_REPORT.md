# Phase 3 Execution & Quality Control Report

**Phase:** Phase 3 — Trick Classification Engine & Land/Bail Verification  
**Status:** In Progress / Pending Execution  
**Target Gate:** Gate 3 (Recognition Macro F1 $\ge 85\%$) & Gate 4 (Land/Bail Gating F1 $\ge 0.80$)  

---

## 1. Executive Summary & Work Completed
* **Objective:** Classify the 9 primary flatground trick classes using decoupled kinematic features evaluated on unseen skaters, and filter out failed attempts via a post-landing Land/Bail gatekeeper.
* **Work Completed:**
  - [ ] Construct flight-phase kinematic feature extractor (`src/classification/feature_extractor.py`).
  - [ ] Implement Model A deterministic kinematic baseline (`src/classification/rule_classifier.py`).
  - [ ] Train Model B XGBoost / LightGBM tree ensemble (`src/classification/tree_classifier.py`).
  - [ ] Build Model C ST-GCN graph neural network (`src/classification/graph_classifier.py`).
  - [ ] Execute formal Ablation Study Matrix (A1–A5) across identical splits (`src/classification/ablation_runner.py`).
  - [ ] Train post-landing Land vs. Bail classifier (`src/classification/bail_gatekeeper.py`).
* **Deliverable Artifacts:**
  - Trained Models: `models/trick_classifier/*.json`, `models/bail_gatekeeper/*.joblib`
  - Ablation Results: `docs/reports/ablation_matrix_results.csv`
  - Confusion Matrices: `docs/reports/confusion_matrix_test.png`

---

## 2. Explicit Assumptions Made
1. **Decoupled Recognition:** Sub-optimal execution does not invalidate categorical trick identity provided the rotation and flip mechanics occur.
2. **Unseen Skater Generalization:** Models must be tested strictly on unseen skaters to prevent memorization of individual clothing, shoe color, or idiosyncratic styling.
3. **Flatground Primary Scope:** 9 primary classes capture flatground street skateboarding mechanics; transitions and grinds are out of current scope.
4. **Bail Filtration Invariance:** Bailed attempts are categorically disqualified from aesthetic cleanliness evaluation.

---

## 3. Systems & Artifacts Built on Assumptions
* **Ablation Matrix Harness (A1–A5):** Empirical isolation proving incremental utility of angular velocity and board-local coordinates over raw 2D landmarks.
* **SHAP Feature Importance Module:** Verification that model decisions align with known skateboarding mechanics (e.g. front foot flick angle identifying Kickflip vs. Heelflip).
* **Land/Bail Guardrail:** Binary decision node intercepting the pipeline before Phase 4.

---

## 4. Quality Control (QC) & Evaluation Results

| Metric Name | Operational Gate Target | Aspirational Target | Expected Result | Actual Achieved Result | QC Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Trick Recognition Macro F1** | $\ge 85.0\%$ | $\ge 90.0\%$ | $\approx 87.5\%$ | *Pending evaluation* | PENDING |
| **Minimum Class F1** | $\ge 75.0\%$ | $\ge 82.0\%$ | $\approx 78.0\%$ | *Pending evaluation* | PENDING |
| **Land vs. Bail Gating F1** | $\ge 0.80$ | $\ge 0.90$ | $\approx 0.86$ | *Pending evaluation* | PENDING |
| **Bail False Positive Rate** | $< 8.0\%$ | $< 3.0\%$ | $< 5.0\%$ | *Pending evaluation* | PENDING |
| **Inference Latency** | $< 40\text{ ms}$ | $< 15\text{ ms}$ | $\approx 20\text{ ms}$ | *Pending evaluation* | PENDING |

---

## 5. Failure Analysis, Edge Cases & Mitigation
* **Heelflip vs Kickflip Confusion:** Inverted flick trajectories on switch stance skaters addressed by incorporating stance detection flags.
* **Varial Kickflip vs Tre Flip:** Under-rotated $360^\circ$ Shuvits addressed by integrating cumulative angular velocity $\int |\omega| dt$.

---

## 6. Phase Gate Sign-off
* **Gate Decision:** PENDING
* **Approver:** Lead Research Engineer
* **Next Step:** Advance to Phase 4 upon achieving Macro F1 $\ge 85\%$ on unseen skaters and Land/Bail F1 $\ge 0.80$.
