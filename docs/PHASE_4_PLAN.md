# Phase 4 Implementation Plan: Kinematic Cleanliness Regression & Expert Calibration

---

## 1. Overview & Scope

Phase 4 constructs the interpretable scoring engine of SkateKine. Rather than estimating an opaque aesthetic score using black-box neural networks, this phase extracts measurable physical and geometric candidate features, calibrates their weights and non-linear penalties against human expert agreement, and applies confidence gating during critical visual occlusions.

### Key Deliverables of Phase 4:
1. **Candidate Kinematic Feature Engine:** Mathematical extraction of Catch Elevation Ratio, Truck Placement Offset (Bolts), Landing Stability Drift, and Angular Residuals.
2. **Dual-Tier Expert Annotation Protocol & ICC Validation:** Multi-rater consensus aggregation across 80–120 verified landed attempts with Intraclass Correlation Coefficient ($\text{ICC}(2, k) \ge 0.75$).
3. **Sub-Score Regression Models:** Independent models mapping physical predictors to domain-specific ratings (Catch, Bolts, Stability).
4. **Overall Cleanliness Calibration Head:** ElasticNet / Ordinal Ridge regression producing a calibrated 0–100 Cleanliness Index.
5. **Critical-Moment Confidence Gating Module:** Automatic reliability attenuation detecting occlusion or keypoint degradation during the catch phase.
6. **Heuristic Baseline Statistical Benchmark:** Formal hypothesis testing confirming learned models outperform naive equal-weighted heuristics ($p < 0.01$).

---

## 2. Core Assumptions & Boundary Constraints

1. **No Uncalibrated Heuristics:** Kinematic metrics are strictly explanatory candidates. Their relative importance, non-linear penalties, and interactions must be learned from expert adjudication.
2. **Metric Conversion Bounds:** Centimeter scaling ($s(t) = \frac{80.0\text{ cm}}{\Vert\mathbf{p}_N(t) - \mathbf{p}_T(t)\Vert_2}$) applies when the deck longitudinal axis is near-parallel to the camera plane. When foreshortened, dimensionless normalized ratios $d_{\text{norm}} = \frac{d_{\text{pixels}}}{\Vert\mathbf{p}_N - \mathbf{p}_T\Vert_2}$ must be substituted.
3. **Inter-Rater Consensus Reliability:** Human subjective scoring is inherently noisy. Only attempts where 3+ expert skaters achieve high consensus ($\text{ICC} \ge 0.75$) may be used as ground truth for regression training.
4. **Fail-Safe Integrity (Confidence Gating):** If tracking confidence falls below $\tau_{\text{conf}} = 0.50$ in the critical window around the catch $[t_{\text{catch}}-\delta, t_{\text{catch}}+\delta]$, the engine must output `Cleanliness: Incomplete, Reliability: LOW` rather than fabricating a false score.

---

## 3. Step-by-Step Implementation Sequence

### Step 4.1: Candidate Kinematic Feature Extraction (`src/cleanliness/feature_extractor.py`)
Implement the four core physical cleanliness dimensions:
1. **Catch Elevation Ratio ($r_{\text{catch}}$):**
   $$r_{\text{catch}} = \frac{y_{\text{deck}}(t_{\text{catch}}) - y_{\text{ground}}}{y_{\text{deck}}(t_{\text{apex}}) - y_{\text{ground}}}$$
   Measures whether the trick was caught at the peak of flight ($r \to 1.0$) or on the ground ($r \to 0.0$).
2. **Truck Landing Offset / Bolts Proximity ($d_{\text{bolts}}$):**
   $$d_{\text{bolts}} = s(t_{\text{catch}}) \left( \Vert\mathbf{p}_{\text{front\_foot}} - \mathbf{p}_{\text{front\_truck}}\Vert_2 + \Vert\mathbf{p}_{\text{rear\_foot}} - \mathbf{p}_{\text{rear\_truck}}\Vert_2 \right)$$
   Measures foot deviation from the front and rear truck mounting bolts.
3. **Landing Stability & Drift ($\sigma_{\text{stability}}^2$):**
   $$\sigma_{\text{stability}}^2 = \text{Var}_{t \in [t_{\text{land}}, t_{\text{land}}+30]}(\theta_{\text{board}}(t)) + \lambda \cdot \text{Var}_{t \in [t_{\text{land}}, t_{\text{land}}+30]}(x_{\text{CoM}}(t))$$
   Quantifies wheel wobble, tic-tacing, and skater balance recovery.
4. **Angular Residual Completion ($\Delta\theta_{\text{residual}}$):**
   $$\Delta\theta_{\text{residual}} = |\theta_{\text{board}}(t_{\text{land}}) - \theta_{\text{target}}|$$
   Quantifies under- or over-rotation at ground impact.

### Step 4.2: Expert Annotation Curation & ICC Analysis (`src/cleanliness/icc_validator.py`)
* Ingest multi-rater scores from 3 independent expert skaters across 80–120 clips.
* Compute Two-Way Random Effects Intraclass Correlation Coefficient ($\text{ICC}(2, k)$).
* Identify and discard or re-adjudicate raters/clips with severe divergence ($> 2.5$ points deviation).
* Compute target consensus score as mean of verified expert ratings.

### Step 4.3: Sub-Score Regression Models (`src/cleanliness/subscore_models.py`)
* Fit univariate and regularized multivariate regressors:
  * $f_{\text{catch}}(r_{\text{catch}}) \to \text{Catch Elevation Rating } (1\text{--}10)$
  * $f_{\text{bolts}}(d_{\text{bolts}}) \to \text{Foot Placement Rating } (1\text{--}10)$
  * $f_{\text{stability}}(\sigma_{\text{stability}}^2) \to \text{Landing Stability Rating } (1\text{--}10)$
  * $f_{\text{rotation}}(\Delta\theta_{\text{residual}}) \to \text{Rotational Completeness Rating } (1\text{--}10)$
* Evaluate monotonicity and non-linear saturation curves (e.g. logarithmic penalty for extreme bolt misses).

### Step 4.4: Composite Overall Cleanliness Model (`src/cleanliness/overall_model.py`)
* Train ElasticNet / Ordinal Ridge regression:
  $$\hat{Y}_{\text{overall}} = \beta_0 + \sum_{i} \beta_i \phi_i(\mathbf{x}) + \sum_{j} \gamma_j \psi_j(\text{SubScores})$$
* Enforce boundary clamping: $\hat{Y}_{\text{overall}} \in [0, 100]$.
* Perform Leave-One-Skater-Out Cross Validation (LOSOCV).
* Perform Wilcoxon signed-rank test against equal-weighted naive heuristic baseline.

### Step 4.5: Critical-Moment Confidence Gating (`src/cleanliness/confidence_gate.py`)
* Monitor mean tracking confidence for board and lower limbs over $[t_{\text{catch}}-5, t_{\text{catch}}+5]$:
  $$C_{\text{critical}} = \min_{t \in [t_{\text{catch}}-5, t_{\text{catch}}+5]} \left(c_{\text{board}}(t), c_{\text{ankles}}(t)\right)$$
* If $C_{\text{critical}} < \tau_{\text{conf}}$ (default: 0.50):
  * Suppress composite score.
  * Emit warning flag: `LOW_RELIABILITY_OCCLUSION`.
  * Return diagnostic explanation detailing the specific keypoint that degraded.

---

## 4. Post-Phase Checkpoint: QC, Evaluation & Acceptance Criteria

### Quality Control (QC) Gates

| Metric / Checkpoint | Hard Operational Gate | Aspirational Target | Verification Method |
| :--- | :--- | :--- | :--- |
| **Inter-Rater Consensus ($\text{ICC}(2, k)$)** | $\ge 0.75$ | $\ge 0.85$ | Intraclass correlation coefficient across 3 expert raters. |
| **Cleanliness Correlation (Spearman $\rho$)** | $\ge 0.78$ | $\ge 0.85$ | Spearman rank correlation between model score and expert consensus. |
| **Score Prediction Error ($\text{MAE}$)** | $\le 8.0\text{ points}$ (on 0–100 scale) | $\le 5.0\text{ points}$ | Mean Absolute Error on held-out evaluation set. |
| **Heuristic Improvement ($p$-value)** | $p < 0.01$ | $p < 0.001$ | Statistically significant lift over equal-weighted baseline. |
| **Confidence Gate Reliability** | 100% suppression on corrupted clips | 100% suppression | Zero fabricated scores emitted on synthetically occluded test clips. |

---

## 5. Phase Documentation & Reporting Requirements

Upon completing Phase 4, assemble `docs/reports/PHASE_4_REPORT.md` including:
1. **Executive Summary:** Overview of candidate feature performance, expert consensus validation, and regression model scores.
2. **Work Completed:** Code implementations for feature extractors, ICC validation harness, sub-score heads, composite model, and confidence gatekeeper.
3. **Assumptions Made:** Formal documentation of metric scale conversion assumptions, expert rating reliability criteria, and occlusion thresholds.
4. **Artifacts Built on Assumptions:** Mathematical formulas for candidate features ($r_{\text{catch}}, d_{\text{bolts}}, \sigma_{\text{stability}}^2$), learned regression coefficients ($\beta_i$), and confidence gating rules.
5. **QC Results vs. Expected Targets:** Quantitative table detailing actual Spearman $\rho$, MAE, ICC scores, and heuristic baseline $p$-value comparison against Gate 5 criteria.
6. **Failure Analysis & Edge Cases:** Cases of expert divergence (e.g. style vs. strict mechanics) and handling of extreme camera perspectives.
7. **Sign-off Decision:** Clear GO / NO-GO recommendation for advancing to Phase 5.
