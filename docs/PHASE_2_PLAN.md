# Phase 2 Implementation Plan: Temporal Event Localization & Phase Segmentation

---

## 1. Overview & Scope

Phase 2 focuses on segmenting continuous tracking trajectories into distinct physical biomechanical phases. By localizing the exact frame boundaries of **Pop**, **Apex**, **Catch**, and **Landing**, the engine partitions a skateboard trick into structured temporal windows. Downstream trick classification and cleanliness scoring depend entirely on the accuracy and mathematical consistency of these temporal milestones.

### Key Deliverables of Phase 2:
1. **Kinematic Signal Conditioning Pipeline:** Multi-order numerical differentiation ($\dot{y}, \ddot{y}, \omega$) and adaptive peak-detection filters.
2. **Probabilistic Event Boundary Localizer:** Algorithmic detector for $t_{\text{pop}}$, $t_{\text{apex}}$, $t_{\text{catch}}$, and $t_{\text{land}}$.
3. **Temporal Constraint & Sequence Validator:** Enforcer of physical progression ($t_{\text{pop}} < t_{\text{apex}} < t_{\text{catch}} \le t_{\text{land}}$).
4. **Phase Slice Generator:** Structured window extractor producing normalized sub-sequences for phase-specific feature engineering.
5. **Phase Segmentation QC & Evaluation Harness:** Automated benchmark comparing predicted frame indices against ground-truth human annotations.

---

## 2. Core Assumptions & Boundary Constraints

The algorithms in Phase 2 are predicated on the following physical assumptions:

1. **Parabolic Ballistic Assumption:** Between $t_{\text{pop}}$ and $t_{\text{land}}$, the skateboard operates under ballistic projectile motion governed by gravity (modified by skater foot interaction during flick/catch). The apex uniquely corresponds to zero vertical velocity ($\dot{y}_{\text{board}} \approx 0$) at peak altitude.
2. **Impulse Pop Dynamics:** The pop event produces an instantaneous peak in downward rear ankle acceleration coincident with minimum tail-to-ground distance.
3. **Probabilistic Catch Thresholding:** Foot-deck Euclidean proximity does not drop to zero due to shoe geometry and keypoint tracking offsets. The catch is identified when foot-board relative velocity converges ($\Delta v < \tau_v$) simultaneously with rotational deceleration ($|\frac{d\omega}{dt}| < \tau_\alpha$) within distance threshold $\tau_d$.
4. **Ground Plane Invariance:** The ground surface remains approximately horizontal throughout the flatground trick execution window in near-side profile views.
5. **Temporal Monotonicity:** A physically valid landed trick must exhibit strictly monotonically increasing event timestamps: $t_{\text{pop}} < t_{\text{apex}} < t_{\text{catch}} \le t_{\text{land}}$.

---

## 3. Step-by-Step Implementation Sequence

### Step 2.1: Signal Conditioning & Derivative Computation (`src/segmentation/signal_processor.py`)
* Ingest smoothed trajectory series from Phase 1 Parquet feature stores.
* Compute deck centroid trajectory $y_{\text{deck}}(t) = \frac{y_N(t) + y_T(t)}{2}$.
* Extract vertical board velocity $\dot{y}_{\text{deck}}(t)$ and acceleration $\ddot{y}_{\text{deck}}(t)$ using central difference stencils after Savitzky-Golay filtering.
* Compute instantaneous deck in-plane angular velocity $\omega(t) = \frac{\Delta \theta_{\text{board}}}{\Delta t}$ and angular acceleration $\alpha(t) = \frac{d\omega}{dt}$.

### Step 2.2: Pop and Apex Localization (`src/segmentation/event_detector.py`)
* **Pop Detection ($t_{\text{pop}}$):**
  * Detect local minimum in tail-to-ground distance $y_T(t) - y_{\text{ground}}$.
  * Correlate with downward acceleration spike in the rear ankle landmark $\ddot{y}_{\text{rear\_ankle}}$.
  * Mark $t_{\text{pop}}$ at the moment of upward impulse initiation.
* **Apex Detection ($t_{\text{apex}}$):**
  * Identify highest vertical deck elevation $\min(y_{\text{deck}}(t))$ (in image coordinates where $y$ is inverted).
  * Confirm zero-crossing of vertical velocity: $\dot{y}_{\text{deck}}(t_{\text{apex}} - 1) < 0 \le \dot{y}_{\text{deck}}(t_{\text{apex}} + 1)$.

### Step 2.3: Probabilistic Catch Localization (`src/segmentation/catch_detector.py`)
* Search within descent temporal window $[t_{\text{apex}}, t_{\text{land}}]$.
* Formulate triple-condition scoring function:
  $$S_{\text{catch}}(t) = w_1 \cdot \exp\left(-\frac{\Delta d(t)}{\sigma_d}\right) + w_2 \cdot \exp\left(-\frac{\Delta v(t)}{\sigma_v}\right) + w_3 \cdot \exp\left(-\frac{|\alpha(t)|}{\sigma_\alpha}\right)$$
  where:
  * $\Delta d(t) = \min_{f \in \{\text{front, rear}\}} \Vert\mathbf{p}_f(t) - \mathbf{p}_{\text{deck}}(t)\Vert_2$
  * $\Delta v(t) = \Vert\dot{\mathbf{p}}_{\text{feet}}(t) - \dot{\mathbf{p}}_{\text{deck}}(t)\Vert_2$
  * $\alpha(t) = \left|\frac{d\omega_{\text{board}}}{dt}\right|$
* Calibrate thresholds $(\tau_d, \tau_v, \tau_\alpha)$ against the 80–120 verified expert annotations.
* Designate $t_{\text{catch}}$ as the earliest frame satisfying $S_{\text{catch}}(t) \ge \tau_{\text{catch\_score}}$.

### Step 2.4: Landing Localization & Ground Impact (`src/segmentation/landing_detector.py`)
* Detect point where board vertical velocity stabilizes near zero post-descent.
* Monitor front and rear truck altitude relative to estimated ground reference $y_{\text{ground}}$.
* Detect deceleration shockwave in skater knee/hip joints (compression phase).
* Mark $t_{\text{land}}$ as touchdown frame.

### Step 2.5: Sequence Validation & Window Slicing (`src/segmentation/validator.py`)
* Validate temporal consistency: $t_{\text{pop}} < t_{\text{apex}} < t_{\text{catch}} \le t_{\text{land}}$.
* If ordering is violated (e.g. catch detected after landing, or pop after apex), flag attempt with `TEMPORAL_INVERSION_ERROR` and route to confidence gatekeeper.
* Generate window slices:
  * *Pop Phase:* $[t_{\text{pop}} - 15, t_{\text{pop}}]$
  * *Flight / Rotation Phase:* $[t_{\text{pop}}, t_{\text{catch}}]$
  * *Landing / Rollout Phase:* $[t_{\text{land}}, t_{\text{land}} + 30]$

---

## 4. Post-Phase Checkpoint: QC, Evaluation & Acceptance Criteria

At the end of Phase 2, the segmentation pipeline must be evaluated against ground-truth event labels across the test set:

### Quality Control (QC) Gates

| Metric / Checkpoint | Hard Operational Gate | Aspirational Target | Verification Method |
| :--- | :--- | :--- | :--- |
| **Pop Event Localization Error** | $\text{MAE} \le 5.0\text{ frames}$ ($\le 83\text{ ms}$) | $\text{MAE} \le 3.0\text{ frames}$ ($\le 50\text{ ms}$) | Absolute frame difference against human consensus annotations. |
| **Apex Event Localization Error** | $\text{MAE} \le 4.0\text{ frames}$ ($\le 67\text{ ms}$) | $\text{MAE} \le 2.0\text{ frames}$ ($\le 33\text{ ms}$) | Absolute frame difference against human consensus annotations. |
| **Catch Event Localization Error** | $\text{MAE} \le 7.0\text{ frames}$ ($\le 116\text{ ms}$) | $\text{MAE} \le 5.0\text{ frames}$ ($\le 83\text{ ms}$) | Absolute frame difference against human consensus annotations. |
| **Landing Localization Error** | $\text{MAE} \le 5.0\text{ frames}$ ($\le 83\text{ ms}$) | $\text{MAE} \le 3.0\text{ frames}$ ($\le 50\text{ ms}$) | Absolute frame difference against human consensus annotations. |
| **Temporal Monotonicity Rate** | $\ge 95.0\%$ of landed attempts | $\ge 98.0\%$ of landed attempts | Percentage of clips strictly adhering to $t_{\text{pop}} < t_{\text{apex}} < t_{\text{catch}} \le t_{\text{land}}$. |

---

## 5. Phase Documentation & Reporting Requirements

Upon completing Phase 2, compile `docs/reports/PHASE_2_REPORT.md` covering:
1. **Executive Summary:** Overview of event localization modules implemented, signal filters tuned, and test dataset evaluated.
2. **Work Completed:** Code implementations for signal processing, pop/apex/catch/landing detectors, and temporal order validation.
3. **Assumptions Made:** Formal documentation of physical heuristics (parabolic apex zero-crossing, triple-condition catch thresholds, ground plane stability).
4. **Artifacts Built on Assumptions:** Mathematical formulations of scoring functions, calibrated threshold parameters $(\tau_d, \tau_v, \tau_\alpha)$, and window partition schemas.
5. **QC Results vs. Expected Targets:** Quantitative table detailing actual MAE errors for Pop, Apex, Catch, Land, and temporal monotonicity compliance rate against Gate 2 targets.
6. **Failure Analysis & Edge Cases:** Identification of misidentified catch frames (e.g., foot hovering above deck, early stomp, loose grip tape) and recovery strategies.
7. **Sign-off Decision:** Clear GO / NO-GO recommendation for advancing to Phase 3.
