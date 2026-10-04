# Phase 2 Implementation Plan: Temporal Event Localization & Phase Segmentation

---

## 1. Overview & Scope

Phase 2 focuses on segmenting continuous tracking trajectories into distinct physical biomechanical phases. By localizing the exact frame boundaries of **Pop**, **Apex**, **Catch**, and **Landing**, the engine partitions a skateboard trick into structured temporal windows. Downstream trick classification and cleanliness scoring depend entirely on the accuracy and mathematical consistency of these temporal milestones.

### Key Deliverables of Phase 2:
1. **Kinematic Signal Conditioning Pipeline (`src/segmentation/signal_processor.py`):** Canonical $z$-up coordinate framing ($z = -y$), multi-order numerical differentiation ($\dot{z}, \ddot{z}, \omega, \alpha$), and adaptive peak-detection filters.
2. **Probabilistic Event Boundary Localizers:**
   - **Pop & Apex Detector (`src/segmentation/event_detector.py`):** Multi-cue pop fusion score [A1] and zero-crossing parabolic apex detector [A2].
   - **Persistent Catch Detector (`src/segmentation/catch_detector.py`):** Temporally sustained contact validator over $K = \text{round}(\tau_{\text{persist}} \times \text{FPS})$ frames [A3].
   - **Impact-Driven Landing Detector (`src/segmentation/landing_detector.py`):** Causal shockwave compression and ground proximity detector [A5].
3. **Temporal Constraint & Sequence Validator (`src/segmentation/validator.py`):** Enforcer of strict physical progression ($t_{\text{pop}} < t_{\text{apex}} < t_{\text{catch}} \le t_{\text{land}}$).
4. **Continuous Window Slice Generator (`src/segmentation/slicer.py`):** Time-scaled window extractor ($\tau$ in seconds) invariant to 60 vs 120 FPS frame rates [A6].
5. **Phase Segmentation QC & Evaluation Harness (`tests/test_phase2_segmentation.py`):** Disjoint calibration (50%) vs frozen evaluation test set (50%) [A4] reporting dual-unit errors (frames and milliseconds) [A6].

---

## 2. The 6 Non-Negotiable Operational Amendments [A1–A6]

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PHASE 2 IMPLEMENTATION AMENDMENTS                     │
│                                                                             │
│  [A1] Multi-Cue Pop Formulation : Fusion score over tail, impulse, ascent   │
│  [A2] Canonical "Z-Up" Framing  : Unified world height convention (z = -y)  │
│  [A3] Persistent Catch Window   : Temporal stability over N frames (K >= 3) │
│  [A4] Disjoint Calibration Split: Calibration set != Frozen evaluation set  │
│  [A5] Ground Impact vs. v_z ~ 0 : Descent + proximity + deceleration shock  │
│  [A6] Time-Scaled Temporal Cuts : Windows in seconds (tau), dual metrics    │
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **[A1] Multi-Cue Pop Formulation:** Replacing fragile single-cue tail-distance minimum with an ensemble evidence score:
   $$S_{\text{pop}}(t) = w_1 S_{\text{tail-ground}}(t) + w_2 S_{\text{ankle-impulse}}(t) + w_3 S_{\text{board-ascent}}(t) + w_4 S_{\text{temporal-prior}}(t)$$
   where pop is marked at the maximum evidence peak within the approach-to-flight transition window.
2. **[A2] Canonical "Z-Up" Framing:** All trajectory vertical coordinates are converted immediately upon ingestion to a canonical world convention:
   $$z(t) = -y(t) \implies \text{positive } \dot{z} = \text{upward motion}, \quad \text{negative } \dot{z} = \text{downward descent}$$
   Apex is unambiguously defined as $\max(z(t))$ where $\dot{z}$ crosses zero from positive to negative.
3. **[A3] Temporally Persistent Catch Detection:** Eliminating false single-frame optical crossing artifacts by enforcing sustained multi-frame contact:
   $$\text{Catch is valid at } t_{\text{catch}} \iff S_{\text{catch}}(t) \ge \tau_{\text{catch}} \quad \forall \, t \in [t_{\text{catch}}, \, t_{\text{catch}} + K]$$
   where $K = \text{round}(\tau_{\text{persist}} \times \text{FPS})$ represents a sustained physical contact window ($\tau_{\text{persist}} \approx 30\text{--}50\text{ ms}$, corresponding to 2–3 frames at 60 FPS or 4–6 frames at 120 FPS).
4. **[A4] Disjoint Calibration vs. Evaluation Sets:** Enforcing strict skater-disjoint splitting on the annotated event dataset:
   - **Calibration Set (50%):** Optimize weights and thresholds $(\tau_d, \tau_v, \tau_\alpha)$.
   - **Frozen Evaluation Set (50%):** Run final benchmark without any parameter tuning.
5. **[A5] Impact-Driven Landing Detection:** Replacing near-zero velocity ($\dot{z} \approx 0$) with a causal kinematic chain:
   $$\text{Descent Phase } (\dot{z} < -\epsilon) \longrightarrow \text{Ground / Wheel Proximity} \longrightarrow \text{Impact Shockwave } (\ddot{z}_{\text{knees}} > \tau_{\text{shock}}) \longrightarrow \text{Post-Contact Rollout}$$
6. **[A6] Time-Scaled Temporal Windows & Dual-Unit Error Metrics:**
   - Window boundaries defined in continuous seconds ($\tau$), converted dynamically: $N = \text{round}(\tau \times \text{FPS})$.
   - Localization error reported in both **frames** and **milliseconds** ($\text{MAE}_{\text{frames}}$ and $\text{MAE}_{\text{ms}}$).

---

## 3. Step-by-Step Implementation Sequence

### Step 2.1: Signal Conditioning & Canonical Z-Up Framing (`src/segmentation/signal_processor.py`)
* Ingest smoothed trajectory series from Phase 1 Parquet feature stores.
* Invert image $y$ into canonical $z$-up coordinates ($z = -y$).
* Compute deck centroid trajectory $z_{\text{deck}}(t) = \frac{z_N(t) + z_T(t)}{2}$.
* Extract vertical board velocity $\dot{z}_{\text{deck}}(t)$ and acceleration $\ddot{z}_{\text{deck}}(t)$ using central difference stencils after Savitzky-Golay filtering.
* Compute instantaneous deck in-plane angular velocity $\omega(t) = \frac{d\theta_{\text{unwrapped}}}{dt}$ and angular acceleration $\alpha(t) = \frac{d\omega}{dt}$.

### Step 2.2: Pop and Apex Localization (`src/segmentation/event_detector.py`)
* **Pop Detection ($t_{\text{pop}}$) [A1]:**
  * Fuse tail-ground proximity, rear ankle downward acceleration impulse, and upward deck centroid ascent into $S_{\text{pop}}(t)$.
  * Select peak evidence score within pre-apex window.
* **Apex Detection ($t_{\text{apex}}$) [A2]:**
  * Identify global maximum vertical deck elevation $\max(z_{\text{deck}}(t))$ during flight.
  * Confirm zero-crossing of vertical velocity: $\dot{z}_{\text{deck}}(t_{\text{apex}} - 1) > 0 \ge \dot{z}_{\text{deck}}(t_{\text{apex}} + 1)$.

### Step 2.3: Persistent Probabilistic Catch Localization (`src/segmentation/catch_detector.py`)
* Search within descent temporal window $[t_{\text{apex}}, t_{\text{land}}]$.
* Compute instantaneous contact evidence score:
  $$S_{\text{catch}}(t) = w_1 \exp\left(-\frac{\Delta d(t)}{\sigma_d}\right) + w_2 \exp\left(-\frac{\Delta v(t)}{\sigma_v}\right) + w_3 \exp\left(-\frac{|\alpha(t)|}{\sigma_\alpha}\right)$$
* Apply persistence constraint over $K = \text{round}(\tau_{\text{persist}} \times \text{FPS})$ consecutive frames [A3].
* Mark $t_{\text{catch}}$ as the onset of sustained contact.

### Step 2.4: Impact-Driven Landing Localization (`src/segmentation/landing_detector.py`)
* Monitor descent phase ($\dot{z} < -\epsilon$) followed by deck leveling and ground proximity.
* Detect deceleration shockwave in skater knee/hip joints ($\ddot{z}_{\text{knee}} > \tau_{\text{shock}}$) [A5].
* Mark $t_{\text{land}}$ as the initial ground impact frame.

### Step 2.5: Sequence Validation & Continuous Window Slicing (`src/segmentation/validator.py`, `src/segmentation/slicer.py`)
* Validate temporal consistency: $t_{\text{pop}} < t_{\text{apex}} < t_{\text{catch}} \le t_{\text{land}}$.
* If ordering is violated, flag attempt with `TEMPORAL_INVERSION_ERROR` and log failure mode.
* Generate time-scaled window slices [A6]:
  * *Approach Phase:* $[t_{\text{pop}} - \tau_{\text{pre}}, t_{\text{pop}}]$
  * *Flight / Trick Execution Phase:* $[t_{\text{pop}}, t_{\text{catch}}]$
  * *Landing / Rollout Phase:* $[t_{\text{land}}, t_{\text{land}} + \tau_{\text{post}}]$

---

## 4. Quality Control (QC) & Acceptance Criteria (Gate 2)

Evaluated across the frozen evaluation test set (skater-disjoint from calibration set [A4]):

| Metric / Checkpoint | Hard Operational Gate | Aspirational Target | Verification Method |
| :--- | :--- | :--- | :--- |
| **Pop Event Localization Error** | $\text{MAE} \le 5.0\text{ frames}$ ($\le 83\text{ ms}$) | $\text{MAE} \le 3.0\text{ frames}$ ($\le 50\text{ ms}$) | Absolute frame & ms difference against human consensus annotations. |
| **Apex Event Localization Error** | $\text{MAE} \le 4.0\text{ frames}$ ($\le 67\text{ ms}$) | $\text{MAE} \le 2.0\text{ frames}$ ($\le 33\text{ ms}$) | Absolute frame & ms difference against human consensus annotations. |
| **Catch Event Localization Error** | $\text{MAE} \le 7.0\text{ frames}$ ($\le 116\text{ ms}$) | $\text{MAE} \le 5.0\text{ frames}$ ($\le 83\text{ ms}$) | Absolute frame & ms difference against human consensus annotations. |
| **Landing Localization Error** | $\text{MAE} \le 5.0\text{ frames}$ ($\le 83\text{ ms}$) | $\text{MAE} \le 3.0\text{ frames}$ ($\le 50\text{ ms}$) | Absolute frame & ms difference against human consensus annotations. |
| **Temporal Monotonicity Rate** | $\ge 95.0\%$ of landed attempts | $\ge 98.0\%$ of landed attempts | Percentage of clips strictly adhering to $t_{\text{pop}} < t_{\text{apex}} < t_{\text{catch}} \le t_{\text{land}}$. |

---

## 5. Phase Documentation & Reporting Requirements

Upon completing Phase 2, compile `docs/reports/PHASE_2_REPORT.md` covering:
1. Executive summary of temporal event localization results.
2. Code implementations and mathematical formulations for A1–A6.
3. Parameter calibration table on disjoint calibration set and test evaluation on frozen set.
4. QC Results vs. Gate 2 targets (dual-unit MAE in frames and milliseconds).
5. Failure analysis and edge cases (e.g. bails, early stomps).
6. Sign-off decision for advancing to Phase 3.

