# Technical Analysis: 3 New Skateboarding Research Papers & Strategic Synthesis for Phase 3.5.9

**Author:** Antigravity Engineering (Pair Programming with TungDucVu)  
**Date:** October 7, 2026  
**Status:** Completed & Cataloged  
**Reference Directory:** [`data/references/`](file:///d:/Future%20Career/Skateboard%20Trick%20Recognition%20&%20Cleanliness%20Scoring%20Engine/data/references/)

---

## 1. Executive Summary

To strengthen the SkateKine trick recognition engine and resolve remaining cross-skater failure modes (specifically Ollie skater concentration, BS180 vs. FS180 stance confusion, and the Ollie/Shove-it planar rotation boundary), we analyzed three newly introduced peer-reviewed academic papers in `data/references/`:

1. **Shilaskar et al. (IEEE 2023):** *Action Recognition of Skateboarding Tricks – Ollie and Kickflip Using Neural Network*
2. **Hollaus et al. (IEEE 2023):** *Motion Based Trick Classification in Skateboarding Using Machine Learning*
3. **Abdullah, Zakaria et al. (PeerJ Comp. Sci. 2021):** *The classification of skateboarding tricks via transfer learning pipelines*

Together, these papers evaluate both **pure monocular video action recognition** and **wearable/deck-mounted inertial kinematics** across amateur and competitive skateboarders. Their empirical findings directly confirm our architectural hypotheses from Phase 3.5.8 and provide concrete mathematical formulations for Phase 3.5.9.

---

## 2. Deep Dive: Paper 1 — Shilaskar et al. (IEEE 2023)

- **Full Citation:** Swati Shilaskar, Shivpriya Deshmukh, Shripad Bhatlawande, Jayesh Deshmukh, Harshal Dhande, Manoj Dohale. *"Action Recognition of Skateboarding Tricks – Ollie and Kickflip Using Neural Network."* IEEE 2023.
- **Reference File:** [`Action_Recognition_of_Skateboarding_Tricks__Ollie_and_Kickflip_Using_Neural_Network.pdf`](file:///d:/Future%20Career/Skateboard%20Trick%20Recognition%20&%20Cleanliness%20Scoring%20Engine/data/references/Action_Recognition_of_Skateboarding_Tricks__Ollie_and_Kickflip_Using_Neural_Network.pdf)
- **Extracted Text:** [`scratch/Action_Recognition_of_Skateboarding_Tricks__Ollie_and_Kickflip_Using_Neural_Network.txt`](file:///d:/Future%20Career/Skateboard%20Trick%20Recognition%20&%20Cleanliness%20Scoring%20Engine/scratch/Action_Recognition_of_Skateboarding_Tricks__Ollie_and_Kickflip_Using_Neural_Network.txt)

### 2.1 Dataset Origin & Properties
The authors utilized the open dataset [`LightningDrop/SkateboardML: SkateboardML 1.0`](https://github.com/LightningDrop/SkateboardML) containing **222 short video clips** in `.mov` format:
- **Ollie:** 108 clips
- **Kickflip:** 114 clips
- **Duration:** $\approx 2.0\text{--}2.5$ seconds per clip
- **Resolution:** Variable (640×360 up to 1920×1080)
- **Pre-processing:** Extracted 20 frames per video sequence, resized to $128 \times 128 \times 3$, normalized pixel values to $[0, 1]$, and applied standard scaling. 80:20 train/test split.

### 2.2 Model Architectures & Empirical Results

| Architecture | Description | Accuracy | AUC | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|
| **CNN** | 4 Convolutional layers + MaxPooling + Dense | 72% | 0.79 | 0.74 | 0.70 | 0.72 |
| **CRNN (ConvLSTM2D)** | 4 ConvLSTM2D layers (filters: 4, 8, 14, 16) + MaxPooling3D + Dense | **79%** | **0.86** | **0.82** | 0.77 | **0.80** |
| **MobileNet + BiLSTM** | Pretrained MobileNet feature extractor + Bidirectional LSTM | 75% | 0.84 | 0.77 | **0.80** | 0.78 |

### 2.3 Critical Takeaway for SkateKine
1. **The Ceilings of End-to-End Monocular RGB Vision:**
   Even when simplified to a binary 2-class problem (Ollie vs. Kickflip) on clean trimmed videos, pure deep appearance networks saturate at **75%–79% accuracy**. The spatial convolutions fail to reliably capture the rapid longitudinal flip rotation of the deck (which occurs in under 200 ms during flight) without explicit geometric tracking of the board edges.
2. **Validation of SkateKine's Kinematic Strategy:**
   This empirical result strongly vindicates SkateKine's core principle: **Monocular Video $\to$ 2D/3D Kinematic Pose & Rigid Body Geometry $\to$ Physical Invariants $\to$ ML Classification**. Black-box RGB models conflate the skater's bodily jump with the deck's rotation.
3. **Direct Data Utility:**
   This 222-video dataset (`Kickflip` [114] and `Ollie` [108]) has been organized into [`data/raw_videos/skateboard_ml/`](file:///d:/Future%20Career/Skateboard%20Trick%20Recognition%20&%20Cleanliness%20Scoring%20Engine/data/raw_videos/skateboard_ml/) and cataloged in [`skateboard_ml_manifest.csv`](file:///d:/Future%20Career/Skateboard%20Trick%20Recognition%20&%20Cleanliness%20Scoring%20Engine/data/metadata/skateboard_ml_manifest.csv). It directly relieves our critical skater bottleneck for Ollie in Phase 3.5.9A.

---

## 3. Deep Dive: Paper 2 — Hollaus et al. (IEEE 2023)

- **Full Citation:** Bernhard Hollaus, Ephraim Westenberger, Jonas Kreiner, Gabriel Freitas, Lennart Fresen. *"Motion Based Trick Classification in Skateboarding Using Machine Learning."* 2023 World Symposium on Digital Intelligence for Systems and Machines (DISA), IEEE 2023.
- **Reference File:** [`Motion_Based_Trick_Classification_in_Skateboarding_Using_Machine_Learning.pdf`](file:///d:/Future%20Career/Skateboard%20Trick%20Recognition%20&%20Cleanliness%20Scoring%20Engine/data/references/Motion_Based_Trick_Classification_in_Skateboarding_Using_Machine_Learning.pdf)
- **Extracted Text:** [`scratch/Motion_Based_Trick_Classification_in_Skateboarding_Using_Machine_Learning.txt`](file:///d:/Future%20Career/Skateboard%20Trick%20Recognition%20&%20Cleanliness%20Scoring%20Engine/scratch/Motion_Based_Trick_Classification_in_Skateboarding_Using_Machine_Learning.txt)

### 3.1 Experimental Setup & Kinematic Triggering
- **Participants:** 12 skaters performing tricks on 5 distinct skateboard setups (decks from 8.0" to 8.5", Independent/Thunder/Destructo trucks).
- **Sensor:** XSens Dot 6-DOF IMU (120 Hz) placed **underneath the axle arm of the front truck** (solid PLA + flexible TPU housing). This placement leaves the open deck between trucks unobstructed for grinds and slides.
- **Trick Set (5 classes):** Ollie, 180° Backside (BS), 180° Frontside (FS), Kickflip, 360° Flip.
- **Dataset Size:** 862 valid recorded attempts.
- **Kinematic Pop Trigger:**
  - The authors established an automated temporal trigger using the **angular spin rate around the y-axis** ($\omega_y$, truck axle axis).
  - Physical rationale: Every trick requires popping the board by lifting the nose and driving the tail into the ground. This sharp pitch rotation around the truck axle produces an unambiguous peak in $\omega_y$.
  - Windowing: Exactly $[-20, +99]$ time steps around the trigger (120 time steps total = 1.0 second duration at 120 Hz).
- **Data Augmentation:** The raw 862 samples were strongly imbalanced (e.g., 360 Flip was only executed by 1 skater). The authors used `librosa.effects.pitch_shift` + Gaussian noise to synthesize balanced 1,300 samples (260 per class).
- **Model:** 1D CNN (PyTorch, 118,757 parameters) with Adam optimizer. Overall test accuracy: **95.1%**.

### 3.2 Key Biomechanical Findings & Insights for SkateKine

#### Finding 1: The Ollie "Black Hole" / Fallback Fallacy
- **Empirical Observation:** In the test confusion matrix, **Ollie was the most frequently mispredicted class**. Low-confidence executions of rotational tricks (specifically 180° FS) collapsed into Ollie.
- **Biomechanical Explanation:** An Ollie is not a distinct isolated motion—it is the **fundamental sub-phase of almost every flatground skateboarding trick**. Every pop trick begins with an ollie-like crouch, tail snap, and front-foot slide.
- **Application to SkateKine:** In our B4 baseline, when rotation or flip signals are noisy or ambiguous, the classifier naturally defaults to Ollie.
  *Design Rule:* We must implement an explicit physical gate before fallback: if any measurable body/deck rotation or foreshortening occurs during flight ($|\Delta \psi| > \tau$ or $\min(L_{\text{board}}/L_0) < 0.65$), the candidate pool **must exclude Ollie**.

#### Finding 2: Stance Symmetry & Inversion (Goofy vs. Regular)
- **Empirical Observation:** The authors observed systematic confusion between 180° BS and 180° FS.
- **Biomechanical Explanation:** Skater stance directly mirrors the rotational coordinate system:
  - For a **Regular** skater (left foot forward):
    - Backside 180 rotates clockwise (positive yaw direction).
    - Frontside 180 rotates counter-clockwise (negative yaw direction).
  - For a **Goofy** skater (right foot forward):
    - Backside 180 rotates counter-clockwise (negative yaw direction).
    - Frontside 180 rotates clockwise (positive yaw direction).
  - Therefore, a Goofy skater performing a 180° BS exhibits angular kinematics identical to a Regular skater performing a 180° FS!
- **Application to SkateKine:** This perfectly explains our remaining confusion between BS180 and FS180 in Phase 3.5.8.
  *Formulation:* We must apply **stance sign normalization** to all body yaw $\Delta \psi_{\text{body}}$ and deck yaw $\Delta \psi_{\text{board}}$ features:
  $$\Delta \psi_{\text{normalized}} = \text{sign}(\text{stance}) \cdot \Delta \psi$$
  where $\text{sign}(\text{regular}) = +1$ and $\text{sign}(\text{goofy}) = -1$.

---

## 4. Deep Dive: Paper 3 — Abdullah, Zakaria et al. (PeerJ Comp. Sci. 2021)

- **Full Citation:** Muhammad Amirul Abdullah, Muhammad Ar Rahim Ibrahim, Muhammad Nur Aiman Shapiee, Muhammad Aizzat Zakaria, Mohd Azraai Mohd Razman, Rabiu Muazu Musa, Noor Azuan Abu Osman, Anwar P.P. Abdul Majeed. *"The classification of skateboarding tricks via transfer learning pipelines."* PeerJ Computer Science 7:e680, 2021.
- **Reference File:** [`peerj-cs-07-680.pdf`](file:///d:/Future%20Career/Skateboard%20Trick%20Recognition%20&%20Cleanliness%20Scoring%20Engine/data/references/peerj-cs-07-680.pdf)
- **Extracted Text:** [`scratch/peerj-cs-07-680.txt`](file:///d:/Future%20Career/Skateboard%20Trick%20Recognition%20&%20Cleanliness%20Scoring%20Engine/scratch/peerj-cs-07-680.txt)

### 4.1 Experimental Setup & Continuous Wavelet Transform
- **Participants:** 6 amateur skateboarders ($20 \pm 7$ years old, $\ge 5$ years experience) executing 5 trials each across 5 tricks on flat concrete:
  - Ollie (O)
  - Nollie Frontside Shuvit (NFS)
  - Frontside 180 (FS180)
  - Pop Shove-it (PS)
  - Kickflip (KF)
- **Sensor:** MPU6050 6-axis IMU ($a_x, a_y, a_z, \omega_x, \omega_y, \omega_z$) embedded in an ABS riser pad mounted behind the front truck.
- **Signal Transformations Evaluated:**
  1. **RAW:** 6 raw signals stacked into a 2D matrix.
  2. **CWT (Continuous Wavelet Transform):** Converted 1D time-series into 2D time-frequency scalograms using the **Morlet Mother Wavelet**:
     $$\psi(t) = \pi^{-1/4} e^{i \omega_0 t} e^{-t^2 / 2}$$
     Captures multi-scale transient bursts (high frequency at small scales, low frequency at large scales).
- **Classification Pipeline:**
  - Feature extraction: Pre-trained CNNs (MobileNet, MobileNetV2, NasNetLarge, NasNetMobile, ResNet101, ResNet101V2).
  - Classifier: Grid-searched SVM (Linear, Polynomial degrees 2–6, RBF; $C \in [0.01, 100]$, $\gamma \in [0.1, 100]$).
  - Evaluated 1,500 pipeline combinations across 60:20:20 train/validation/test splits.

### 4.2 Benchmark Results & Findings

| Input Type | Feature Extractor | Classifier | Test Accuracy | Precision | Recall | F1 | Prediction Time |
|---|---|---|---|---|---|---|---|
| **CWT** | **MobileNet** | **Linear SVM ($C=0.01$)** | **1.00 (100%)** | **1.00** | **1.00** | **1.00** | **0.109 s** |
| **RAW** | MobileNet | Linear SVM ($C=0.01$) | 1.00 (100%) | 1.00 | 1.00 | 1.00 | 0.125 s |
| **CWT** | ResNet101 | Linear SVM ($C=0.01$) | 1.00 (100%) | 1.00 | 1.00 | 1.00 | 0.250 s |
| **CWT** | NasNetLarge | Linear SVM | 0.96 | 0.97 | 0.96 | 0.96 | 1.438 s |

### 4.3 Key Insights for SkateKine

#### Finding 1: Time-Frequency Energy Bursts for Planar Shove-its
- **Observation:** In the CWT scalograms, **Pop Shove-it and Nollie FS Shuvit produce distinct high-frequency energy bursts during the scoop/snap phase** (0.0–0.2s post-pop) that are completely absent in pure vertical pops (Ollie) and axial barrel rolls (Kickflip).
- **Application to SkateKine:**
  Planar shove-its currently suffer in monocular video because visual deck foreshortening is noisy. However, the **temporal derivative / jerk of the aspect ratio** $d(L_{\text{board}}/L_0)/dt$ and the transient lateral foot sweep acceleration contain high-frequency burst energy during the scoop phase.
  *Design Rule:* Incorporating a wavelet/spectral energy or high-frequency transient metric into the Kinematic Tokenizer for the scoop phase directly separates Pop Shove-it and FS Shove-it from Ollie.

#### Finding 2: Decoupled Feature Extraction + Margin Maximization
- Replacing end-to-end deep learning with an explicit **feature extraction $\to$ margin-maximizing classifier (Linear/RBF SVM)** was dramatically faster and avoided overfitting on small skater cohorts.
- Directly supports our architectural approach: SkateKine computes explicit kinematic features and medoid prototype distances, followed by a constrained resolver.

---

## 5. Comparative Synthesis Matrix Across All 5 Reference Papers

| Paper | Domain & Modality | Dataset Scope | Key Tricks Covered | Primary Strength | Key Vulnerability / Lesson | SkateKine Adoption |
|---|---|---|---|---|---|---|
| **Shilaskar et al. (2023)** | Monocular 2D Video | 222 clips (web) | Ollie, Kickflip | Video action recognition benchmark | Saturated at 79% on 2 classes without 3D/kinematic tracking | Ingest 222 clips to resolve Ollie data bottleneck; validate kinematic superiority |
| **Hollaus et al. (2023)** | Front-truck IMU (120 Hz) | 12 skaters, 862 attempts | Ollie, BS180, FS180, Kickflip, 360 Flip | Axle pitch trigger ($\omega_y$); stance bias analysis | Ollie black-hole fallback; BS/FS 180 stance sign confusion | Implement stance sign normalization; add non-rotation gate for Ollie fallback |
| **Abdullah et al. (2021)** | Riser-pad IMU (6-axis) | 6 skaters, 150 samples | Ollie, Nollie FS Shuv, FS180, Pop Shove, Kickflip | Continuous Wavelet Transform (CWT) time-frequency scalograms | Small cohort; lab-like flat ground setup | Use transient scoop energy / spectral variance for planar shove-it tokens |
| **Juriga (2021)** | Motion Capture & Video | Lab biomechanics | Ollie, Kickflip, Pop Shove-it | Precise phase boundaries (pop, apex, catch, land) | Rigid-body markers require lab environment | Ground-truth event segmentation thresholds and physical deck lengths |
| **SkateboardAI (2023)** | Monocular Video | Wild video clips | Multi-trick | Temporal proposal bounding boxes | Lacks rigid-body physical constraints; misses planar shove-its | Camera-adaptive coordinate normalization |

---

## 6. Actionable Blueprint for Phase 3.5.9 Implementation

Based on these findings, we establish 4 concrete updates to the **Phase 3.5.9 Plan**:

### 6.1 Action Item 1: Ingest `skateboard_ml` to Break the Ollie Bottleneck (Phase 3.5.9A)
- **Status:** Videos moved to `data/raw_videos/skateboard_ml/` and cataloged in `data/metadata/skateboard_ml_manifest.csv` (108 Ollie, 114 Kickflip).
- **Execution:** Extract 2D/3D tracking coordinates through Phase 1 & 2 pipelines to expand the Ollie training/evaluation pool from 2 skaters to $>10$ diverse skaters.

### 6.2 Action Item 2: Stance Sign Normalization for 180s (Phase 3.5.9C)
- **Problem:** Goofy BS180 looks identical to Regular FS180 in absolute rotation coordinates.
- **Formulation:**
  $$\psi_{\text{body, norm}}(t) = s \cdot \psi_{\text{body}}(t), \quad s = \begin{cases} +1 & \text{if Regular} \\ -1 & \text{if Goofy} \end{cases}$$
  $$\psi_{\text{board, norm}}(t) = s \cdot \psi_{\text{board}}(t)$$
- This mirrors the rotational coordinate system so that BS180 always corresponds to $+180^\circ$ and FS180 always corresponds to $-180^\circ$, eliminating cross-stance classification error.

### 6.3 Action Item 3: Ollie Fallback Guard / Physical Gating Rule (Phase 3.5.9B)
- **Problem:** The Hollaus et al. "Ollie Black Hole" causes low-confidence 180s and Shove-its to collapse into Ollie.
- **Formulation:**
  ```python
  def filter_candidate_families(kinematics):
      # Physical foreshortening and rotation check
      has_deck_foreshortening = (kinematics.min_board_len_ratio < 0.65)
      has_significant_body_yaw = (abs(kinematics.delta_body_yaw_norm) > 45.0)
      has_significant_flip = (kinematics.flip_energy > threshold)
      
      if has_deck_foreshortening or has_significant_body_yaw or has_significant_flip:
          # Prohibit falling back into Ollie
          return [trick for trick in CANDIDATES if trick != 'ollie']
      return CANDIDATES
  ```

### 6.4 Action Item 4: Transient High-Frequency Scoop Token for Shove-its (Phase 3.5.9C)
- **Inspiration:** Abdullah et al. CWT time-frequency scalograms.
- **Formulation:**
  Compute the transient high-frequency energy $E_{\text{scoop}}$ of the board aspect ratio velocity during the pop-to-apex window:
  $$E_{\text{scoop}} = \sum_{t=t_{\text{pop}}}^{t_{\text{apex}}} \left( \frac{d}{dt} \frac{L_{\text{board}}(t)}{L_0} \right)^2$$
  Tricks with $E_{\text{scoop}} > \tau_{\text{scoop}}$ and minimal body yaw are strongly routed to the `{pop_shuvit, frontside_shuvit}` family, preventing confusion with clean Ollies.
