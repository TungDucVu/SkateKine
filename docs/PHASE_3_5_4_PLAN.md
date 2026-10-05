# Phase 3.5.4 Implementation Plan: Cross-Skater Data Acquisition for Flatground Basics & Benchmark Unfreezing

**Creation Date:** 2026-10-05  
**Baseline State (Phase 3.5.3 Frozen Benchmark):**
* **Active Trajectory Store:** 125 parquet files (110 canonical 9-class attempts + 15 non-canonical/bail attempts) across 14 pro skaters.
* **Gate 4 (Post-Impact Land/Bail):** **PASS (Pilot Certified / Provisional)** — 92.55% F1, 70.0% bail recall (21/30 bails caught), 5.43% FPR.
* **Gate 3 (Trick Recognition Engine):** **FAILED / RECOVERY IN PROGRESS** — 15.63% (Unweighted) / 16.16% (Class-Weighted) Macro F1 $\ll$ 82.0% target; Top-2: 38.18%.
* **Empirical Skater-Confounding Evidence:** +18.77% performance gap between Stratified 5-Fold (34.40% Macro F1) and strict Nested GroupKFold (15.63% Macro F1).
* **Varial / Hardflip Proof-of-Concept:** Recovered from $0.0\% \rightarrow 55.00\%$ F1 with 19 clips across 7 pro skaters and rotational interaction features.

---

## 1. Executive Summary & Problem Diagnosis

Phase 3.5.3 definitively answered the core algorithmic question: **inverse-frequency class weighting and model architecture changes cannot resolve the zero-F1 class collapse when the underlying training folds lack cross-skater variation.**

Our manifest audit (`scratch/inspect_candidates.py`) revealed that for the four collapsed or sparse basic flatground tricks:
* **Ollie:** Only 2 skaters exist in the entire 1,035-clip manifest (`player_a`: 48, `player_b`: 4) — 100% saturated in store.
* **Backside 180:** Only 3 skaters exist in the entire manifest (`player_a`: 51, `jack_colbourn`: 1, `sean_malto`: 1) — 100% saturated in store.
* **Frontside 180:** Only 4 skaters exist in the entire manifest (`player_a`: 53, `shane_oneill`: 1, `jack_colbourn`: 1, `sean_malto`: 1) — 100% saturated in store.
* **Pop Shove-it:** Only 3 skaters exist in the entire manifest (`player_a`: 53, `luan_oliveira`: 1, `torey_pudwill`: 1) — 100% saturated in store.

In strict Nested GroupKFold by skater, Fold 0 isolates `player_a` ($N=46$). When `player_a` is held out, the training fold contains only 1–3 clips from 1–2 skaters for these basic tricks, causing Fold 0 accuracy to drop to **8.7%** and freezing `Frontside 180`, `Frontside Shove-it`, and `Pop Shove-it` at **0.0% F1**.

**Phase 3.5.4 Mandate:** Acquire and materialize clean, independent multi-skater video trajectories for the four bottleneck flatground basics to establish true cross-skater statistical invariance.

---

## 2. Target Acquisition Matrix

| Canonical Class | Current Skaters in Store | Target New Skaters | Total Post-Expansion Target | Target External Skater Sources |
| :--- | :---: | :---: | :---: | :--- |
| **Ollie** | 2 (`player_a`, `player_b`) | $+4$ independent skaters | $\ge 6$ skaters | Mitchie Brusco (SkateIQ), Aaron Kyro (Braille), Jonny Giger, Spencer Barton |
| **Backside 180** | 3 (`player_a`, `colbourn`, `malto`) | $+3$ independent skaters | $\ge 6$ skaters | Mitchie Brusco, Jonny Giger, Spencer Nuzzi, Nigel Jones |
| **Frontside 180** | 4 (`player_a`, `oneill`, `colbourn`, `malto`) | $+3$ independent skaters | $\ge 7$ skaters | Mitchie Brusco, Aaron Kyro, Jonny Giger, Chris Cole |
| **Pop Shove-it** | 3 (`player_a`, `oliveira`, `pudwill`) | $+3$ independent skaters | $\ge 6$ skaters | Mitchie Brusco, Jonny Giger, Spencer Barton, Nigel Jones |

### Quality Ingestion Criteria:
1. **Camera Framing:** Wide/medium shot capturing complete skater body (head to feet) and full board throughout $[t_{\text{pop}} - 15, t_{\text{land}} + 30]$.
2. **Frame Rate:** Native $\ge 30\text{ fps}$ (preferably 60 fps).
3. **Execution Purity:** Clean flatground execution on smooth ground (no stairs, rails, or obstacles).
4. **Stance & Perspective Diversity:** Mixed regular and goofy stances, filmed from side, front-quarter, and rear-quarter angles to test canonicalization robustness.

---

## 3. Systematic Ingestion & Pipeline Protocol

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 3.5.4 DATA INGESTION & BENCHMARK FLOW              │
│                                                                             │
│  [Step 1] Acquire Video Clips via yt-dlp & ffmpeg                           │
│           • Directory: data/raw_videos/phase354_expansion/                  │
│           • Clean re-encoding: H.264 / yuv420p / 30-60 fps                  │
│                                                                             │
│  [Step 2] Register in Manifest (`data/metadata/video_manifest.csv`)         │
│           • Add unique clip_id (e.g., ext_ollie_mitchie_01)                │
│           • Annotate skater_id, stance, trick_name, outcome='land'          │
│                                                                             │
│  [Step 3] Materialize Trajectories via Phase1Pipeline                       │
│           • YOLOv8-pose + YOLOv8-board tracking                             │
│           • Canonical coordinate transformation & SG derivative filtering   │
│           • Output: features/v1_trajectories/{clip_id}.parquet              │
│                                                                             │
│  [Step 4] Re-run Frozen Benchmark (`phase3_pipeline.py`)                    │
│           • Evaluate on expanded trajectory store (N >= 135-140)            │
│           • Re-measure Nested GroupKFold vs. Stratified 5-Fold              │
│           • Test for unfreezing of FS180, BS180, Pop Shove-it               │
│                                                                             │
│  [Step 5] Document Findings in Updated PHASE_3_REPORT.md                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Key Hypotheses to Test

1. **The Representation Unfreezing Hypothesis:**  
   Providing $\ge 5\text{--}6$ independent pro skaters per class will eliminate the $0.0\%$ F1 collapse for `Frontside 180`, `Pop Shove-it`, and `Frontside Shove-it` without altering tree classifier hyperparameters.
2. **The Fold 0 Skater-Isolation Hypothesis:**  
   With diverse non-Player A skaters available in training folds when `player_a` is held out, Fold 0 accuracy will lift significantly from its current $8.7\%$ floor.
3. **The ST-GCN Generalization Check:**  
   Re-evaluating the audited `CompactSTGCN` on the expanded multi-skater dataset will determine whether additional cross-skater trajectories improve graph convolution generalization beyond the current $9.76\%$ baseline.

---

## 5. Exit Criteria for Phase 3.5.4

* [ ] $\ge 5\text{--}6$ independent pro/expert skaters materialized for all 9 canonical trick classes.
* [ ] Minimum per-class F1 across all 9 classes strictly $> 0.0\%$.
* [ ] Formal re-evaluation of Gate 3 against the operational target ($\ge 82.0\%$ Macro F1).
* [ ] Gate 4 provisional certification maintained with $\text{F1} \ge 80\%$ and $\text{FPR} < 8\%$.
