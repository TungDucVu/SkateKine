# Phase 3.5 Implementation Plan: Trajectory Store Expansion & Recognition Recovery

---

## 1. Executive Summary & Objective

Phase 3 established an exemplary, leakage-free architectural foundation (9.5/10 pipeline hygiene, stance/viewpoint canonicalization, zero raw RGB leakage, and a provisionally approved Post-Impact Land/Bail Verification Gate with 94.7% F1). However, the pre-training audit and empirical benchmarks confirmed that **Gate 3 (Trick Recognition) failed due to Trajectory Store Starvation and Skater Confounding** ($N=19\text{--}26$ attempts across 9 classes, with basic tricks heavily dominated by single recording subjects).

**Phase 3.5 is the targeted operational recovery phase.** Its objective is to scale the trajectory store from the small pilot batch to a statistically powered multi-skater corpus ($\ge 250\text{--}300$ clips) across all 9 canonical classes, rerun the nested benchmarks, and achieve Gate 3 operational certification before unlocking Phase 4.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PHASE 3.5 OPERATIONAL SEQUENCE                     │
│                                                                             │
│  [Step 3.5.1] Batch Trajectory Expansion (Parallel / Checkpointed Pipeline) │
│               • Ingest remaining tracking-eligible clips from manifest      │
│               • Target: >= 250-300 canonical clips (>= 20-30 per class)     │
│               • All 9 classes populated with multiple skaters               │
│                                                                             │
│  [Step 3.5.2] Skater Stratification Audit                                   │
│               • Verify non-Player-A coverage for Pop Shuv, FS Shuv, 180s    │
│               • Construct strict outer test split with diverse skaters      │
│                                                                             │
│  [Step 3.5.3] Rerun Dual-Mode Benchmarks & Systematic Ablations             │
│               • Model A (Calibrated Rules), Model B (GBDT), Model C (ST-GCN)│
│               • Target: Model C Top-1 > 65%, Top-2 > 85% on unseen skaters  │
│               • Formal comparison of A0 (Sparse Positional) vs A5 (Dynamics)│
│                                                                             │
│  [Step 3.5.4] Gate 3 Certification & Phase 4 Unlock                         │
│               • Output: docs/reports/PHASE_3_5_REPORT.md                    │
│               • Formally certify Gate 3 and unlock Phase 4 Cleanliness      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Step-by-Step Execution Sequence

### Step 3.5.1: High-Throughput Batch Trajectory Ingestion (`src/tracking/batch_processor.py`)
* Construct robust batch processing script with progress bar, frame skip / resolution checks, and automatic checkpointing to resume without re-running existing parquet files.
* Select tracking-eligible clips from `data/metadata/video_manifest.csv` prioritizing:
  1. Under-represented canonical classes: `pop_shuvit`, `frontside_shuvit`, `360_flip`, `backside180`, `frontside180`, `ollie`, `hardflip`.
  2. Diverse skaters from BATB (`chris_joslin`, `shane_oneill`, `sewa_kroetkov`, `luan_oliveira`, `chris_cole`, `cody_cepeda`, `sean_malto`, `pj_ladd`, `torey_pudwill`).
* Output verified Parquet trajectory files into `features/v1_trajectories/`.

### Step 3.5.2: Skater Stratification Audit
* Re-run `src/classification/dataset_audit.py` on the expanded trajectory store.
* Confirm that each class contains at least 3 distinct skaters in training folds and at least 2 distinct skaters in test folds.

### Step 3.5.3: Model Re-Training & Benchmark Execution
* Re-run `src/classification/phase3_pipeline.py` on the expanded multi-skater corpus:
  * **Model A:** Calibrated rule baseline with refined threshold boundaries.
  * **Model B:** LightGBM / XGBoost with nested 5-fold GroupKFold by `skater_id` and SHAP explanations.
  * **Model C:** ST-GCN trained with cosine annealing learning rate scheduler across 50 epochs.
  * **Ablation Suite:** A0 (Sparse Phase-Summary Positional Baseline) through A5 (Full Dynamics).

### Step 3.5.4: Formal Certification & Phase 4 Transition
* Produce `docs/reports/PHASE_3_5_REPORT.md` and updated `phase3_benchmark_results.json`.
* Hard Exit Criteria:
  * Gate 3: End-to-End Macro F1 $\ge 70\%$, Model C Top-2 $\ge 85\%$ on unseen skaters.
  * Gate 4: Post-Impact Land/Bail verification F1 $\ge 90\%$, FPR $< 5\%$.
* Upon passing, unlock Phase 4 Cleanliness Scoring.
