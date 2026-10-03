# Phase 5 Implementation Plan: Telemetry HUD Overlay & Interactive Dashboard

---

## 1. Overview & Scope

Phase 5 delivers the user-facing presentation and analytical inspectability layer of SkateKine. By translating internal mathematical state representations into an interactive dashboard and telemetry-augmented video, Phase 5 allows coaches, skaters, and researchers to visually audit every intermediate step of the engine—from raw keypoint trajectories to phase transitions and metric cleanliness attributions.

### Key Deliverables of Phase 5:
1. **High-Performance Video Telemetry Renderer:** Frame-by-frame OpenCV renderer burning trajectory trails, skeletal landmarks, 8-point board polygons, and real-time HUD telemetry into an annotated MP4.
2. **Kinematic Signal Plotting Engine:** Interactive time-series visualizer (Plotly / Matplotlib) plotting vertical board displacement and angular velocity curves with phase event lines.
3. **Interactive Inspection Application (Streamlit):** Web interface supporting drag-and-drop video upload, side-by-side synchronized video playback, scorecard breakdown, and failure diagnostics.
4. **Structured Data Export Subsystem:** Automated packager exporting annotated MP4 video, raw feature `.parquet`, and standardized JSON scorecards.
5. **Real-Time Throughput Benchmark & Resilience Suite:** Automated load test verifying latency ($\le 1.5\times$ real-time on target GPU) and zero unhandled crashes across 50 consecutive clips.

---

## 2. Core Assumptions & Boundary Constraints

1. **Deterministic Visual Synchronization:** Telemetry overlay text and graphics must strictly synchronize with video frame timestamps. Audio-video desynchronization or frame-rate aliasing is unacceptable.
2. **Transparent Diagnostic Flags:** If the confidence gatekeeper flagged the attempt (e.g. `LOW_RELIABILITY_OCCLUSION` or `BAILED`), the overlay and dashboard must prominently render high-contrast warning badges rather than displaying misleading partial scores.
3. **Target Execution Environment:** Real-time processing benchmarks are calibrated for an NVIDIA RTX 4070 (12GB VRAM) or equivalent host system running 1080p60 inputs.
4. **Export Completeness:** Exported JSON scorecards must adhere to a standardized schema enabling downstream database integration and automated ranking.

---

## 3. Step-by-Step Implementation Sequence

### Step 5.1: Video Telemetry Rendering Engine (`src/visualization/telemetry_renderer.py`)
* Implement multi-layered visual overlay using OpenCV:
  * **Layer 1 (Skeleton & Board Polygons):** Colored joint bones (green for high confidence, yellow/red for degraded confidence) and 8-point board wireframe.
  * **Layer 2 (Motion Trails):** Fading trajectory trails (last 15 frames) for skateboard nose, tail, and skater center-of-mass.
  * **Layer 3 (Phase Markers):** Visual popups at $t_{\text{pop}}$ (red), $t_{\text{apex}}$ (cyan), $t_{\text{catch}}$ (magenta), and $t_{\text{land}}$ (green).
  * **Layer 4 (Telemetry HUD Header):** Top banner with Trick Name, Classification Confidence %, Land/Bail Status, and Cleanliness Index.
  * **Layer 5 (Kinematic Metric Card):** Lower corner HUD card displaying Catch Elevation Ratio %, Metric Truck Landing Offset (cm), Projected Roll Error ($^\circ$), and Rollout Stability %.
* Optimize frame encoding pipeline using NVENC hardware acceleration when available.

### Step 5.2: Synchronized Analytical Signal Visualizer (`src/visualization/signal_plotter.py`)
* Build interactive Plotly chart module generating dual-pane synchronized plots:
  * **Pane A:** Vertical displacement curve $y_{\text{deck}}(t)$ with vertical milestone dashed lines for Pop, Apex, Catch, and Landing.
  * **Pane B:** Angular velocity curve $\omega_{\text{board}}(t)$ showing rotation acceleration and stabilization plateau at catch.
* Add interactive scrubber linking plot cursor to video playback frame.

### Step 5.3: Streamlit Interactive Dashboard (`src/app/streamlit_app.py`)
* Construct multi-panel layout:
  * **Panel 1 (Upload & Configuration):** Video drag-and-drop, camera perspective selector, stance override (Regular/Goofy), and model selection (Model A / B / C).
  * **Panel 2 (Playback & Inspection):** Side-by-side player comparing original raw clip against rendered telemetry clip with step-frame forward/backward controls.
  * **Panel 3 (Biomechanical Scorecard):** Radar chart displaying sub-scores (Catch, Bolts, Stability, Rotation) alongside the composite 0–100 Cleanliness rating.
  * **Panel 4 (Feature Attribution):** Waterfall plot showing positive contributions (e.g. high catch elevation) and negative deductions (e.g. landing off-bolts).
  * **Panel 5 (Diagnostic Log):** Detailed table of detected keypoint tracking confidences and gatekeeper warning messages.

### Step 5.4: Export & Serialization Engine (`src/app/exporter.py`)
* Implement export handlers:
  * **Rendered Video:** Export as web-compatible H.264 MP4 (`outputs/{clip_id}_telemetry.mp4`).
  * **Kinematic Trace:** Export full trajectory data as `.parquet` (`outputs/{clip_id}_features.parquet`).
  * **Scorecard JSON:** Export structured metrics adhering to formal schema (`outputs/{clip_id}_scorecard.json`).

### Step 5.5: Throughput & Stress Benchmark (`tests/benchmark_pipeline.py`)
* Construct test runner ingesting 50 diverse video clips (clean attempts, bails, occlusions, varying lighting).
* Measure total pipeline latency (tracking + normalization + segmentation + classification + scoring + rendering).
* Verify zero uncaught exceptions and validate error flag propagation.

---

## 4. Post-Phase Checkpoint: QC, Evaluation & Acceptance Criteria

### Quality Control (QC) Gates

| Metric / Checkpoint | Hard Operational Gate | Aspirational Target | Verification Method |
| :--- | :--- | :--- | :--- |
| **Pipeline Latency** | $\le 1.5\times$ video real-time ($\le 25\text{ ms/frame}$ on RTX 4070) | $\le 1.0\times$ video real-time ($\le 16.6\text{ ms/frame}$) | Total wall-clock time divided by input clip duration across 50 test clips. |
| **Pipeline Reliability / Crash Rate** | 0 unhandled crashes across 50 clips ($100\%$ completion) | 0 unhandled crashes | Automated test run on uncurated challenging skate clips. |
| **Audio-Visual Sync Drift** | $\le 1\text{ frame}$ drift | $0\text{ frame}$ drift | Synchronization check between rendered overlay frame index and underlying video. |
| **Confidence Flag Display Integrity** | $100\%$ accuracy in rendering `LOW RELIABILITY` badge when triggered | $100\%$ accuracy | Visual audit of clips flagged with tracking dropouts or bails. |
| **Export Validity** | $100\%$ compliance with JSON scorecard schema | $100\%$ compliance | Pydantic JSON schema validator on all generated exports. |

---

## 5. Phase Documentation & Reporting Requirements

Upon completing Phase 5, prepare `docs/reports/PHASE_5_REPORT.md` including:
1. **Executive Summary:** Overview of the telemetry rendering engine, Streamlit UI, throughput benchmarks, and export pipeline.
2. **Work Completed:** Details on OpenCV HUD rendering classes, Plotly signal integrations, Streamlit dashboard architecture, and exporter routines.
3. **Assumptions Made:** Formal documentation of UI assumptions (synchronous single-clip processing, hardware acceleration availability, standard JSON output schema).
4. **Artifacts Built on Assumptions:** Video layout layouts, HUD typography and color palettes, interactive Plotly dashboard components, and Pydantic export schemas.
5. **QC Results vs. Expected Targets:** Quantitative table detailing measured latency (ms/frame and real-time factor), stability crash-test results (0/50 crashes), and schema validation results against Gate 6 criteria.
6. **Failure Analysis & Edge Cases:** Handling of irregular aspect ratios, high resolution (4K) downsampling overhead, and browser video decoding bottlenecks.
7. **Sign-off Decision:** Clear GO / NO-GO recommendation for advancing to Phase 6.
