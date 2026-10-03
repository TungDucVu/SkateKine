# Phase Documentation & Reporting Standard

This document defines the mandatory structure and quality criteria for all phase completion reports in the **SkateKine** project. Every phase report must be written in Markdown and saved in `docs/reports/PHASE_{X}_REPORT.md`.

---

## Required Report Structure

Each phase report must strictly contain the following six sections:

### 1. Executive Summary & Work Completed
* **Objective:** High-level summary of what was planned for the phase and the primary accomplishments.
* **Work Completed:** Granular, bulleted breakdown of all tasks executed, modules developed, algorithms coded, and datasets processed.
* **Repository Deliverables:** Direct file links to newly created source code, configuration files, model weights, test suites, and data manifests.

### 2. Explicit Assumptions Made
* **Context:** Scientific models and computer vision pipelines always depend on simplifying operational assumptions.
* **Requirements:** Document every single assumption applied during the phase. Categories include:
  * Camera geometry and perspective assumptions (e.g., side profile within $\pm 15^\circ$, fixed focal distance).
  * Biomechanical and physical assumptions (e.g., standard deck length 80cm, rigid body assumptions, parabolic apex trajectory).
  * Keypoint visibility and sensor limitations (e.g., anatomical ankle offsets from the deck, optical motion blur tolerances).
  * Data and annotation assumptions (e.g., inter-rater reliability thresholds, skater identity disjointness).

### 3. Artifacts & Systems Built Upon Those Assumptions
* **Context:** Trace how each assumption directly informed software and architectural design decisions.
* **Requirements:** Explicitly detail:
  * Feature formulations and coordinate transformations derived from specific assumptions.
  * Threshold parameters, filter lengths, or loss weights calibrated under those assumptions.
  * Fallbacks, interpolation routines, or gating logic designed to catch violations of those assumptions.

### 4. Quality Control (QC) & Evaluation Results
* **Context:** Empirical verification against the quantitative phase gates established in the Master Specification.
* **Requirements:** Include a comparative evaluation table displaying:
  * **Metric Name**
  * **Operational Gate Target (Minimum Acceptable)**
  * **Aspirational Target**
  * **Actual Achieved Result**
  * **Status (PASS / FAIL / CONDITIONAL)**
* In-depth narrative discussing metric interpretation, statistical significance, and convergence behaviors.

### 5. Failure Analysis, Edge Cases & Mitigation
* **Context:** Transparent identification of edge cases, erroneous predictions, and pipeline vulnerabilities.
* **Requirements:**
  * Categorization of observed errors using the project's standard 4-tier taxonomy (Optical Failure, Tracking Failure, Kinematic Ambiguity, Boundary Localization Error).
  * Specific video clips or conditions that triggered failure.
  * Mitigation strategies implemented or scheduled for subsequent phases.

### 6. Phase Gate Sign-off & Next Phase Handover
* **Decision:** Formal declaration:
  * **APPROVED (GO):** All operational gate criteria met; advance to the next phase.
  * **CONDITIONAL APPROVAL:** Minor non-critical deviations noted with explicit remediation actions.
  * **REJECTED (NO-GO):** Operational gate failure; re-engineering required before advancing.
* **Sign-off Checklist:** Verification that code is committed, unit tests pass, and data manifests are preserved.
