# SkateKine Reference Literature & Datasets Directory

This directory contains external academic papers, reference kinematic datasets, and literature guides informing the SkateKine trick recognition and cleanliness scoring engine.

---

## 1. Reference Papers

| Filename | Title | Authors & Venue | Key Focus & Relevance |
|---|---|---|---|
| [`Action_Recognition_of_Skateboarding_Tricks__Ollie_and_Kickflip_Using_Neural_Network.pdf`](file:///d:/Future%20Career/Skateboard%20Trick%20Recognition%20&%20Cleanliness%20Scoring%20Engine/data/references/Action_Recognition_of_Skateboarding_Tricks__Ollie_and_Kickflip_Using_Neural_Network.pdf) | *Action Recognition of Skateboarding Tricks – Ollie and Kickflip Using Neural Network* | Swati Shilaskar et al. (IEEE 2023) | Benchmark on 222 raw video clips (108 Ollie, 114 Kickflip) from `LightningDrop/SkateboardML`. Evaluates CNN, ConvLSTM2D (79% acc), and MobileNet+BiLSTM (75% acc). Demonstrates saturation of end-to-end 2D vision models without kinematic tracking. |
| [`Motion_Based_Trick_Classification_in_Skateboarding_Using_Machine_Learning.pdf`](file:///d:/Future%20Career/Skateboard%20Trick%20Recognition%20&%20Cleanliness%20Scoring%20Engine/data/references/Motion_Based_Trick_Classification_in_Skateboarding_Using_Machine_Learning.pdf) | *Motion Based Trick Classification in Skateboarding Using Machine Learning* | Bernhard Hollaus et al. (IEEE 2023) | 12 skaters, 5 canonical tricks (Ollie, BS 180, FS 180, Kickflip, 360 Flip) using 120 Hz front-truck axle IMU. Identifies the **Ollie Black Hole effect** (fallback class when rotational confidence is low) and **Stance Inversion** (Goofy vs Regular rotational mirroring between BS 180 and FS 180). |
| [`peerj-cs-07-680.pdf`](file:///d:/Future%20Career/Skateboard%20Trick%20Recognition%20&%20Cleanliness%20Scoring%20Engine/data/references/peerj-cs-07-680.pdf) | *The classification of skateboarding tricks via transfer learning pipelines* | Muhammad Amirul Abdullah, Muhammad Aizzat Zakaria et al. (PeerJ Comp. Sci. 2021) | 6 skaters, 5 tricks (Ollie, Nollie FS Shuvit, FS 180, Pop Shove-it, Kickflip). Evaluates Continuous Wavelet Transform (CWT Morlet scalograms) + MobileNet/ResNet feature extractors + Linear/RBF SVM. Demonstrates distinct time-frequency energy bursts for planar shove-its. |
| [`2311.11467v2.pdf`](file:///d:/Future%20Career/Skateboard%20Trick%20Recognition%20&%20Cleanliness%20Scoring%20Engine/data/references/2311.11467v2.pdf) | *SkateboardAI: Monocular Vision-Based Skateboarding Analysis* | Video action analysis and trick localization literature | Deep monocular tracking approaches, temporal action proposal boundaries, and trick event segmentation. |
| [`MA-MED-21-VZ_paper_Juriga_Patrick-1.pdf`](file:///d:/Future%20Career/Skateboard%20Trick%20Recognition%20&%20Cleanliness%20Scoring%20Engine/data/references/MA-MED-21-VZ_paper_Juriga_Patrick-1.pdf) | *Master Thesis: Kinematic Analysis of Skateboarding Manoeuvres* | Patrick Juriga (2021) | Ground-truth kinematic markers, biomechanical trick phases (crouch, pop, flight, catch, land), and rigid-body deck planar equations. |

---

## 2. Reference Tabular Datasets

- [`Skateboarding_Kinematic_Dataset.csv`](file:///d:/Future%20Career/Skateboard%20Trick%20Recognition%20&%20Cleanliness%20Scoring%20Engine/data/references/Skateboarding_Kinematic_Dataset.csv): High-frequency kinematic trajectories with joint angles and body velocities.
- [`Skateboarding_Trick_Attempt_Dataset.csv`](file:///d:/Future%20Career/Skateboard%20Trick%20Recognition%20&%20Cleanliness%20Scoring%20Engine/data/references/Skateboarding_Trick_Attempt_Dataset.csv): Multi-skater trick attempt logs, outcomes (land/bail), and temporal event annotations.

---

## 3. Raw Video Data Repositories (`data/raw_videos/`)

- `archive_highspeed/`: 413 high-frame-rate clips from Player A and Player B across canonical tricks.
- `thrasher_batb/`: 598 competition clips from Battle at the Berrics pro skaters with clean lands and bails.
- `skaterxl_synthetic/`: 24 synthetic physics game captures used for quarantined reference validation.
- `skateboard_ml/`: 222 clips (108 Ollie, 114 Kickflip) organized into `Kickflip/` and `Ollie/` subdirectories, cataloged in [`skateboard_ml_manifest.csv`](file:///d:/Future%20Career/Skateboard%20Trick%20Recognition%20&%20Cleanliness%20Scoring%20Engine/data/metadata/skateboard_ml_manifest.csv).
