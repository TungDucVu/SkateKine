"""
Phase 3 Step 3.7: Systematic Feature Ablation Matrix (A0–A5).
Systematically compares information contributions across identical nested GroupKFold splits.
Includes A0 common position baseline to isolate feature modality from parameter capacity.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score
import xgboost as xgb

ABLATION_FEATURE_SETS = {
    'A0_raw_positions': [
        'pop_board_y', 'apex_board_y', 'land_board_y',
        'pop_hip_y', 'apex_hip_y', 'land_hip_y',
        'flight_duration_sec'
    ],
    'A1_pose_only': [
        'flight_duration_sec', 'ascent_ratio', 'apex_displacement_px',
        'flick_dx', 'flick_dy', 'flick_vel_mag', 'flick_direction_x',
        'pop_hip_y', 'apex_hip_y', 'land_hip_y'
    ],
    'A2_board_only': [
        'flight_duration_sec', 'ascent_ratio',
        'delta_theta_net', 'canonical_delta_theta_net', 'theta_abs', 'rot_consistency',
        'omega_peak', 'omega_mean', 'flip_cycle_count', 'min_aspect', 'aspect_std', 'aspect_range',
        'bvel_x_mean', 'bvel_y_mean', 'bacc_y_mean',
        'pop_board_y', 'apex_board_y', 'land_board_y'
    ],
    'A3_pose_plus_board': [
        'flight_duration_sec', 'ascent_ratio', 'apex_displacement_px',
        'pop_board_y', 'apex_board_y', 'land_board_y',
        'pop_hip_y', 'apex_hip_y', 'land_hip_y',
        'left_foot_dist', 'right_foot_dist'
    ],
    'A4_coords_plus_velocities': [
        'flight_duration_sec', 'ascent_ratio', 'apex_displacement_px',
        'pop_board_y', 'apex_board_y', 'land_board_y',
        'pop_hip_y', 'apex_hip_y', 'land_hip_y',
        'bvel_x_mean', 'bvel_x_std', 'bvel_y_mean', 'bvel_y_std',
        'flick_vel_mag', 'omega_mean', 'omega_peak'
    ],
    'A5_full_engineered_dynamics': [
        'flight_duration_sec', 'ascent_ratio', 'apex_displacement_px', 'has_catch',
        'delta_theta_net', 'canonical_delta_theta_net', 'theta_abs', 'rot_consistency',
        'omega_peak', 'omega_mean', 'canonical_omega_mean',
        'flip_cycle_count', 'min_aspect', 'aspect_std', 'aspect_range',
        'flick_dx', 'flick_dy', 'flick_vel_mag', 'flick_direction_x',
        'bvel_x_mean', 'bvel_x_std', 'bvel_x_max',
        'bvel_y_mean', 'bvel_y_std', 'bvel_y_max',
        'bacc_y_mean', 'bacc_y_std', 'bacc_y_max',
        'jerk_mean', 'jerk_std', 'jerk_max',
        'left_foot_dist', 'right_foot_dist',
        'pop_board_y', 'apex_board_y', 'land_board_y',
        'pop_hip_y', 'apex_hip_y', 'land_hip_y'
    ]
}


class SystematicAblationRunner:
    """Runs the formal A0–A5 ablation matrix across identical skater splits."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def run_ablations(
        self,
        X_df: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        n_splits: int = 4
    ) -> Dict[str, Dict[str, Any]]:
        classes = sorted(y.unique().tolist())
        class_map = {c: i for i, c in enumerate(classes)}
        y_enc = y.map(class_map).values

        unique_groups = groups.unique()
        actual_splits = min(n_splits, len(unique_groups))
        gkf = GroupKFold(n_splits=actual_splits)

        results = {}

        for exp_id, feat_cols in ABLATION_FEATURE_SETS.items():
            valid_cols = [c for c in feat_cols if c in X_df.columns]
            X = X_df[valid_cols].fillna(0.0).values

            y_trues = []
            y_preds = []

            for train_idx, test_idx in gkf.split(X, y_enc, groups=groups):
                X_train, y_train = X[train_idx], y_enc[train_idx]
                X_test, y_test = X[test_idx], y_enc[test_idx]

                from sklearn.preprocessing import LabelEncoder
                le_fold = LabelEncoder()
                y_tr_enc = le_fold.fit_transform(y_train)

                clf = xgb.XGBClassifier(
                    n_estimators=40,
                    max_depth=4,
                    learning_rate=0.08,
                    eval_metric='mlogloss',
                    random_state=self.seed,
                    n_jobs=1
                )
                clf.fit(X_train, y_tr_enc)
                p_test_enc = clf.predict(X_test)
                preds = le_fold.inverse_transform(p_test_enc)

                y_trues.extend(y_test)
                y_preds.extend(preds)

            macro_f1 = float(f1_score(y_trues, y_preds, average='macro', zero_division=0))
            weighted_f1 = float(f1_score(y_trues, y_preds, average='weighted', zero_division=0))
            acc = float(accuracy_score(y_trues, y_preds))

            results[exp_id] = {
                'num_features': len(valid_cols),
                'macro_f1': macro_f1,
                'weighted_f1': weighted_f1,
                'accuracy': acc,
                'feature_list': valid_cols
            }

        return results
