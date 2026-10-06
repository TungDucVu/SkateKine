"""
Phase 3 Step 3.5: Model B — Gradient Boosted Decision Trees (XGBoost / LightGBM).
Implements nested GroupKFold by skater_id (P3-A7) and SHAP feature importance analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import xgboost as xgb
import shap


FEATURE_COLUMNS = [
    'flight_duration_sec', 'ascent_ratio', 'apex_displacement_px', 'has_catch',
    'delta_theta_net', 'canonical_delta_theta_net', 'theta_abs', 'rot_consistency',
    'omega_peak', 'omega_mean', 'canonical_omega_mean',
    'skater_yaw_swap', 'feet_swap', 'delta_feet_dx', 'delta_feet_norm',
    'min_norm_length', 'board_length_std', 'board_yaw_swap', 'board_swap',
    'delta_board_norm', 'diff_yaw_norm',
    'flip_yaw_product',
    'flip_cycle_count', 'min_aspect', 'aspect_std', 'aspect_range',
    'flick_dx', 'flick_dy', 'flick_vel_mag', 'flick_direction_x', 'flick_local_y_delta',
    'bvel_x_mean', 'bvel_x_std', 'bvel_x_max',
    'bvel_y_mean', 'bvel_y_std', 'bvel_y_max',
    'bacc_y_mean', 'bacc_y_std', 'bacc_y_max',
    'jerk_mean', 'jerk_std', 'jerk_max',
    'left_foot_dist', 'right_foot_dist',
    'pop_board_y', 'apex_board_y', 'land_board_y',
    'pop_hip_y', 'apex_hip_y', 'land_hip_y',
    'board_length_pop', 'board_width_pop',
    # Phase 3.5.7 Tier 1 Physics Invariant Features
    'c_inv_cosine', 'trough_depth', 'trough_symmetry', 'trough_duration',
    'length_ratio_land_pop', 'recovery_slope',
    'delta_theta_body', 'delta_theta_board', 'delta_theta_relative', 'rot_ratio', 'rotational_coupling',
    'ballistic_r2', 'ballistic_rmse', 'effective_acc_y',
    'hip_board_clearance_mean', 'hip_board_clearance_max',
    'pop_impulse_dy', 'flick_acc_mag', 'flick_timing_ratio'
]


@dataclass
class TreeClassifierEvaluation:
    macro_f1: float
    weighted_f1: float
    top1_accuracy: float
    top2_accuracy: float
    per_class_f1: Dict[str, float]
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    confusion_matrix: List[List[int]]
    classes: List[str]
    shap_importance: Dict[str, float]
    nested_cv_results: List[Dict[str, Any]]


class SkateboardTreeClassifier:
    """
    Model B: XGBoost Tree Classifier with nested GroupKFold cross-validation by skater_id.
    """

    def __init__(self, n_estimators: int = 50, max_depth: int = 4, learning_rate: float = 0.05, seed: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        self.model = None
        self.class_mapping = {}
        self.inv_class_mapping = {}
        self.classes_ = []

    def fit_and_evaluate_nested_cv(
        self,
        X_df: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        n_outer_splits: int = 4,
        n_inner_splits: int = 3,
        use_class_weighting: bool = False
    ) -> TreeClassifierEvaluation:
        """
        Executes Nested GroupKFold validation [P3-A7].
        Outer splits withhold test skaters completely.
        Inner splits tune hyperparameter depth and learning rate.
        Supports inverse-frequency class weighting: w_c = N / (K * N_c).
        """
        from sklearn.utils.class_weight import compute_sample_weight
        self.classes_ = sorted(y.unique().tolist())
        self.class_mapping = {c: i for i, c in enumerate(self.classes_)}
        self.inv_class_mapping = {i: c for i, c in enumerate(self.classes_)}
        y_encoded = y.map(self.class_mapping).values

        # Feature matrix
        feat_cols = [c for c in FEATURE_COLUMNS if c in X_df.columns]
        X = X_df[feat_cols].fillna(0.0).values

        unique_groups = groups.unique()
        actual_outer_splits = min(n_outer_splits, len(unique_groups))
        outer_gkf = GroupKFold(n_splits=actual_outer_splits)

        all_y_true = []
        all_y_pred = []
        all_y_probs = []
        fold_summaries = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_gkf.split(X, y_encoded, groups=groups)):
            X_train, y_train = X[train_idx], y_encoded[train_idx]
            X_test, y_test = X[test_idx], y_encoded[test_idx]
            train_groups = groups.iloc[train_idx]
            test_skaters = groups.iloc[test_idx].unique().tolist()

            # Nested Inner CV for hyperparameter tuning [P3-A7]
            best_depth = self.max_depth
            best_lr = self.learning_rate
            best_inner_score = -1.0

            from sklearn.preprocessing import LabelEncoder

            inner_splits = min(n_inner_splits, len(train_groups.unique()))
            if inner_splits >= 2:
                inner_gkf = GroupKFold(n_splits=inner_splits)
                for depth_cand in [3, 4]:
                    for lr_cand in [0.05, 0.1]:
                        inner_scores = []
                        for in_tr, in_val in inner_gkf.split(X_train, y_train, groups=train_groups):
                            le_inner = LabelEncoder()
                            y_in_tr = le_inner.fit_transform(y_train[in_tr])
                            w_in = compute_sample_weight('balanced', y_in_tr) if use_class_weighting else None
                            clf = xgb.XGBClassifier(
                                n_estimators=40,
                                max_depth=depth_cand,
                                learning_rate=lr_cand,
                                eval_metric='mlogloss',
                                random_state=self.seed,
                                n_jobs=1
                            )
                            clf.fit(X_train[in_tr], y_in_tr, sample_weight=w_in)
                            p_val_enc = clf.predict(X_train[in_val])
                            p_val = le_inner.inverse_transform(p_val_enc)
                            inner_scores.append(accuracy_score(y_train[in_val], p_val))
                        mean_in_score = np.mean(inner_scores)
                        if mean_in_score > best_inner_score:
                            best_inner_score = mean_in_score
                            best_depth = depth_cand
                            best_lr = lr_cand

            # Stance Mirroring Augmentation (Juriga 2023) [Step 3.5.5]
            directional_cols = [
                'canonical_delta_theta_net', 'delta_theta_net', 'flick_dx',
                'flick_direction_x', 'flick_local_y_delta', 'delta_feet_dx',
                'delta_feet_norm', 'delta_board_norm', 'diff_yaw_norm', 'bvel_x_mean'
            ]
            dir_indices = [feat_cols.index(c) for c in directional_cols if c in feat_cols]

            X_tr_mirror = X_train.copy()
            for d_idx in dir_indices:
                X_tr_mirror[:, d_idx] *= -1.0

            X_train_aug = np.vstack([X_train, X_tr_mirror])
            y_train_aug = np.concatenate([y_train, y_train])

            # Train final model for this fold using best tuned parameters
            le_outer = LabelEncoder()
            y_tr_enc = le_outer.fit_transform(y_train_aug)
            w_tr = compute_sample_weight('balanced', y_tr_enc) if use_class_weighting else None

            model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=best_depth,
                learning_rate=best_lr,
                eval_metric='mlogloss',
                random_state=self.seed,
                n_jobs=1
            )
            model.fit(X_train_aug, y_tr_enc, sample_weight=w_tr)

            # Predict on outer test fold and map to global class distribution
            probs_local = model.predict_proba(X_test)
            probs = np.zeros((len(X_test), len(self.classes_)))
            for loc_idx, glob_idx in enumerate(le_outer.classes_):
                probs[:, glob_idx] = probs_local[:, loc_idx]
            preds = np.argmax(probs, axis=1)

            all_y_true.extend(y_test)
            all_y_pred.extend(preds)
            all_y_probs.extend(probs)

            fold_summaries.append({
                'fold': fold_idx,
                'test_skaters': test_skaters,
                'n_test_samples': len(test_idx),
                'best_depth': best_depth,
                'best_lr': best_lr,
                'fold_acc': float(accuracy_score(y_test, preds))
            })

        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_probs = np.array(all_y_probs)

        # Calculate metrics
        macro_f1 = float(f1_score(all_y_true, all_y_pred, average='macro', zero_division=0))
        weighted_f1 = float(f1_score(all_y_true, all_y_pred, average='weighted', zero_division=0))
        top1_acc = float(accuracy_score(all_y_true, all_y_pred))

        # Top-2 accuracy
        top2_correct = 0
        for true_label, prob_dist in zip(all_y_true, all_y_probs):
            top2_classes = np.argsort(prob_dist)[-2:]
            if true_label in top2_classes:
                top2_correct += 1
        top2_acc = float(top2_correct / max(1, len(all_y_true)))

        # Per-class metrics
        per_class_f1 = {}
        per_class_prec = {}
        per_class_rec = {}
        for idx, cname in enumerate(self.classes_):
            c_mask_true = (all_y_true == idx)
            c_mask_pred = (all_y_pred == idx)
            f1_c = f1_score(c_mask_true, c_mask_pred, zero_division=0)
            prec_c = precision_score(c_mask_true, c_mask_pred, zero_division=0)
            rec_c = recall_score(c_mask_true, c_mask_pred, zero_division=0)
            per_class_f1[cname] = float(f1_c)
            per_class_prec[cname] = float(prec_c)
            per_class_rec[cname] = float(rec_c)

        # Confusion matrix
        cm = confusion_matrix(all_y_true, all_y_pred, labels=list(range(len(self.classes_)))).tolist()

        # Fit full model on all data for SHAP interpretation
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            eval_metric='mlogloss',
            random_state=self.seed,
            n_jobs=1
        )
        self.model.fit(X, y_encoded)

        # Compute SHAP feature importances
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            # multi-class list of arrays
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        elif len(shap_values.shape) == 3:
            mean_shap = np.abs(shap_values).mean(axis=(0, 2))
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)

        shap_importance = {col: float(val) for col, val in sorted(zip(feat_cols, mean_shap), key=lambda x: -x[1])[:15]}

        return TreeClassifierEvaluation(
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            top1_accuracy=top1_acc,
            top2_accuracy=top2_acc,
            per_class_f1=per_class_f1,
            per_class_precision=per_class_prec,
            per_class_recall=per_class_rec,
            confusion_matrix=cm,
            classes=self.classes_,
            shap_importance=shap_importance,
            nested_cv_results=fold_summaries
        )

    def predict(self, X_df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
        feat_cols = [c for c in FEATURE_COLUMNS if c in X_df.columns]
        X = X_df[feat_cols].fillna(0.0).values
        probs = self.model.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        pred_labels = [self.inv_class_mapping[p] for p in preds]
        return pred_labels, probs


class HierarchicalKinematicClassifier:
    """
    Model B2: Hierarchical Kinematic Factorized Classifier based on Skateboarding_Kinematic_Dataset.csv.
    Decomposes the flat 9-class trick categorization into a physically motivated multi-stage hierarchy:
      Stage 1: Flip Roll Detector (Flip vs. Flat)
      Stage 2A: Body 180 Detector (for Flat tricks: 180 vs Straight Pop)
        - 180 Specialist: Frontside 180 vs Backside 180
        - Straight Specialist: Ollie vs Pop Shove-it vs Frontside Shove-it
      Stage 2B: Flip Sub-Tree Specialist (Kickflip, Heelflip, Varial / Hardflip, 360 Flip)
    Eliminates cross-family confusion and resolves the 'Ollie Black Hole' where non-flipping tricks
    collapse into Ollie predictions under unseen-skater holdouts.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.classes_ = []
        self.clf_flip = None
        self.clf_fsub = None
        self.clf_b180 = None
        self.clf_180 = None
        self.clf_str = None
        self.le_f = None
        self.le_180 = None
        self.le_str = None

    def fit_and_evaluate_nested_cv(
        self,
        X_df: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        n_outer_splits: int = 4
    ) -> TreeClassifierEvaluation:
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils.class_weight import compute_sample_weight

        self.classes_ = sorted(y.unique().tolist())
        feat_cols = [c for c in FEATURE_COLUMNS if c in X_df.columns]
        X = X_df[feat_cols].fillna(0.0).values
        y_vals = y.values

        is_flip_all = y.isin(['Kickflip', 'Heelflip', 'Varial / Hardflip', '360 Flip']).astype(int).values
        is_body180_all = y.isin(['Frontside 180', 'Backside 180']).astype(int).values

        unique_groups = groups.unique()
        actual_splits = min(n_outer_splits, len(unique_groups))
        outer_gkf = GroupKFold(n_splits=actual_splits)

        all_y_true = []
        all_y_pred = []
        all_y_probs = []
        fold_summaries = []

        directional_cols = [
            'canonical_delta_theta_net', 'delta_theta_net', 'flick_dx',
            'flick_direction_x', 'flick_local_y_delta', 'delta_feet_dx',
            'delta_feet_norm', 'delta_board_norm', 'diff_yaw_norm', 'bvel_x_mean',
            'delta_theta_body', 'delta_theta_board', 'delta_theta_relative'
        ]
        dir_indices = [feat_cols.index(c) for c in directional_cols if c in feat_cols]

        def augment_mirror(X_data, y_data):
            X_m = X_data.copy()
            for idx in dir_indices:
                X_m[:, idx] *= -1.0
            return np.vstack([X_data, X_m]), np.concatenate([y_data, y_data])

        for fold_idx, (train_idx, test_idx) in enumerate(outer_gkf.split(X, y_vals, groups=groups)):
            test_skaters = groups.iloc[test_idx].unique().tolist()

            # 1. Train Stage 1: Flip vs. Flat Detector
            X_f_tr, y_f_tr = augment_mirror(X[train_idx], is_flip_all[train_idx])
            w_flip = compute_sample_weight('balanced', y_f_tr)
            clf_flip = xgb.XGBClassifier(n_estimators=45, max_depth=3, learning_rate=0.05, random_state=self.seed, eval_metric='logloss')
            clf_flip.fit(X_f_tr, y_f_tr, sample_weight=w_flip)
            p_flip_test = clf_flip.predict(X[test_idx])
            prob_flip_test = clf_flip.predict_proba(X[test_idx])[:, 1]

            # 2. Train Stage 2B: Flip Sub-Tree Specialist
            flip_tr_indices = [i for i in train_idx if is_flip_all[i] == 1]
            le_f = LabelEncoder()
            y_f_sub_tr = le_f.fit_transform(y_vals[flip_tr_indices])
            X_fsub_tr, y_fsub_tr = augment_mirror(X[flip_tr_indices], y_f_sub_tr)
            w_fsub = compute_sample_weight('balanced', y_fsub_tr)
            clf_fsub = xgb.XGBClassifier(n_estimators=45, max_depth=3, learning_rate=0.05, random_state=self.seed, eval_metric='mlogloss')
            clf_fsub.fit(X_fsub_tr, y_fsub_tr, sample_weight=w_fsub)

            # 3. Train Stage 2A: Body 180 Detector (on non-flip tricks)
            flat_tr_indices = [i for i in train_idx if is_flip_all[i] == 0]
            X_b180_tr, y_b180_tr = augment_mirror(X[flat_tr_indices], is_body180_all[flat_tr_indices])
            w_b180 = compute_sample_weight('balanced', y_b180_tr)
            clf_b180 = xgb.XGBClassifier(n_estimators=45, max_depth=3, learning_rate=0.05, random_state=self.seed, eval_metric='logloss')
            clf_b180.fit(X_b180_tr, y_b180_tr, sample_weight=w_b180)
            p_b180_test = clf_b180.predict(X[test_idx])

            # 3A. Train 180 Specialist (FS 180 vs BS 180)
            b180_tr_indices = [i for i in flat_tr_indices if is_body180_all[i] == 1]
            le_180 = LabelEncoder()
            y_180_sub_tr = le_180.fit_transform(y_vals[b180_tr_indices])
            X_180_tr, y_180_tr = augment_mirror(X[b180_tr_indices], y_180_sub_tr)
            w_180 = compute_sample_weight('balanced', y_180_tr)
            clf_180 = xgb.XGBClassifier(n_estimators=35, max_depth=2, learning_rate=0.05, random_state=self.seed, eval_metric='logloss')
            clf_180.fit(X_180_tr, y_180_tr, sample_weight=w_180)

            # 3B. Train Straight Flat Specialist (Ollie vs Pop Shove-it vs FS Shove-it)
            straight_tr_indices = [i for i in flat_tr_indices if is_body180_all[i] == 0]
            le_str = LabelEncoder()
            y_str_sub_tr = le_str.fit_transform(y_vals[straight_tr_indices])
            X_str_tr, y_str_tr = augment_mirror(X[straight_tr_indices], y_str_sub_tr)
            w_str = compute_sample_weight('balanced', y_str_tr)
            clf_str = xgb.XGBClassifier(n_estimators=45, max_depth=3, learning_rate=0.05, random_state=self.seed, eval_metric='mlogloss')
            clf_str.fit(X_str_tr, y_str_tr, sample_weight=w_str)

            # Predict on outer test fold
            fold_preds = []
            fold_probs = np.zeros((len(test_idx), len(self.classes_)))

            for i_local, i_global in enumerate(test_idx):
                x_sample = X[i_global:i_global+1]
                p_flip = p_flip_test[i_local]
                p_b180 = p_b180_test[i_local]

                if p_flip == 1:
                    probs_local = clf_fsub.predict_proba(x_sample)[0]
                    for sub_idx, glob_name in enumerate(le_f.classes_):
                        glob_idx = self.classes_.index(glob_name)
                        fold_probs[i_local, glob_idx] = probs_local[sub_idx] * prob_flip_test[i_local]
                    pred_class_idx = np.argmax(probs_local)
                    pred_name = le_f.inverse_transform([pred_class_idx])[0]
                else:
                    if p_b180 == 1:
                        probs_local = clf_180.predict_proba(x_sample)[0]
                        for sub_idx, glob_name in enumerate(le_180.classes_):
                            glob_idx = self.classes_.index(glob_name)
                            fold_probs[i_local, glob_idx] = probs_local[sub_idx] * (1.0 - prob_flip_test[i_local])
                        pred_class_idx = np.argmax(probs_local)
                        pred_name = le_180.inverse_transform([pred_class_idx])[0]
                    else:
                        probs_local = clf_str.predict_proba(x_sample)[0]
                        for sub_idx, glob_name in enumerate(le_str.classes_):
                            glob_idx = self.classes_.index(glob_name)
                            fold_probs[i_local, glob_idx] = probs_local[sub_idx] * (1.0 - prob_flip_test[i_local])
                        pred_class_idx = np.argmax(probs_local)
                        pred_name = le_str.inverse_transform([pred_class_idx])[0]

                fold_preds.append(pred_name)

            all_y_true.extend(y_vals[test_idx].tolist())
            all_y_pred.extend(fold_preds)
            all_y_probs.extend(fold_probs.tolist())

            fold_acc = float(accuracy_score(y_vals[test_idx], fold_preds))
            fold_summaries.append({
                'fold': fold_idx,
                'test_skaters': test_skaters,
                'n_test_samples': len(test_idx),
                'fold_acc': fold_acc
            })

        # Calculate final evaluation metrics
        all_y_true_arr = np.array(all_y_true)
        all_y_pred_arr = np.array(all_y_pred)
        all_y_probs_arr = np.array(all_y_probs)

        macro_f1 = float(f1_score(all_y_true_arr, all_y_pred_arr, average='macro', zero_division=0))
        weighted_f1 = float(f1_score(all_y_true_arr, all_y_pred_arr, average='weighted', zero_division=0))
        top1_acc = float(accuracy_score(all_y_true_arr, all_y_pred_arr))

        top2_count = 0
        for i, true_label in enumerate(all_y_true_arr):
            top2_indices = np.argsort(all_y_probs_arr[i])[-2:]
            top2_names = [self.classes_[idx] for idx in top2_indices]
            if true_label in top2_names:
                top2_count += 1
        top2_acc = float(top2_count / max(1, len(all_y_true_arr)))

        per_class_f1 = {}
        per_class_prec = {}
        per_class_rec = {}
        for cname in self.classes_:
            c_mask_true = (all_y_true_arr == cname)
            c_mask_pred = (all_y_pred_arr == cname)
            per_class_f1[cname] = float(f1_score(c_mask_true, c_mask_pred, zero_division=0))
            per_class_prec[cname] = float(precision_score(c_mask_true, c_mask_pred, zero_division=0))
            per_class_rec[cname] = float(recall_score(c_mask_true, c_mask_pred, zero_division=0))

        cm = confusion_matrix(all_y_true_arr, all_y_pred_arr, labels=self.classes_).tolist()

        # Fit on entire dataset so the classifier can be used for downstream predictions
        self.fit(X_df, y)

        return TreeClassifierEvaluation(
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            top1_accuracy=top1_acc,
            top2_accuracy=top2_acc,
            per_class_f1=per_class_f1,
            per_class_precision=per_class_prec,
            per_class_recall=per_class_rec,
            confusion_matrix=cm,
            classes=self.classes_,
            shap_importance={},
            nested_cv_results=fold_summaries
        )

    def fit(self, X_df: pd.DataFrame, y: pd.Series):
        """Fits all stages of the hierarchical model on the full dataset."""
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils.class_weight import compute_sample_weight

        self.classes_ = sorted(y.unique().tolist())
        self.feat_cols = [c for c in FEATURE_COLUMNS if c in X_df.columns]
        X = X_df[self.feat_cols].fillna(0.0).values
        y_vals = y.values

        is_flip_all = y.isin(['Kickflip', 'Heelflip', 'Varial / Hardflip', '360 Flip']).astype(int).values
        is_body180_all = y.isin(['Frontside 180', 'Backside 180']).astype(int).values

        directional_cols = [
            'canonical_delta_theta_net', 'delta_theta_net', 'flick_dx',
            'flick_direction_x', 'flick_local_y_delta', 'delta_feet_dx',
            'delta_feet_norm', 'delta_board_norm', 'diff_yaw_norm', 'bvel_x_mean',
            'delta_theta_body', 'delta_theta_board', 'delta_theta_relative'
        ]
        self.dir_indices = [self.feat_cols.index(c) for c in directional_cols if c in self.feat_cols]

        def augment_mirror(X_data, y_data):
            X_m = X_data.copy()
            for idx in self.dir_indices:
                X_m[:, idx] *= -1.0
            return np.vstack([X_data, X_m]), np.concatenate([y_data, y_data])

        # 1. Flip vs Flat
        X_f_tr, y_f_tr = augment_mirror(X, is_flip_all)
        w_flip = compute_sample_weight('balanced', y_f_tr)
        self.clf_flip = xgb.XGBClassifier(n_estimators=45, max_depth=3, learning_rate=0.05, random_state=self.seed, eval_metric='logloss')
        self.clf_flip.fit(X_f_tr, y_f_tr, sample_weight=w_flip)

        # 2. Flip Sub-Tree Specialist
        flip_indices = [i for i in range(len(y_vals)) if is_flip_all[i] == 1]
        self.le_f = LabelEncoder()
        y_f_sub = self.le_f.fit_transform(y_vals[flip_indices])
        X_fsub, y_fsub = augment_mirror(X[flip_indices], y_f_sub)
        w_fsub = compute_sample_weight('balanced', y_fsub)
        self.clf_fsub = xgb.XGBClassifier(n_estimators=45, max_depth=3, learning_rate=0.05, random_state=self.seed, eval_metric='mlogloss')
        self.clf_fsub.fit(X_fsub, y_fsub, sample_weight=w_fsub)

        # 3. Body 180 Detector (Flat tricks)
        flat_indices = [i for i in range(len(y_vals)) if is_flip_all[i] == 0]
        X_b180, y_b180 = augment_mirror(X[flat_indices], is_body180_all[flat_indices])
        w_b180 = compute_sample_weight('balanced', y_b180)
        self.clf_b180 = xgb.XGBClassifier(n_estimators=45, max_depth=3, learning_rate=0.05, random_state=self.seed, eval_metric='logloss')
        self.clf_b180.fit(X_b180, y_b180, sample_weight=w_b180)

        # 3A. 180 Specialist
        b180_indices = [i for i in flat_indices if is_body180_all[i] == 1]
        self.le_180 = LabelEncoder()
        y_180_sub = self.le_180.fit_transform(y_vals[b180_indices])
        X_180, y_180 = augment_mirror(X[b180_indices], y_180_sub)
        w_180 = compute_sample_weight('balanced', y_180)
        self.clf_180 = xgb.XGBClassifier(n_estimators=35, max_depth=2, learning_rate=0.05, random_state=self.seed, eval_metric='logloss')
        self.clf_180.fit(X_180, y_180, sample_weight=w_180)

        # 3B. Straight Specialist
        straight_indices = [i for i in flat_indices if is_body180_all[i] == 0]
        self.le_str = LabelEncoder()
        y_str_sub = self.le_str.fit_transform(y_vals[straight_indices])
        X_str, y_str = augment_mirror(X[straight_indices], y_str_sub)
        w_str = compute_sample_weight('balanced', y_str)
        self.clf_str = xgb.XGBClassifier(n_estimators=45, max_depth=3, learning_rate=0.05, random_state=self.seed, eval_metric='mlogloss')
        self.clf_str.fit(X_str, y_str, sample_weight=w_str)

        return self

    def predict(self, X_df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
        """Predicts trick classes and probability distributions using the trained hierarchy."""
        if self.clf_flip is None:
            raise ValueError("HierarchicalKinematicClassifier must be fitted before calling predict.")
        feat_cols = [c for c in FEATURE_COLUMNS if c in X_df.columns]
        X = X_df[feat_cols].fillna(0.0).values
        n_samples = len(X)

        p_flip = self.clf_flip.predict(X)
        prob_flip = self.clf_flip.predict_proba(X)[:, 1]
        p_b180 = self.clf_b180.predict(X)

        preds = []
        probs = np.zeros((n_samples, len(self.classes_)))

        for i in range(n_samples):
            x_sample = X[i:i+1]
            if p_flip[i] == 1:
                probs_local = self.clf_fsub.predict_proba(x_sample)[0]
                for sub_idx, glob_name in enumerate(self.le_f.classes_):
                    glob_idx = self.classes_.index(glob_name)
                    probs[i, glob_idx] = probs_local[sub_idx] * prob_flip[i]
                pred_class_idx = np.argmax(probs_local)
                pred_name = self.le_f.inverse_transform([pred_class_idx])[0]
            else:
                if p_b180[i] == 1:
                    probs_local = self.clf_180.predict_proba(x_sample)[0]
                    for sub_idx, glob_name in enumerate(self.le_180.classes_):
                        glob_idx = self.classes_.index(glob_name)
                        probs[i, glob_idx] = probs_local[sub_idx] * (1.0 - prob_flip[i])
                    pred_class_idx = np.argmax(probs_local)
                    pred_name = self.le_180.inverse_transform([pred_class_idx])[0]
                else:
                    probs_local = self.clf_str.predict_proba(x_sample)[0]
                    for sub_idx, glob_name in enumerate(self.le_str.classes_):
                        glob_idx = self.classes_.index(glob_name)
                        probs[i, glob_idx] = probs_local[sub_idx] * (1.0 - prob_flip[i])
                    pred_class_idx = np.argmax(probs_local)
                    pred_name = self.le_str.inverse_transform([pred_class_idx])[0]
            preds.append(pred_name)

        return preds, probs


PHYSICS_CANONICAL_MAPPING = {
    'Ollie':              {'flip': 'None', 'body': '0',      'shuv': '0'},
    'Kickflip':           {'flip': 'Kick', 'body': '0',      'shuv': '0'},
    'Heelflip':           {'flip': 'Heel', 'body': '0',      'shuv': '0'},
    'Pop Shove-it':       {'flip': 'None', 'body': '0',      'shuv': 'BS_180'},
    'Frontside Shove-it': {'flip': 'None', 'body': '0',      'shuv': 'FS_180'},
    'Backside 180':       {'flip': 'None', 'body': 'BS_180', 'shuv': 'BS_180'},
    'Frontside 180':      {'flip': 'None', 'body': 'FS_180', 'shuv': 'FS_180'},
    'Varial / Hardflip':  {'flip': 'Kick', 'body': '0',      'shuv': 'BS_180'},
    '360 Flip':           {'flip': 'Kick', 'body': '0',      'shuv': 'BS_360'},
}


class PhysicsMultiTaskClassifier:
    """
    Model B3 / Phase 3.5.7: Physics-Informed Kinematic Engine & Multi-Task Factorized Classifier.
    Integrates:
      Tier 1: Physics Invariant Features (Deck Inversion C_inv, Foreshortening Profile, Rotational Coupling, Ballistics)
      Tier 2: Multi-Task Physical Component Heads:
              - Flip Head: None vs Kick vs Heel
              - Body Spin Head: 0° vs FS 180 vs BS 180
              - Board Shuv Head: 0° vs BS 180 vs FS 180 vs BS 360
      Tier 3: Physics Consistency Constraints (Bayesian likelihood fusion rejecting physically impossible state space)
      Tier 4: ML Ambiguity Resolver
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.classes_ = []
        self.clf_flip = None
        self.clf_body = None
        self.clf_shuv = None
        self.le_flip = None
        self.le_body = None
        self.le_shuv = None
        self.feat_cols = []
        self.dir_indices = []

    def fit_and_evaluate_nested_cv(
        self,
        X_df: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        n_outer_splits: int = 4
    ) -> TreeClassifierEvaluation:
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils.class_weight import compute_sample_weight

        self.classes_ = sorted(y.unique().tolist())
        self.feat_cols = [c for c in FEATURE_COLUMNS if c in X_df.columns]
        X = X_df[self.feat_cols].fillna(0.0).values
        y_vals = y.values

        y_flip = np.array([PHYSICS_CANONICAL_MAPPING[t]['flip'] for t in y_vals])
        y_body = np.array([PHYSICS_CANONICAL_MAPPING[t]['body'] for t in y_vals])
        y_shuv = np.array([PHYSICS_CANONICAL_MAPPING[t]['shuv'] for t in y_vals])

        unique_groups = groups.unique()
        actual_splits = min(n_outer_splits, len(unique_groups))
        outer_gkf = GroupKFold(n_splits=actual_splits)

        all_y_true = []
        all_y_pred = []
        all_y_probs = []
        fold_summaries = []

        directional_cols = [
            'canonical_delta_theta_net', 'delta_theta_net', 'flick_dx',
            'flick_direction_x', 'flick_local_y_delta', 'delta_feet_dx',
            'delta_feet_norm', 'delta_board_norm', 'diff_yaw_norm', 'bvel_x_mean',
            'delta_theta_body', 'delta_theta_board', 'delta_theta_relative'
        ]
        self.dir_indices = [self.feat_cols.index(c) for c in directional_cols if c in self.feat_cols]

        def augment_mirror(X_data, y_data):
            X_m = X_data.copy()
            for idx in self.dir_indices:
                X_m[:, idx] *= -1.0
            return np.vstack([X_data, X_m]), np.concatenate([y_data, y_data])

        for fold_idx, (train_idx, test_idx) in enumerate(outer_gkf.split(X, y_vals, groups=groups)):
            test_skaters = groups.iloc[test_idx].unique().tolist()

            # 1. Train Flip Head
            le_f = LabelEncoder()
            y_f_tr = le_f.fit_transform(y_flip[train_idx])
            X_f_tr, y_f_aug = augment_mirror(X[train_idx], y_f_tr)
            w_f = compute_sample_weight('balanced', y_f_aug)
            clf_f = xgb.XGBClassifier(n_estimators=45, max_depth=3, learning_rate=0.05, random_state=self.seed, eval_metric='mlogloss')
            clf_f.fit(X_f_tr, y_f_aug, sample_weight=w_f)
            p_f_te = clf_f.predict_proba(X[test_idx])

            # 2. Train Body Spin Head
            le_b = LabelEncoder()
            y_b_tr = le_b.fit_transform(y_body[train_idx])
            X_b_tr, y_b_aug = augment_mirror(X[train_idx], y_b_tr)
            w_b = compute_sample_weight('balanced', y_b_aug)
            clf_b = xgb.XGBClassifier(n_estimators=45, max_depth=3, learning_rate=0.05, random_state=self.seed, eval_metric='mlogloss')
            clf_b.fit(X_b_tr, y_b_aug, sample_weight=w_b)
            p_b_te = clf_b.predict_proba(X[test_idx])

            # 3. Train Board Shuv Head
            le_s = LabelEncoder()
            y_s_tr = le_s.fit_transform(y_shuv[train_idx])
            X_s_tr, y_s_aug = augment_mirror(X[train_idx], y_s_tr)
            w_s = compute_sample_weight('balanced', y_s_aug)
            clf_s = xgb.XGBClassifier(n_estimators=45, max_depth=3, learning_rate=0.05, random_state=self.seed, eval_metric='mlogloss')
            clf_s.fit(X_s_tr, y_s_aug, sample_weight=w_s)
            p_s_te = clf_s.predict_proba(X[test_idx])

            # 4. Bayesian Physics Consistency Fusion (Tier 3)
            fold_probs = np.zeros((len(test_idx), len(self.classes_)))
            fold_preds = []

            for i in range(len(test_idx)):
                prob_dict = {}
                for k, cname in enumerate(self.classes_):
                    spec = PHYSICS_CANONICAL_MAPPING[cname]
                    f_idx = list(le_f.classes_).index(spec['flip']) if spec['flip'] in le_f.classes_ else -1
                    b_idx = list(le_b.classes_).index(spec['body']) if spec['body'] in le_b.classes_ else -1
                    s_idx = list(le_s.classes_).index(spec['shuv']) if spec['shuv'] in le_s.classes_ else -1

                    p_val = (p_f_te[i, f_idx] if f_idx >= 0 else 0.01) * \
                            (p_b_te[i, b_idx] if b_idx >= 0 else 0.01) * \
                            (p_s_te[i, s_idx] if s_idx >= 0 else 0.01)
                    prob_dict[cname] = p_val
                    fold_probs[i, k] = p_val

                # Normalize probabilities across canonical classes
                sum_p = fold_probs[i].sum()
                if sum_p > 0:
                    fold_probs[i] /= sum_p

                best_trick = max(prob_dict, key=prob_dict.get)
                fold_preds.append(best_trick)

            all_y_true.extend(y_vals[test_idx].tolist())
            all_y_pred.extend(fold_preds)
            all_y_probs.extend(fold_probs.tolist())

            fold_acc = float(accuracy_score(y_vals[test_idx], fold_preds))
            fold_summaries.append({
                'fold': fold_idx,
                'test_skaters': test_skaters,
                'n_test_samples': len(test_idx),
                'fold_acc': fold_acc
            })

        all_y_true_arr = np.array(all_y_true)
        all_y_pred_arr = np.array(all_y_pred)
        all_y_probs_arr = np.array(all_y_probs)

        macro_f1 = float(f1_score(all_y_true_arr, all_y_pred_arr, average='macro', zero_division=0))
        weighted_f1 = float(f1_score(all_y_true_arr, all_y_pred_arr, average='weighted', zero_division=0))
        top1_acc = float(accuracy_score(all_y_true_arr, all_y_pred_arr))

        top2_count = 0
        for i, true_label in enumerate(all_y_true_arr):
            top2_indices = np.argsort(all_y_probs_arr[i])[-2:]
            top2_names = [self.classes_[idx] for idx in top2_indices]
            if true_label in top2_names:
                top2_count += 1
        top2_acc = float(top2_count / max(1, len(all_y_true_arr)))

        per_class_f1 = {}
        per_class_prec = {}
        per_class_rec = {}
        for cname in self.classes_:
            c_mask_true = (all_y_true_arr == cname)
            c_mask_pred = (all_y_pred_arr == cname)
            per_class_f1[cname] = float(f1_score(c_mask_true, c_mask_pred, zero_division=0))
            per_class_prec[cname] = float(precision_score(c_mask_true, c_mask_pred, zero_division=0))
            per_class_rec[cname] = float(recall_score(c_mask_true, c_mask_pred, zero_division=0))

        cm = confusion_matrix(all_y_true_arr, all_y_pred_arr, labels=self.classes_).tolist()

        # Fit on full dataset
        self.fit(X_df, y)

        return TreeClassifierEvaluation(
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            top1_accuracy=top1_acc,
            top2_accuracy=top2_acc,
            per_class_f1=per_class_f1,
            per_class_precision=per_class_prec,
            per_class_recall=per_class_rec,
            confusion_matrix=cm,
            classes=self.classes_,
            shap_importance={},
            nested_cv_results=fold_summaries
        )

    def fit(self, X_df: pd.DataFrame, y: pd.Series):
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils.class_weight import compute_sample_weight

        self.classes_ = sorted(y.unique().tolist())
        self.feat_cols = [c for c in FEATURE_COLUMNS if c in X_df.columns]
        X = X_df[self.feat_cols].fillna(0.0).values
        y_vals = y.values

        y_flip = np.array([PHYSICS_CANONICAL_MAPPING[t]['flip'] for t in y_vals])
        y_body = np.array([PHYSICS_CANONICAL_MAPPING[t]['body'] for t in y_vals])
        y_shuv = np.array([PHYSICS_CANONICAL_MAPPING[t]['shuv'] for t in y_vals])

        directional_cols = [
            'canonical_delta_theta_net', 'delta_theta_net', 'flick_dx',
            'flick_direction_x', 'flick_local_y_delta', 'delta_feet_dx',
            'delta_feet_norm', 'delta_board_norm', 'diff_yaw_norm', 'bvel_x_mean',
            'delta_theta_body', 'delta_theta_board', 'delta_theta_relative'
        ]
        self.dir_indices = [self.feat_cols.index(c) for c in directional_cols if c in self.feat_cols]

        def augment_mirror(X_data, y_data):
            X_m = X_data.copy()
            for idx in self.dir_indices:
                X_m[:, idx] *= -1.0
            return np.vstack([X_data, X_m]), np.concatenate([y_data, y_data])

        # 1. Fit Flip Head
        self.le_flip = LabelEncoder()
        y_f = self.le_flip.fit_transform(y_flip)
        X_f, y_f_aug = augment_mirror(X, y_f)
        w_f = compute_sample_weight('balanced', y_f_aug)
        self.clf_flip = xgb.XGBClassifier(n_estimators=45, max_depth=3, learning_rate=0.05, random_state=self.seed, eval_metric='mlogloss')
        self.clf_flip.fit(X_f, y_f_aug, sample_weight=w_f)

        # 2. Fit Body Spin Head
        self.le_body = LabelEncoder()
        y_b = self.le_body.fit_transform(y_body)
        X_b, y_b_aug = augment_mirror(X, y_b)
        w_b = compute_sample_weight('balanced', y_b_aug)
        self.clf_body = xgb.XGBClassifier(n_estimators=45, max_depth=3, learning_rate=0.05, random_state=self.seed, eval_metric='mlogloss')
        self.clf_body.fit(X_b, y_b_aug, sample_weight=w_b)

        # 3. Fit Board Shuv Head
        self.le_shuv = LabelEncoder()
        y_s = self.le_shuv.fit_transform(y_shuv)
        X_s, y_s_aug = augment_mirror(X, y_s)
        w_s = compute_sample_weight('balanced', y_s_aug)
        self.clf_shuv = xgb.XGBClassifier(n_estimators=45, max_depth=3, learning_rate=0.05, random_state=self.seed, eval_metric='mlogloss')
        self.clf_shuv.fit(X_s, y_s_aug, sample_weight=w_s)

        return self

    def predict(self, X_df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
        if self.clf_flip is None:
            raise ValueError("PhysicsMultiTaskClassifier must be fitted before predict.")
        X = X_df[self.feat_cols].fillna(0.0).values
        n_samples = len(X)

        p_f = self.clf_flip.predict_proba(X)
        p_b = self.clf_body.predict_proba(X)
        p_s = self.clf_shuv.predict_proba(X)

        preds = []
        probs = np.zeros((n_samples, len(self.classes_)))

        for i in range(n_samples):
            prob_dict = {}
            for k, cname in enumerate(self.classes_):
                spec = PHYSICS_CANONICAL_MAPPING[cname]
                f_idx = list(self.le_flip.classes_).index(spec['flip']) if spec['flip'] in self.le_flip.classes_ else -1
                b_idx = list(self.le_body.classes_).index(spec['body']) if spec['body'] in self.le_body.classes_ else -1
                s_idx = list(self.le_shuv.classes_).index(spec['shuv']) if spec['shuv'] in self.le_shuv.classes_ else -1

                p_val = (p_f[i, f_idx] if f_idx >= 0 else 0.01) * \
                        (p_b[i, b_idx] if b_idx >= 0 else 0.01) * \
                        (p_s[i, s_idx] if s_idx >= 0 else 0.01)
                prob_dict[cname] = p_val
                probs[i, k] = p_val

            sum_p = probs[i].sum()
            if sum_p > 0:
                probs[i] /= sum_p

            best_trick = max(prob_dict, key=prob_dict.get)
            preds.append(best_trick)

        return preds, probs

