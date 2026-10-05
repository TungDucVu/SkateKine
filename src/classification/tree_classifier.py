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
    'skater_yaw_swap', 'delta_feet_dx',
    'min_norm_length', 'board_length_std', 'board_yaw_swap',
    'flip_yaw_product',
    'flip_cycle_count', 'min_aspect', 'aspect_std', 'aspect_range',
    'flick_dx', 'flick_dy', 'flick_vel_mag', 'flick_direction_x',
    'bvel_x_mean', 'bvel_x_std', 'bvel_x_max',
    'bvel_y_mean', 'bvel_y_std', 'bvel_y_max',
    'bacc_y_mean', 'bacc_y_std', 'bacc_y_max',
    'jerk_mean', 'jerk_std', 'jerk_max',
    'left_foot_dist', 'right_foot_dist',
    'pop_board_y', 'apex_board_y', 'land_board_y',
    'pop_hip_y', 'apex_hip_y', 'land_hip_y',
    'board_length_pop', 'board_width_pop'
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

            # Train final model for this fold using best tuned parameters
            le_outer = LabelEncoder()
            y_tr_enc = le_outer.fit_transform(y_train)
            w_tr = compute_sample_weight('balanced', y_tr_enc) if use_class_weighting else None

            model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=best_depth,
                learning_rate=best_lr,
                eval_metric='mlogloss',
                random_state=self.seed,
                n_jobs=1
            )
            model.fit(X_train, y_tr_enc, sample_weight=w_tr)

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
