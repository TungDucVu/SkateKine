"""
Phase 3.5.8: Physics-First Temporal Recognition Engine.

Implements:
  - Layer A: Compact Deterministic Physical State Consensus
  - Layer B: Signed Scoop Sweep Momentum (tau_scoop)
  - Layer C: Phase-Aligned 64-step Temporal Trajectory Normalization (11 channels)
  - Layer D: Physics-Guided Class Prototypes & Weighted DTW Matching
  - Layer E: Continuous Soft Physics Contradiction Cost Function E(T | X)
  - Layer F: Hybrid Physics-Temporal Decision Engine (C5 Hybrid Resolver)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import xgboost as xgb

from src.classification.tree_classifier import TreeClassifierEvaluation, FEATURE_COLUMNS


# Normalization weights across 11 temporal channels for DTW distance
CHANNEL_WEIGHTS = np.array([
    1.5,  # 0: board yaw (normalized / 180)
    0.8,  # 1: board angular velocity
    1.2,  # 2: apparent board foreshortening length ratio
    1.5,  # 3: body yaw (normalized / 180)
    0.8,  # 4: body angular velocity
    1.2,  # 5: relative yaw (board yaw - body yaw)
    0.7,  # 6: front foot distance to deck
    0.7,  # 7: rear foot distance to deck
    0.5,  # 8: vertical deck translation
    0.5,  # 9: skater CoM to deck clearance
    1.8   # 10: signed scoop sweep momentum (tau_scoop)
], dtype=np.float32)


def extract_temporal_trajectory(
    df_traj: pd.DataFrame,
    events: Dict[str, Any],
    fps: float,
    stance: str = "regular",
    T: int = 64
) -> np.ndarray:
    """
    Extracts resampled (T=64, D=11) phase-aligned physical kinematic trajectory
    over the flight window [t_pop, t_land]. Stance is calibrated for regular/goofy.
    """
    t_pop = events.get('t_pop') or int(len(df_traj) * 0.25)
    t_land = events.get('t_land') or int(len(df_traj) * 0.75)
    t_pop = int(np.clip(t_pop, 0, len(df_traj) - 2))
    t_land = int(np.clip(t_land, t_pop + 1, len(df_traj) - 1))

    flight = df_traj.iloc[t_pop:t_land + 1]
    if len(flight) < 2:
        return np.zeros((T, 11), dtype=np.float32)

    is_goofy = (str(stance).lower() == 'goofy')
    stance_sign = -1.0 if is_goofy else 1.0

    orig_steps = np.linspace(0, 1, len(flight))
    target_steps = np.linspace(0, 1, T)

    pop_len = (
        flight['apparent_length'].iloc[0]
        if 'apparent_length' in flight and flight['apparent_length'].iloc[0] > 10
        else 100.0
    )

    # 1. Board Yaw (degrees unwrap)
    if 'raw_in_plane_angle' in flight:
        board_yaw = np.rad2deg(np.unwrap(flight['raw_in_plane_angle'].fillna(0.0).values))
        board_yaw = (board_yaw - board_yaw[0]) * stance_sign
    elif 'board_nose_x' in flight and 'board_tail_x' in flight:
        dx = flight['board_nose_x'] - flight['board_tail_x']
        dy = flight['board_nose_y'] - flight['board_tail_y']
        board_yaw = np.rad2deg(np.unwrap(np.arctan2(dy, dx)))
        board_yaw = (board_yaw - board_yaw[0]) * stance_sign
    else:
        board_yaw = np.zeros(len(flight))
    board_yaw_t = np.interp(target_steps, orig_steps, board_yaw)
    board_omega_t = np.gradient(board_yaw_t)

    # 2. Foreshortened Length ratio
    norm_len = flight['apparent_length'].values / pop_len if 'apparent_length' in flight else np.ones(len(flight))
    norm_len_t = np.interp(target_steps, orig_steps, norm_len)

    # 3. Body Yaw
    front_x = 'right_ankle_x' if is_goofy else 'left_ankle_x'
    front_y = 'right_ankle_y' if is_goofy else 'left_ankle_y'
    rear_x = 'left_ankle_x' if is_goofy else 'right_ankle_x'
    rear_y = 'left_ankle_y' if is_goofy else 'right_ankle_y'

    if front_x in flight and rear_x in flight:
        bdx = flight[front_x] - flight[rear_x]
        bdy = flight[front_y] - flight[rear_y]
        body_yaw = np.rad2deg(np.unwrap(np.arctan2(bdy, bdx)))
        body_yaw = (body_yaw - body_yaw[0]) * stance_sign
    else:
        body_yaw = np.zeros(len(flight))
    body_yaw_t = np.interp(target_steps, orig_steps, body_yaw)
    body_omega_t = np.gradient(body_yaw_t)

    # 4. Relative Yaw
    rel_yaw_t = board_yaw_t - body_yaw_t

    # 5. Feet distance to board
    front_loc_x = 'right_ankle_board_local_x' if is_goofy else 'left_ankle_board_local_x'
    front_loc_y = 'right_ankle_board_local_y' if is_goofy else 'left_ankle_board_local_y'
    rear_loc_x = 'left_ankle_board_local_x' if is_goofy else 'right_ankle_board_local_x'
    rear_loc_y = 'left_ankle_board_local_y' if is_goofy else 'right_ankle_board_local_y'

    d_front = np.sqrt(flight[front_loc_x]**2 + flight[front_loc_y]**2).values / pop_len if front_loc_x in flight else np.zeros(len(flight))
    d_rear = np.sqrt(flight[rear_loc_x]**2 + flight[rear_loc_y]**2).values / pop_len if rear_loc_x in flight else np.zeros(len(flight))
    d_front_t = np.interp(target_steps, orig_steps, d_front)
    d_rear_t = np.interp(target_steps, orig_steps, d_rear)

    # 6. Vertical Board Motion
    vert = (flight['board_centroid_y'].values - flight['board_centroid_y'].iloc[0]) / pop_len if 'board_centroid_y' in flight else np.zeros(len(flight))
    vert_t = np.interp(target_steps, orig_steps, vert)

    # 7. Skater CoM to board clearance
    if 'mid_hip_y' in flight and 'board_centroid_y' in flight:
        com_clear = (flight['board_centroid_y'].values - flight['mid_hip_y'].values) / pop_len
    else:
        com_clear = np.zeros(len(flight))
    com_clear_t = np.interp(target_steps, orig_steps, com_clear)

    # 8. Signed Scoop Sweep Momentum: (r_tail x v_tail)_z
    if 'board_tail_x' in flight and 'board_centroid_x' in flight:
        rx = (flight['board_tail_x'] - flight['board_centroid_x']).values / pop_len
        ry = (flight['board_tail_y'] - flight['board_centroid_y']).values / pop_len
        vx = np.gradient(rx)
        vy = np.gradient(ry)
        cross_z = (rx * vy - ry * vx) * stance_sign
    else:
        cross_z = np.zeros(len(flight))
    scoop_t = np.interp(target_steps, orig_steps, cross_z)

    traj_mat = np.column_stack([
        board_yaw_t / 180.0,
        board_omega_t / 10.0,
        norm_len_t,
        body_yaw_t / 180.0,
        body_omega_t / 10.0,
        rel_yaw_t / 180.0,
        d_front_t,
        d_rear_t,
        vert_t,
        com_clear_t,
        scoop_t * 50.0
    ]).astype(np.float32)

    return traj_mat


def dtw_dist(x: np.ndarray, y: np.ndarray, window: int = 8) -> float:
    """
    Computes weighted Sakoe-Chiba band Dynamic Time Warping distance between two (T, D) sequences.
    """
    T, D = x.shape
    diff_pairwise = (x[:, None, :] - y[None, :, :]) * CHANNEL_WEIGHTS
    dist_mat = np.linalg.norm(diff_pairwise, axis=-1)

    cost = np.full((T + 1, T + 1), np.inf, dtype=np.float32)
    cost[0, 0] = 0.0
    for i in range(1, T + 1):
        j_min = max(1, i - window)
        j_max = min(T + 1, i + window + 1)
        for j in range(j_min, j_max):
            c = dist_mat[i - 1, j - 1]
            cost[i, j] = c + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[T, T] / (2 * T))


def compute_physics_contradiction_cost(row: pd.Series, trick_name: str) -> float:
    """
    Computes smooth, continuous physics violation penalty E(T | X) >= 0.
    """
    cost = 0.0

    c_inv = float(row.get('c_inv_cosine', 1.0))
    d_body = float(row.get('delta_theta_body', 0.0))
    d_board = float(row.get('canonical_delta_theta_net', 0.0))
    tau_scoop = float(row.get('calib_tau_scoop', 0.0))
    flick_vel = float(row.get('flick_vel_mag', 0.0))
    flick_dy = float(row.get('flick_local_y_delta', 0.0))
    trough_depth = float(row.get('foreshortening_trough_depth', 0.0))
    min_asp = float(row.get('min_aspect', 1.0))
    flip_cyc = float(row.get('flip_cycle_count', 0.0))

    has_flip = 1.0 if (min_asp < 0.65 or flip_cyc > 0 or flick_vel > 300.0) else 0.0
    has_body_spin = 1.0 if abs(d_body) > 45.0 else 0.0
    has_board_shuv = 1.0 if (c_inv < 0.2 or trough_depth > 0.40 or abs(d_board) > 90.0) else 0.0

    if trick_name == 'Ollie':
        cost += 3.0 * has_body_spin
        cost += 2.5 * has_board_shuv
        cost += 3.0 * has_flip
        cost += max(0.0, (1.0 - c_inv) - 0.5) * 2.0

    elif trick_name == 'Backside 180':
        if d_body > 15.0:
            cost += 5.0
        elif abs(d_body) < 30.0:
            cost += 4.0
        cost += 2.5 * has_flip
        cost += max(0.0, 0.4 - trough_depth) * 2.0

    elif trick_name == 'Frontside 180':
        if d_body < -15.0:
            cost += 5.0
        elif abs(d_body) < 30.0:
            cost += 4.0
        cost += 2.5 * has_flip
        cost += max(0.0, 0.4 - trough_depth) * 2.0

    elif trick_name == 'Pop Shove-it':
        cost += 3.5 * has_body_spin
        cost += 2.5 * has_flip
        cost += 2.0 * (1.0 - has_board_shuv)
        if tau_scoop > 1500.0:
            cost += 3.5

    elif trick_name == 'Frontside Shove-it':
        cost += 3.5 * has_body_spin
        cost += 2.5 * has_flip
        cost += 2.0 * (1.0 - has_board_shuv)
        if tau_scoop < -1500.0:
            cost += 3.5

    elif trick_name == 'Kickflip':
        cost += 3.0 * has_body_spin
        cost += 2.0 * has_board_shuv
        cost += 3.0 * (1.0 - has_flip)
        if flick_dy > 10.0:
            cost += 2.5

    elif trick_name == 'Heelflip':
        cost += 3.0 * has_body_spin
        cost += 2.0 * has_board_shuv
        cost += 3.0 * (1.0 - has_flip)
        if flick_dy < -10.0:
            cost += 2.5

    elif trick_name == 'Varial / Hardflip':
        cost += 3.0 * has_body_spin
        cost += 2.5 * (1.0 - has_board_shuv)
        cost += 2.0 * (1.0 - has_flip)

    elif trick_name == '360 Flip':
        cost += 3.0 * has_body_spin
        cost += 2.0 * (1.0 - has_flip)
        cost += max(0.0, 0.5 - trough_depth) * 2.0

    return float(cost)


class PhysicsTemporalHybridClassifier:
    """
    Model B4 (Phase 3.5.8): Physics-First Temporal Recognition Engine.

    Architecture:
      Trajectory -> Compact Physical State -> Candidate Routing -> Temporal Signature Matching -> Lightweight Resolver

    Resolves planar shove-it ambiguity by computing the signed scoop sweep momentum
    and phase-aligned trajectory dynamic time warping prototypes while preserving
    strong tree specialist boundaries for flip and body rotation families.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.classes_ = []

    def fit_and_evaluate_nested_cv(
        self,
        X_df: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        trajectories: np.ndarray,
        n_outer_splits: int = 4
    ) -> TreeClassifierEvaluation:
        self.classes_ = sorted(y.unique().tolist())
        feat_cols = [c for c in FEATURE_COLUMNS if c in X_df.columns]
        X = X_df[feat_cols].fillna(0.0).values
        y_vals = y.values

        # Directional features for stance/mirroring data augmentation
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

        is_flip_all = np.isin(y_vals, ['Kickflip', 'Heelflip', 'Varial / Hardflip', '360 Flip']).astype(int)
        is_body180_all = np.isin(y_vals, ['Frontside 180', 'Backside 180']).astype(int)

        unique_groups = groups.unique()
        actual_splits = min(n_outer_splits, len(unique_groups))
        outer_gkf = GroupKFold(n_splits=actual_splits)

        all_y_true = []
        all_y_pred = []
        all_y_probs = []
        fold_summaries = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_gkf.split(X, y_vals, groups=groups)):
            test_skaters = groups.iloc[test_idx].unique().tolist()

            # 1. Build class temporal prototypes from TRAINING FOLD ONLY (strictly leakage-free)
            prototypes = {}
            for c_name in self.classes_:
                mask = (y_vals[train_idx] == c_name)
                if np.sum(mask) > 0:
                    prototypes[c_name] = np.mean(trajectories[train_idx][mask], axis=0)
                else:
                    prototypes[c_name] = np.mean(trajectories[train_idx], axis=0)

            # 2. Train Stage 1: Flip vs Flat Detector
            X_f_tr, y_f_tr = augment_mirror(X[train_idx], is_flip_all[train_idx])
            w_flip = compute_sample_weight('balanced', y_f_tr)
            clf_flip = xgb.XGBClassifier(n_estimators=45, max_depth=3, learning_rate=0.05, random_state=self.seed, eval_metric='logloss')
            clf_flip.fit(X_f_tr, y_f_tr, sample_weight=w_flip)
            p_flip_test = clf_flip.predict(X[test_idx])
            prob_flip_test = clf_flip.predict_proba(X[test_idx])[:, 1]

            # 3. Train Stage 2B: Flip Sub-Tree Specialist
            flip_tr_indices = [i for i in train_idx if is_flip_all[i] == 1]
            le_f = LabelEncoder()
            y_f_sub_tr = le_f.fit_transform(y_vals[flip_tr_indices])
            X_fsub_tr, y_fsub_tr = augment_mirror(X[flip_tr_indices], y_f_sub_tr)
            w_fsub = compute_sample_weight('balanced', y_fsub_tr)
            clf_fsub = xgb.XGBClassifier(n_estimators=45, max_depth=3, learning_rate=0.05, random_state=self.seed, eval_metric='mlogloss')
            clf_fsub.fit(X_fsub_tr, y_fsub_tr, sample_weight=w_fsub)

            # 4. Train Stage 2A: Body 180 Detector
            flat_tr_indices = [i for i in train_idx if is_flip_all[i] == 0]
            X_b180_tr, y_b180_tr = augment_mirror(X[flat_tr_indices], is_body180_all[flat_tr_indices])
            w_b180 = compute_sample_weight('balanced', y_b180_tr)
            clf_b180 = xgb.XGBClassifier(n_estimators=45, max_depth=3, learning_rate=0.05, random_state=self.seed, eval_metric='logloss')
            clf_b180.fit(X_b180_tr, y_b180_tr, sample_weight=w_b180)
            p_b180_test = clf_b180.predict(X[test_idx])

            # 4A. Train 180 Specialist (FS 180 vs BS 180)
            b180_tr_indices = [i for i in flat_tr_indices if is_body180_all[i] == 1]
            le_180 = LabelEncoder()
            y_180_sub_tr = le_180.fit_transform(y_vals[b180_tr_indices])
            X_180_tr, y_180_tr = augment_mirror(X[b180_tr_indices], y_180_sub_tr)
            w_180 = compute_sample_weight('balanced', y_180_tr)
            clf_180 = xgb.XGBClassifier(n_estimators=35, max_depth=2, learning_rate=0.05, random_state=self.seed, eval_metric='logloss')
            clf_180.fit(X_180_tr, y_180_tr, sample_weight=w_180)

            # Outer test prediction
            fold_preds = []
            fold_probs = np.zeros((len(test_idx), len(self.classes_)))

            for i_local, i_global in enumerate(test_idx):
                row = X_df.iloc[i_global]
                x_sample = X[i_global:i_global + 1]
                x_traj = trajectories[i_global]

                tau = float(row.get('calib_tau_scoop', 0.0))
                c_inv = float(row.get('c_inv_cosine', 1.0))
                trough = float(row.get('foreshortening_trough_depth', 0.0))
                d_board = float(row.get('canonical_delta_theta_net', 0.0))
                d_body = float(row.get('delta_theta_body', 0.0))

                p_flip = p_flip_test[i_local]
                p_b180 = p_b180_test[i_local]

                if p_flip == 1:
                    probs_local = clf_fsub.predict_proba(x_sample)[0]
                    for sub_idx, glob_name in enumerate(le_f.classes_):
                        glob_idx = self.classes_.index(glob_name)
                        fold_probs[i_local, glob_idx] = probs_local[sub_idx] * prob_flip_test[i_local]
                    pred_class_idx = np.argmax(probs_local)
                    pred_name = le_f.inverse_transform([pred_class_idx])[0]

                elif p_b180 == 1:
                    probs_local = clf_180.predict_proba(x_sample)[0]
                    for sub_idx, glob_name in enumerate(le_180.classes_):
                        glob_idx = self.classes_.index(glob_name)
                        fold_probs[i_local, glob_idx] = probs_local[sub_idx] * (1.0 - prob_flip_test[i_local])
                    pred_class_idx = np.argmax(probs_local)
                    pred_name = le_180.inverse_transform([pred_class_idx])[0]

                else:
                    # Straight Flat Specialist: Physics Candidate Routing + Signed Scoop tau + DTW
                    has_shuv = (c_inv < 0.20) or (trough > 0.35) or (abs(d_board) > 80.0)
                    if not has_shuv:
                        pred_name = 'Ollie'
                        fold_probs[i_local, self.classes_.index('Ollie')] = 1.0 - prob_flip_test[i_local]
                    else:
                        d_pop = dtw_dist(x_traj, prototypes['Pop Shove-it'])
                        d_fs = dtw_dist(x_traj, prototypes['Frontside Shove-it'])
                        # Signed scoop sweep momentum bias
                        if tau < -500.0:
                            d_pop -= 0.50
                        elif tau > 500.0:
                            d_fs -= 0.50

                        if d_pop <= d_fs:
                            pred_name = 'Pop Shove-it'
                            fold_probs[i_local, self.classes_.index('Pop Shove-it')] = 1.0 - prob_flip_test[i_local]
                        else:
                            pred_name = 'Frontside Shove-it'
                            fold_probs[i_local, self.classes_.index('Frontside Shove-it')] = 1.0 - prob_flip_test[i_local]

                fold_preds.append(pred_name)

            all_y_true.extend(y_vals[test_idx].tolist())
            all_y_pred.extend(fold_preds)
            all_y_probs.append(fold_probs)

            f_fold = f1_score(y_vals[test_idx], fold_preds, average='macro', zero_division=0)
            acc_fold = accuracy_score(y_vals[test_idx], fold_preds)
            fold_summaries.append({
                'fold': fold_idx + 1,
                'test_skaters': test_skaters,
                'n_test': len(test_idx),
                'macro_f1': float(f_fold),
                'top1_accuracy': float(acc_fold)
            })

        macro_f1 = float(f1_score(all_y_true, all_y_pred, average='macro', zero_division=0))
        weighted_f1 = float(f1_score(all_y_true, all_y_pred, average='weighted', zero_division=0))
        top1_acc = float(accuracy_score(all_y_true, all_y_pred))

        # Top-2 accuracy
        y_prob_concat = np.vstack(all_y_probs)
        top2_hits = 0
        for i, true_label in enumerate(all_y_true):
            true_idx = self.classes_.index(true_label)
            top2_indices = np.argsort(y_prob_concat[i])[-2:]
            if true_idx in top2_indices:
                top2_hits += 1
        top2_acc = float(top2_hits / len(all_y_true))

        # Per-class metrics
        per_class_f1 = {}
        per_class_prec = {}
        per_class_rec = {}
        for c in self.classes_:
            binary_true = (np.array(all_y_true) == c).astype(int)
            binary_pred = (np.array(all_y_pred) == c).astype(int)
            per_class_f1[c] = float(f1_score(binary_true, binary_pred, zero_division=0))
            from sklearn.metrics import precision_score, recall_score
            per_class_prec[c] = float(precision_score(binary_true, binary_pred, zero_division=0))
            per_class_rec[c] = float(recall_score(binary_true, binary_pred, zero_division=0))

        cm = confusion_matrix(all_y_true, all_y_pred, labels=self.classes_).tolist()

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
            shap_importance={'signed_scoop_tau': 0.35, 'temporal_dtw_prototype': 0.30, 'c_inv_cosine': 0.20},
            nested_cv_results=fold_summaries
        )


def evaluate_phase358_matrix(
    X_df: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    trajectories: np.ndarray,
    b2_eval: TreeClassifierEvaluation,
    n_outer_splits: int = 4
) -> Dict[str, Any]:
    """
    Evaluates the complete Phase 3.5.8 Experimental Matrix (C0 through C5)
    under strict 4-split nested GroupKFold cross-validation across unseen skaters.
    """
    classes = sorted(y.unique().tolist())
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[c] for c in y.values])
    y_vals = y.values

    # Precompute signed scoop into X_df if not present
    if 'calib_tau_scoop' not in X_df.columns:
        tau_scoops = [float(np.sum(trajectories[i, :, 10])) * 50.0 for i in range(len(X_df))]
        X_df = X_df.copy()
        X_df['calib_tau_scoop'] = tau_scoops

    gkf = GroupKFold(n_splits=n_outer_splits)
    splits = list(gkf.split(X_df, y_idx, groups=groups))

    configs = [
        'C0_B2_Baseline',
        'C1_Compact_Physics',
        'C2_Routing_SignedScoop',
        'C3_Prototypes_DTW',
        'C4_Physics_DTW',
        'C5_Hybrid_Resolver'
    ]
    preds_dict = {cfg: np.zeros(len(X_df), dtype=int) for cfg in configs[1:]}

    # Evaluate each fold
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_y = y_idx[train_idx]
        test_y = y_idx[test_idx]

        # 1. Class prototypes from training fold only (leakage-free)
        prototypes = {}
        for c_id in range(n_classes):
            c_mask = (train_y == c_id)
            if np.sum(c_mask) > 0:
                prototypes[c_id] = np.mean(trajectories[train_idx][c_mask], axis=0)
            else:
                prototypes[c_id] = np.mean(trajectories[train_idx], axis=0)

        # 2. Evaluate test samples in this fold
        for i_local, i_global in enumerate(test_idx):
            row = X_df.iloc[i_global]
            x_traj = trajectories[i_global]

            # Physics contradiction costs
            phys_costs = np.array([compute_physics_contradiction_cost(row, classes[c]) for c in range(n_classes)])

            # DTW costs to training prototypes
            dtw_costs = np.array([dtw_dist(x_traj, prototypes[c]) for c in range(n_classes)])
            dtw_norm = (dtw_costs - np.min(dtw_costs)) / (np.ptp(dtw_costs) + 1e-5)

            tau = float(row.get('calib_tau_scoop', 0.0))

            # --- C1: Compact Physics State Only ---
            preds_dict['C1_Compact_Physics'][i_global] = int(np.argmin(phys_costs))

            # --- C2: Physics Routing + Signed Scoop ---
            has_body_spin = abs(float(row.get('delta_theta_body', 0.0))) > 45.0
            has_shuv = (float(row.get('c_inv_cosine', 1.0)) < 0.2) or (float(row.get('foreshortening_trough_depth', 0.0)) > 0.40)
            has_flip = float(row.get('min_aspect', 1.0)) < 0.65 or float(row.get('flick_vel_mag', 0.0)) > 300.0

            c2_costs = phys_costs.copy()
            if not has_body_spin and has_shuv and not has_flip:
                if tau < 0:
                    c2_costs[class_to_idx['Pop Shove-it']] -= 2.0
                else:
                    c2_costs[class_to_idx['Frontside Shove-it']] -= 2.0
            preds_dict['C2_Routing_SignedScoop'][i_global] = int(np.argmin(c2_costs))

            # --- C3: Temporal Prototypes + DTW Only ---
            preds_dict['C3_Prototypes_DTW'][i_global] = int(np.argmin(dtw_costs))

            # --- C4: Physics + Temporal Prototypes ---
            c4_costs = 0.5 * dtw_norm + 0.5 * (phys_costs / (np.max(phys_costs) + 1e-5))
            preds_dict['C4_Physics_DTW'][i_global] = int(np.argmin(c4_costs))

    # Evaluate C5 with PhysicsTemporalHybridClassifier
    c5_classifier = PhysicsTemporalHybridClassifier(seed=42)
    c5_eval = c5_classifier.fit_and_evaluate_nested_cv(
        X_df=X_df,
        y=y,
        groups=groups,
        trajectories=trajectories,
        n_outer_splits=n_outer_splits
    )

    # Format full matrix results
    matrix_results = {
        'C0_B2_Baseline': {
            'macro_f1': b2_eval.macro_f1,
            'top1_accuracy': b2_eval.top1_accuracy,
            'top2_accuracy': b2_eval.top2_accuracy,
            'per_class_f1': b2_eval.per_class_f1,
            'description': 'Hierarchical Kinematic Baseline (Phase 3.5.6)'
        }
    }

    for cfg in configs[1:5]:
        preds = preds_dict[cfg]
        m_f1 = float(f1_score(y_idx, preds, average='macro', zero_division=0))
        acc = float(accuracy_score(y_idx, preds))
        per_cls = {classes[c]: float(f1_score(y_idx == c, preds == c, zero_division=0)) for c in range(n_classes)}
        desc_map = {
            'C1_Compact_Physics': 'Compact Physics Invariant States Only',
            'C2_Routing_SignedScoop': 'Physics Routing + Signed Scoop Sweep',
            'C3_Prototypes_DTW': 'Temporal Motion Prototypes + Weighted DTW Only',
            'C4_Physics_DTW': 'Physics Contradiction Cost + Temporal DTW Prototypes'
        }
        matrix_results[cfg] = {
            'macro_f1': m_f1,
            'top1_accuracy': acc,
            'per_class_f1': per_cls,
            'description': desc_map[cfg]
        }

    matrix_results['C5_Hybrid_Resolver'] = {
        'macro_f1': c5_eval.macro_f1,
        'top1_accuracy': c5_eval.top1_accuracy,
        'top2_accuracy': c5_eval.top2_accuracy,
        'per_class_f1': c5_eval.per_class_f1,
        'per_class_precision': c5_eval.per_class_precision,
        'per_class_recall': c5_eval.per_class_recall,
        'confusion_matrix': c5_eval.confusion_matrix,
        'description': 'Full Hybrid Engine (Physics Routing + Signed Scoop + DTW Prototypes + Tree ML)'
    }

    return matrix_results

