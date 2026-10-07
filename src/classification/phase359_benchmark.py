"""
Phase 3.5.9 Controlled Cross-Skater Recognition Benchmark:
Implements and evaluates the controlled ablation tree:
  - Exp 0: Frozen Baseline B4 (Reference: 28.08% Macro F1)
  - Exp 1: B4 + 15 Audited Ollie Clips (Data expansion impact)
  - Exp 2: B4 + Ollie Physical Candidate Gate (Hollaus fallback guard)
  - Exp 3: B4 + Stance Sign Normalization (BS180 / FS180 coordinate alignment)
  - Exp 4: B4 + Visual Scoop Transient Energy (Abdullah time-frequency metric)
  - Exp 5: B4 + ALL Integrated Mechanisms

All evaluated strictly under the identical 4-split nested GroupKFold by skater.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.classification.phase3_pipeline import Phase3Engine
from src.classification.temporal_matcher import (
    PhysicsTemporalHybridClassifier,
    extract_temporal_trajectory,
    dtw_dist,
    compute_physics_contradiction_cost
)
from src.classification.tree_classifier import FEATURE_COLUMNS


def compute_scoop_transient_energy(df_traj: pd.DataFrame, t_pop: int, t_apex: int) -> float:
    """
    Computes visual transient scoop energy E_scoop during the pop-to-apex window:
    E_scoop = 1 / (t_apex - t_pop) * sum( (d(L_board / L0) / dt)^2 )
    """
    if t_apex <= t_pop or len(df_traj) == 0:
        return 0.0
    window = df_traj.iloc[max(0, t_pop):min(len(df_traj), t_apex + 1)]
    if len(window) < 2 or 'board_length' not in window.columns:
        return 0.0
    lengths = window['board_length'].values
    l0 = np.percentile(lengths, 90) if len(lengths) > 0 else 1.0
    if l0 <= 1e-3 or np.isnan(l0):
        l0 = 1.0
    ratios = lengths / l0
    diffs = np.diff(ratios)
    energy = float(np.mean(diffs ** 2)) * 1000.0  # Scaled for numerical conditioning
    return energy


def run_phase359_experiments():
    print("=" * 80)
    print("PHASE 3.5.9: CONTROLLED CROSS-SKATER RECOGNITION BENCHMARK")
    print("=" * 80)

    # 1. Load full dataset
    engine = Phase3Engine()
    df_pred, df_oracle, bail_records, graph_arr, compact_arr, temporal_arr = engine.extract_dataset_records()

    canon_mask = df_pred['canonical_class'].notna()
    df_c = df_pred[canon_mask].copy().reset_index(drop=True)
    temporal_c = temporal_arr[canon_mask.values]

    y_vals = df_c['canonical_class'].values
    groups = df_c['skater_id'].values
    classes = sorted(list(set(y_vals)))
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[c] for c in y_vals])

    print(f"Total Canonical Attempts: {len(df_c)} across {len(set(groups))} Skaters/Sessions")
    print("Class Distribution:")
    for c in classes:
        print(f"  {c:<20}: {np.sum(y_vals == c)}")
    print("=" * 80)

    # Compute Scoop Transient Energy for each clip
    scoop_energies = []
    for i, row in df_c.iterrows():
        cid = row['clip_id']
        pq_path = os.path.join(engine.trajectory_dir, f"{cid}.parquet")
        if os.path.exists(pq_path):
            df_t = pd.read_parquet(pq_path)
            e_sc = compute_scoop_transient_energy(df_t, int(row.get('t_pop', 0)), int(row.get('t_apex', 20)))
        else:
            e_sc = 0.0
        scoop_energies.append(e_sc)
    df_c['transient_scoop_energy'] = scoop_energies

    # Standard Feature Matrix
    feat_cols = [c for c in FEATURE_COLUMNS if c in df_c.columns]
    X_raw = df_c[feat_cols].fillna(0.0).values

    # Sanitize temporal tensors against NaN/Inf
    clean_temporal = np.nan_to_num(temporal_c, nan=0.0, posinf=0.0, neginf=0.0)

    # Directional features for mirroring
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

    # Setup strict 4-split nested GroupKFold
    gkf = GroupKFold(n_splits=4)
    splits = list(gkf.split(X_raw, y_idx, groups=groups))

    experiment_results = {}

    # Define Experiment Configurations
    # Format: (exp_id, name, use_physical_gate, use_stance_norm, use_scoop_energy)
    configs = [
        ("Exp1_Data_Expansion", "B4 Baseline on Expanded Dataset (125 Clips)", False, False, False),
        ("Exp2_Ollie_Gate", "B4 + Calibrated Ollie Physical Candidate Gate", True, False, False),
        ("Exp3_Stance_Norm", "B4 + Stance Sign Normalization", False, True, False),
        ("Exp4_Scoop_Energy", "B4 + Transient Scoop Energy Feature", False, False, True),
        ("Exp5_Full_Phase359", "B4 + ALL Integrated Phase 3.5.9 Mechanisms", True, True, True)
    ]

    for exp_id, exp_name, use_gate, use_stance, use_scoop in configs:
        print(f"\nEvaluating {exp_id}: {exp_name}...")
        t0_exp = time.time()

        all_y_true = []
        all_y_pred = []
        all_y_probs = []

        # Prepare feature matrix for this configuration
        X_exp = X_raw.copy()
        current_feat_cols = list(feat_cols)

        if use_scoop:
            X_exp = np.hstack([X_exp, np.array(scoop_energies)[:, None]])
            current_feat_cols.append('transient_scoop_energy')

        if use_stance:
            # Stance sign normalization: regular=+1, goofy=-1
            stances = df_c['stance'].fillna('regular').str.lower().values
            signs = np.where(stances == 'goofy', -1.0, 1.0)
            for d_idx in dir_indices:
                X_exp[:, d_idx] *= signs

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            # 1. Class temporal prototypes from TRAINING FOLD ONLY (strictly leakage-free)
            prototypes = {}
            for c_name in classes:
                mask = (y_vals[train_idx] == c_name)
                if np.sum(mask) > 0:
                    prototypes[c_name] = np.mean(clean_temporal[train_idx][mask], axis=0)
                else:
                    prototypes[c_name] = np.mean(clean_temporal[train_idx], axis=0)

            # 2. Calibrate Ollie Physical Gate Thresholds strictly on TRAINING FOLD
            if use_gate:
                tr_ollie_mask = (y_vals[train_idx] == 'Ollie')
                if np.sum(tr_ollie_mask) > 0:
                    ollie_min_aspects = df_c.iloc[train_idx[tr_ollie_mask]]['min_aspect'].values
                    ollie_body_yaws = np.abs(df_c.iloc[train_idx[tr_ollie_mask]]['delta_theta_body'].values)
                    tau_gate_aspect = float(np.percentile(ollie_min_aspects, 5)) if len(ollie_min_aspects) > 0 else 0.65
                    tau_gate_yaw = float(np.percentile(ollie_body_yaws, 95)) if len(ollie_body_yaws) > 0 else 45.0
                else:
                    tau_gate_aspect = 0.65
                    tau_gate_yaw = 45.0
            else:
                tau_gate_aspect = 0.0
                tau_gate_yaw = 999.0

            # 3. Train Flip vs Flat Detector
            X_f_tr, y_f_tr = augment_mirror(X_exp[train_idx], is_flip_all[train_idx])
            w_flip = compute_sample_weight('balanced', y_f_tr)
            clf_flip = xgb.XGBClassifier(n_estimators=45, max_depth=3, learning_rate=0.05, random_state=42, eval_metric='logloss')
            clf_flip.fit(X_f_tr, y_f_tr, sample_weight=w_flip)
            prob_flip_test = clf_flip.predict_proba(X_exp[test_idx])[:, 1]

            # 4. Train Flip Sub-Tree Specialist
            flip_tr_indices = [i for i in train_idx if is_flip_all[i] == 1]
            le_f = LabelEncoder()
            y_f_sub_tr = le_f.fit_transform(y_vals[flip_tr_indices])
            X_fsub_tr, y_fsub_tr = augment_mirror(X_exp[flip_tr_indices], y_f_sub_tr)
            w_fsub = compute_sample_weight('balanced', y_fsub_tr)
            clf_fsub = xgb.XGBClassifier(n_estimators=45, max_depth=3, learning_rate=0.05, random_state=42, eval_metric='mlogloss')
            clf_fsub.fit(X_fsub_tr, y_fsub_tr, sample_weight=w_fsub)

            # 5. Train Body 180 Detector
            flat_tr_indices = [i for i in train_idx if is_flip_all[i] == 0]
            X_b180_tr, y_b180_tr = augment_mirror(X_exp[flat_tr_indices], is_body180_all[flat_tr_indices])
            w_b180 = compute_sample_weight('balanced', y_b180_tr)
            clf_b180 = xgb.XGBClassifier(n_estimators=45, max_depth=3, learning_rate=0.05, random_state=42, eval_metric='logloss')
            clf_b180.fit(X_b180_tr, y_b180_tr, sample_weight=w_b180)
            prob_b180_test = clf_b180.predict_proba(X_exp[test_idx])[:, 1]

            # 6. Train 180 Direction Specialist
            b180_tr_indices = [i for i in flat_tr_indices if is_body180_all[i] == 1]
            le_180 = LabelEncoder()
            y_180_sub_tr = le_180.fit_transform(y_vals[b180_tr_indices])
            X_180_tr, y_180_tr = augment_mirror(X_exp[b180_tr_indices], y_180_sub_tr)
            w_180 = compute_sample_weight('balanced', y_180_tr)
            clf_180 = xgb.XGBClassifier(n_estimators=45, max_depth=3, learning_rate=0.05, random_state=42, eval_metric='logloss')
            clf_180.fit(X_180_tr, y_180_tr, sample_weight=w_180)

            # 7. Train Planar Resolver
            shuv_tr_indices = [i for i in flat_tr_indices if is_body180_all[i] == 0]
            le_shuv = LabelEncoder()
            y_shuv_sub_tr = le_shuv.fit_transform(y_vals[shuv_tr_indices])
            X_shuv_tr, y_shuv_tr = augment_mirror(X_exp[shuv_tr_indices], y_shuv_sub_tr)
            w_shuv = compute_sample_weight('balanced', y_shuv_tr)
            clf_shuv = xgb.XGBClassifier(n_estimators=45, max_depth=3, learning_rate=0.05, random_state=42, eval_metric='mlogloss')
            clf_shuv.fit(X_shuv_tr, y_shuv_tr, sample_weight=w_shuv)

            p_flip_test = clf_flip.predict(X_exp[test_idx])
            prob_flip_test = clf_flip.predict_proba(X_exp[test_idx])[:, 1]

            p_b180_test = clf_b180.predict(X_exp[test_idx])
            prob_b180_test = clf_b180.predict_proba(X_exp[test_idx])[:, 1]

            # Evaluate each test sample in this fold
            for i_local, i_global in enumerate(test_idx):
                row = df_c.iloc[i_global]
                x_vec = X_exp[i_global].reshape(1, -1)
                x_traj = clean_temporal[i_global]

                fold_probs = np.zeros(n_classes)
                p_flip = p_flip_test[i_local]
                p_b180 = p_b180_test[i_local]

                if p_flip == 1:
                    # Routed to Flip Sub-Tree Specialist
                    probs_local = clf_fsub.predict_proba(x_vec)[0]
                    for sub_idx, glob_name in enumerate(le_f.classes_):
                        glob_idx = classes.index(glob_name)
                        fold_probs[glob_idx] = probs_local[sub_idx] * prob_flip_test[i_local]
                    pred_class_idx = np.argmax(probs_local)
                    pred_name = le_f.inverse_transform([pred_class_idx])[0]

                elif p_b180 == 1:
                    # Routed to Body 180 Specialist
                    probs_local = clf_180.predict_proba(x_vec)[0]
                    for sub_idx, glob_name in enumerate(le_180.classes_):
                        glob_idx = classes.index(glob_name)
                        fold_probs[glob_idx] = probs_local[sub_idx] * (1.0 - prob_flip_test[i_local])
                    pred_class_idx = np.argmax(probs_local)
                    pred_name = le_180.inverse_transform([pred_class_idx])[0]

                else:
                    # Routed to Straight Flat Family: {Ollie, Pop Shove-it, Frontside Shove-it}
                    c_inv = float(row.get('c_inv_cosine', 1.0))
                    trough = float(row.get('trough_depth', 0.0))
                    min_len = float(row.get('min_norm_length', 1.0))
                    d_board = float(row.get('delta_theta_board', 0.0))
                    tau = float(np.nan_to_num(row.get('calib_tau_scoop', 0.0)))
                    e_sc = float(row.get('transient_scoop_energy', 0.0))

                    if use_stance:
                        st = str(row.get('stance', 'regular')).lower()
                        sign_st = -1.0 if st == 'goofy' else 1.0
                        d_board *= sign_st
                        tau *= sign_st

                    # Physical Foreshortening Gate (Phase 3.5.9B / Hollaus Fallback Guard)
                    has_shuv = (c_inv < 0.20) or (trough > 0.35) or (min_len < 0.65) or (abs(d_board) > 80.0)
                    if use_scoop and e_sc > 5.0:
                        has_shuv = True

                    if use_gate and (has_shuv or min_len < tau_gate_aspect or abs(d_board) > tau_gate_yaw):
                        has_shuv = True

                    if not has_shuv:
                        pred_name = 'Ollie'
                        fold_probs[classes.index('Ollie')] = 1.0 - prob_flip_test[i_local]
                    else:
                        d_pop = dtw_dist(x_traj, prototypes['Pop Shove-it'])
                        d_fs = dtw_dist(x_traj, prototypes['Frontside Shove-it'])

                        # Signed scoop sweep momentum bias
                        if tau < -100.0:
                            d_pop -= 0.35
                        elif tau > 100.0:
                            d_fs -= 0.35

                        if use_scoop and e_sc > 8.0:
                            # Strong transient scoop energy further separates Shove-its
                            if tau <= 0:
                                d_pop -= 0.25
                            else:
                                d_fs -= 0.25

                        if d_pop <= d_fs:
                            pred_name = 'Pop Shove-it'
                            fold_probs[classes.index('Pop Shove-it')] = 1.0 - prob_flip_test[i_local]
                        else:
                            pred_name = 'Frontside Shove-it'
                            fold_probs[classes.index('Frontside Shove-it')] = 1.0 - prob_flip_test[i_local]

                prob_vector = fold_probs / (np.sum(fold_probs) + 1e-6)
                all_y_true.append(y_vals[i_global])
                all_y_pred.append(pred_name)
                all_y_probs.append(prob_vector)

        # Compute Performance Metrics
        m_f1 = float(f1_score(all_y_true, all_y_pred, average='macro', zero_division=0))
        top1_acc = float(accuracy_score(all_y_true, all_y_pred))

        # Top-2 Accuracy
        y_prob_concat = np.vstack(all_y_probs)
        top2_hits = 0
        for i, true_label in enumerate(all_y_true):
            true_idx = classes.index(true_label)
            top2_indices = np.argsort(y_prob_concat[i])[-2:]
            if true_idx in top2_indices:
                top2_hits += 1
        top2_acc = float(top2_hits / len(all_y_true))

        per_class_f1 = {}
        for c in classes:
            bin_t = (np.array(all_y_true) == c).astype(int)
            bin_p = (np.array(all_y_pred) == c).astype(int)
            per_class_f1[c] = round(float(f1_score(bin_t, bin_p, zero_division=0)) * 100.0, 2)

        min_f1 = min(per_class_f1.values())
        cm = confusion_matrix(all_y_true, all_y_pred, labels=classes).tolist()
        t_el = time.time() - t0_exp

        print(f"[{exp_id}] Finished in {t_el:.1f}s:")
        print(f"  Macro F1: {m_f1*100:.2f}% | Top-1: {top1_acc*100:.2f}% | Top-2: {top2_acc*100:.2f}% | Min-Class F1: {min_f1:.2f}%")
        print(f"  Per-Class F1 (%): {per_class_f1}")

        experiment_results[exp_id] = {
            'experiment_name': exp_name,
            'macro_f1': round(m_f1 * 100.0, 2),
            'top1_accuracy': round(top1_acc * 100.0, 2),
            'top2_accuracy': round(top2_acc * 100.0, 2),
            'min_class_f1': min_f1,
            'per_class_f1': per_class_f1,
            'confusion_matrix': cm,
            'classes': classes
        }

    # Reference Frozen B4 Baseline
    baseline_b4 = {
        'experiment_name': 'Frozen Baseline B4 (110 Clips / 14 Skaters)',
        'macro_f1': 28.08,
        'top1_accuracy': 30.00,
        'top2_accuracy': 46.36,
        'min_class_f1': 9.09,
        'per_class_f1': {
            '360 Flip': 50.00,
            'Backside 180': 43.48,
            'Frontside 180': 28.57,
            'Frontside Shove-it': 26.67,
            'Heelflip': 18.18,
            'Kickflip': 23.53,
            'Ollie': 9.09,
            'Pop Shove-it': 12.50,
            'Varial / Hardflip': 53.66
        }
    }

    full_output = {
        'baseline_b4_frozen': baseline_b4,
        'experiments': experiment_results
    }

    out_json = "docs/reports/phase359_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print("=" * 80)
    print(f"BENCHMARK COMPLETE: Results saved to {out_json}")
    print("=" * 80)

    return full_output


if __name__ == "__main__":
    run_phase359_experiments()
