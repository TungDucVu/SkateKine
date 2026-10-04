"""
Phase 1 Visualizer: Generates high-fidelity visual telemetry charts and annotated video frames.
Outputs:
1. docs/visualizations/phase1_pilot_summary.png (25-clip benchmark dashboard)
2. docs/visualizations/kinematic_signals_kickflip.png (4-pane trajectory signals for Chris Joslin Kickflip)
3. docs/visualizations/telemetry_annotated_frames.png (Multi-frame annotated HUD montage)
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set clean scientific plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10


def generate_pilot_summary_dashboard(json_path: str = 'docs/reports/pilot_benchmark_results.json', out_dir: str = 'docs/visualizations'):
    os.makedirs(out_dir, exist_ok=True)
    with open(json_path, 'r') as f:
        data = json.load(f)

    clips = data['detailed_clip_results']
    df = pd.DataFrame(clips)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=200)
    fig.suptitle('SkateKine — Phase 1 Pilot Batch Certification Dashboard (25 Diverse Clips)', fontsize=16, fontweight='bold', y=0.98)

    # 1. Detection Rates per Clip
    ax = axes[0, 0]
    x = np.arange(len(df))
    width = 0.38
    ax.bar(x - width/2, df['skater_detection_rate'] * 100, width, label='Skater Detection Rate', color='#2ecc71', alpha=0.85)
    ax.bar(x + width/2, df['board_detection_rate'] * 100, width, label='Board Detection Rate', color='#3498db', alpha=0.85)
    ax.axhline(90, color='#e74c3c', linestyle='--', linewidth=1.5, label='Operational Target (90%)')
    ax.set_xticks(x)
    ax.set_xticklabels(df['clip_id'], rotation=65, ha='right', fontsize=8)
    ax.set_ylabel('Detection Rate (%)', fontweight='bold')
    ax.set_ylim(50, 105)
    ax.set_title('Skater & Board Detection Rates Across Pilot Clips', fontweight='bold')
    ax.legend(loc='lower right', frameon=True)

    # 2. Tracking Provenance Distribution across all frames
    ax = axes[0, 1]
    prov_totals = {0: 0, 1: 0, 2: 0, -1: 0}
    for p in df['provenance']:
        for k, v in p.items():
            ik = int(k)
            if ik in prov_totals:
                prov_totals[ik] += v
    
    prov_labels = ['Direct Detection (0)', 'Optical Flow (1)', 'Kalman Predicted (2)', 'Lost (-1)']
    prov_counts = [prov_totals[0], prov_totals[1], prov_totals[2], prov_totals[-1]]
    colors = ['#27ae60', '#f39c12', '#9b59b6', '#e74c3c']
    
    wedges, texts, autotexts = ax.pie(
        prov_counts,
        labels=prov_labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=140,
        textprops=dict(fontweight='bold'),
        wedgeprops=dict(width=0.45, edgecolor='w')
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(10)
    ax.set_title(f'Tracking Provenance Distribution ({data["total_frames_processed"]} Total Frames)', fontweight='bold')

    # 3. Torso-Normalized Jerk Index [G6]
    ax = axes[1, 0]
    valid_jerk = df['mean_torso_jerk'].fillna(0.0)
    bars = ax.bar(x, valid_jerk, color='#1abc9c', alpha=0.85, edgecolor='#16a085')
    ax.axhline(0.15, color='#e74c3c', linestyle='--', linewidth=1.5, label='Gate 1 Max Jerk (0.15)')
    ax.set_xticks(x)
    ax.set_xticklabels(df['clip_id'], rotation=65, ha='right', fontsize=8)
    ax.set_ylabel('Jerk [torso-lengths / frame³]', fontweight='bold')
    ax.set_title('Resolution-Normalized Skater Jitter (Torso Jerk Index [G6])', fontweight='bold')
    ax.legend(loc='upper right', frameon=True)

    # 4. Metric Scale Factor & Validity [G3]
    ax = axes[1, 1]
    valid_mask = df['scale_valid']
    colors_scale = ['#2ecc71' if v else '#e74c3c' for v in valid_mask]
    scales = df['scale_cm_per_px'].fillna(0.0)
    ax.bar(x, scales, color=colors_scale, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df['clip_id'], rotation=65, ha='right', fontsize=8)
    ax.set_ylabel('Metric Scale Factor (cm / pixel)', fontweight='bold')
    ax.set_title('Pre-Pop Metric Scale Calibration ([G3] Green=Valid, Red=Fallback)', fontweight='bold')

    custom_lines = [
        patches.Patch(facecolor='#2ecc71', label='Scale Valid (84% passed)'),
        patches.Patch(facecolor='#e74c3c', label='Dimensionless Fallback')
    ]
    ax.legend(handles=custom_lines, loc='upper right', frameon=True)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'phase1_pilot_summary.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved pilot summary dashboard: {out_path}')
    return out_path


def generate_kinematic_signal_plots(clip_id: str = 'batb_0005', out_dir: str = 'docs/visualizations'):
    os.makedirs(out_dir, exist_ok=True)
    parquet_path = f'features/v1_trajectories/{clip_id}.parquet'
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f'Missing parquet: {parquet_path}')

    df = pd.read_parquet(parquet_path)
    fps = 60.0 # batb_0005 is 60 fps
    time_sec = df['frame_idx'] / fps

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, dpi=200)
    fig.suptitle(f'SkateKine Kinematic Signal Telemetry — Chris Joslin Kickflip ({clip_id})', fontsize=15, fontweight='bold')

    # Pane 1: Inverted Vertical Trajectory (Altitude)
    ax = axes[0]
    # Invert image y coordinates so up = higher altitude
    ground_ref = df['board_centroid_y'].max()
    board_alt = ground_ref - df['board_centroid_y']
    left_ankle_alt = ground_ref - df['left_ankle_y']
    right_ankle_alt = ground_ref - df['right_ankle_y']

    ax.plot(time_sec, board_alt, label='Skateboard Deck Centroid', color='#e74c3c', linewidth=2.5)
    ax.plot(time_sec, left_ankle_alt, label='Front Ankle (Left)', color='#3498db', linewidth=1.5, linestyle='--')
    ax.plot(time_sec, right_ankle_alt, label='Rear Ankle (Right)', color='#f39c12', linewidth=1.5, linestyle=':')
    ax.set_ylabel('Relative Altitude (px)', fontweight='bold')
    ax.set_title('1. Vertical Trajectory & Flight Envelope (Pop, Flight, Catch, Land)', fontweight='bold')
    ax.legend(loc='upper right', frameon=True)

    # Pane 2: Phase-Unwrapped Angle & Angular Velocity [G4]
    ax = axes[1]
    ax.plot(time_sec, np.rad2deg(df['theta_unwrapped']), label='Unwrapped In-Plane Angle [G4] (deg)', color='#9b59b6', linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(time_sec, df['omega_rad_per_sec'], label='Angular Velocity ω (rad/s)', color='#2ecc71', linewidth=1.8, linestyle='-.')
    ax.set_ylabel('Angle (deg)', fontweight='bold', color='#9b59b6')
    ax2.set_ylabel('ω (rad/s)', fontweight='bold', color='#2ecc71')
    ax.set_title('2. Board Rotation Dynamics: Continuous Angle θ & Angular Velocity ω', fontweight='bold')

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right', frameon=True)

    # Pane 3: Frame Tracking Provenance State [G5]
    ax = axes[2]
    prov = df['tracking_source']
    colors_map = {0: '#27ae60', 1: '#f39c12', 2: '#9b59b6', -1: '#e74c3c'}
    bar_colors = [colors_map.get(int(x), '#95a5a6') for x in prov]
    ax.bar(time_sec, np.ones_like(time_sec), width=1.0/fps, color=bar_colors, edgecolor='none')
    ax.set_yticks([])
    ax.set_ylabel('Provenance [G5]', fontweight='bold')
    ax.set_title('3. Frame State Provenance Timeline (Green: Detected, Orange: Optical Flow, Purple: Kalman)', fontweight='bold')

    # Pane 4: Board-Local Foot Positions relative to Truck Bolts
    ax = axes[3]
    ax.plot(time_sec, df['left_ankle_board_local_x'], label='Front Foot Offset X (Along Deck)', color='#2980b9', linewidth=2)
    ax.plot(time_sec, df['right_ankle_board_local_x'], label='Rear Foot Offset X (Along Deck)', color='#d35400', linewidth=2)
    ax.axhline(0, color='gray', linestyle=':', label='Deck Center')
    ax.set_xlabel('Time (seconds)', fontweight='bold')
    ax.set_ylabel('Offset from Center (px)', fontweight='bold')
    ax.set_title('4. Board-Local Projected Foot Placement Relative to Deck Center', fontweight='bold')
    ax.legend(loc='lower right', frameon=True)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'kinematic_signals_{clip_id}.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved kinematic signal plot: {out_path}')
    return out_path


def generate_annotated_frame_montage(clip_id: str = 'batb_0005', out_dir: str = 'docs/visualizations'):
    os.makedirs(out_dir, exist_ok=True)
    df_manifest = pd.read_csv('data/metadata/video_manifest.csv')
    row = df_manifest[df_manifest['clip_id'] == clip_id].iloc[0]
    video_path = row['filepath'].replace('/', os.sep)
    parquet_path = f'features/v1_trajectories/{clip_id}.parquet'

    df_traj = pd.read_parquet(parquet_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Select 5 representative key moments: Approach, Pop, Peak Flip, Catch, Landing
    # For batb_0005 (164 frames): 20, 60, 85, 105, 135
    frame_indices = [20, 60, 85, 105, 135]
    frame_titles = ['1. Pre-Pop Approach', '2. Pop Impulse Initiation', '3. Mid-Air In-Plane Flip', '4. Foot Catch Elevation', '5. 4-Wheel Touchdown']

    annotated_images = []

    for f_target in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_target)
        ret, frame = cap.read()
        if not ret:
            continue

        r = df_traj[df_traj['frame_idx'] == f_target].iloc[0]

        # Draw board 8-point polygon if available
        if not np.isnan(r['board_nose_x']) and not np.isnan(r['board_tail_x']):
            nose = (int(r['board_nose_x']), int(r['board_nose_y']))
            tail = (int(r['board_tail_x']), int(r['board_tail_y']))
            ft = (int(r['board_ft_x']), int(r['board_ft_y']))
            rt = (int(r['board_rt_x']), int(r['board_rt_y']))

            # Longitudinal spine
            cv2.line(frame, nose, tail, (0, 0, 255), 3, cv2.LINE_AA) # Red deck spine
            # Nose circle (green) & Tail circle (yellow)
            cv2.circle(frame, nose, 7, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, tail, 7, (0, 255, 255), -1, cv2.LINE_AA)
            # Trucks (magenta circles)
            cv2.circle(frame, ft, 6, (255, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, rt, 6, (255, 0, 255), -1, cv2.LINE_AA)

        # Draw skater ankles and hip
        if not np.isnan(r['left_ankle_x']):
            la = (int(r['left_ankle_x']), int(r['left_ankle_y']))
            ra = (int(r['right_ankle_x']), int(r['right_ankle_y']))
            hip = (int(r['mid_hip_x']), int(r['mid_hip_y']))
            cv2.circle(frame, la, 6, (255, 200, 0), -1, cv2.LINE_AA) # Cyan left ankle
            cv2.circle(frame, ra, 6, (255, 100, 0), -1, cv2.LINE_AA) # Blue right ankle
            cv2.circle(frame, hip, 8, (0, 255, 100), -1, cv2.LINE_AA) # Green mid-hip
            cv2.line(frame, hip, la, (255, 200, 0), 2, cv2.LINE_AA)
            cv2.line(frame, hip, ra, (255, 100, 0), 2, cv2.LINE_AA)

        # Draw Telemetry HUD Card in top-left
        cv2.rectangle(frame, (20, 20), (380, 150), (15, 15, 15), -1)
        cv2.rectangle(frame, (20, 20), (380, 150), (0, 255, 255), 2)
        
        cv2.putText(frame, f"TRICK: {row['trick_name'].upper()} ({row['skater_id']})", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FRAME: {f_target:03d} | TIME: {f_target/60.0:.2f}s", (35, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        
        prov_map = {0: 'DETECTED', 1: 'OPTICAL_FLOW', 2: 'KALMAN_PRED', -1: 'LOST'}
        prov_str = prov_map.get(int(r['tracking_source']), 'UNKNOWN')
        prov_color = (0, 255, 0) if r['tracking_source'] == 0 else (0, 165, 255)
        cv2.putText(frame, f"SOURCE: {prov_str} [G5]", (35, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, prov_color, 2, cv2.LINE_AA)
        
        omega_val = r['omega_rad_per_sec']
        omega_str = f"{omega_val:.1f} rad/s" if not np.isnan(omega_val) else "N/A"
        cv2.putText(frame, f"OMEGA: {omega_str} | SCALE: {r['scale_cm_per_px']:.3f} cm/px", (35, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

        annotated_images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    # Create composite montage plot
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), dpi=200)
    fig.suptitle(f'SkateKine Phase 1 Telemetry Keyframe Sequence — Chris Joslin Kickflip ({clip_id})', fontsize=16, fontweight='bold', y=0.98)

    for i, (img, title) in enumerate(zip(annotated_images, frame_titles)):
        axes[i].imshow(img)
        axes[i].set_title(title, fontweight='bold', fontsize=11)
        axes[i].axis('off')

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'telemetry_annotated_frames.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved annotated frame montage: {out_path}')
    return out_path


if __name__ == '__main__':
    print('Generating Phase 1 Visualizations...')
    f1 = generate_pilot_summary_dashboard()
    f2 = generate_kinematic_signal_plots('batb_0005')
    f3 = generate_annotated_frame_montage('batb_0005')
    print('All visualizations generated successfully!')
