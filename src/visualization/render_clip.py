"""
Video Telemetry Renderer: Renders full video clips with burned HUD overlay, motion trails,
and kinematic telemetry. Capable of batch rendering all 25 pilot benchmark clips and
generating a comprehensive 25-clip peak-action contact sheet grid.
"""

import os
import cv2
import json
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple


def render_telemetry_video(
    clip_id: str,
    output_dir: str = 'outputs/telemetry_videos',
    extract_keyframe: bool = True
) -> Tuple[str, Optional[np.ndarray], Dict]:
    """
    Renders full telemetry video for a specific clip with HUD and wireframe overlays.
    Returns: (output_video_path, apex_keyframe_image, clip_metadata)
    """
    os.makedirs(output_dir, exist_ok=True)
    df_manifest = pd.read_csv('data/metadata/video_manifest.csv')
    matched = df_manifest[df_manifest['clip_id'] == clip_id]
    if len(matched) == 0:
        raise ValueError(f"Clip {clip_id} not found in manifest.")
    row = matched.iloc[0]

    video_path = row['filepath'].replace('/', os.sep)
    parquet_path = f'features/v1_trajectories/{clip_id}.parquet'

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet trajectory file not found: {parquet_path}")

    df_traj = pd.read_parquet(parquet_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_fps = float(cap.get(cv2.CAP_PROP_FPS))
    # Round fps to avoid MPEG-4 timebase denominator overflow (>65535) on highspeed footage
    out_fps = float(max(1, min(120, round(raw_fps))))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = os.path.join(output_dir, f'{clip_id}_telemetry.mp4').replace(os.sep, '/')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, out_fps, (w, h))

    if not out.isOpened():
        raise RuntimeError(f"VideoWriter failed to open for {out_path} with fps {out_fps}")

    # Resolution scaling factors (baseline: 1280x720)
    scale_factor = max(0.65, min(1.6, w / 1280.0))
    hud_w = int(460 * scale_factor)
    hud_h = int(185 * scale_factor)
    font_scale = 0.52 * scale_factor
    font_thickness = max(1, int(1.5 * scale_factor))

    trail_nose: List[Tuple[int, int]] = []
    trail_tail: List[Tuple[int, int]] = []
    trail_len = int(15 * (raw_fps / 60.0))  # Scale trail length to frame rate

    # Apex frame tracking (lowest Y coord of board = highest altitude, or peak spin rate)
    min_board_y = 1e9
    apex_keyframe = None
    apex_frame_idx = len(df_traj) // 2

    # Pre-determine apex candidate from parquet if available
    valid_board_y = df_traj['board_centroid_y'].dropna()
    if len(valid_board_y) > 0:
        apex_frame_idx = int(valid_board_y.idxmin())

    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if f_idx < len(df_traj):
            r = df_traj.iloc[f_idx]

            # 1. Motion trails & 8-point board wireframe
            nose_valid = not np.isnan(r['board_nose_x']) and not np.isnan(r['board_nose_y'])
            tail_valid = not np.isnan(r['board_tail_x']) and not np.isnan(r['board_tail_y'])

            if nose_valid and tail_valid:
                nose_pt = (int(r['board_nose_x']), int(r['board_nose_y']))
                tail_pt = (int(r['board_tail_x']), int(r['board_tail_y']))
                trail_nose.append(nose_pt)
                trail_tail.append(tail_pt)
                if len(trail_nose) > trail_len:
                    trail_nose.pop(0)
                    trail_tail.pop(0)

                # Draw fading trail
                for t in range(1, len(trail_nose)):
                    alpha = t / float(len(trail_nose))
                    trail_thick = max(1, int(3 * alpha * scale_factor))
                    # Nose trail: Emerald green
                    cv2.line(frame, trail_nose[t-1], trail_nose[t], (0, int(255 * alpha), int(120 * alpha)), trail_thick, cv2.LINE_AA)
                    # Tail trail: Amber / Gold
                    cv2.line(frame, trail_tail[t-1], trail_tail[t], (0, int(200 * alpha), int(255 * alpha)), trail_thick, cv2.LINE_AA)

                # Trucks
                ft_pt = (int(r['board_ft_x']), int(r['board_ft_y'])) if not np.isnan(r['board_ft_x']) else None
                rt_pt = (int(r['board_rt_x']), int(r['board_rt_y'])) if not np.isnan(r['board_rt_x']) else None

                # Central deck spine (Tail -> Nose)
                cv2.line(frame, tail_pt, nose_pt, (0, 0, 240), max(2, int(3 * scale_factor)), cv2.LINE_AA)

                # Nose marker (Green, with label N)
                cv2.circle(frame, nose_pt, int(7 * scale_factor), (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, nose_pt, int(9 * scale_factor), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "N", (nose_pt[0] - 4, nose_pt[1] - int(10 * scale_factor)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale_factor, (0, 255, 0), 1, cv2.LINE_AA)

                # Tail marker (Gold, with label T)
                cv2.circle(frame, tail_pt, int(7 * scale_factor), (0, 215, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, tail_pt, int(9 * scale_factor), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "T", (tail_pt[0] - 4, tail_pt[1] - int(10 * scale_factor)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale_factor, (0, 215, 255), 1, cv2.LINE_AA)

                # Trucks markers (Front Truck: Magenta, Rear Truck: Cyan)
                if ft_pt is not None:
                    cv2.circle(frame, ft_pt, int(5 * scale_factor), (255, 0, 255), -1, cv2.LINE_AA)
                if rt_pt is not None:
                    cv2.circle(frame, rt_pt, int(5 * scale_factor), (255, 255, 0), -1, cv2.LINE_AA)

                # Centroid crosshair
                if not np.isnan(r['board_centroid_x']) and not np.isnan(r['board_centroid_y']):
                    c_pt = (int(r['board_centroid_x']), int(r['board_centroid_y']))
                    c_size = int(6 * scale_factor)
                    cv2.line(frame, (c_pt[0] - c_size, c_pt[1]), (c_pt[0] + c_size, c_pt[1]), (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.line(frame, (c_pt[0], c_pt[1] - c_size), (c_pt[0], c_pt[1] + c_size), (255, 255, 255), 1, cv2.LINE_AA)

            # 2. Skater Lower Body Kinematics
            if not np.isnan(r['mid_hip_x']) and not np.isnan(r['mid_hip_y']):
                hip = (int(r['mid_hip_x']), int(r['mid_hip_y']))
                cv2.circle(frame, hip, int(7 * scale_factor), (0, 255, 120), -1, cv2.LINE_AA)

                if not np.isnan(r['left_ankle_x']):
                    la = (int(r['left_ankle_x']), int(r['left_ankle_y']))
                    cv2.circle(frame, la, int(6 * scale_factor), (255, 200, 0), -1, cv2.LINE_AA)
                    cv2.line(frame, hip, la, (255, 200, 0), max(1, int(2 * scale_factor)), cv2.LINE_AA)
                    cv2.putText(frame, "L_Ankle", (la[0] + 5, la[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * scale_factor, (255, 200, 0), 1, cv2.LINE_AA)

                if not np.isnan(r['right_ankle_x']):
                    ra = (int(r['right_ankle_x']), int(r['right_ankle_y']))
                    cv2.circle(frame, ra, int(6 * scale_factor), (0, 140, 255), -1, cv2.LINE_AA)
                    cv2.line(frame, hip, ra, (0, 140, 255), max(1, int(2 * scale_factor)), cv2.LINE_AA)
                    cv2.putText(frame, "R_Ankle", (ra[0] + 5, ra[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * scale_factor, (0, 140, 255), 1, cv2.LINE_AA)

            # 3. HUD Overlay Panel (Semi-transparent dark glass)
            overlay = frame.copy()
            x1, y1 = int(20 * scale_factor), int(20 * scale_factor)
            x2, y2 = x1 + hud_w, y1 + hud_h

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (12, 14, 18), -1)
            cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
            # Glowing border
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), max(1, int(2 * scale_factor)), cv2.LINE_AA)

            # HUD Line 1: Header
            ty = y1 + int(24 * scale_factor)
            cv2.putText(frame, "SKATEKINE TELEMETRY [PHASE 1]", (x1 + int(15 * scale_factor), ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.05, (0, 255, 255), font_thickness, cv2.LINE_AA)

            # HUD Line 2: Trick & Skater Info
            ty += int(25 * scale_factor)
            trick_str = str(row['trick_name']).upper()
            skater_str = str(row['skater_id']).upper()
            outcome_str = str(row['outcome']).upper()
            outcome_color = (0, 255, 120) if outcome_str == 'LAND' else (0, 100, 255)
            cv2.putText(frame, f"TRICK: {trick_str} | {skater_str}", (x1 + int(15 * scale_factor), ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.95, (255, 255, 255), font_thickness, cv2.LINE_AA)

            # HUD Line 3: Frame Counter & Timestamp
            ty += int(24 * scale_factor)
            t_sec = f_idx / raw_fps
            cv2.putText(frame, f"FRAME: {f_idx:03d}/{total_frames:03d} | TIME: {t_sec:5.2f}s | {raw_fps:.0f} FPS",
                        (x1 + int(15 * scale_factor), ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.88, (200, 200, 200), 1, cv2.LINE_AA)

            # HUD Line 4: Tracking Provenance Pill [G5]
            ty += int(25 * scale_factor)
            prov_map = {0: 'DETECTED', 1: 'OPTICAL_FLOW', 2: 'KALMAN_PRED', -1: 'LOST'}
            prov_code = int(r['tracking_source'])
            prov_str = prov_map.get(prov_code, 'LOST')
            prov_colors = {
                0: (0, 255, 0),      # Bright Green
                1: (0, 165, 255),    # Amber
                2: (255, 220, 0),    # Cyan/Blue
                -1: (0, 0, 255)      # Red
            }
            p_col = prov_colors.get(prov_code, (0, 0, 255))
            cv2.putText(frame, f"PROVENANCE: {prov_str} [G5]", (x1 + int(15 * scale_factor), ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.92, p_col, font_thickness, cv2.LINE_AA)

            # HUD Line 5: Angular Velocity & Scale [G3, G4]
            ty += int(24 * scale_factor)
            omega_val = r['omega_rad_per_sec']
            omega_str = f"{omega_val:+.1f} rad/s" if not np.isnan(omega_val) else "0.0 rad/s"
            scale_val = r['scale_cm_per_px']
            scale_valid = bool(r['scale_valid']) if not np.isnan(r['scale_valid']) else False
            scale_str = f"{scale_val:.3f} cm/px" if not np.isnan(scale_val) else "Torso Norm"
            scale_status = "VALID [G3]" if scale_valid else "FALLBACK"
            cv2.putText(frame, f"OMEGA: {omega_str} | SCALE: {scale_str} ({scale_status})",
                        (x1 + int(15 * scale_factor), ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.80, (230, 230, 230), 1, cv2.LINE_AA)

            # HUD Line 6: Torso Normalized Jerk [G6]
            ty += int(23 * scale_factor)
            jerk_val = r['norm_jerk_torso']
            jerk_str = f"{jerk_val:.4f}" if not np.isnan(jerk_val) else "N/A"
            cv2.putText(frame, f"TORSO JERK: {jerk_str} [G6] | STATUS: {outcome_str}",
                        (x1 + int(15 * scale_factor), ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.80, outcome_color, 1, cv2.LINE_AA)

        # Check if current frame is apex keyframe
        if extract_keyframe and (f_idx == apex_frame_idx or apex_keyframe is None):
            apex_keyframe = frame.copy()

        out.write(frame)
        f_idx += 1

    cap.release()
    out.release()

    meta = {
        'clip_id': clip_id,
        'trick_name': str(row['trick_name']),
        'skater_id': str(row['skater_id']),
        'outcome': str(row['outcome']),
        'total_frames': total_frames,
        'rendered_frames': f_idx,
        'out_fps': out_fps,
        'out_path': out_path
    }
    return out_path, apex_keyframe, meta


def render_all_pilot_clips(output_dir: str = 'outputs/telemetry_videos') -> List[Dict]:
    """
    Renders telemetry videos for all 25 pilot benchmark clips and generates
    a 25-clip apex keyframe contact sheet for rapid visual inspection.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('docs/visualizations', exist_ok=True)

    with open('docs/reports/pilot_benchmark_results.json', 'r') as f:
        bench_data = json.load(f)

    pilot_clips = bench_data['detailed_clip_results']
    print(f"============================================================")
    print(f"BATCH TELEMETRY RENDERER: Rendering all {len(pilot_clips)} pilot clips")
    print(f"Output directory: {output_dir}")
    print(f"============================================================")

    results = []
    apex_thumbnails = []

    for i, clip_info in enumerate(pilot_clips):
        cid = clip_info['clip_id']
        t_name = clip_info['trick_name']
        outcome = clip_info['outcome']
        print(f"[{i+1:02d}/{len(pilot_clips):02d}] Rendering {cid} ({t_name} | {outcome})...", end='', flush=True)

        try:
            video_path, keyframe, meta = render_telemetry_video(cid, output_dir=output_dir, extract_keyframe=True)
            results.append(meta)
            if keyframe is not None:
                # Resize keyframe thumbnail to standard 640x360 for contact sheet
                thumb = cv2.resize(keyframe, (640, 360), interpolation=cv2.INTER_AREA)
                apex_thumbnails.append((cid, t_name, outcome, thumb))
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            print(f" Done ({file_size_mb:.2f} MB, {meta['rendered_frames']} frames)")
        except Exception as e:
            print(f" FAILED: {e}")

    # Generate 5x5 High-Resolution Contact Sheet Grid (3200 x 1800 px)
    if len(apex_thumbnails) > 0:
        print("\nGenerating 25-Clip Apex Telemetry Contact Sheet Grid...")
        grid_cols = 5
        grid_rows = int(np.ceil(len(apex_thumbnails) / grid_cols))
        thumb_w, thumb_h = 640, 360
        grid_img = np.zeros((grid_rows * thumb_h, grid_cols * thumb_w, 3), dtype=np.uint8)

        for idx, (cid, t_name, outcome, thumb) in enumerate(apex_thumbnails):
            r = idx // grid_cols
            c = idx % grid_cols
            y_start = r * thumb_h
            x_start = c * thumb_w
            grid_img[y_start:y_start+thumb_h, x_start:x_start+thumb_w] = thumb

            # Label banner at top of thumbnail
            banner_col = (0, 160, 40) if outcome == 'land' else (0, 40, 180)
            cv2.rectangle(grid_img, (x_start, y_start), (x_start + thumb_w, y_start + 32), (20, 20, 20), -1)
            cv2.rectangle(grid_img, (x_start, y_start), (x_start + 6, y_start + 32), banner_col, -1)
            label_text = f"#{idx+1:02d} {cid} | {t_name.upper()} [{outcome.upper()}]"
            cv2.putText(grid_img, label_text, (x_start + 14, y_start + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
            # Grid border
            cv2.rectangle(grid_img, (x_start, y_start), (x_start + thumb_w, y_start + thumb_h), (80, 80, 80), 1)

        contact_path = 'docs/visualizations/telemetry_25_clips_grid.png'
        cv2.imwrite(contact_path, grid_img)
        print(f"Saved 25-clip contact sheet to: {contact_path}")

    print(f"\nSuccessfully rendered {len(results)}/{len(pilot_clips)} telemetry videos.")
    return results


if __name__ == '__main__':
    render_all_pilot_clips()

