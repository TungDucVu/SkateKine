"""
Video Telemetry Renderer: Renders full video clip with burned HUD overlay and motion trails.
Outputs: outputs/{clip_id}_telemetry.mp4
"""

import os
import cv2
import numpy as np
import pandas as pd


def render_telemetry_video(clip_id: str = 'batb_0005', output_dir: str = 'outputs'):
    os.makedirs(output_dir, exist_ok=True)
    df_manifest = pd.read_csv('data/metadata/video_manifest.csv')
    row = df_manifest[df_manifest['clip_id'] == clip_id].iloc[0]
    video_path = row['filepath'].replace('/', os.sep)
    parquet_path = f'features/v1_trajectories/{clip_id}.parquet'

    df_traj = pd.read_parquet(parquet_path)
    cap = cv2.VideoCapture(video_path)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = os.path.join(output_dir, f'{clip_id}_telemetry.mp4').replace(os.sep, '/')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    trail_nose = []
    trail_tail = []
    trail_len = 15

    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if f_idx < len(df_traj):
            r = df_traj.iloc[f_idx]

            # 1. Update and draw motion trails
            if not np.isnan(r['board_nose_x']) and not np.isnan(r['board_tail_x']):
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
                    thickness = max(1, int(3 * alpha))
                    cv2.line(frame, trail_nose[t-1], trail_nose[t], (0, int(255 * alpha), int(100 * alpha)), thickness, cv2.LINE_AA)
                    cv2.line(frame, trail_tail[t-1], trail_tail[t], (0, int(200 * alpha), int(255 * alpha)), thickness, cv2.LINE_AA)

                # Draw 8-point board wireframe
                ft_pt = (int(r['board_ft_x']), int(r['board_ft_y']))
                rt_pt = (int(r['board_rt_x']), int(r['board_rt_y']))

                # Deck spine
                cv2.line(frame, nose_pt, tail_pt, (0, 0, 255), 3, cv2.LINE_AA)
                # Nose & Tail markers
                cv2.circle(frame, nose_pt, 6, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, tail_pt, 6, (0, 255, 255), -1, cv2.LINE_AA)
                # Trucks
                cv2.circle(frame, ft_pt, 5, (255, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, rt_pt, 5, (255, 0, 255), -1, cv2.LINE_AA)

            # 2. Draw Skater Legs
            if not np.isnan(r['left_ankle_x']):
                la = (int(r['left_ankle_x']), int(r['left_ankle_y']))
                ra = (int(r['right_ankle_x']), int(r['right_ankle_y']))
                hip = (int(r['mid_hip_x']), int(r['mid_hip_y']))
                cv2.circle(frame, la, 6, (255, 200, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, ra, 6, (255, 100, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, hip, 7, (0, 255, 100), -1, cv2.LINE_AA)
                cv2.line(frame, hip, la, (255, 200, 0), 2, cv2.LINE_AA)
                cv2.line(frame, hip, ra, (255, 100, 0), 2, cv2.LINE_AA)

            # 3. HUD Overlay Card (Top Left)
            cv2.rectangle(frame, (20, 20), (420, 160), (15, 15, 15), -1)
            cv2.rectangle(frame, (20, 20), (420, 160), (0, 255, 255), 2)
            
            cv2.putText(frame, f"SKATEKINE HUD [PHASE 1]", (35, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"TRICK: {row['trick_name'].upper()} | {row['skater_id'].upper()}", (35, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"FRAME: {f_idx:03d}/{total_frames:03d} | TIME: {f_idx/fps:.2f}s", (35, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

            prov_map = {0: 'DETECTED', 1: 'OPTICAL_FLOW', 2: 'KALMAN_PRED', -1: 'LOST'}
            prov_str = prov_map.get(int(r['tracking_source']), 'UNKNOWN')
            prov_color = (0, 255, 0) if r['tracking_source'] == 0 else (0, 165, 255)
            cv2.putText(frame, f"PROVENANCE: {prov_str} [G5]", (35, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.52, prov_color, 2, cv2.LINE_AA)
            
            omega_val = r['omega_rad_per_sec']
            omega_str = f"{omega_val:.1f} rad/s" if not np.isnan(omega_val) else "N/A"
            scale_val = r['scale_cm_per_px']
            scale_str = f"{scale_val:.3f} cm/px" if not np.isnan(scale_val) else "torso-ratio"
            cv2.putText(frame, f"OMEGA: {omega_str} | SCALE: {scale_str}", (35, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

        out.write(frame)
        f_idx += 1

    cap.release()
    out.release()
    print(f'Rendered full telemetry video to: {out_path}')
    return out_path


if __name__ == '__main__':
    render_telemetry_video('batb_0005')
