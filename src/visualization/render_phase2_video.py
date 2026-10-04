"""
Phase 2 Video Telemetry & Temporal Phase Renderer:
Renders full video clips with burned Phase 2 temporal phase indicators,
interactive timeline scrubbers, event milestone badges (Pop, Apex, Catch, Land),
and vertical ballistic altitude meters.
Outputs: outputs/phase2_videos/{clip_id}_phase2.mp4
"""

import os
import sys
import cv2
import json
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.segmentation.phase2_pipeline import Phase2Pipeline


def render_phase2_video(
    clip_id: str,
    output_dir: str = 'outputs/phase2_videos',
    pipeline: Optional[Phase2Pipeline] = None
) -> Tuple[str, Dict[str, np.ndarray], Dict]:
    """
    Renders video for a single clip with Phase 2 temporal event and phase overlays.
    Returns: (output_path, dict_of_event_keyframes, clip_metadata)
    """
    os.makedirs(output_dir, exist_ok=True)
    if pipeline is None:
        pipeline = Phase2Pipeline()

    df_manifest = pd.read_csv('data/metadata/video_manifest.csv')
    matched = df_manifest[df_manifest['clip_id'] == clip_id]
    if len(matched) == 0:
        raise ValueError(f"Clip {clip_id} not found in manifest.")
    row = matched.iloc[0]

    video_path = row['filepath'].replace('/', os.sep)
    out_p2 = pipeline.process_clip_by_id(clip_id)
    ev = out_p2.events
    df_cond = out_p2.conditioned_df

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_fps = float(cap.get(cv2.CAP_PROP_FPS))
    out_fps = float(max(1, min(120, round(raw_fps))))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = os.path.join(output_dir, f'{clip_id}_phase2.mp4').replace(os.sep, '/')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, out_fps, (w, h))

    if not out.isOpened():
        raise RuntimeError(f"VideoWriter failed to open for {out_path} at {out_fps} FPS")

    # Scaling parameters
    scale = max(0.65, min(1.6, w / 1280.0))
    hud_w = int(490 * scale)
    hud_h = int(200 * scale)
    font_scale = 0.50 * scale
    thick = max(1, int(1.5 * scale))

    t_pop = ev.t_pop
    t_apex = ev.t_apex
    t_catch = ev.t_catch if ev.t_catch is not None else ev.t_land - 2
    t_land = ev.t_land

    trail_nose = []
    trail_tail = []
    trail_len = int(14 * (raw_fps / 60.0))

    event_keyframes = {}
    max_altitude = max(1.0, float(df_cond['deck_altitude'].max()))

    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if f_idx < len(df_cond):
            r = df_cond.iloc[f_idx]

            # -------------------------------------------------------------------------
            # 1. Determine Current Temporal Phase
            # -------------------------------------------------------------------------
            if f_idx < t_pop:
                phase_name = "APPROACH"
                phase_color = (0, 165, 255)       # Amber
                phase_desc = "Rolling Approach (Pre-Pop)"
            elif f_idx == t_pop:
                phase_name = "EVENT: POP SNAP"
                phase_color = (0, 0, 255)         # Bright Red
                phase_desc = "Impulse Pop Initiation [t_pop]"
            elif f_idx < t_apex:
                phase_name = "FLIGHT: ASCENT"
                phase_color = (255, 190, 0)       # Cyan/Blue
                phase_desc = "Ballistic Ascent & Rotation"
            elif f_idx == t_apex:
                phase_name = "EVENT: APEX PEAK"
                phase_color = (255, 255, 0)       # Cyan
                phase_desc = "Zero-Crossing Apex [t_apex]"
            elif f_idx < t_catch:
                phase_name = "FLIGHT: DESCENT"
                phase_color = (255, 100, 150)     # Sky Blue
                phase_desc = "Descent & Flip Completion"
            elif f_idx == t_catch:
                phase_name = "EVENT: FEET CATCH"
                phase_color = (255, 0, 255)       # Magenta/Purple
                phase_desc = "Persistent Foot Catch [t_catch]"
            elif f_idx < t_land:
                phase_name = "PRE-LANDING"
                phase_color = (200, 50, 255)      # Violet
                phase_desc = "Final Touchdown Approach"
            elif f_idx == t_land:
                phase_name = "EVENT: TOUCHDOWN"
                phase_color = (0, 255, 0)         # Bright Green
                phase_desc = "Ground Impact Shockwave [t_land]"
            else:
                phase_name = "ROLLOUT"
                phase_color = (0, 220, 100)       # Emerald Green
                phase_desc = "Post-Landing Compression & Rollout"

            # -------------------------------------------------------------------------
            # 2. Board Wireframe & Motion Trails
            # -------------------------------------------------------------------------
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

                for t_i in range(1, len(trail_nose)):
                    alpha_trail = t_i / float(len(trail_nose))
                    t_thick = max(1, int(3 * alpha_trail * scale))
                    cv2.line(frame, trail_nose[t_i-1], trail_nose[t_i], (0, int(255 * alpha_trail), int(120 * alpha_trail)), t_thick, cv2.LINE_AA)
                    cv2.line(frame, trail_tail[t_i-1], trail_tail[t_i], (0, int(200 * alpha_trail), int(255 * alpha_trail)), t_thick, cv2.LINE_AA)

                # Spine & Markers
                cv2.line(frame, tail_pt, nose_pt, (0, 0, 240), max(2, int(3 * scale)), cv2.LINE_AA)
                cv2.circle(frame, nose_pt, int(7 * scale), (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, tail_pt, int(7 * scale), (0, 215, 255), -1, cv2.LINE_AA)

                if not np.isnan(r['board_ft_x']):
                    cv2.circle(frame, (int(r['board_ft_x']), int(r['board_ft_y'])), int(5 * scale), (255, 0, 255), -1, cv2.LINE_AA)
                if not np.isnan(r['board_rt_x']):
                    cv2.circle(frame, (int(r['board_rt_x']), int(r['board_rt_y'])), int(5 * scale), (255, 255, 0), -1, cv2.LINE_AA)

            # Skater limbs
            if not np.isnan(r['mid_hip_x']) and not np.isnan(r['left_ankle_x']) and not np.isnan(r['right_ankle_x']):
                hip = (int(r['mid_hip_x']), int(r['mid_hip_y']))
                la = (int(r['left_ankle_x']), int(r['left_ankle_y']))
                ra = (int(r['right_ankle_x']), int(r['right_ankle_y']))
                cv2.circle(frame, hip, int(6 * scale), (0, 255, 120), -1, cv2.LINE_AA)
                cv2.circle(frame, la, int(5 * scale), (255, 200, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, ra, int(5 * scale), (0, 140, 255), -1, cv2.LINE_AA)
                cv2.line(frame, hip, la, (255, 200, 0), max(1, int(2 * scale)), cv2.LINE_AA)
                cv2.line(frame, hip, ra, (0, 140, 255), max(1, int(2 * scale)), cv2.LINE_AA)

            # -------------------------------------------------------------------------
            # 3. Main HUD Overlay Panel (Top Left)
            # -------------------------------------------------------------------------
            overlay = frame.copy()
            x1, y1 = int(20 * scale), int(20 * scale)
            x2, y2 = x1 + hud_w, y1 + hud_h
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (12, 14, 18), -1)
            cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), max(1, int(2 * scale)), cv2.LINE_AA)

            ty = y1 + int(24 * scale)
            cv2.putText(frame, "SKATEKINE [PHASE 2: EVENT SEGMENTATION]", (x1 + int(14 * scale), ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.02, (0, 255, 255), thick, cv2.LINE_AA)

            ty += int(26 * scale)
            t_name = str(row['trick_name']).upper()
            skater = str(row['skater_id']).upper()
            outcome = str(row['outcome']).upper()
            out_col = (0, 255, 120) if outcome == 'LAND' else (0, 80, 255)
            cv2.putText(frame, f"TRICK: {t_name} | {skater}", (x1 + int(14 * scale), ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.95, (255, 255, 255), thick, cv2.LINE_AA)

            ty += int(25 * scale)
            t_sec = f_idx / raw_fps
            cv2.putText(frame, f"FRAME: {f_idx:03d}/{total_frames:03d} | TIME: {t_sec:5.2f}s | {raw_fps:.0f} FPS",
                        (x1 + int(14 * scale), ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, (200, 200, 200), 1, cv2.LINE_AA)

            ty += int(27 * scale)
            # Active Phase Indicator Pill
            cv2.rectangle(frame, (x1 + int(12 * scale), ty - int(16 * scale)), (x1 + int(470 * scale), ty + int(7 * scale)), (25, 28, 35), -1)
            cv2.rectangle(frame, (x1 + int(12 * scale), ty - int(16 * scale)), (x1 + int(470 * scale), ty + int(7 * scale)), phase_color, 1)
            cv2.putText(frame, f"PHASE: [{phase_name}]", (x1 + int(20 * scale), ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.95, phase_color, thick, cv2.LINE_AA)

            ty += int(27 * scale)
            flt_ms = ev.flight_duration_ms
            cv2.putText(frame, f"AIRTIME: {flt_ms:.0f} ms | RESULT: {outcome}",
                        (x1 + int(14 * scale), ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, out_col, thick, cv2.LINE_AA)

            ty += int(25 * scale)
            catch_lbl = f"f{ev.t_catch}" if ev.t_catch is not None else "N/A"
            cv2.putText(frame, f"POP: f{t_pop} | APEX: f{t_apex} | CATCH: {catch_lbl} | LAND: f{t_land}",
                        (x1 + int(14 * scale), ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.78, (220, 220, 220), 1, cv2.LINE_AA)

            # -------------------------------------------------------------------------
            # 4. Interactive Biomechanical Timeline Scrubber (Bottom of Frame)
            # -------------------------------------------------------------------------
            bar_margin = int(40 * scale)
            bar_y = h - int(45 * scale)
            bar_h = int(14 * scale)
            bar_w = w - (2 * bar_margin)

            # Background bar
            cv2.rectangle(frame, (bar_margin - 4, bar_y - 22), (bar_margin + bar_w + 4, bar_y + bar_h + 18), (15, 17, 22), -1)
            cv2.rectangle(frame, (bar_margin - 4, bar_y - 22), (bar_margin + bar_w + 4, bar_y + bar_h + 18), (60, 65, 75), 1)

            # Phase color segments
            x_pop = bar_margin + int(bar_w * (t_pop / max(1, total_frames)))
            x_apex = bar_margin + int(bar_w * (t_apex / max(1, total_frames)))
            x_catch = bar_margin + int(bar_w * (t_catch / max(1, total_frames)))
            x_land = bar_margin + int(bar_w * (t_land / max(1, total_frames)))

            # 1. Approach segment (bar_margin to x_pop)
            cv2.rectangle(frame, (bar_margin, bar_y), (x_pop, bar_y + bar_h), (0, 140, 220), -1)
            # 2. Flight segment (x_pop to x_land)
            cv2.rectangle(frame, (x_pop, bar_y), (x_land, bar_y + bar_h), (255, 160, 0), -1)
            # 3. Rollout segment (x_land to end)
            cv2.rectangle(frame, (x_land, bar_y), (bar_margin + bar_w, bar_y + bar_h), (0, 200, 80), -1)

            # Event tick marks & labels
            events_to_mark = [
                (x_pop, "POP", (0, 0, 255)),
                (x_apex, "APEX", (255, 255, 0)),
                (x_catch, "CATCH", (255, 0, 255)),
                (x_land, "LAND", (0, 255, 0))
            ]
            for ev_x, ev_label, ev_c in events_to_mark:
                cv2.line(frame, (ev_x, bar_y - 5), (ev_x, bar_y + bar_h + 5), ev_c, max(1, int(2 * scale)), cv2.LINE_AA)
                cv2.putText(frame, ev_label, (ev_x - int(12 * scale), bar_y - int(8 * scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38 * scale, ev_c, 1, cv2.LINE_AA)

            # Live needle playhead
            playhead_x = bar_margin + int(bar_w * (f_idx / max(1, total_frames)))
            cv2.line(frame, (playhead_x, bar_y - 12), (playhead_x, bar_y + bar_h + 12), (255, 255, 255), max(2, int(2.5 * scale)), cv2.LINE_AA)
            cv2.circle(frame, (playhead_x, bar_y + bar_h // 2), int(5 * scale), (255, 255, 255), -1, cv2.LINE_AA)

            # -------------------------------------------------------------------------
            # 5. Vertical Ballistic Altitude Meter (Right Edge)
            # -------------------------------------------------------------------------
            gauge_w = int(22 * scale)
            gauge_h = int(180 * scale)
            gx1 = w - int(45 * scale)
            gy1 = int(70 * scale)
            gx2 = gx1 + gauge_w
            gy2 = gy1 + gauge_h

            # Gauge background
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (20, 22, 28), -1)
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (100, 105, 115), 1)
            cv2.putText(frame, "ALT", (gx1, gy1 - int(8 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.40 * scale, (200, 200, 200), 1, cv2.LINE_AA)

            cur_alt = float(r['deck_altitude']) if not np.isnan(r['deck_altitude']) else 0.0
            fill_pct = max(0.0, min(1.0, cur_alt / max_altitude))
            fill_h = int(gauge_h * fill_pct)
            cv2.rectangle(frame, (gx1 + 2, gy2 - fill_h), (gx2 - 2, gy2 - 1), phase_color, -1)

            # Ground baseline mark
            cv2.line(frame, (gx1 - 4, gy2), (gx2 + 4, gy2), (0, 255, 120), 2, cv2.LINE_AA)

            # -------------------------------------------------------------------------
            # 6. Save Keyframes for Event Strip Montage
            # -------------------------------------------------------------------------
            if f_idx == t_pop:
                event_keyframes['pop'] = frame.copy()
            elif f_idx == t_apex:
                event_keyframes['apex'] = frame.copy()
            elif f_idx == t_catch:
                event_keyframes['catch'] = frame.copy()
            elif f_idx == t_land:
                event_keyframes['land'] = frame.copy()

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
        'out_path': out_path,
        't_pop': t_pop,
        't_apex': t_apex,
        't_catch': t_catch,
        't_land': t_land,
        'flight_duration_ms': ev.flight_duration_ms
    }
    return out_path, event_keyframes, meta


def render_all_phase2_pilot_clips(output_dir: str = 'outputs/phase2_videos') -> List[Dict]:
    """
    Renders Phase 2 temporal event and phase segmented videos for all 25 pilot benchmark clips
    and generates a 4-event phase progression strip montage for key tricks.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('docs/visualizations', exist_ok=True)

    with open('docs/reports/pilot_benchmark_results.json', 'r') as f:
        bench_data = json.load(f)

    pilot_clips = bench_data['detailed_clip_results']
    pipeline = Phase2Pipeline()

    print("============================================================")
    print(f"PHASE 2 VIDEO RENDERER: Rendering all {len(pilot_clips)} pilot clips")
    print(f"Output directory: {output_dir}")
    print("============================================================")

    results = []
    selected_strips = []

    for i, clip_info in enumerate(pilot_clips):
        cid = clip_info['clip_id']
        t_name = clip_info['trick_name']
        outcome = clip_info['outcome']
        print(f"[{i+1:02d}/{len(pilot_clips):02d}] Rendering Phase 2 {cid} ({t_name} | {outcome})...", end='', flush=True)

        try:
            video_path, kfs, meta = render_phase2_video(cid, output_dir=output_dir, pipeline=pipeline)
            results.append(meta)
            f_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            print(f" Done ({f_size_mb:.2f} MB, {meta['rendered_frames']} frames)")

            # Collect for progression strip montage (e.g. for key iconic tricks)
            if cid in ['batb_0005', 'batb_0006', 'batb_0007', 'archive_0156', 'archive_0206']:
                selected_strips.append((cid, t_name, outcome, kfs))
        except Exception as e:
            print(f" FAILED: {e}")

    # Generate 5-Trick Phase Progression Strip Montage
    if len(selected_strips) > 0:
        print("\nGenerating Phase 2 Progression Strip Montage...")
        thumb_w, thumb_h = 320, 180
        cols = 4  # Pop, Apex, Catch, Land
        rows = len(selected_strips)
        montage = np.zeros((rows * (thumb_h + 30), cols * thumb_w, 3), dtype=np.uint8)

        for r_idx, (cid, t_name, outcome, kfs) in enumerate(selected_strips):
            y_base = r_idx * (thumb_h + 30)
            # Row header banner
            cv2.rectangle(montage, (0, y_base), (cols * thumb_w, y_base + 24), (20, 24, 30), -1)
            out_c = (0, 255, 120) if outcome == 'land' else (0, 80, 255)
            cv2.putText(montage, f"CLIP: {cid.upper()} | TRICK: {t_name.upper()} [{outcome.upper()}]", (15, y_base + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, out_c, 1, cv2.LINE_AA)

            for c_idx, ev_key in enumerate(['pop', 'apex', 'catch', 'land']):
                x_pos = c_idx * thumb_w
                y_pos = y_base + 24
                if ev_key in kfs and kfs[ev_key] is not None:
                    thumb = cv2.resize(kfs[ev_key], (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                    montage[y_pos:y_pos+thumb_h, x_pos:x_pos+thumb_w] = thumb
                # Event label header
                cv2.rectangle(montage, (x_pos, y_pos), (x_pos + 100, y_pos + 20), (10, 12, 16), -1)
                cv2.putText(montage, ev_key.upper(), (x_pos + 8, y_pos + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.rectangle(montage, (x_pos, y_pos), (x_pos + thumb_w, y_pos + thumb_h), (60, 60, 60), 1)

        strip_path = 'docs/visualizations/phase2_progression_strips.png'
        cv2.imwrite(strip_path, montage)
        print(f"Saved Phase 2 progression strip montage to: {strip_path}")

    print(f"\nSuccessfully rendered {len(results)}/{len(pilot_clips)} Phase 2 telemetry videos.")
    return results


if __name__ == '__main__':
    render_all_phase2_pilot_clips()
