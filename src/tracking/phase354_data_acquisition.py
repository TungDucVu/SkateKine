"""
Phase 3.5.4 Automated Data Acquisition & Ingestion Pipeline.
Acquires independent multi-skater clips for the four bottleneck flatground basics:
- Ollie
- Backside 180
- Frontside 180
- Pop Shove-it

Downloads clips via yt-dlp, trims & cleans with ffmpeg (h264/yuv420p/30fps),
registers in video_manifest.csv, and runs Phase1Pipeline tracking.
"""

import os
import sys
import subprocess
import cv2
import pandas as pd
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.tracking.phase1_pipeline import Phase1Pipeline


EXPANSION_TARGETS = [
    # 1. Ollie
    {
        "clip_id": "ext_ollie_mitchie_01",
        "video_url": "https://www.youtube.com/watch?v=arS7-YTrYA4",
        "ss": "00:36:46",
        "duration": "00:00:04",
        "skater_id": "mitchie_brusco",
        "stance": "regular",
        "trick_name": "ollie",
        "source": "skateiq"
    },
    {
        "clip_id": "ext_ollie_spencer_01",
        "video_url": "https://www.youtube.com/watch?v=XRqetLJBE6I",
        "ss": "00:01:25",
        "duration": "00:00:04",
        "skater_id": "spencer_nuzzi",
        "stance": "regular",
        "trick_name": "ollie",
        "source": "spencer_nuzzi"
    },

    # 2. Frontside 180
    {
        "clip_id": "ext_fs180_mitchie_01",
        "video_url": "https://www.youtube.com/watch?v=FaoSG1P-AOA",
        "ss": "00:01:10",
        "duration": "00:00:04",
        "skater_id": "mitchie_brusco",
        "stance": "regular",
        "trick_name": "frontside_180",
        "source": "skateiq"
    },
    {
        "clip_id": "ext_fs180_aaron_01",
        "video_url": "https://www.youtube.com/watch?v=ZSe9vPoXKiU",
        "ss": "00:01:15",
        "duration": "00:00:04",
        "skater_id": "aaron_kyro",
        "stance": "regular",
        "trick_name": "frontside_180",
        "source": "braille"
    },
    {
        "clip_id": "ext_fs180_spencer_01",
        "video_url": "https://www.youtube.com/watch?v=Ize_UuHKxJ4",
        "ss": "00:00:50",
        "duration": "00:00:04",
        "skater_id": "spencer_nuzzi",
        "stance": "regular",
        "trick_name": "frontside_180",
        "source": "spencer_nuzzi"
    },

    # 3. Backside 180
    {
        "clip_id": "ext_bs180_mitchie_01",
        "video_url": "https://www.youtube.com/watch?v=kYJ_b1Q5i4Y",
        "ss": "00:01:05",
        "duration": "00:00:04",
        "skater_id": "mitchie_brusco",
        "stance": "regular",
        "trick_name": "backside_180",
        "source": "skateiq"
    },
    {
        "clip_id": "ext_bs180_aaron_01",
        "video_url": "https://www.youtube.com/watch?v=C4pNwV2loW4",
        "ss": "00:00:45",
        "duration": "00:00:04",
        "skater_id": "aaron_kyro",
        "stance": "regular",
        "trick_name": "backside_180",
        "source": "braille"
    },
    {
        "clip_id": "ext_bs180_spencer_01",
        "video_url": "https://www.youtube.com/watch?v=1q4cS1yH2II",
        "ss": "00:00:40",
        "duration": "00:00:04",
        "skater_id": "spencer_nuzzi",
        "stance": "regular",
        "trick_name": "backside_180",
        "source": "spencer_nuzzi"
    },

    # 4. Pop Shove-it
    {
        "clip_id": "ext_popshov_spencer_01",
        "video_url": "https://www.youtube.com/watch?v=JYuwXjlTqZw",
        "ss": "00:00:35",
        "duration": "00:00:04",
        "skater_id": "spencer_nuzzi",
        "stance": "regular",
        "trick_name": "pop_shuvit",
        "source": "spencer_nuzzi"
    },
    {
        "clip_id": "ext_popshov_whytrick_01",
        "video_url": "https://www.youtube.com/watch?v=V7na_ALn2vA",
        "ss": "00:00:15",
        "duration": "00:00:04",
        "skater_id": "whythetrick_skater",
        "stance": "regular",
        "trick_name": "pop_shuvit",
        "source": "whythetrick"
    }
]


def download_and_clip(target: Dict[str, Any], output_dir: str = "data/raw_videos/phase354_expansion") -> str:
    """
    Downloads raw video segment using yt-dlp and trims cleanly with ffmpeg.
    """
    os.makedirs(output_dir, exist_ok=True)
    cid = target["clip_id"]
    final_mp4 = os.path.join(output_dir, f"{cid}.mp4")

    if os.path.exists(final_mp4):
        print(f"[{cid}] Clean video already exists: {final_mp4}")
        return final_mp4

    temp_raw = os.path.join(output_dir, f"temp_{cid}.mp4")

    # 1. Download segment via yt-dlp
    print(f"[{cid}] Downloading from {target['source']} ({target['video_url']})...")
    # yt-dlp with android client
    cmd_dl = [
        "yt-dlp",
        "--no-update",
        "--extractor-args", "youtube:player_client=android",
        "-f", "18/best[height<=720]/best",
        "-o", temp_raw,
        target["video_url"]
    ]
    try:
        subprocess.run(cmd_dl, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"[{cid}] yt-dlp download failed: {e.stderr.decode('utf-8', errors='ignore')[:200]}")
        return ""

    if not os.path.exists(temp_raw):
        print(f"[{cid}] Temp raw file not found.")
        return ""

    # 2. Trim and normalize with ffmpeg
    print(f"[{cid}] Trimming with ffmpeg at {target['ss']} for {target['duration']}...")
    cmd_ff = [
        "ffmpeg",
        "-y",
        "-ss", target["ss"],
        "-i", temp_raw,
        "-t", target["duration"],
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-r", "30",
        "-an",
        final_mp4
    ]
    try:
        subprocess.run(cmd_ff, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"[{cid}] ffmpeg trim failed: {e.stderr.decode('utf-8', errors='ignore')[:200]}")
        return ""
    finally:
        if os.path.exists(temp_raw):
            try: os.remove(temp_raw)
            except: pass

    # Verify OpenCV readability
    cap = cv2.VideoCapture(final_mp4)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if n < 15:
        print(f"[{cid}] Error: Clean clip has only {n} frames.")
        return ""

    print(f"[{cid}] Success: {w:.0f}x{h:.0f} @ {fps:.1f} fps, {n} frames.")
    return final_mp4


def update_manifest_with_new_clips(new_records: List[Dict[str, Any]], manifest_path: str = "data/metadata/video_manifest.csv"):
    """
    Registers new external expansion clips into video_manifest.csv.
    """
    df_manifest = pd.read_csv(manifest_path)
    existing_ids = set(df_manifest['clip_id'])

    rows_to_add = []
    for r in new_records:
        cid = r['clip_id']
        if cid in existing_ids:
            continue
        rows_to_add.append({
            'clip_id': cid,
            'source_dataset': 'external_expansion_p354',
            'filepath': r['filepath'],
            'skater_id': r['skater_id'],
            'stance': r['stance'],
            'original_label': r['trick_name'],
            'trick_name': r['trick_name'],
            'category': 'flatground',
            'phase1_eligible': True,
            'outcome': 'land',
            'outcome_source': 'verified_tutorial',
            'width': r['width'],
            'height': r['height'],
            'fps': r['fps'],
            'total_frames': r['total_frames'],
            'duration_sec': r['duration_sec'],
            'quality_flag': 'valid',
            'split': 'expansion',
            'tracking_eligible': True,
            'temporal_high_speed': False,
            'pilot_set': False
        })

    if len(rows_to_add) > 0:
        df_new = pd.DataFrame(rows_to_add)
        df_updated = pd.concat([df_manifest, df_new], ignore_index=True)
        df_updated.to_csv(manifest_path, index=False)
        print(f"Updated {manifest_path} with {len(rows_to_add)} new expansion clips. Total manifest rows: {len(df_updated)}.")
    else:
        print("No new manifest rows needed (all clips already registered).")


def run_phase354_pipeline():
    """
    Orchestrates the Phase 3.5.4 acquisition, registration, and tracking.
    """
    print("=" * 80)
    print("PHASE 3.5.4 CROSS-SKATER DATA ACQUISITION & INGESTION")
    print("=" * 80)

    downloaded = []
    for target in EXPANSION_TARGETS:
        cid = target["clip_id"]
        vpath = download_and_clip(target)
        if vpath and os.path.exists(vpath):
            cap = cv2.VideoCapture(vpath)
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            downloaded.append({
                'clip_id': cid,
                'filepath': vpath,
                'skater_id': target['skater_id'],
                'stance': target['stance'],
                'trick_name': target['trick_name'],
                'width': w,
                'height': h,
                'fps': fps,
                'total_frames': n,
                'duration_sec': float(n / max(1, fps))
            })

    print(f"\nDownloaded & verified {len(downloaded)} / {len(EXPANSION_TARGETS)} external clips.")

    # Update manifest
    update_manifest_with_new_clips(downloaded)

    # Track with Phase1Pipeline
    print("\n" + "=" * 80)
    print("MATERIALIZING TRAJECTORIES VIA Phase1Pipeline")
    print("=" * 80)

    p1 = Phase1Pipeline()
    tracked_count = 0

    for rec in downloaded:
        cid = rec['clip_id']
        fp = rec['filepath']
        parquet_file = os.path.join("features/v1_trajectories", f"{cid}.parquet")

        if os.path.exists(parquet_file):
            print(f"[{cid}] Parquet already exists: {parquet_file}")
            tracked_count += 1
            continue

        print(f"[{cid}] Tracking {fp} -> {parquet_file}...")
        try:
            res = p1.process_clip(fp, clip_id=cid, output_dir="features/v1_trajectories")
            print(f"  -> Success: {res.processed_frames} frames, skater det: {res.skater_detection_rate*100:.1f}%, board det: {res.board_detection_rate*100:.1f}%")
            tracked_count += 1
        except Exception as e:
            print(f"  -> Tracking failed for {cid}: {e}")

    print("=" * 80)
    print(f"PHASE 3.5.4 DATA INGESTION COMPLETE: {tracked_count} trajectories ready in features/v1_trajectories")
    print("=" * 80)


if __name__ == "__main__":
    run_phase354_pipeline()
