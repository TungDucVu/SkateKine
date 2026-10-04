"""
SkateKine Manifest Builder (Step 0)
Scans, extracts technical video parameters, parses trick/stance metadata,
and generates the leak-proof partitioned video_manifest.csv.
"""

import os
import re
import cv2
import pandas as pd


SKATER_LOOKUP = {
    'sewa': 'sewa_kroetkov',
    'joslin': 'chris_joslin',
    'josllin': 'chris_joslin',
    'luan': 'luan_oliveira',
    'shane': 'shane_oneill',
    'pj': 'pj_ladd',
    'sean': 'sean_malto',
    'colbourn': 'jack_colbourn',
    'chris cole': 'chris_cole',
    'cole': 'chris_cole',
    'cody': 'cody_cepeda',
    'cepeda': 'cody_cepeda',
    'pudwill': 'torey_pudwill',
    'busenitz': 'dennis_busenitz',
    'fyock': 'tom_fyock',
    'tucker': 'nick_tucker',
    'nick': 'nick_tucker',
    'ishod': 'ishod_wair'
}

SPLIT_MAP = {
    # Train
    'player_a': 'train',
    'sean_malto': 'train',
    'jack_colbourn': 'train',
    'chris_cole': 'train',
    'cody_cepeda': 'train',
    'torey_pudwill': 'train',
    'dennis_busenitz': 'train',
    'tom_fyock': 'train',
    'nick_tucker': 'train',
    'ishod_wair': 'train',
    'thrasher_unknown': 'train',
    # Val
    'shane_oneill': 'val',
    'pj_ladd': 'val',
    'player_b': 'val',
    # Test
    'sewa_kroetkov': 'test',
    'chris_joslin': 'test',
    'luan_oliveira': 'test',
    # Synthetic
    'virtual_avatar': 'quarantine_synthetic'
}

# The 9 primary classes defined in MASTER_FINAL.md
CORE_9_CLASSES = {
    'ollie',
    'kickflip',
    'heelflip',
    'pop_shuvit',
    'frontside_shuvit',
    'frontside_180',
    'backside_180',
    'varial_kickflip',
    '360_flip'
}


def parse_stance(text: str, default: str = 'regular') -> str:
    t = text.lower()
    if 'nollie' in t:
        return 'nollie'
    elif 'fakie' in t or 'halfcab' in t or 'caballerial' in t:
        return 'fakie'
    elif 'switch' in t:
        return 'switch'
    return default


def parse_trick_and_category(raw_str: str) -> tuple[str, str, bool]:
    s = raw_str.lower().replace('-', ' ').replace('_', ' ')
    
    # Strip stances and descriptors from raw string for canonical trick extraction
    cleaned = re.sub(r'\b(nollie|fakie|switch|halfcab|caballerial|bs|fs|frontside|backside|bail\d*|bail)\b', ' ', s)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Matching rules
    if '360 double flip' in s:
        trick = '360_double_flip'
        cat = 'coupled_trick'
    elif '360 flip' in s or 'tre flip' in s or 'treflip' in s:
        trick = '360_flip'
        cat = 'coupled_trick'
    elif 'varial kickflip' in s:
        trick = 'varial_kickflip'
        cat = 'coupled_trick'
    elif 'varial heel' in s or 'varial heelflip' in s:
        trick = 'varial_heelflip'
        cat = 'coupled_trick'
    elif 'hardflip' in s:
        trick = 'hardflip'
        cat = 'coupled_trick'
    elif 'inward heel' in s or 'inward heelflip' in s:
        trick = 'inward_heelflip'
        cat = 'coupled_trick'
    elif 'biggerspin' in s:
        trick = 'biggerspin'
        cat = 'coupled_trick'
    elif 'bigspin' in s:
        trick = 'bigspin'
        cat = 'coupled_trick'
    elif 'big heel' in s or 'big heelflip' in s:
        trick = 'big_heelflip'
        cat = 'coupled_trick'
    elif 'pressure' in s:
        trick = 'pressureflip'
        cat = 'flip_trick'
    elif 'double kickflip' in s or 'double flip' in s:
        trick = 'double_kickflip'
        cat = 'flip_trick'
    elif 'kickflip' in s or 'kick' in s or 'flip' in s and 'heelflip' not in s:
        trick = 'kickflip'
        cat = 'flip_trick'
    elif 'heelflip' in s or 'heel' in s:
        trick = 'heelflip'
        cat = 'flip_trick'
    elif 'frontside shuv' in s or 'front shuv' in s or 'frontshuvit' in s or 'fs shuv' in s:
        trick = 'frontside_shuvit'
        cat = 'shuv_trick'
    elif '540 shuv' in s:
        trick = '540_shuvit'
        cat = 'shuv_trick'
    elif '360 shuv' in s or '3 shuv' in s:
        trick = '360_shuvit'
        cat = 'shuv_trick'
    elif 'shuvit' in s or 'shuv' in s:
        trick = 'pop_shuvit'
        cat = 'shuv_trick'
    elif 'frontside 180' in s or 'fs 180' in s or 'front180' in s:
        trick = 'frontside_180'
        cat = '180_trick'
    elif 'backside 180' in s or 'bs 180' in s or 'back180' in s:
        trick = 'backside_180'
        cat = '180_trick'
    elif 'frontside 360' in s or 'fs 360' in s:
        trick = 'frontside_360'
        cat = '180_trick'
    elif 'backside 360' in s or 'bs 360' in s:
        trick = 'backside_360'
        cat = '180_trick'
    elif 'impossible' in s:
        trick = 'impossible'
        cat = 'coupled_trick'
    elif 'lazer' in s:
        trick = 'lazer_flip'
        cat = 'coupled_trick'
    elif 'ollie' in s:
        trick = 'ollie'
        cat = 'linear'
    else:
        trick = cleaned.replace(' ', '_') if cleaned else 'unknown'
        cat = 'unsupported'

    phase1_eligible = (trick in CORE_9_CLASSES)
    return trick, cat, phase1_eligible


def extract_video_metadata(filepath: str) -> tuple[int, int, float, int, float, str]:
    if not os.path.exists(filepath):
        return 0, 0, 0.0, 0, 0.0, 'corrupted'
    
    sz = os.path.getsize(filepath)
    if sz == 0:
        return 0, 0, 0.0, 0, 0.0, 'corrupted'

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return 0, 0, 0.0, 0, 0.0, 'corrupted'

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if frames <= 0 or w <= 0 or h <= 0:
        return w, h, fps, frames, 0.0, 'corrupted'

    duration = round(frames / fps, 2) if fps > 0 else 0.0
    quality = 'pass'
    if fps < 50.0:
        quality = 'low_fps'

    return w, h, round(fps, 2), frames, duration, quality


def build_manifest(output_csv: str = 'data/metadata/video_manifest.csv') -> pd.DataFrame:
    rows = []
    
    # -------------------------------------------------------------
    # 1. Archive Highspeed Dataset (413 clips on disk)
    # -------------------------------------------------------------
    archive_dir = r'data\raw_videos\archive_highspeed'
    archive_label_file = r'data\metadata\archive_original_label3.csv'
    archive_meta = {}
    if os.path.exists(archive_label_file):
        df_arc = pd.read_csv(archive_label_file)
        for _, r in df_arc.iterrows():
            norm_rel = r['filepath'].replace('/', os.sep).strip()
            archive_meta[norm_rel.lower()] = r

    arc_idx = 1
    for root, dirs, files in os.walk(archive_dir):
        for f in files:
            if f.lower().endswith(('.mov', '.mp4')):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, archive_dir)
                norm_key = rel_path.lower()
                
                # Metadata from CSV if available, else derive from folder
                if norm_key in archive_meta:
                    meta = archive_meta[norm_key]
                    player = meta['player'].strip()
                    skater_id = 'player_a' if player == 'A' else 'player_b'
                    stance = meta['stance'].strip().lower() if pd.notna(meta.get('stance')) else 'goofy'
                    orig_label = str(meta['trick_name']).strip()
                else:
                    skater_id = 'player_a'
                    stance = 'goofy'
                    orig_label = os.path.basename(os.path.dirname(full_path))

                trick_name, category, phase1_elig = parse_trick_and_category(orig_label)
                
                w, h, fps, frames, dur, qflag = extract_video_metadata(full_path)
                split = SPLIT_MAP.get(skater_id, 'train')

                canonical_rel_path = full_path.replace(os.sep, '/')
                rows.append({
                    'clip_id': f'archive_{arc_idx:04d}',
                    'source_dataset': 'archive_highspeed',
                    'filepath': canonical_rel_path,
                    'skater_id': skater_id,
                    'stance': stance,
                    'original_label': orig_label,
                    'trick_name': trick_name,
                    'category': category,
                    'phase1_eligible': phase1_elig,
                    'outcome': 'land',
                    'outcome_source': 'label_metadata',
                    'width': w,
                    'height': h,
                    'fps': fps,
                    'total_frames': frames,
                    'duration_sec': dur,
                    'quality_flag': qflag,
                    'split': split
                })
                arc_idx += 1

    # -------------------------------------------------------------
    # 2. Thrasher BATB Dataset (598 clips)
    # -------------------------------------------------------------
    thrasher_dir = r'data\raw_videos\thrasher_batb'
    batb_idx = 1
    for root, dirs, files in os.walk(thrasher_dir):
        for f in files:
            if f.lower().endswith(('.mov', '.mp4')):
                full_path = os.path.join(root, f)
                rel_dir = os.path.relpath(root, thrasher_dir)
                full_str = (rel_dir + ' ' + f).lower()
                
                # Match Skater
                skater_id = 'thrasher_unknown'
                for k, v in SKATER_LOOKUP.items():
                    if k in full_str:
                        skater_id = v
                        break
                
                # Match Outcome (Bail vs Land)
                is_bail = ('bails' in rel_dir.lower() or 'bail' in f.lower())
                outcome = 'bail' if is_bail else 'land'
                
                # Stance
                stance = parse_stance(full_str, default='regular')
                
                # Original label (use directory name or filename)
                orig_label = rel_dir if rel_dir != '.' else os.path.splitext(f)[0]
                trick_name, category, phase1_elig = parse_trick_and_category(full_str)
                
                w, h, fps, frames, dur, qflag = extract_video_metadata(full_path)
                split = SPLIT_MAP.get(skater_id, 'train')

                canonical_rel_path = full_path.replace(os.sep, '/')
                rows.append({
                    'clip_id': f'batb_{batb_idx:04d}',
                    'source_dataset': 'thrasher_batb',
                    'filepath': canonical_rel_path,
                    'skater_id': skater_id,
                    'stance': stance,
                    'original_label': orig_label,
                    'trick_name': trick_name,
                    'category': category,
                    'phase1_eligible': phase1_elig,
                    'outcome': outcome,
                    'outcome_source': 'directory_heuristic',
                    'width': w,
                    'height': h,
                    'fps': fps,
                    'total_frames': frames,
                    'duration_sec': dur,
                    'quality_flag': qflag,
                    'split': split
                })
                batb_idx += 1

    # -------------------------------------------------------------
    # 3. SkaterXL Synthetic Dataset (24 clips)
    # -------------------------------------------------------------
    sxl_dir = r'data\raw_videos\skaterxl_synthetic'
    sxl_idx = 1
    if os.path.exists(sxl_dir):
        for root, dirs, files in os.walk(sxl_dir):
            for f in files:
                if f.lower().endswith(('.mov', '.mp4')):
                    full_path = os.path.join(root, f)
                    w, h, fps, frames, dur, qflag = extract_video_metadata(full_path)
                    canonical_rel_path = full_path.replace(os.sep, '/')
                    rows.append({
                        'clip_id': f'sxl_{sxl_idx:04d}',
                        'source_dataset': 'skaterxl_synthetic',
                        'filepath': canonical_rel_path,
                        'skater_id': 'virtual_avatar',
                        'stance': 'unknown',
                        'original_label': os.path.splitext(f)[0],
                        'trick_name': 'synthetic_action',
                        'category': 'unsupported',
                        'phase1_eligible': False,
                        'outcome': 'unknown',
                        'outcome_source': 'directory_heuristic',
                        'width': w,
                        'height': h,
                        'fps': fps,
                        'total_frames': frames,
                        'duration_sec': dur,
                        'quality_flag': qflag,
                        'split': 'quarantine_synthetic'
                    })
                    sxl_idx += 1

    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f'Successfully exported manifest with {len(df_out)} rows to {output_csv}')
    return df_out


if __name__ == '__main__':
    build_manifest()
