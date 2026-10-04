"""
Pilot Set Selector (Guardrails G2 & G9)
Appends tracking_eligible and temporal_high_speed flags to video_manifest.csv
and selects a balanced, certified 25-clip pilot set:
- 10 High-Speed Player A clips (Ollie, Kickflip, Heelflip, FS180, BS180)
- 10 BATB Pro clips (Shane O'Neill, Sewa Kroetkov, Chris Joslin)
- 5 Real-World Bails with occlusions/dynamic camera
"""

import os
import pandas as pd


MANIFEST_PATH = 'data/metadata/video_manifest.csv'


def update_manifest_and_select_pilot(manifest_path: str = MANIFEST_PATH) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    
    # [G2] Non-Destructive Filtering
    df['tracking_eligible'] = (df['quality_flag'] == 'pass') & (df['duration_sec'].between(1.5, 10.0))
    df['temporal_high_speed'] = df['fps'] >= 60.0
    
    # Default pilot_set = False
    df['pilot_set'] = False

    # [G9] Pilot Selection Criteria
    # 1. 10 High-speed Player A clips (archive_highspeed)
    p10_archive_ids = []
    archive_pass = df[(df['source_dataset'] == 'archive_highspeed') & (df['tracking_eligible']) & (df['skater_id'] == 'player_a')]
    
    target_tricks = ['ollie', 'kickflip', 'heelflip', 'frontside180', 'backside180']
    for t in target_tricks:
        matching = archive_pass[archive_pass['trick_name'] == t]
        if len(matching) >= 2:
            p10_archive_ids.extend(matching.head(2)['clip_id'].tolist())
        elif len(matching) > 0:
            p10_archive_ids.extend(matching.head(len(matching))['clip_id'].tolist())
            
    # Ensure exactly 10
    if len(p10_archive_ids) < 10:
        remaining = archive_pass[~archive_pass['clip_id'].isin(p10_archive_ids)]
        needed = 10 - len(p10_archive_ids)
        p10_archive_ids.extend(remaining.head(needed)['clip_id'].tolist())
    p10_archive_ids = p10_archive_ids[:10]

    # 2. 10 BATB Pro clips (Shane, Sewa, Joslin) - landed attempts
    p10_batb_ids = []
    batb_land = df[(df['source_dataset'] == 'thrasher_batb') & (df['outcome'] == 'land') & (df['tracking_eligible'])]
    for skater in ['sewa_kroetkov', 'chris_joslin', 'shane_oneill']:
        sk_clips = batb_land[batb_land['skater_id'] == skater]
        p10_batb_ids.extend(sk_clips.head(3)['clip_id'].tolist())
    # Add 1 more to make 10
    rem_batb = batb_land[(batb_land['skater_id'].isin(['sewa_kroetkov', 'chris_joslin', 'shane_oneill'])) & (~batb_land['clip_id'].isin(p10_batb_ids))]
    if len(rem_batb) > 0:
        p10_batb_ids.append(rem_batb.iloc[0]['clip_id'])
    p10_batb_ids = p10_batb_ids[:10]

    # 3. 5 Real-world bails
    p5_bail_ids = []
    batb_bails = df[(df['source_dataset'] == 'thrasher_batb') & (df['outcome'] == 'bail') & (df['tracking_eligible'])]
    # Diverse skaters in bails
    for skater in ['chris_joslin', 'sewa_kroetkov', 'sean_malto', 'torey_pudwill', 'jack_colbourn']:
        sk_bails = batb_bails[batb_bails['skater_id'] == skater]
        if len(sk_bails) > 0:
            p5_bail_ids.append(sk_bails.iloc[0]['clip_id'])
    if len(p5_bail_ids) < 5:
        rem_bails = batb_bails[~batb_bails['clip_id'].isin(p5_bail_ids)]
        needed = 5 - len(p5_bail_ids)
        p5_bail_ids.extend(rem_bails.head(needed)['clip_id'].tolist())
    p5_bail_ids = p5_bail_ids[:5]

    all_pilot_ids = set(p10_archive_ids + p10_batb_ids + p5_bail_ids)
    assert len(all_pilot_ids) == 25, f'Expected 25 pilot clips, got {len(all_pilot_ids)}'

    df.loc[df['clip_id'].isin(all_pilot_ids), 'pilot_set'] = True
    df.to_csv(manifest_path, index=False)
    
    print(f'Successfully updated manifest: {len(df)} total rows.')
    print(f'  Tracking Eligible: {df["tracking_eligible"].sum()}')
    print(f'  Temporal High Speed: {df["temporal_high_speed"].sum()}')
    print(f'  Pilot Set Selected: {df["pilot_set"].sum()} clips')
    return df[df['pilot_set']]


if __name__ == '__main__':
    pilot_df = update_manifest_and_select_pilot()
    print('\nPilot Set Composition:')
    for _, r in pilot_df.iterrows():
        print(f"  [{r['clip_id']}] {r['source_dataset']:<18} | {r['skater_id']:<15} | {r['trick_name']:<16} | Outcome: {r['outcome']:<5} | {r['fps']}fps, {r['duration_sec']}s")
