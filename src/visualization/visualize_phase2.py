"""
Visualization generator for Phase 2 Temporal Event Localization and Phase Segmentation.
Generates: docs/visualizations/temporal_event_segmentation_batb_0005.png
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.segmentation.phase2_pipeline import Phase2Pipeline


def plot_phase2_segmentation(clip_id: str = "batb_0005", out_path: str = "docs/visualizations/temporal_event_segmentation_batb_0005.png"):
    pipeline = Phase2Pipeline()
    out = pipeline.process_clip_by_id(clip_id)
    ev = out.events
    df = out.conditioned_df
    fps = ev.fps

    t = df['frame_idx'].values
    t_sec = t / fps

    # Setup 4-panel publication-grade layout
    plt.style.use('dark_background')
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.patch.set_facecolor('#0d1117')

    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.grid(True, linestyle='--', alpha=0.3, color='#8b949e')

    # Color palette
    c_pop = '#ff7b72'     # Coral Red
    c_apex = '#79c0ff'    # Sky Blue
    c_catch = '#d2a8ff'   # Purple
    c_land = '#7ee787'    # Bright Green

    # Window slices shading
    sl = out.slices
    app_s, app_e = sl.approach_indices
    flt_s, flt_e = sl.flight_indices
    lnd_s, lnd_e = sl.landing_indices

    for ax in axes:
        ax.axvspan(app_s, app_e, color='#f0883e', alpha=0.10, label='Approach Window' if ax == axes[0] else "")
        ax.axvspan(flt_s, flt_e, color='#58a6ff', alpha=0.15, label='Flight Execution Window' if ax == axes[0] else "")
        ax.axvspan(lnd_s, lnd_e, color='#3fb950', alpha=0.10, label='Rollout Window' if ax == axes[0] else "")

        # Event vertical lines
        ax.axvline(ev.t_pop, color=c_pop, linestyle='-', linewidth=2.0)
        ax.axvline(ev.t_apex, color=c_apex, linestyle='-', linewidth=2.0)
        if ev.t_catch is not None:
            ax.axvline(ev.t_catch, color=c_catch, linestyle='-', linewidth=2.0)
        ax.axvline(ev.t_land, color=c_land, linestyle='-', linewidth=2.0)

    # Subplot 1: Canonical Z-Up Altitude
    ax1 = axes[0]
    ax1.plot(t, df['deck_altitude'], color='#58a6ff', linewidth=2.2, label='Deck Altitude z_deck (px)')
    ax1.plot(t, df['tail_altitude'], color='#ffa657', linewidth=1.5, linestyle=':', label='Tail Altitude (px)')
    ax1.axhline(0, color='#8b949e', linestyle='--', linewidth=1.0, alpha=0.6, label='Ground Plane Reference')
    ax1.set_ylabel('Altitude (px)', fontsize=11, fontweight='bold', color='#c9d1d9')
    ax1.set_title(f'Phase 2 Temporal Segmentation: {clip_id.upper()} (Chris Joslin Kickflip | 60 FPS)\nFlight Duration: {ev.flight_duration_ms:.0f} ms | Monotonic Order: Pop ({ev.t_pop}) < Apex ({ev.t_apex}) < Catch ({ev.t_catch}) <= Land ({ev.t_land})',
                  fontsize=13, fontweight='bold', pad=12, color='#f0f6fc')
    ax1.legend(loc='upper right', framealpha=0.6, fontsize=9)

    # Subplot 2: Vertical Velocity (dz/dt) Zero-Crossing [A2]
    ax2 = axes[1]
    ax2.plot(t, df['vel_z_deck'], color='#388bfd', linewidth=2.0, label='Vertical Velocity dz/dt (px/s)')
    ax2.axhline(0, color='#f85149', linestyle='--', linewidth=1.5, alpha=0.8, label='Zero-Crossing Line (Apex)')
    ax2.scatter([ev.t_apex], [0], color='#f85149', s=70, zorder=5)
    ax2.set_ylabel('Velocity (px/s)', fontsize=11, fontweight='bold', color='#c9d1d9')
    ax2.legend(loc='upper right', framealpha=0.6, fontsize=9)

    # Subplot 3: Angular Velocity & Catch Deceleration [A3]
    ax3 = axes[2]
    ax3.plot(t, df['omega_rad_per_sec'], color='#d2a8ff', linewidth=2.0, label='Angular Velocity omega (rad/s)')
    if ev.t_catch is not None:
        ax3.scatter([ev.t_catch], [df['omega_rad_per_sec'].iloc[ev.t_catch]], color=c_catch, s=70, zorder=5, label=f'Catch Frame ({ev.t_catch})')
    ax3.set_ylabel('omega (rad/s)', fontsize=11, fontweight='bold', color='#c9d1d9')
    ax3.legend(loc='upper right', framealpha=0.6, fontsize=9)

    # Subplot 4: Ankle Impulse & Impact Deceleration Shockwave [A1, A5]
    ax4 = axes[3]
    ax4.plot(t, df['acc_z_left_ankle'], color='#e3b341', linewidth=1.6, label='Left Ankle Accel d2z/dt2')
    ax4.plot(t, df['acc_z_deck'], color='#3fb950', linewidth=1.8, label='Board Deceleration Accel (Impact Shockwave [A5])')
    ax4.set_ylabel('Accel (px/s^2)', fontsize=11, fontweight='bold', color='#c9d1d9')
    ax4.set_xlabel('Frame Index', fontsize=11, fontweight='bold', color='#c9d1d9')
    ax4.legend(loc='upper right', framealpha=0.6, fontsize=9)

    # Add custom event badge legend to bottom
    axes[0].text(ev.t_pop - 1, ax1.get_ylim()[1] * 0.85, 'POP', color=c_pop, fontweight='bold', fontsize=10, ha='right')
    axes[0].text(ev.t_apex, ax1.get_ylim()[1] * 0.90, 'APEX', color=c_apex, fontweight='bold', fontsize=10, ha='center')
    if ev.t_catch is not None:
        axes[0].text(ev.t_catch + 0.5, ax1.get_ylim()[1] * 0.75, 'CATCH', color=c_catch, fontweight='bold', fontsize=10, ha='left')
    axes[0].text(ev.t_land + 0.5, ax1.get_ylim()[1] * 0.60, 'LAND', color=c_land, fontweight='bold', fontsize=10, ha='left')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved Phase 2 temporal event segmentation plot to: {out_path}")


if __name__ == '__main__':
    plot_phase2_segmentation()
