#!/usr/bin/env python3
"""Example: Generate a small cohort and visualize results.

Usage:
    python examples/generate_cohort.py

Generates 5 subjects with 1-hour recordings (for speed) and prints
summary statistics. Optionally plots hypnograms and PSDs if matplotlib
is available.
"""

import sys
import numpy as np

sys.path.insert(0, ".")
from sleepsim import SleepDataGenerator
from sleepsim.stages import STAGE_NAMES


def main():
    print("=" * 60)
    print("Synthetic PSG Data Generator - Example")
    print("=" * 60)

    # Generate a small dataset (5 subjects, 1 hour, 128 Hz for speed)
    gen = SleepDataGenerator(
        n_subjects=5,
        sampling_rate=128,
        duration_hours=1.0,
        n_roi=20,
        seed=42,
    )

    print(f"\nGenerating data for {gen.n_subjects} subjects...")
    print(f"  Sampling rate: {gen.sampling_rate} Hz")
    print(f"  Duration: {gen.duration_hours} hours")
    print(f"  Epochs per subject: {gen.n_epochs}")
    print(f"  FC matrix size: {gen.n_roi} x {gen.n_roi}")

    for i, subject_data in enumerate(gen.generate_subject_iter()):
        traits = subject_data["traits"]
        hypnogram = subject_data["hypnogram"]
        psg = subject_data["psg_data"]
        fc = subject_data["fc_matrix"]

        # Stage distribution
        unique, counts = np.unique(hypnogram, return_counts=True)
        stage_dist = {STAGE_NAMES[s]: f"{c / len(hypnogram) * 100:.1f}%"
                      for s, c in zip(unique, counts)}

        print(f"\n--- Subject {traits.subject_id} ---")
        print(f"  Traits: delta_power={traits.delta_power:.2f}, "
              f"spindle_density={traits.spindle_density:.2f}, "
              f"rem_density={traits.rem_density:.2f}, "
              f"sleep_efficiency={traits.sleep_efficiency:.2f}")
        print(f"  Stage distribution: {stage_dist}")
        print(f"  PSG shape: {psg.shape}  "
              f"(channels={subject_data['channel_names']})")
        print(f"  FC matrix: shape={fc.shape}, "
              f"range=[{fc.min():.3f}, {fc.max():.3f}]")

    # Demonstrate epoch-batch generation (for RNN training)
    print("\n--- Epoch Batch Demo (for RNN training) ---")
    batch = gen.generate_epoch_batch(
        subject_indices=[0, 1, 2],
        epoch_indices=[0, 10, 20, 50, 100],
    )
    print(f"  PSG epochs shape: {batch['psg_epochs'].shape}")
    print(f"  Stages shape: {batch['stages'].shape}")
    print(f"  FC matrices shape: {batch['fc_matrices'].shape}")

    # Optional plotting
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _plot_example(gen, plt)
        print("\nPlots saved to examples/output/")
    except ImportError:
        print("\n(Install matplotlib for visualization: pip install matplotlib)")

    print("\nDone!")


def _plot_example(gen, plt):
    """Generate example plots for the first subject."""
    import os
    os.makedirs("examples/output", exist_ok=True)

    subject_data = gen.generate_subject(gen.subjects[0])
    hypnogram = subject_data["hypnogram"]
    psg = subject_data["psg_data"]
    fs = subject_data["sampling_rate"]
    fc = subject_data["fc_matrix"]

    # 1. Hypnogram
    fig, ax = plt.subplots(figsize=(12, 3))
    epochs = np.arange(len(hypnogram))
    ax.step(epochs, hypnogram, where="mid", linewidth=0.8)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["W", "N1", "N2", "N3", "REM"])
    ax.invert_yaxis()
    ax.set_xlabel("Epoch (30s)")
    ax.set_title("Hypnogram - Subject 0")
    fig.tight_layout()
    fig.savefig("examples/output/hypnogram.png", dpi=150)
    plt.close()

    # 2. PSG snippet (first 30 seconds = 1 epoch)
    n_show = fs * 30
    fig, axes = plt.subplots(6, 1, figsize=(14, 10), sharex=True)
    ch_names = subject_data["channel_names"]
    t = np.arange(n_show) / fs
    for ch_idx, (ax, name) in enumerate(zip(axes, ch_names)):
        ax.plot(t, psg[ch_idx, :n_show], linewidth=0.5)
        ax.set_ylabel(name, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("PSG Signals - First Epoch (Subject 0)", fontsize=11)
    fig.tight_layout()
    fig.savefig("examples/output/psg_signals.png", dpi=150)
    plt.close()

    # 3. FC matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(fc, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Virtual FC Matrix - Subject 0")
    ax.set_xlabel("ROI")
    ax.set_ylabel("ROI")
    fig.tight_layout()
    fig.savefig("examples/output/fc_matrix.png", dpi=150)
    plt.close()

    # 4. EEG PSD by stage
    from scipy.signal import welch
    fig, ax = plt.subplots(figsize=(10, 5))
    for stage_val, stage_name in STAGE_NAMES.items():
        mask = hypnogram == stage_val
        if not mask.any():
            continue
        stage_epochs = np.where(mask)[0]
        # Concatenate a few epochs of this stage
        n_use = min(5, len(stage_epochs))
        eeg_concat = []
        for ep in stage_epochs[:n_use]:
            start = ep * fs * 30
            end = start + fs * 30
            if end <= psg.shape[1]:
                eeg_concat.append(psg[0, start:end])
        if not eeg_concat:
            continue
        eeg_concat = np.concatenate(eeg_concat)
        freqs, pxx = welch(eeg_concat, fs=fs, nperseg=min(fs * 4, len(eeg_concat)))
        ax.semilogy(freqs, pxx, label=stage_name, linewidth=1.2)

    ax.set_xlim(0, 40)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("EEG Power Spectral Density by Sleep Stage")
    ax.legend()
    fig.tight_layout()
    fig.savefig("examples/output/eeg_psd_by_stage.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
