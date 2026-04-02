#!/usr/bin/env python3
"""Comprehensive data validation visualization for sleepsim.

Generates a multi-panel validation report covering:
1. Hypnograms for multiple subjects (sleep architecture comparison)
2. Stage distribution across subjects (bar chart)
3. EEG PSD by stage (spectral validation)
4. Per-channel signal snippets across all 5 stages
5. FC matrix comparison across subjects
6. Trait distribution and correlation structure
7. Sleep cycle structure (N3 and REM across the night)
8. EMG amplitude by stage (atonia validation)

Usage:
    python examples/validate_data.py
"""

import sys
import os

import numpy as np
from scipy.signal import welch

sys.path.insert(0, ".")
from sleepsim import SleepDataGenerator
from sleepsim.stages import STAGE_NAMES, W, N1, N2, N3, REM
from sleepsim.channels import CHANNEL_NAMES
from sleepsim.traits import TRAIT_NAMES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

OUT_DIR = "examples/output/validation"
os.makedirs(OUT_DIR, exist_ok=True)

# Color scheme for sleep stages
STAGE_COLORS = {W: "#E64B35", N1: "#4DBBD5", N2: "#00A087", N3: "#3C5488", REM: "#F39B7F"}


def main():
    print("Generating 8-hour data for 10 subjects (128 Hz)...")
    gen = SleepDataGenerator(
        n_subjects=10,
        sampling_rate=128,
        duration_hours=8.0,
        n_roi=20,
        seed=42,
    )

    # Collect all data
    all_data = []
    for data in gen.generate_subject_iter():
        all_data.append(data)
        sid = data["traits"].subject_id
        print(f"  Subject {sid} done.")

    fs = all_data[0]["sampling_rate"]

    print("\nGenerating validation plots...")

    plot_multi_hypnograms(all_data)
    plot_stage_distributions(all_data)
    plot_eeg_psd_by_stage(all_data, fs)
    plot_per_stage_signals(all_data, fs)
    plot_fc_matrices(all_data, gen)
    plot_trait_distributions(gen.subjects)
    plot_trait_correlations(gen.subjects)
    plot_sleep_cycle_dynamics(all_data)
    plot_emg_by_stage(all_data, fs)
    plot_ecg_by_stage(all_data, fs)

    print(f"\nAll plots saved to {OUT_DIR}/")


# ---- Plot functions ----

def plot_multi_hypnograms(all_data):
    """Plot hypnograms for all subjects stacked vertically."""
    n = len(all_data)
    fig, axes = plt.subplots(n, 1, figsize=(16, 2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, data in enumerate(all_data):
        ax = axes[i]
        hyp = data["hypnogram"]
        epochs = np.arange(len(hyp))
        hours = epochs * 30 / 3600

        # Color-code by stage
        for stage_val, color in STAGE_COLORS.items():
            mask = hyp == stage_val
            ax.fill_between(hours, stage_val - 0.4, stage_val + 0.4,
                            where=mask, color=color, alpha=0.7, step="mid")
        ax.step(hours, hyp, where="mid", color="black", linewidth=0.5, alpha=0.5)
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(["W", "N1", "N2", "N3", "REM"], fontsize=8)
        ax.invert_yaxis()
        ax.set_ylabel(f"S{data['traits'].subject_id}", fontsize=9)
        ax.set_ylim(4.5, -0.5)

    axes[-1].set_xlabel("Time (hours)")
    fig.suptitle("Hypnograms - All Subjects", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/01_hypnograms.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [1/10] Hypnograms")


def plot_stage_distributions(all_data):
    """Bar chart of stage proportions per subject."""
    n = len(all_data)
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(n)
    width = 0.15
    for si, (stage_val, stage_name) in enumerate(STAGE_NAMES.items()):
        proportions = []
        for data in all_data:
            hyp = data["hypnogram"]
            proportions.append(np.sum(hyp == stage_val) / len(hyp) * 100)
        ax.bar(x + si * width, proportions, width, label=stage_name,
               color=STAGE_COLORS[stage_val], alpha=0.85)

    ax.set_xlabel("Subject")
    ax.set_ylabel("% of Total Recording")
    ax.set_title("Sleep Stage Distribution by Subject")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f"S{d['traits'].subject_id}" for d in all_data])
    ax.legend()
    ax.set_ylim(0, 65)

    # Add reference ranges as horizontal bands
    ax.axhspan(45, 55, color=STAGE_COLORS[N2], alpha=0.08, label="_N2 norm")
    ax.axhspan(15, 25, color=STAGE_COLORS[REM], alpha=0.08, label="_REM norm")
    ax.axhspan(13, 23, color=STAGE_COLORS[N3], alpha=0.08, label="_N3 norm")

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/02_stage_distribution.png", dpi=150)
    plt.close()
    print("  [2/10] Stage distributions")


def plot_eeg_psd_by_stage(all_data, fs):
    """Average EEG PSD across subjects for each stage."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-stage average PSD
    ax = axes[0]
    for stage_val, stage_name in STAGE_NAMES.items():
        psd_all = []
        for data in all_data:
            hyp = data["hypnogram"]
            psg = data["psg_data"]
            mask = hyp == stage_val
            if not mask.any():
                continue
            stage_epochs = np.where(mask)[0][:10]  # use up to 10 epochs
            for ep in stage_epochs:
                start = ep * fs * 30
                end = start + fs * 30
                if end <= psg.shape[1]:
                    freqs, pxx = welch(psg[0, start:end], fs=fs, nperseg=fs * 2)
                    psd_all.append(pxx)
        if psd_all:
            mean_psd = np.mean(psd_all, axis=0)
            sem_psd = np.std(psd_all, axis=0) / np.sqrt(len(psd_all))
            ax.semilogy(freqs, mean_psd, label=stage_name,
                        color=STAGE_COLORS[stage_val], linewidth=1.5)
            ax.fill_between(freqs, mean_psd - sem_psd, mean_psd + sem_psd,
                            color=STAGE_COLORS[stage_val], alpha=0.2)

    ax.set_xlim(0, 40)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (mean +/- SEM)")
    ax.set_title("EEG_C3 Power Spectral Density by Stage")
    ax.legend()

    # Band power comparison
    ax2 = axes[1]
    bands = {"Delta\n(0.5-4)": (0.5, 4), "Theta\n(4-8)": (4, 8),
             "Alpha\n(8-12)": (8, 12), "Sigma\n(12-16)": (12, 16),
             "Beta\n(16-30)": (16, 30)}

    band_powers = {s: {b: [] for b in bands} for s in STAGE_NAMES.values()}

    for data in all_data:
        hyp = data["hypnogram"]
        psg = data["psg_data"]
        for stage_val, stage_name in STAGE_NAMES.items():
            mask = hyp == stage_val
            if not mask.any():
                continue
            stage_epochs = np.where(mask)[0][:10]
            for ep in stage_epochs:
                start = ep * fs * 30
                end = start + fs * 30
                if end <= psg.shape[1]:
                    freqs_w, pxx_w = welch(psg[0, start:end], fs=fs, nperseg=fs * 2)
                    for bname, (flow, fhigh) in bands.items():
                        band_mask = (freqs_w >= flow) & (freqs_w <= fhigh)
                        band_powers[stage_name][bname].append(np.mean(pxx_w[band_mask]))

    x = np.arange(len(bands))
    width = 0.15
    for si, (stage_name, stage_val) in enumerate(
            zip(STAGE_NAMES.values(), STAGE_NAMES.keys())):
        means = [np.mean(band_powers[stage_name][b]) if band_powers[stage_name][b] else 0
                 for b in bands]
        ax2.bar(x + si * width, means, width, label=stage_name,
                color=STAGE_COLORS[stage_val], alpha=0.85)

    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels(bands.keys(), fontsize=9)
    ax2.set_ylabel("Mean Band Power")
    ax2.set_title("EEG Band Power by Stage")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/03_eeg_psd.png", dpi=150)
    plt.close()
    print("  [3/10] EEG PSD by stage")


def plot_per_stage_signals(all_data, fs):
    """Show 10-second snippets of all channels for each sleep stage."""
    data = all_data[0]
    hyp = data["hypnogram"]
    psg = data["psg_data"]
    n_show = int(10 * fs)  # 10 seconds

    fig, axes = plt.subplots(5, 6, figsize=(20, 14))

    for row, (stage_val, stage_name) in enumerate(STAGE_NAMES.items()):
        mask = np.where(hyp == stage_val)[0]
        if len(mask) == 0:
            for col in range(6):
                axes[row, col].text(0.5, 0.5, "N/A", ha="center", va="center",
                                    transform=axes[row, col].transAxes)
            continue

        ep = mask[len(mask) // 2]  # middle epoch of this stage
        start = ep * fs * 30
        end = start + n_show
        if end > psg.shape[1]:
            start = max(0, psg.shape[1] - n_show)
            end = psg.shape[1]

        t = np.arange(end - start) / fs

        for col, ch_name in enumerate(CHANNEL_NAMES):
            ax = axes[row, col]
            sig = psg[col, start:end]
            ax.plot(t, sig, linewidth=0.4, color=STAGE_COLORS[stage_val])
            if row == 0:
                ax.set_title(ch_name, fontsize=9)
            if col == 0:
                ax.set_ylabel(stage_name, fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=6)
            if row == 4:
                ax.set_xlabel("Time (s)", fontsize=7)

    fig.suptitle("PSG Channel Signals by Sleep Stage (10s snippets, Subject 0)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/04_per_stage_signals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [4/10] Per-stage signal snippets")


def plot_fc_matrices(all_data, gen):
    """Compare FC matrices across subjects + show inter-subject variability."""
    n_show = min(5, len(all_data))
    fig, axes = plt.subplots(1, n_show + 1, figsize=(4 * (n_show + 1), 4))

    fc_list = []
    for i in range(n_show):
        fc = all_data[i]["fc_matrix"]
        fc_list.append(fc)
        ax = axes[i]
        im = ax.imshow(fc, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_title(f"Subject {all_data[i]['traits'].subject_id}", fontsize=9)
        ax.tick_params(labelsize=6)

    # Inter-subject variability (std of FC across subjects)
    fc_stack = np.stack(fc_list)
    fc_std = np.std(fc_stack, axis=0)
    ax = axes[-1]
    im2 = ax.imshow(fc_std, cmap="YlOrRd", aspect="equal")
    ax.set_title("Inter-subject SD", fontsize=9)
    ax.tick_params(labelsize=6)
    plt.colorbar(im2, ax=ax, fraction=0.046)

    fig.suptitle("Virtual FC Matrices", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/05_fc_matrices.png", dpi=150)
    plt.close()
    print("  [5/10] FC matrices")


def plot_trait_distributions(subjects):
    """Histogram of each trait parameter across subjects."""
    n_traits = len(TRAIT_NAMES)
    cols = 4
    rows = (n_traits + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3 * rows))
    axes = axes.flatten()

    for i, name in enumerate(TRAIT_NAMES):
        vals = [getattr(s, name) for s in subjects]
        axes[i].hist(vals, bins=15, color="#3C5488", alpha=0.7, edgecolor="white")
        axes[i].set_title(name, fontsize=9)
        axes[i].axvline(np.mean(vals), color="red", linestyle="--", linewidth=1)
        axes[i].tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Trait Parameter Distributions (red = mean)", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/06_trait_distributions.png", dpi=150)
    plt.close()
    print("  [6/10] Trait distributions")


def plot_trait_correlations(subjects):
    """Trait-trait correlation matrix (observed from generated subjects)."""
    mat = np.array([s.to_normalized_vector() for s in subjects])
    corr = np.corrcoef(mat.T)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(TRAIT_NAMES)))
    ax.set_yticks(range(len(TRAIT_NAMES)))
    ax.set_xticklabels(TRAIT_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(TRAIT_NAMES, fontsize=8)

    # Annotate values
    for i in range(len(TRAIT_NAMES)):
        for j in range(len(TRAIT_NAMES)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if abs(corr[i, j]) > 0.5 else "black")

    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Observed Trait Correlation Matrix")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/07_trait_correlations.png", dpi=150)
    plt.close()
    print("  [7/10] Trait correlations")


def plot_sleep_cycle_dynamics(all_data):
    """Show how N3 and REM proportions evolve across the night."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bin epochs into 30-min segments
    bin_size = 60  # 60 epochs = 30 minutes
    n_bins = len(all_data[0]["hypnogram"]) // bin_size

    for stage_val, stage_name, ax_idx in [(N3, "N3", 0), (REM, "REM", 1)]:
        ax = axes[ax_idx]
        all_proportions = []
        for data in all_data:
            hyp = data["hypnogram"]
            proportions = []
            for b in range(n_bins):
                seg = hyp[b * bin_size:(b + 1) * bin_size]
                proportions.append(np.sum(seg == stage_val) / len(seg) * 100)
            all_proportions.append(proportions)

        all_proportions = np.array(all_proportions)
        mean_p = np.mean(all_proportions, axis=0)
        std_p = np.std(all_proportions, axis=0)
        hours = (np.arange(n_bins) + 0.5) * bin_size * 30 / 3600

        ax.plot(hours, mean_p, "-o", color=STAGE_COLORS[stage_val],
                linewidth=2, markersize=5)
        ax.fill_between(hours, mean_p - std_p, mean_p + std_p,
                         color=STAGE_COLORS[stage_val], alpha=0.2)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel(f"{stage_name} %")
        ax.set_title(f"{stage_name} Proportion Across the Night (mean +/- SD)")
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/08_sleep_cycle_dynamics.png", dpi=150)
    plt.close()
    print("  [8/10] Sleep cycle dynamics")


def plot_emg_by_stage(all_data, fs):
    """Box plot of EMG RMS amplitude by sleep stage."""
    fig, ax = plt.subplots(figsize=(8, 5))

    emg_ch = CHANNEL_NAMES.index("EMG_chin")
    stage_rms = {s: [] for s in STAGE_NAMES.values()}

    for data in all_data:
        hyp = data["hypnogram"]
        psg = data["psg_data"]
        for stage_val, stage_name in STAGE_NAMES.items():
            mask = np.where(hyp == stage_val)[0][:15]
            for ep in mask:
                start = ep * fs * 30
                end = start + fs * 30
                if end <= psg.shape[1]:
                    rms = np.sqrt(np.mean(psg[emg_ch, start:end] ** 2))
                    stage_rms[stage_name].append(rms)

    positions = range(len(STAGE_NAMES))
    bp_data = [stage_rms[s] for s in STAGE_NAMES.values()]
    bp = ax.boxplot(bp_data, positions=positions, patch_artist=True, widths=0.6)
    for patch, (stage_val, color) in zip(bp["boxes"], STAGE_COLORS.items()):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(STAGE_NAMES.values())
    ax.set_ylabel("EMG RMS Amplitude")
    ax.set_title("Chin EMG Amplitude by Sleep Stage\n(REM atonia validation)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/09_emg_by_stage.png", dpi=150)
    plt.close()
    print("  [9/10] EMG by stage")


def plot_ecg_by_stage(all_data, fs):
    """Heart rate estimation by stage."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ecg_ch = CHANNEL_NAMES.index("ECG")
    stage_hr = {s: [] for s in STAGE_NAMES.values()}

    for data in all_data:
        hyp = data["hypnogram"]
        psg = data["psg_data"]
        for stage_val, stage_name in STAGE_NAMES.items():
            mask = np.where(hyp == stage_val)[0][:10]
            for ep in mask:
                start = ep * fs * 30
                end = start + fs * 30
                if end <= psg.shape[1]:
                    ecg_seg = psg[ecg_ch, start:end]
                    # Simple peak detection: count threshold crossings
                    threshold = 0.5 * np.max(ecg_seg)
                    peaks = np.where((ecg_seg[:-1] < threshold) &
                                     (ecg_seg[1:] >= threshold))[0]
                    if len(peaks) > 1:
                        rr_intervals = np.diff(peaks) / fs
                        hr = 60.0 / np.mean(rr_intervals)
                        if 30 < hr < 200:  # reasonable range
                            stage_hr[stage_name].append(hr)

    positions = range(len(STAGE_NAMES))
    bp_data = [stage_hr[s] if stage_hr[s] else [0] for s in STAGE_NAMES.values()]
    bp = ax.boxplot(bp_data, positions=positions, patch_artist=True, widths=0.6)
    for patch, (stage_val, color) in zip(bp["boxes"], STAGE_COLORS.items()):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(STAGE_NAMES.values())
    ax.set_ylabel("Estimated Heart Rate (bpm)")
    ax.set_title("Heart Rate by Sleep Stage\n(Autonomic modulation validation)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/10_ecg_by_stage.png", dpi=150)
    plt.close()
    print("  [10/10] ECG/HR by stage")


if __name__ == "__main__":
    main()
