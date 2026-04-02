#!/usr/bin/env python3
"""Condition comparison validation visualization.

Generates diagnostic plots comparing Healthy, RBD, OSA, and Insomnia groups
to verify that each condition produces physiologically distinct PSG patterns.

Usage:
    python examples/validate_conditions.py
"""

import sys
import os

import numpy as np
from scipy.signal import welch

sys.path.insert(0, ".")
from sleepsim import SleepDataGenerator, VALID_CONDITIONS, CONDITION_DESCRIPTIONS
from sleepsim.stages import STAGE_NAMES, W, N1, N2, N3, REM
from sleepsim.channels import CHANNEL_NAMES
from sleepsim.traits import TRAIT_NAMES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

OUT_DIR = "examples/output/conditions"
os.makedirs(OUT_DIR, exist_ok=True)

# Color scheme for conditions
COND_COLORS = {
    "healthy":  "#00A087",
    "rbd":      "#E64B35",
    "osa":      "#4DBBD5",
    "insomnia": "#7E6148",
}
STAGE_COLORS = {W: "#E64B35", N1: "#4DBBD5", N2: "#00A087", N3: "#3C5488", REM: "#F39B7F"}

N_SUBJECTS = 8
FS = 128
DURATION = 8.0


def main():
    print("Generating 8-hour data for 4 conditions x 8 subjects...")
    all_groups = {}
    for cond in VALID_CONDITIONS:
        print(f"  Generating {cond}...")
        gen = SleepDataGenerator(
            n_subjects=N_SUBJECTS, sampling_rate=FS,
            duration_hours=DURATION, seed=42, condition=cond,
        )
        group_data = list(gen.generate_subject_iter())
        all_groups[cond] = {"generator": gen, "data": group_data}

    print("\nGenerating comparison plots...")

    plot_01_representative_hypnograms(all_groups)
    plot_02_stage_distribution_by_condition(all_groups)
    plot_03_sleep_architecture_summary(all_groups)
    plot_04_eeg_psd_by_condition(all_groups)
    plot_05_eeg_band_power_heatmap(all_groups)
    plot_06_emg_condition_comparison(all_groups)
    plot_07_respiratory_and_spo2(all_groups)
    plot_08_fc_matrix_by_condition(all_groups)
    plot_09_fc_difference_from_healthy(all_groups)
    plot_10_trait_shift_by_condition(all_groups)
    plot_11_sleep_onset_and_waso(all_groups)
    plot_12_condition_signal_gallery(all_groups)

    print(f"\nAll 12 plots saved to {OUT_DIR}/")


# =====================================================================
# Plot functions
# =====================================================================

def plot_01_representative_hypnograms(all_groups):
    """One representative hypnogram per condition, stacked."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 8), sharex=True)

    for ax, cond in zip(axes, VALID_CONDITIONS):
        hyp = all_groups[cond]["data"][0]["hypnogram"]
        hours = np.arange(len(hyp)) * 30 / 3600
        for sv, color in STAGE_COLORS.items():
            mask = hyp == sv
            ax.fill_between(hours, sv - 0.4, sv + 0.4,
                            where=mask, color=color, alpha=0.7, step="mid")
        ax.step(hours, hyp, where="mid", color="black", lw=0.4, alpha=0.4)
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(["W", "N1", "N2", "N3", "REM"], fontsize=8)
        ax.invert_yaxis()
        ax.set_ylim(4.5, -0.5)
        ax.set_ylabel(cond.upper(), fontsize=10, fontweight="bold",
                      color=COND_COLORS[cond])

    axes[-1].set_xlabel("Time (hours)")
    fig.suptitle("Representative Hypnograms by Condition", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/01_hypnograms_by_condition.png", dpi=150)
    plt.close()
    print("  [1/12] Representative hypnograms")


def plot_02_stage_distribution_by_condition(all_groups):
    """Grouped bar chart: stage proportions per condition."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(VALID_CONDITIONS))
    width = 0.15
    for si, (sv, sname) in enumerate(STAGE_NAMES.items()):
        means, sems = [], []
        for cond in VALID_CONDITIONS:
            props = []
            for d in all_groups[cond]["data"]:
                h = d["hypnogram"]
                props.append(np.sum(h == sv) / len(h) * 100)
            means.append(np.mean(props))
            sems.append(np.std(props) / np.sqrt(len(props)))
        ax.bar(x + si * width, means, width, yerr=sems, capsize=3,
               label=sname, color=STAGE_COLORS[sv], alpha=0.85)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([CONDITION_DESCRIPTIONS[c] for c in VALID_CONDITIONS],
                       fontsize=9)
    ax.set_ylabel("% of Total Recording")
    ax.set_title("Sleep Stage Distribution by Condition (mean +/- SEM)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 65)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/02_stage_distribution.png", dpi=150)
    plt.close()
    print("  [2/12] Stage distributions")


def plot_03_sleep_architecture_summary(all_groups):
    """Key sleep architecture metrics across conditions."""
    metrics = {
        "Sleep\nEfficiency (%)": lambda h: np.sum(h != W) / len(h) * 100,
        "N3 (%)": lambda h: np.sum(h == N3) / len(h) * 100,
        "REM (%)": lambda h: np.sum(h == REM) / len(h) * 100,
        "WASO\n(epochs)": lambda h: _count_waso_epochs(h),
        "Sleep Onset\n(epochs)": lambda h: _sleep_onset_epoch(h),
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.5 * len(metrics), 5))

    for ax, (mname, mfunc) in zip(axes, metrics.items()):
        data_by_cond = []
        for cond in VALID_CONDITIONS:
            vals = [mfunc(d["hypnogram"]) for d in all_groups[cond]["data"]]
            data_by_cond.append(vals)

        bp = ax.boxplot(data_by_cond, patch_artist=True, widths=0.6)
        for patch, cond in zip(bp["boxes"], VALID_CONDITIONS):
            patch.set_facecolor(COND_COLORS[cond])
            patch.set_alpha(0.6)
        ax.set_xticklabels([c[:3].upper() for c in VALID_CONDITIONS], fontsize=8)
        ax.set_title(mname, fontsize=9)

    fig.suptitle("Sleep Architecture Metrics by Condition", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/03_sleep_architecture.png", dpi=150)
    plt.close()
    print("  [3/12] Sleep architecture summary")


def plot_04_eeg_psd_by_condition(all_groups):
    """EEG PSD for N2 and REM stages, comparing conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, target_stage, title in [
        (axes[0], N2, "N2 (NREM) EEG PSD"),
        (axes[1], REM, "REM EEG PSD"),
    ]:
        for cond in VALID_CONDITIONS:
            psds = _collect_psds(all_groups[cond]["data"], target_stage, FS)
            if psds:
                freqs = psds[0][0]
                pxx_all = np.array([p[1] for p in psds])
                mean_psd = np.mean(pxx_all, axis=0)
                sem = np.std(pxx_all, axis=0) / np.sqrt(len(pxx_all))
                ax.semilogy(freqs, mean_psd, label=CONDITION_DESCRIPTIONS[cond],
                            color=COND_COLORS[cond], lw=1.5)
                ax.fill_between(freqs, mean_psd - sem, mean_psd + sem,
                                color=COND_COLORS[cond], alpha=0.15)
        ax.set_xlim(0, 40)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD")
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/04_eeg_psd_by_condition.png", dpi=150)
    plt.close()
    print("  [4/12] EEG PSD by condition")


def plot_05_eeg_band_power_heatmap(all_groups):
    """Heatmap: band power x condition x stage."""
    bands = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 12),
             "Sigma": (12, 16), "Beta": (16, 30)}

    # Compute mean band power for each condition x stage x band
    matrix = np.zeros((len(VALID_CONDITIONS), len(STAGE_NAMES), len(bands)))
    for ci, cond in enumerate(VALID_CONDITIONS):
        for si, (sv, sname) in enumerate(STAGE_NAMES.items()):
            psds = _collect_psds(all_groups[cond]["data"], sv, FS)
            if psds:
                freqs = psds[0][0]
                pxx_mean = np.mean([p[1] for p in psds], axis=0)
                for bi, (bname, (fl, fh)) in enumerate(bands.items()):
                    mask = (freqs >= fl) & (freqs <= fh)
                    matrix[ci, si, bi] = np.mean(pxx_mean[mask])

    # Normalize per band for visualization
    for bi in range(len(bands)):
        bmax = matrix[:, :, bi].max()
        if bmax > 0:
            matrix[:, :, bi] /= bmax

    fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True)
    for ci, (cond, ax) in enumerate(zip(VALID_CONDITIONS, axes)):
        im = ax.imshow(matrix[ci].T, cmap="YlOrRd", aspect="auto",
                       vmin=0, vmax=1)
        ax.set_xticks(range(len(STAGE_NAMES)))
        ax.set_xticklabels(STAGE_NAMES.values(), fontsize=8)
        ax.set_yticks(range(len(bands)))
        ax.set_yticklabels(bands.keys(), fontsize=8)
        ax.set_title(CONDITION_DESCRIPTIONS[cond], fontsize=9,
                     color=COND_COLORS[cond], fontweight="bold")

    plt.colorbar(im, ax=axes, fraction=0.02, label="Relative Power")
    fig.suptitle("EEG Band Power (normalized per band)", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/05_band_power_heatmap.png", dpi=150)
    plt.close()
    print("  [5/12] Band power heatmap")


def plot_06_emg_condition_comparison(all_groups):
    """EMG RMS by stage x condition — key for RBD validation."""
    fig, ax = plt.subplots(figsize=(12, 6))
    emg_ch = CHANNEL_NAMES.index("EMG_chin")

    x = np.arange(len(STAGE_NAMES))
    width = 0.18
    for ci, cond in enumerate(VALID_CONDITIONS):
        means, sems = [], []
        for sv in STAGE_NAMES.keys():
            rms_vals = []
            for d in all_groups[cond]["data"]:
                h = d["hypnogram"]
                psg = d["psg_data"]
                epochs = np.where(h == sv)[0][:15]
                for ep in epochs:
                    s = ep * FS * 30
                    e = s + FS * 30
                    if e <= psg.shape[1]:
                        rms_vals.append(np.sqrt(np.mean(psg[emg_ch, s:e] ** 2)))
            means.append(np.mean(rms_vals) if rms_vals else 0)
            sems.append(np.std(rms_vals) / np.sqrt(len(rms_vals)) if len(rms_vals) > 1 else 0)

        ax.bar(x + ci * width, means, width, yerr=sems, capsize=2,
               label=CONDITION_DESCRIPTIONS[cond], color=COND_COLORS[cond], alpha=0.8)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(STAGE_NAMES.values())
    ax.set_ylabel("EMG RMS Amplitude")
    ax.set_title("Chin EMG by Stage and Condition\n(RBD: elevated REM EMG = REM without atonia)")
    ax.legend()

    # Annotate the key finding
    ax.annotate("RBD: REM without atonia",
                xy=(4 + 0.18, 0), xytext=(3.5, ax.get_ylim()[1] * 0.85),
                fontsize=9, color=COND_COLORS["rbd"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COND_COLORS["rbd"]))

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/06_emg_by_condition.png", dpi=150)
    plt.close()
    print("  [6/12] EMG by condition (RBD validation)")


def plot_07_respiratory_and_spo2(all_groups):
    """Respiratory effort and SpO2 traces — key for OSA validation."""
    resp_ch = CHANNEL_NAMES.index("Resp_effort")
    spo2_ch = CHANNEL_NAMES.index("SpO2")

    fig, axes = plt.subplots(4, 2, figsize=(18, 12))

    for row, cond in enumerate(VALID_CONDITIONS):
        d = all_groups[cond]["data"][0]
        psg = d["psg_data"]
        hyp = d["hypnogram"]

        # Find a NREM epoch to show (N2 preferred)
        n2_epochs = np.where(hyp == N2)[0]
        if len(n2_epochs) < 3:
            n2_epochs = np.where(hyp != W)[0]
        # Show 3 consecutive epochs (90 seconds)
        ep_start = n2_epochs[len(n2_epochs) // 2]
        sample_start = ep_start * FS * 30
        show_dur = 3 * FS * 30  # 90 seconds
        sample_end = min(sample_start + show_dur, psg.shape[1])
        t = np.arange(sample_end - sample_start) / FS

        # Respiratory effort
        ax_resp = axes[row, 0]
        ax_resp.plot(t, psg[resp_ch, sample_start:sample_end],
                     color=COND_COLORS[cond], lw=0.6)
        ax_resp.set_ylabel(cond.upper(), fontsize=10, fontweight="bold",
                           color=COND_COLORS[cond])
        if row == 0:
            ax_resp.set_title("Respiratory Effort (90s snippet)")
        if row == 3:
            ax_resp.set_xlabel("Time (s)")

        # SpO2
        ax_spo2 = axes[row, 1]
        spo2_data = psg[spo2_ch, sample_start:sample_end] * 100  # to percentage
        ax_spo2.plot(t, spo2_data, color=COND_COLORS[cond], lw=0.8)
        ax_spo2.set_ylim(82, 100)
        if row == 0:
            ax_spo2.set_title("SpO2 (%)")
        if row == 3:
            ax_spo2.set_xlabel("Time (s)")

    fig.suptitle("Respiratory Signals by Condition\n(OSA: apnea events with SpO2 desaturation)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/07_respiratory_spo2.png", dpi=150)
    plt.close()
    print("  [7/12] Respiratory & SpO2 (OSA validation)")


def plot_08_fc_matrix_by_condition(all_groups):
    """Mean FC matrix per condition."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    for ax, cond in zip(axes, VALID_CONDITIONS):
        fc_all = np.stack([d["fc_matrix"] for d in all_groups[cond]["data"]])
        fc_mean = np.mean(fc_all, axis=0)
        im = ax.imshow(fc_mean, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_title(CONDITION_DESCRIPTIONS[cond], fontsize=10,
                     color=COND_COLORS[cond], fontweight="bold")
        ax.tick_params(labelsize=6)

    plt.colorbar(im, ax=axes, fraction=0.02)
    fig.suptitle("Mean FC Matrix by Condition", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/08_fc_by_condition.png", dpi=150)
    plt.close()
    print("  [8/12] FC matrices by condition")


def plot_09_fc_difference_from_healthy(all_groups):
    """FC difference from healthy (disease - healthy) for each condition."""
    fc_healthy = np.mean(
        np.stack([d["fc_matrix"] for d in all_groups["healthy"]["data"]]), axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    conditions_disease = ["rbd", "osa", "insomnia"]

    for ax, cond in zip(axes, conditions_disease):
        fc_cond = np.mean(
            np.stack([d["fc_matrix"] for d in all_groups[cond]["data"]]), axis=0)
        diff = fc_cond - fc_healthy
        vmax = max(0.15, np.abs(diff).max())
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_title(f"{CONDITION_DESCRIPTIONS[cond]} - Healthy", fontsize=10,
                     color=COND_COLORS[cond], fontweight="bold")
        ax.tick_params(labelsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Add ROI group boundaries
        for boundary in [4, 8, 12, 16]:
            ax.axhline(boundary - 0.5, color="gray", lw=0.5, ls="--")
            ax.axvline(boundary - 0.5, color="gray", lw=0.5, ls="--")

    fig.suptitle("FC Difference from Healthy Control\n"
                 "(dashed lines = ROI group boundaries: Frontal|Central|Temporal|Occipital|Subcortical)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/09_fc_diff_from_healthy.png", dpi=150)
    plt.close()
    print("  [9/12] FC difference from healthy")


def plot_10_trait_shift_by_condition(all_groups):
    """Show how trait parameters shift relative to healthy."""
    # Compute mean normalized trait vector per condition
    trait_means = {}
    for cond in VALID_CONDITIONS:
        vecs = [d["traits"].to_normalized_vector() for d in all_groups[cond]["data"]]
        trait_means[cond] = np.mean(vecs, axis=0)

    healthy_mean = trait_means["healthy"]
    conditions_disease = ["rbd", "osa", "insomnia"]

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(TRAIT_NAMES))
    width = 0.25
    for ci, cond in enumerate(conditions_disease):
        diff = trait_means[cond] - healthy_mean
        bars = ax.bar(x + ci * width, diff, width,
                      label=CONDITION_DESCRIPTIONS[cond],
                      color=COND_COLORS[cond], alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(TRAIT_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Normalized Trait Shift (vs Healthy)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Trait Parameter Shift by Condition (relative to Healthy)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/10_trait_shift.png", dpi=150)
    plt.close()
    print("  [10/12] Trait shift by condition")


def plot_11_sleep_onset_and_waso(all_groups):
    """Sleep onset latency and WASO comparison — key for Insomnia."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sleep onset latency
    ax = axes[0]
    data_onset = []
    for cond in VALID_CONDITIONS:
        vals = [_sleep_onset_epoch(d["hypnogram"]) * 0.5
                for d in all_groups[cond]["data"]]  # convert to minutes
        data_onset.append(vals)
    bp = ax.boxplot(data_onset, patch_artist=True, widths=0.6)
    for patch, cond in zip(bp["boxes"], VALID_CONDITIONS):
        patch.set_facecolor(COND_COLORS[cond])
        patch.set_alpha(0.6)
    ax.set_xticklabels([c[:3].upper() for c in VALID_CONDITIONS])
    ax.set_ylabel("Minutes")
    ax.set_title("Sleep Onset Latency")

    # WASO
    ax = axes[1]
    data_waso = []
    for cond in VALID_CONDITIONS:
        vals = [_count_waso_epochs(d["hypnogram"]) * 0.5
                for d in all_groups[cond]["data"]]  # convert to minutes
        data_waso.append(vals)
    bp = ax.boxplot(data_waso, patch_artist=True, widths=0.6)
    for patch, cond in zip(bp["boxes"], VALID_CONDITIONS):
        patch.set_facecolor(COND_COLORS[cond])
        patch.set_alpha(0.6)
    ax.set_xticklabels([c[:3].upper() for c in VALID_CONDITIONS])
    ax.set_ylabel("Minutes")
    ax.set_title("Wake After Sleep Onset (WASO)")

    fig.suptitle("Sleep Continuity Metrics by Condition\n"
                 "(Insomnia: prolonged onset + high WASO; OSA: high WASO from arousals)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/11_onset_waso.png", dpi=150)
    plt.close()
    print("  [11/12] Sleep onset & WASO (Insomnia validation)")


def plot_12_condition_signal_gallery(all_groups):
    """REM epoch signals across conditions — shows key differences at a glance."""
    fig, axes = plt.subplots(4, 4, figsize=(20, 12))

    show_channels = ["EEG_C3", "EMG_chin", "Resp_effort", "SpO2"]
    ch_indices = [CHANNEL_NAMES.index(c) for c in show_channels]

    for row, cond in enumerate(VALID_CONDITIONS):
        d = all_groups[cond]["data"][0]
        psg = d["psg_data"]
        hyp = d["hypnogram"]

        # Find a REM epoch
        rem_epochs = np.where(hyp == REM)[0]
        if len(rem_epochs) == 0:
            rem_epochs = np.where(hyp == N2)[0]  # fallback
        ep = rem_epochs[len(rem_epochs) // 2]

        show_samples = 10 * FS  # 10 seconds
        start = ep * FS * 30
        end = min(start + show_samples, psg.shape[1])
        t = np.arange(end - start) / FS

        for col, (ch_name, ch_idx) in enumerate(zip(show_channels, ch_indices)):
            ax = axes[row, col]
            sig = psg[ch_idx, start:end]
            if ch_name == "SpO2":
                sig = sig * 100  # percentage
                ax.set_ylim(82, 100)
            ax.plot(t, sig, color=COND_COLORS[cond], lw=0.6)
            if row == 0:
                ax.set_title(ch_name, fontsize=10)
            if col == 0:
                ax.set_ylabel(cond.upper(), fontsize=10, fontweight="bold",
                              color=COND_COLORS[cond])
            ax.tick_params(labelsize=6)
            if row == 3:
                ax.set_xlabel("Time (s)", fontsize=8)

    stage_label = "REM"
    fig.suptitle(f"Signal Comparison During {stage_label} Sleep (10s snippets)\n"
                 "RBD: high EMG | OSA: apnea + desaturation | Insomnia: EEG hyperarousal",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/12_condition_signal_gallery.png", dpi=150)
    plt.close()
    print("  [12/12] Condition signal gallery")


# =====================================================================
# Helper functions
# =====================================================================

def _collect_psds(group_data, stage, fs, max_epochs=10):
    """Collect PSD estimates for a given stage across subjects."""
    results = []
    for d in group_data:
        h = d["hypnogram"]
        psg = d["psg_data"]
        epochs = np.where(h == stage)[0][:max_epochs]
        for ep in epochs:
            s = ep * fs * 30
            e = s + fs * 30
            if e <= psg.shape[1]:
                freqs, pxx = welch(psg[0, s:e], fs=fs, nperseg=fs * 2)
                results.append((freqs, pxx))
    return results


def _sleep_onset_epoch(hypnogram):
    """Find the first epoch of sustained sleep (3+ consecutive non-Wake)."""
    for i in range(len(hypnogram) - 2):
        if (hypnogram[i] != W and hypnogram[i + 1] != W and hypnogram[i + 2] != W):
            return i
    return len(hypnogram)


def _count_waso_epochs(hypnogram):
    """Count Wake epochs after sleep onset."""
    onset = _sleep_onset_epoch(hypnogram)
    if onset >= len(hypnogram):
        return 0
    return np.sum(hypnogram[onset:] == W)


if __name__ == "__main__":
    main()
