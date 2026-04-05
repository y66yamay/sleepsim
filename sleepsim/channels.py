"""Per-channel PSG signal generation with stage-dependent characteristics.

Generates realistic synthetic EEG, EOG, EMG, and ECG signals whose
spectral and temporal properties change according to sleep stage and
individual trait parameters.
"""

from typing import List, Optional

import numpy as np

from .traits import SubjectTraits
from .stages import W, N1, N2, N3, REM
from .conditions import get_signal_modifiers
from .utils import bandpass_filter, pink_noise, normalize_rms, crossfade


# Non-EEG channels (fixed across all configurations)
NON_EEG_CHANNELS = ["EOG_L", "EOG_R", "EMG_chin", "ECG", "Resp_effort", "SpO2"]

# Default EEG configuration (backward compatible)
DEFAULT_EEG_CHANNELS = ["C3", "C4"]

# Full 8-channel list (for backward compatibility with code referencing
# CHANNEL_NAMES and N_CHANNELS as module-level constants)
CHANNEL_NAMES = ["EEG_C3", "EEG_C4"] + NON_EEG_CHANNELS
N_CHANNELS = len(CHANNEL_NAMES)


# ============================================================
# 10-20 system topographical characteristics
# ============================================================
# Per-channel band-amplitude modulators relative to C3/C4 baseline.
# Based on known spatial distributions:
#   - Delta:    strongest frontally, weakest occipitally
#   - Alpha:    strongest occipitally, weakest frontally
#   - Spindle:  strongest at central, weaker at periphery
#   - Beta:     broadly distributed, slightly stronger frontally
#   - Theta:    relatively uniform
EEG_TOPOGRAPHY = {
    # Prefrontal
    "Fp1": {"delta": 1.10, "theta": 1.00, "alpha": 0.30, "spindle": 0.35, "beta": 1.05},
    "Fp2": {"delta": 1.10, "theta": 1.00, "alpha": 0.30, "spindle": 0.35, "beta": 1.05},
    # Frontal
    "F3":  {"delta": 1.20, "theta": 1.05, "alpha": 0.45, "spindle": 0.55, "beta": 1.05},
    "F4":  {"delta": 1.20, "theta": 1.05, "alpha": 0.45, "spindle": 0.55, "beta": 1.05},
    "F7":  {"delta": 1.05, "theta": 1.00, "alpha": 0.35, "spindle": 0.40, "beta": 1.10},
    "F8":  {"delta": 1.05, "theta": 1.00, "alpha": 0.35, "spindle": 0.40, "beta": 1.10},
    "Fz":  {"delta": 1.25, "theta": 1.05, "alpha": 0.45, "spindle": 0.70, "beta": 1.05},
    # Central (reference)
    "C3":  {"delta": 1.00, "theta": 1.00, "alpha": 0.80, "spindle": 1.00, "beta": 1.00},
    "C4":  {"delta": 1.00, "theta": 1.00, "alpha": 0.80, "spindle": 1.00, "beta": 1.00},
    "Cz":  {"delta": 1.05, "theta": 1.00, "alpha": 0.80, "spindle": 1.10, "beta": 1.00},
    # Temporal
    "T3":  {"delta": 0.85, "theta": 0.95, "alpha": 0.60, "spindle": 0.60, "beta": 0.85},
    "T4":  {"delta": 0.85, "theta": 0.95, "alpha": 0.60, "spindle": 0.60, "beta": 0.85},
    "T5":  {"delta": 0.80, "theta": 0.95, "alpha": 0.95, "spindle": 0.55, "beta": 0.80},
    "T6":  {"delta": 0.80, "theta": 0.95, "alpha": 0.95, "spindle": 0.55, "beta": 0.80},
    # Parietal
    "P3":  {"delta": 0.90, "theta": 0.95, "alpha": 1.05, "spindle": 0.90, "beta": 0.90},
    "P4":  {"delta": 0.90, "theta": 0.95, "alpha": 1.05, "spindle": 0.90, "beta": 0.90},
    "Pz":  {"delta": 0.95, "theta": 0.95, "alpha": 1.05, "spindle": 0.95, "beta": 0.90},
    # Occipital
    "O1":  {"delta": 0.70, "theta": 0.90, "alpha": 1.50, "spindle": 0.45, "beta": 0.75},
    "O2":  {"delta": 0.70, "theta": 0.90, "alpha": 1.50, "spindle": 0.45, "beta": 0.75},
    "Oz":  {"delta": 0.70, "theta": 0.90, "alpha": 1.55, "spindle": 0.45, "beta": 0.75},
    # Auricular / mastoid (used as reference, minimal brain activity)
    "A1":  {"delta": 0.30, "theta": 0.30, "alpha": 0.30, "spindle": 0.20, "beta": 0.40},
    "A2":  {"delta": 0.30, "theta": 0.30, "alpha": 0.30, "spindle": 0.20, "beta": 0.40},
    "M1":  {"delta": 0.30, "theta": 0.30, "alpha": 0.30, "spindle": 0.20, "beta": 0.40},
    "M2":  {"delta": 0.30, "theta": 0.30, "alpha": 0.30, "spindle": 0.20, "beta": 0.40},
}

# Approximate 2D positions (head-space, x=right-left, y=anterior-posterior)
# used to compute inter-channel spatial correlation.
EEG_POSITIONS = {
    "Fp1": (-0.3, 0.9), "Fp2": (0.3, 0.9),
    "F3":  (-0.5, 0.6), "F4":  (0.5, 0.6),
    "F7":  (-0.8, 0.5), "F8":  (0.8, 0.5),
    "Fz":  (0.0, 0.6),
    "C3":  (-0.5, 0.0), "C4":  (0.5, 0.0),
    "Cz":  (0.0, 0.0),
    "T3":  (-1.0, 0.0), "T4":  (1.0, 0.0),
    "T5":  (-0.9, -0.5), "T6":  (0.9, -0.5),
    "P3":  (-0.5, -0.6), "P4":  (0.5, -0.6),
    "Pz":  (0.0, -0.6),
    "O1":  (-0.3, -0.9), "O2":  (0.3, -0.9),
    "Oz":  (0.0, -0.95),
    "A1":  (-1.1, 0.0), "A2":  (1.1, 0.0),
    "M1":  (-1.0, -0.2), "M2":  (1.0, -0.2),
}

AVAILABLE_EEG_CHANNELS = tuple(EEG_TOPOGRAPHY.keys())


def validate_eeg_channels(eeg_channels: List[str]) -> List[str]:
    """Validate and canonicalize an EEG channel name list.

    Args:
        eeg_channels: List of channel names (case-insensitive).

    Returns:
        Canonical channel names in the provided order.

    Raises:
        ValueError: If an unrecognized channel name is provided.
    """
    if not eeg_channels:
        raise ValueError("eeg_channels must contain at least one channel")
    canonical = []
    # Build a case-insensitive lookup (the canonical keys are Title-case)
    lookup = {k.lower(): k for k in EEG_TOPOGRAPHY.keys()}
    for ch in eeg_channels:
        key = ch.lower().strip()
        if key not in lookup:
            raise ValueError(
                f"Unknown EEG channel '{ch}'. Available channels: "
                f"{sorted(EEG_TOPOGRAPHY.keys())}"
            )
        canonical.append(lookup[key])
    return canonical


def build_channel_names(eeg_channels: List[str]) -> List[str]:
    """Build the full channel name list (EEG + fixed non-EEG)."""
    return [f"EEG_{ch}" for ch in eeg_channels] + list(NON_EEG_CHANNELS)


class PSGChannelGenerator:
    """Generate multi-channel PSG signals for a single subject.

    The non-EEG channels (EOG, EMG, ECG, Resp, SpO2) are fixed, but the
    set of EEG channels is configurable via the ``eeg_channels`` parameter.
    Supported channel names are any key in ``EEG_TOPOGRAPHY`` (standard
    10-20 positions: Fp1/2, F3/4/7/8/z, C3/4/z, T3/4/5/6, P3/4/z, O1/2/z,
    A1/2, M1/2).
    """

    def __init__(self, traits: SubjectTraits, sampling_rate: int = 256,
                 epoch_sec: float = 30.0,
                 rng: Optional[np.random.Generator] = None,
                 eeg_channels: Optional[List[str]] = None):
        self.traits = traits
        self.fs = sampling_rate
        self.epoch_sec = epoch_sec
        self.epoch_samples = int(sampling_rate * epoch_sec)
        self.rng = rng or np.random.default_rng(traits.subject_id + 1000)
        self.signal_mods = get_signal_modifiers(traits.condition)

        if eeg_channels is None:
            eeg_channels = list(DEFAULT_EEG_CHANNELS)
        self.eeg_channels = validate_eeg_channels(eeg_channels)
        self.channel_names = build_channel_names(self.eeg_channels)
        self.n_channels = len(self.channel_names)
        self._spatial_mix = self._compute_spatial_mixing_matrix()

    def _compute_spatial_mixing_matrix(self) -> np.ndarray:
        """Compute a mixing matrix that introduces spatial correlation
        between nearby EEG channels.

        Each output channel is a weighted mix of all channel-specific
        sources, with weights decaying with inter-channel distance.
        """
        n_eeg = len(self.eeg_channels)
        M = np.eye(n_eeg)
        positions = [EEG_POSITIONS.get(ch, (0.0, 0.0))
                     for ch in self.eeg_channels]
        decay = 0.6  # smaller -> more local, larger -> more global mixing
        for i in range(n_eeg):
            for j in range(n_eeg):
                if i == j:
                    continue
                d = float(np.hypot(positions[i][0] - positions[j][0],
                                   positions[i][1] - positions[j][1]))
                M[i, j] = np.exp(-d / decay) * 0.3
        # Normalize rows to unit sum of squared weights so that output
        # amplitude is comparable to input.
        M = M / np.sqrt((M ** 2).sum(axis=1, keepdims=True))
        return M

    def generate_epoch(self, stage: int, epoch_index: int = 0) -> np.ndarray:
        """Generate one epoch of all PSG channels.

        Args:
            stage: Sleep stage (0=W, 1=N1, 2=N2, 3=N3, 4=REM).
            epoch_index: Position in the night (for subtle time effects).

        Returns:
            np.ndarray of shape (n_channels, epoch_samples).
        """
        # Generate per-channel EEG sources with topographical weights
        n_eeg = len(self.eeg_channels)
        eeg_sources = np.zeros((n_eeg, self.epoch_samples), dtype=np.float64)
        for i, ch in enumerate(self.eeg_channels):
            eeg_sources[i] = self._generate_eeg(stage, topography=EEG_TOPOGRAPHY[ch])
        # Apply spatial mixing to create physiologically plausible correlations
        eeg_channels_data = self._spatial_mix @ eeg_sources

        eog_l, eog_r = self._generate_eog(stage)
        emg = self._generate_emg(stage)
        ecg = self._generate_ecg(stage)
        resp, spo2 = self._generate_respiratory(stage)

        # Apply condition-specific post-processing to all EEG channels
        if self.signal_mods.get("arousal_eeg_burst") and stage != W:
            eeg_channels_data = self._apply_arousal_bursts_multichannel(
                eeg_channels_data, stage)
        if self.signal_mods.get("alpha_intrusion_nrem") and stage in (N2, N3):
            for i in range(n_eeg):
                eeg_channels_data[i] = self._apply_alpha_intrusion(
                    eeg_channels_data[i])

        return np.concatenate(
            [eeg_channels_data,
             np.stack([eog_l, eog_r, emg, ecg, resp, spo2], axis=0)],
            axis=0,
        )

    def generate_all(self, stages: np.ndarray) -> np.ndarray:
        """Generate continuous multi-channel PSG for a full night.

        Applies cross-fading between epochs to ensure continuity.

        Args:
            stages: Array of stage labels, one per epoch.

        Returns:
            np.ndarray of shape (n_channels, total_samples).
        """
        n_epochs = len(stages)
        overlap = int(0.5 * self.fs)  # 0.5 second crossfade

        # Generate all epochs first
        epoch_data = []
        for i, stage in enumerate(stages):
            epoch_data.append(self.generate_epoch(int(stage), epoch_index=i))

        # Assemble with crossfade per channel
        result_channels = []
        for ch in range(self.n_channels):
            segments = [epoch_data[i][ch] for i in range(n_epochs)]
            continuous = segments[0]
            for seg in segments[1:]:
                continuous = crossfade(continuous, seg, overlap)
            result_channels.append(continuous)

        # Trim/pad to exact length
        target_len = n_epochs * self.epoch_samples
        result = np.zeros((self.n_channels, target_len), dtype=np.float32)
        for ch in range(self.n_channels):
            n = min(len(result_channels[ch]), target_len)
            result[ch, :n] = result_channels[ch][:n]

        return result

    # --- EEG Generation ---

    def _generate_eeg(self, stage: int,
                      topography: Optional[dict] = None) -> np.ndarray:
        """Generate one EEG channel for one epoch.

        Args:
            stage: Sleep stage code.
            topography: Optional per-band multipliers for this channel
                position (see ``EEG_TOPOGRAPHY``). If None, behaves like
                a C3/C4-equivalent signal.
        """
        n = self.epoch_samples

        # Base noise
        noise = self.rng.standard_normal(n)
        signal = np.zeros(n, dtype=np.float64)

        # Stage-dependent band amplitudes, scaled by topography
        amp = self._eeg_band_amplitudes(stage)
        topo = topography or {}
        delta_w = topo.get("delta", 1.0)
        theta_w = topo.get("theta", 1.0)
        alpha_w = topo.get("alpha", 1.0)
        spindle_w = topo.get("spindle", 1.0)
        beta_w = topo.get("beta", 1.0)

        # Delta (0.5-4 Hz)
        signal += delta_w * amp["delta"] * bandpass_filter(noise, 0.5, 4.0, self.fs)

        # Theta (4-8 Hz)
        noise2 = self.rng.standard_normal(n)
        signal += theta_w * amp["theta"] * bandpass_filter(noise2, 4.0, 8.0, self.fs)

        # Alpha (8-12 Hz)
        noise3 = self.rng.standard_normal(n)
        signal += alpha_w * amp["alpha"] * bandpass_filter(noise3, 8.0, 12.0, self.fs)

        # Beta (16-30 Hz)
        noise4 = self.rng.standard_normal(n)
        signal += beta_w * amp["beta"] * bandpass_filter(noise4, 16.0, 30.0, self.fs)

        # Add 1/f noise floor
        signal += 0.3 * pink_noise(n, self.rng)

        # Stage-specific transients
        if stage == N2:
            signal = self._add_spindles(signal, amplitude_scale=spindle_w)
            signal = self._add_k_complexes(signal)

        return signal.astype(np.float64)

    def _eeg_band_amplitudes(self, stage: int) -> dict:
        """Return relative amplitude for each EEG frequency band."""
        t = self.traits
        amps = {
            W:   {"delta": 0.3, "theta": 0.3, "alpha": 1.2 * t.alpha_power, "beta": 0.8},
            N1:  {"delta": 0.4, "theta": 0.8, "alpha": 0.6 * t.alpha_power, "beta": 0.3},
            N2:  {"delta": 0.6, "theta": 0.4, "alpha": 0.2, "beta": 0.2},
            N3:  {"delta": 2.0 * t.delta_power, "theta": 0.3, "alpha": 0.1, "beta": 0.1},
            REM: {"delta": 0.3, "theta": 0.7, "alpha": 0.2, "beta": 0.4},
        }
        a = amps.get(stage, amps[W])

        # Insomnia: elevated beta during NREM (cortical hyperarousal)
        beta_boost = self.signal_mods.get("beta_power_boost", 1.0)
        if beta_boost > 1.0 and stage in (N1, N2, N3):
            a["beta"] *= beta_boost

        return a

    def _add_spindles(self, signal: np.ndarray,
                      amplitude_scale: float = 1.0) -> np.ndarray:
        """Overlay sleep spindles onto N2 EEG signal."""
        t = self.traits
        n = len(signal)
        n_spindles = self.rng.poisson(t.spindle_density)

        for _ in range(n_spindles):
            # Spindle duration: 0.5-2.0 seconds
            dur_sec = self.rng.uniform(0.5, 2.0)
            dur_samples = int(dur_sec * self.fs)
            if dur_samples >= n:
                continue

            # Random position
            pos = self.rng.integers(0, n - dur_samples)

            # Gaussian-windowed sinusoid at spindle_frequency
            t_arr = np.arange(dur_samples) / self.fs
            freq = t.spindle_frequency + self.rng.normal(0, 0.5)
            envelope = np.exp(-0.5 * ((t_arr - dur_sec / 2) / (dur_sec / 4)) ** 2)
            spindle = 1.5 * amplitude_scale * envelope * np.sin(2 * np.pi * freq * t_arr)

            signal[pos:pos + dur_samples] += spindle

        return signal

    def _add_k_complexes(self, signal: np.ndarray) -> np.ndarray:
        """Overlay K-complexes onto N2 EEG signal."""
        n = len(signal)
        n_kc = self.rng.poisson(0.5)  # ~0.5 per epoch on average

        for _ in range(n_kc):
            dur_samples = int(0.5 * self.fs)  # ~0.5 sec
            if dur_samples >= n:
                continue
            pos = self.rng.integers(0, n - dur_samples)

            # Biphasic waveform: sharp negative then broad positive
            t_arr = np.arange(dur_samples) / self.fs
            kc = -3.0 * np.sin(2 * np.pi * 1.0 * t_arr) * np.exp(-2 * t_arr)
            signal[pos:pos + dur_samples] += kc

        return signal

    # --- EOG Generation ---

    def _generate_eog(self, stage: int) -> tuple:
        """Generate left and right EOG channels for one epoch."""
        n = self.epoch_samples
        t = self.traits

        base_l = np.zeros(n)
        base_r = np.zeros(n)

        if stage == W:
            # Saccade-like movements
            n_saccades = self.rng.poisson(5)
            for _ in range(n_saccades):
                upper = max(1, n - int(0.2 * self.fs))
                pos = self.rng.integers(0, upper)
                dur = int(self.rng.uniform(0.05, 0.2) * self.fs)
                dur = min(dur, max(0, n - pos))
                amplitude = self.rng.uniform(0.5, 2.0) * self.rng.choice([-1, 1])
                base_l[pos:pos + dur] += amplitude
                base_r[pos:pos + dur] -= amplitude  # conjugate

        elif stage == REM:
            # Rapid eye movements in bursts
            n_bursts = self.rng.poisson(t.rem_density * 5)
            for _ in range(n_bursts):
                burst_start = self.rng.integers(0, max(1, n - int(2 * self.fs)))
                burst_dur = int(self.rng.uniform(0.5, 2.0) * self.fs)
                n_rems = self.rng.integers(2, 6)
                for _ in range(n_rems):
                    pos = burst_start + self.rng.integers(0, max(1, burst_dur))
                    if pos >= n - int(0.1 * self.fs):
                        continue
                    dur = int(self.rng.uniform(0.03, 0.1) * self.fs)
                    dur = min(dur, n - pos)
                    amplitude = self.rng.uniform(1.0, 3.0) * self.rng.choice([-1, 1])
                    ramp = np.linspace(0, amplitude, dur)
                    base_l[pos:pos + dur] += ramp
                    base_r[pos:pos + dur] -= ramp

        else:
            # Slow rolling eye movements (N1/N2/N3)
            roll_amp = {N1: 0.8, N2: 0.3, N3: 0.1}.get(stage, 0.1)
            freq = self.rng.uniform(0.1, 0.5)
            t_arr = np.arange(n) / self.fs
            roll = roll_amp * np.sin(2 * np.pi * freq * t_arr)
            base_l += roll
            base_r -= roll

        # Add baseline noise
        noise_amp = 0.2
        base_l += noise_amp * self.rng.standard_normal(n)
        base_r += noise_amp * self.rng.standard_normal(n)

        return base_l, base_r

    # --- EMG Generation ---

    def _generate_emg(self, stage: int) -> np.ndarray:
        """Generate chin EMG channel for one epoch."""
        n = self.epoch_samples
        t = self.traits
        mods = self.signal_mods

        # Stage-dependent tonic EMG amplitude
        tonic_amp = {
            W:   1.0,
            N1:  0.6,
            N2:  0.4,
            N3:  0.3,
            REM: 0.1 * (1.0 - t.muscle_atonia_depth),
        }.get(stage, 0.5)

        # RBD: REM without atonia — EMG stays elevated during REM
        if stage == REM:
            rem_floor = mods.get("rem_emg_tonic_floor", 0.0)
            tonic_amp = max(tonic_amp, rem_floor)

        # Broadband noise (20-100 Hz)
        noise = self.rng.standard_normal(n)
        emg = tonic_amp * bandpass_filter(noise, 20.0, min(100.0, self.fs / 2 - 1), self.fs)

        # Phasic activity in REM
        if stage == REM:
            phasic_rate = mods.get("rem_emg_phasic_rate", 2.0)
            phasic_amp = mods.get("rem_emg_phasic_amplitude", 1.0)
            n_twitches = self.rng.poisson(phasic_rate)
            for _ in range(n_twitches):
                pos = self.rng.integers(0, max(1, n - int(0.3 * self.fs)))
                dur = int(self.rng.uniform(0.05, 0.5) * self.fs)
                dur = min(dur, n - pos)
                twitch = phasic_amp * self.rng.uniform(0.5, 1.5) * self.rng.standard_normal(dur)
                emg[pos:pos + dur] += twitch

            # RBD: large movement artifacts (dream enactment)
            if mods.get("rem_movement_artifacts"):
                n_movements = self.rng.poisson(1.5)
                for _ in range(n_movements):
                    pos = self.rng.integers(0, max(1, n - int(2.0 * self.fs)))
                    dur = int(self.rng.uniform(0.5, 2.0) * self.fs)
                    dur = min(dur, n - pos)
                    movement = 3.0 * self.rng.standard_normal(dur)
                    # Envelope shape: ramp up, sustain, ramp down
                    env = np.ones(dur)
                    ramp = min(int(0.1 * self.fs), dur // 3)
                    env[:ramp] = np.linspace(0, 1, ramp)
                    env[-ramp:] = np.linspace(1, 0, ramp)
                    emg[pos:pos + dur] += movement * env

        # OSA: snoring vibration artifact during NREM
        if mods.get("snoring_artifact") and stage in (N1, N2):
            snore_freq = self.rng.uniform(80, 120)  # Hz
            t_arr = np.arange(n) / self.fs
            # Intermittent snoring bursts
            n_snores = self.rng.poisson(3)
            for _ in range(n_snores):
                pos = self.rng.integers(0, max(1, n - int(3 * self.fs)))
                dur = int(self.rng.uniform(1.0, 3.0) * self.fs)
                dur = min(dur, n - pos)
                snore = 0.3 * np.sin(2 * np.pi * snore_freq * t_arr[:dur])
                snore *= np.exp(-0.5 * ((t_arr[:dur] - dur / self.fs / 2) /
                                         (dur / self.fs / 4)) ** 2)
                emg[pos:pos + dur] += snore

        return emg

    # --- ECG Generation ---

    def _generate_ecg(self, stage: int) -> np.ndarray:
        """Generate ECG channel for one epoch."""
        n = self.epoch_samples
        t = self.traits

        # Stage-dependent heart rate modulation
        hr_mod = {
            W:   1.05,
            N1:  0.98,
            N2:  0.95,
            N3:  0.90,
            REM: 1.02,
        }.get(stage, 1.0)

        mean_hr = t.heart_rate_mean * hr_mod
        mean_rr = 60.0 / mean_hr  # seconds

        ecg = np.zeros(n, dtype=np.float64)

        # Place QRS complexes with HRV modulation
        pos = 0.0
        beat_idx = 0
        while pos < n / self.fs:
            # HRV: sinusoidal modulation at respiratory frequency
            hrv_mod = t.hrv_amplitude * 0.05 * np.sin(2 * np.pi * 0.2 * pos)
            rr = mean_rr + hrv_mod + self.rng.normal(0, 0.02)
            rr = max(0.4, min(1.5, rr))  # clamp to reasonable range

            sample_pos = int(pos * self.fs)
            if sample_pos < n:
                ecg = self._place_qrs(ecg, sample_pos)

            pos += rr
            beat_idx += 1

        # Add baseline wander + noise
        t_arr = np.arange(n) / self.fs
        baseline = 0.1 * np.sin(2 * np.pi * 0.15 * t_arr)
        ecg += baseline + 0.05 * self.rng.standard_normal(n)

        return ecg

    def _place_qrs(self, ecg: np.ndarray, pos: int) -> np.ndarray:
        """Place a synthetic QRS complex at the given sample position."""
        qrs_dur = int(0.08 * self.fs)  # ~80ms
        half = qrs_dur // 2

        start = max(0, pos - half)
        end = min(len(ecg), pos + half)
        actual_len = end - start

        if actual_len < 3:
            return ecg

        # Triphasic QRS: small negative Q, tall positive R, small negative S
        t_arr = np.linspace(-1, 1, actual_len)
        q_wave = -0.2 * np.exp(-((t_arr + 0.5) ** 2) / 0.02)
        r_wave = 1.5 * np.exp(-(t_arr ** 2) / 0.01)
        s_wave = -0.3 * np.exp(-((t_arr - 0.4) ** 2) / 0.02)
        qrs = q_wave + r_wave + s_wave

        ecg[start:end] += qrs
        return ecg

    # --- Respiratory / SpO2 Generation ---

    def _generate_respiratory(self, stage: int) -> tuple:
        """Generate respiratory effort and SpO2 channels for one epoch.

        Returns:
            (resp_effort, spo2) tuple of 1-D arrays.
        """
        n = self.epoch_samples
        mods = self.signal_mods

        # Base respiratory effort: sinusoidal at breathing rate
        # Breathing rate varies by stage
        breath_rate = {
            W:   0.25,   # ~15 breaths/min
            N1:  0.23,
            N2:  0.22,
            N3:  0.20,   # slower in deep sleep
            REM: 0.27,   # slightly irregular in REM
        }.get(stage, 0.22)

        t_arr = np.arange(n) / self.fs
        resp = np.sin(2 * np.pi * breath_rate * t_arr)
        # Add some variability
        resp += 0.1 * self.rng.standard_normal(n)

        # REM: more irregular breathing
        if stage == REM:
            freq_mod = 0.05 * np.sin(2 * np.pi * 0.03 * t_arr)
            resp = np.sin(2 * np.pi * (breath_rate + freq_mod) * t_arr)
            resp += 0.15 * self.rng.standard_normal(n)

        # Baseline SpO2: ~96-98% for healthy, encoded as fraction (0.96-0.98)
        baseline_spo2 = mods.get("desaturation_baseline", 0.97)
        spo2 = np.full(n, baseline_spo2) + 0.002 * self.rng.standard_normal(n)

        # OSA: apnea events
        ahi = mods.get("apnea_rate_per_hour", 0.0)
        if ahi > 0 and stage != W:
            dur_range = mods.get("apnea_duration_range", (10.0, 40.0))
            desat_depth = mods.get("desaturation_depth", 0.08)

            # Expected apneas in this epoch
            expected_apneas = ahi * (self.epoch_sec / 3600.0)
            # More apneas in REM and supine
            if stage == REM:
                expected_apneas *= 1.5
            n_apneas = self.rng.poisson(max(0.1, expected_apneas))

            for _ in range(n_apneas):
                apnea_dur_sec = self.rng.uniform(dur_range[0], dur_range[1])
                apnea_dur = int(apnea_dur_sec * self.fs)
                apnea_dur = min(apnea_dur, n // 2)

                margin = apnea_dur + int(5 * self.fs)
                upper = max(1, n - margin) if n > margin else 1
                pos = self.rng.integers(0, upper)

                # Respiratory cessation/reduction
                resp[pos:pos + apnea_dur] *= 0.1  # near-zero effort during apnea
                # Resumption burst
                resume_dur = min(int(2 * self.fs), n - pos - apnea_dur)
                if resume_dur > 0:
                    resume_start = pos + apnea_dur
                    resp[resume_start:resume_start + resume_dur] *= 2.0  # recovery breath

                # SpO2 desaturation (delayed, gradual)
                desat_start = pos + int(apnea_dur * 0.3)
                desat_dur = apnea_dur + int(5 * self.fs)  # extends beyond apnea
                desat_end = min(desat_start + desat_dur, n)
                actual_desat_dur = desat_end - desat_start
                if actual_desat_dur > 0:
                    # V-shaped desaturation
                    half = actual_desat_dur // 2
                    desat_curve = np.concatenate([
                        np.linspace(0, -desat_depth, half),
                        np.linspace(-desat_depth, 0, actual_desat_dur - half),
                    ])
                    # Add individual variation
                    desat_curve *= self.rng.uniform(0.7, 1.3)
                    spo2[desat_start:desat_end] += desat_curve

        # Clamp SpO2 to reasonable range
        spo2 = np.clip(spo2, 0.60, 1.00)

        return resp, spo2

    # --- Condition-specific EEG post-processing ---

    def _apply_arousal_bursts_multichannel(self, eeg: np.ndarray,
                                           stage: int) -> np.ndarray:
        """Add brief high-frequency EEG bursts across all EEG channels."""
        n_eeg, n = eeg.shape
        ahi = self.signal_mods.get("apnea_rate_per_hour", 0.0)
        n_bursts = self.rng.poisson(ahi * self.epoch_sec / 3600.0)

        for _ in range(n_bursts):
            dur = int(self.rng.uniform(1.0, 3.0) * self.fs)
            pos = self.rng.integers(0, max(1, n - dur))
            dur = min(dur, n - pos)

            t_arr = np.arange(dur) / self.fs
            burst = (1.5 * np.sin(2 * np.pi * 10 * t_arr) +
                     0.8 * np.sin(2 * np.pi * 20 * t_arr))
            env = np.exp(-0.5 * ((t_arr - dur / self.fs / 2) /
                                  (dur / self.fs / 4)) ** 2)
            burst *= env

            # Apply burst to each channel with small amplitude variation
            for i in range(n_eeg):
                scale = 1.0 + 0.1 * self.rng.standard_normal()
                eeg[i, pos:pos + dur] += burst * scale

        return eeg

    def _apply_arousal_bursts(self, eeg_c3: np.ndarray,
                              eeg_c4: np.ndarray,
                              stage: int) -> tuple:
        """Add brief high-frequency EEG bursts (post-apnea arousals for OSA)."""
        n = len(eeg_c3)
        # Number of arousal bursts per epoch (correlated with AHI)
        ahi = self.signal_mods.get("apnea_rate_per_hour", 0.0)
        n_bursts = self.rng.poisson(ahi * self.epoch_sec / 3600.0)

        for _ in range(n_bursts):
            dur = int(self.rng.uniform(1.0, 3.0) * self.fs)
            pos = self.rng.integers(0, max(1, n - dur))
            dur = min(dur, n - pos)

            t_arr = np.arange(dur) / self.fs
            # Alpha/beta burst
            burst = (1.5 * np.sin(2 * np.pi * 10 * t_arr) +
                     0.8 * np.sin(2 * np.pi * 20 * t_arr))
            env = np.exp(-0.5 * ((t_arr - dur / self.fs / 2) /
                                  (dur / self.fs / 4)) ** 2)
            burst *= env

            eeg_c3[pos:pos + dur] += burst
            eeg_c4[pos:pos + dur] += burst * 0.9  # slightly different amplitude

        return eeg_c3, eeg_c4

    def _apply_alpha_intrusion(self, eeg: np.ndarray) -> np.ndarray:
        """Add alpha wave intrusion into NREM sleep (insomnia feature)."""
        n = len(eeg)
        amp = self.signal_mods.get("alpha_intrusion_amplitude", 0.4)

        # Intermittent alpha bursts during NREM
        n_intrusions = self.rng.poisson(3)
        for _ in range(n_intrusions):
            dur = int(self.rng.uniform(2.0, 8.0) * self.fs)
            pos = self.rng.integers(0, max(1, n - dur))
            dur = min(dur, n - pos)

            t_arr = np.arange(dur) / self.fs
            freq = self.rng.uniform(8.0, 11.0)
            alpha = amp * self.traits.alpha_power * np.sin(2 * np.pi * freq * t_arr)
            # Gentle envelope
            env = 0.5 * (1 - np.cos(2 * np.pi * np.arange(dur) / dur))
            eeg[pos:pos + dur] += alpha * env

        return eeg
