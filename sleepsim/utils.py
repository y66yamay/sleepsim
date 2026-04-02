"""Shared utility functions: filters, noise generators, signal helpers."""

import warnings

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(data: np.ndarray, low: float, high: float, fs: float,
                    order: int = 4) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter."""
    nyq = fs / 2.0
    low_n = max(low / nyq, 1e-6)
    high_n = min(high / nyq, 0.9999)
    if low_n >= high_n:
        warnings.warn(
            f"bandpass_filter: normalized low ({low_n:.4f}) >= high ({high_n:.4f}), "
            f"returning zeros (low={low:.1f}, high={high:.1f}, fs={fs:.1f})")
        return np.zeros_like(data)
    sos = butter(order, [low_n, high_n], btype='band', output='sos')
    # Pad signal to avoid edge artifacts for short segments
    pad_len = min(3 * (2 * order + 1), len(data) - 1)
    if pad_len < 1 or len(data) < 2 * pad_len + 1:
        warnings.warn(
            f"bandpass_filter: signal too short (len={len(data)}) for filtering, "
            f"returning zeros")
        return np.zeros_like(data)
    return sosfiltfilt(sos, data, padlen=pad_len)


def lowpass_filter(data: np.ndarray, cutoff: float, fs: float,
                   order: int = 4) -> np.ndarray:
    """Apply zero-phase Butterworth lowpass filter."""
    nyq = fs / 2.0
    cutoff_n = min(cutoff / nyq, 0.9999)
    sos = butter(order, cutoff_n, btype='low', output='sos')
    pad_len = min(3 * (2 * order + 1), len(data) - 1)
    if pad_len < 1 or len(data) < 2 * pad_len + 1:
        warnings.warn(
            f"lowpass_filter: signal too short (len={len(data)}) for filtering, "
            f"returning unfiltered copy")
        return data.copy()
    return sosfiltfilt(sos, data, padlen=pad_len)


def pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Generate 1/f (pink) noise via spectral shaping."""
    white = rng.standard_normal(n_samples)
    freq = np.fft.rfftfreq(n_samples)
    freq[0] = 1.0  # avoid division by zero
    spectrum = np.fft.rfft(white)
    spectrum /= np.sqrt(freq)
    result = np.fft.irfft(spectrum, n=n_samples)
    return result / (np.std(result) + 1e-10)


def crossfade(seg_a: np.ndarray, seg_b: np.ndarray,
              overlap_samples: int) -> np.ndarray:
    """Crossfade two segments using a raised-cosine taper.

    Returns concatenated signal with the overlap region blended.
    seg_a and seg_b are 1-D arrays. The last `overlap_samples` of seg_a
    are blended with the first `overlap_samples` of seg_b.
    """
    if overlap_samples <= 0 or overlap_samples > len(seg_a) or overlap_samples > len(seg_b):
        warnings.warn(
            f"crossfade: invalid overlap ({overlap_samples}) for segments "
            f"(len_a={len(seg_a)}, len_b={len(seg_b)}), concatenating without blend")
        return np.concatenate([seg_a, seg_b])

    taper = 0.5 * (1.0 - np.cos(np.pi * np.arange(overlap_samples) / overlap_samples))
    blended = seg_a[-overlap_samples:] * (1.0 - taper) + seg_b[:overlap_samples] * taper
    return np.concatenate([seg_a[:-overlap_samples], blended, seg_b[overlap_samples:]])


def normalize_rms(data: np.ndarray, target_rms: float = 1.0) -> np.ndarray:
    """Normalize signal to a target RMS amplitude."""
    rms = np.sqrt(np.mean(data ** 2))
    if rms < 1e-10:
        return data
    return data * (target_rms / rms)
