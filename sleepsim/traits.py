"""Individual trait parameters for virtual sleep subjects.

Traits are grounded in known sleep physiology variables that differ
meaningfully between individuals. They control sleep stage transitions
and PSG signal characteristics.
"""

import dataclasses
from typing import List, Optional

import numpy as np


# Trait definitions: (name, min, max, description)
TRAIT_SPECS = [
    ("delta_power",          0.5, 2.0,  "Relative delta (0.5-4 Hz) amplitude in N3"),
    ("spindle_density",      0.3, 1.5,  "Sleep spindle events per 30s epoch in N2"),
    ("spindle_frequency",   11.0, 16.0, "Center frequency of sleep spindles (Hz)"),
    ("alpha_power",          0.5, 1.5,  "Relative alpha (8-12 Hz) amplitude in Wake/N1"),
    ("rem_latency",         60.0, 120.0, "Minutes from sleep onset to first REM"),
    ("sleep_cycle_duration", 80.0, 110.0, "NREM-REM cycle length in minutes"),
    ("rem_density",          0.3, 1.0,  "Fraction of REM epochs with rapid eye movements"),
    ("muscle_atonia_depth",  0.5, 1.0,  "Depth of EMG suppression in REM (1=full atonia)"),
    ("sleep_efficiency",     0.80, 0.98, "Fraction of recording that is sleep"),
    ("n3_fraction",          0.10, 0.25, "Proportion of total sleep time in N3"),
    ("heart_rate_mean",     55.0, 75.0, "Mean heart rate during sleep (bpm)"),
    ("hrv_amplitude",        0.3, 1.0,  "Heart rate variability magnitude"),
]

TRAIT_NAMES = [s[0] for s in TRAIT_SPECS]
TRAIT_MINS = np.array([s[1] for s in TRAIT_SPECS])
TRAIT_MAXS = np.array([s[2] for s in TRAIT_SPECS])
N_TRAITS = len(TRAIT_SPECS)


@dataclasses.dataclass
class SubjectTraits:
    """Container for one virtual subject's trait parameters."""

    subject_id: int
    delta_power: float
    spindle_density: float
    spindle_frequency: float
    alpha_power: float
    rem_latency: float
    sleep_cycle_duration: float
    rem_density: float
    muscle_atonia_depth: float
    sleep_efficiency: float
    n3_fraction: float
    heart_rate_mean: float
    hrv_amplitude: float
    condition: str = "healthy"  # "healthy", "rbd", "osa", "insomnia"

    def to_vector(self) -> np.ndarray:
        """Return trait values as a 1-D array (raw, not normalized)."""
        return np.array([getattr(self, name) for name in TRAIT_NAMES],
                        dtype=np.float64)

    def to_normalized_vector(self) -> np.ndarray:
        """Return trait values normalized to [0, 1] range."""
        raw = self.to_vector()
        return (raw - TRAIT_MINS) / (TRAIT_MAXS - TRAIT_MINS + 1e-10)

    @classmethod
    def from_vector(cls, subject_id: int, values: np.ndarray,
                    condition: str = "healthy") -> "SubjectTraits":
        """Create SubjectTraits from a raw trait vector."""
        kwargs = {"subject_id": subject_id, "condition": condition}
        for i, name in enumerate(TRAIT_NAMES):
            kwargs[name] = float(values[i])
        return cls(**kwargs)


# Correlation structure between traits (approximate physiological relationships)
# Positive: delta_power <-> n3_fraction, sleep_efficiency <-> shorter rem_latency
# Negative: high sleep_efficiency <-> lower rem_latency
_TRAIT_CORRELATION = np.eye(N_TRAITS)

def _build_correlation_matrix() -> np.ndarray:
    """Build a plausible inter-trait correlation matrix."""
    C = np.eye(N_TRAITS)
    idx = {name: i for i, name in enumerate(TRAIT_NAMES)}

    def _set_corr(a: str, b: str, r: float):
        C[idx[a], idx[b]] = r
        C[idx[b], idx[a]] = r

    # Physiologically plausible correlations
    _set_corr("delta_power", "n3_fraction", 0.6)
    _set_corr("spindle_density", "spindle_frequency", -0.2)
    _set_corr("sleep_efficiency", "rem_latency", -0.3)
    _set_corr("sleep_efficiency", "n3_fraction", 0.3)
    _set_corr("rem_density", "muscle_atonia_depth", 0.4)
    _set_corr("heart_rate_mean", "hrv_amplitude", -0.3)
    _set_corr("delta_power", "alpha_power", -0.2)
    _set_corr("rem_latency", "sleep_cycle_duration", 0.3)

    # Ensure positive semi-definiteness via eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 0.01)
    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Re-normalize to correlation matrix
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return C


TRAIT_CORRELATION = _build_correlation_matrix()


def generate_subjects(n: int, condition: str = "healthy",
                      seed: Optional[int] = 42,
                      rng: Optional[np.random.Generator] = None,
                      id_offset: int = 0) -> List[SubjectTraits]:
    """Generate n virtual subjects with correlated trait parameters.

    Args:
        n: Number of subjects to generate.
        condition: Clinical condition ("healthy", "rbd", "osa", "insomnia").
        seed: Random seed (used if rng is None).
        rng: Optional numpy random generator.
        id_offset: Starting subject ID (useful for multi-group datasets).

    Returns:
        List of SubjectTraits instances.
    """
    from .conditions import validate_condition, TRAIT_MODIFIERS
    condition = validate_condition(condition)

    if rng is None:
        rng = np.random.default_rng(seed)

    # Sample from multivariate normal in z-space
    L = np.linalg.cholesky(TRAIT_CORRELATION)
    z = rng.standard_normal((n, N_TRAITS))
    z_correlated = z @ L.T  # (n, N_TRAITS)

    # Map from z-scores to [0, 1] via sigmoid-like CDF
    from scipy.stats import norm
    u = norm.cdf(z_correlated)  # uniform [0, 1] with correlation structure

    # Map to trait ranges
    raw_values = TRAIT_MINS[None, :] + u * (TRAIT_MAXS - TRAIT_MINS)[None, :]

    # Apply condition-specific trait modifications
    modifiers = TRAIT_MODIFIERS.get(condition, {})
    if modifiers:
        trait_idx = {name: i for i, name in enumerate(TRAIT_NAMES)}
        for trait_name, (shift, scale) in modifiers.items():
            if trait_name in trait_idx:
                ti = trait_idx[trait_name]
                raw_values[:, ti] = raw_values[:, ti] * scale + shift
                # Clamp to valid ranges (with small margin for condition shifts)
                trait_range = TRAIT_MAXS[ti] - TRAIT_MINS[ti]
                raw_values[:, ti] = np.clip(
                    raw_values[:, ti],
                    TRAIT_MINS[ti] - 0.2 * trait_range,
                    TRAIT_MAXS[ti] + 0.2 * trait_range,
                )

    subjects = []
    for i in range(n):
        subjects.append(SubjectTraits.from_vector(
            subject_id=id_offset + i, values=raw_values[i], condition=condition))

    return subjects
