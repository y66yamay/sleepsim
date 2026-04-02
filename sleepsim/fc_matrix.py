"""Trait-to-FC matrix embedding.

Maps individual trait parameter vectors to symmetric functional connectivity
matrices that mimic resting-state fMRI FC matrices. The mapping is structured
so that a hypernetwork can learn to invert it (FC -> traits -> PSG).
"""

from typing import Optional

import numpy as np

from .traits import SubjectTraits, N_TRAITS, TRAIT_NAMES
from .conditions import get_fc_modifiers


# Default ROI group assignments (which traits influence which brain regions)
# ROIs 0-3: Frontal, 4-7: Central/Parietal, 8-11: Temporal,
# 12-15: Occipital, 16-19: Subcortical/Brainstem
DEFAULT_N_ROI = 20

ROI_LABELS = [
    "L_DLPFC", "R_DLPFC", "L_mPFC", "R_mPFC",        # Frontal (0-3)
    "L_Motor", "R_Motor", "L_Parietal", "R_Parietal",  # Central/Parietal (4-7)
    "L_Temporal", "R_Temporal", "L_Insula", "R_Insula", # Temporal (8-11)
    "L_Occipital", "R_Occipital", "L_Cuneus", "R_Cuneus", # Occipital (12-15)
    "Thalamus", "Hypothalamus", "Brainstem", "Cerebellum", # Subcortical (16-19)
]


def _build_roi_weight_matrix(n_roi: int = DEFAULT_N_ROI) -> np.ndarray:
    """Build the fixed weight matrix mapping traits to ROI activations.

    Returns:
        W of shape (n_roi, n_traits). Each row defines how strongly
        each trait influences that ROI's activation level.
    """
    idx = {name: i for i, name in enumerate(TRAIT_NAMES)}
    W = np.zeros((n_roi, N_TRAITS))

    # Compute ROI group boundaries proportional to n_roi
    group_size = n_roi // 5
    remainder = n_roi % 5
    boundaries = []
    start = 0
    for g in range(5):
        end = start + group_size + (1 if g < remainder else 0)
        boundaries.append((start, end))
        start = end

    # Frontal ROIs: delta_power, spindle_density, sleep_efficiency
    for roi in range(boundaries[0][0], boundaries[0][1]):
        W[roi, idx["delta_power"]] = 0.8
        W[roi, idx["spindle_density"]] = 0.6
        W[roi, idx["sleep_efficiency"]] = 0.5
        W[roi, idx["n3_fraction"]] = 0.4

    # Central/Parietal ROIs: spindle_frequency, alpha_power, spindle_density
    for roi in range(boundaries[1][0], boundaries[1][1]):
        W[roi, idx["spindle_frequency"]] = 0.7
        W[roi, idx["alpha_power"]] = 0.6
        W[roi, idx["spindle_density"]] = 0.5
        W[roi, idx["delta_power"]] = 0.3

    # Temporal ROIs: alpha_power, spindle_density
    for roi in range(boundaries[2][0], boundaries[2][1]):
        W[roi, idx["alpha_power"]] = 0.8
        W[roi, idx["spindle_density"]] = 0.5
        W[roi, idx["spindle_frequency"]] = 0.3

    # Occipital ROIs: alpha_power dominant
    for roi in range(boundaries[3][0], boundaries[3][1]):
        W[roi, idx["alpha_power"]] = 1.0
        W[roi, idx["delta_power"]] = 0.2

    # Subcortical/Brainstem ROIs: REM/autonomic traits
    for roi in range(boundaries[4][0], boundaries[4][1]):
        W[roi, idx["rem_latency"]] = 0.6
        W[roi, idx["rem_density"]] = 0.7
        W[roi, idx["muscle_atonia_depth"]] = 0.5
        W[roi, idx["heart_rate_mean"]] = 0.4
        W[roi, idx["hrv_amplitude"]] = 0.5
        W[roi, idx["sleep_cycle_duration"]] = 0.4

    # Add slight random variation per ROI within groups for differentiation
    rng = np.random.default_rng(12345)
    W += 0.1 * rng.standard_normal(W.shape)
    W = np.clip(W, 0, None)  # Keep non-negative

    return W


class FCMatrixGenerator:
    """Generate virtual rsfMRI functional connectivity matrices from traits.

    The mapping is: traits -> ROI activations -> outer product -> FC matrix.
    This creates a structured, invertible relationship between individual
    traits and the FC pattern.
    """

    def __init__(self, n_roi: int = DEFAULT_N_ROI, noise_scale: float = 0.05,
                 rng: Optional[np.random.Generator] = None):
        self.n_roi = n_roi
        self.noise_scale = noise_scale
        self.rng = rng or np.random.default_rng(99)
        self.W_roi = _build_roi_weight_matrix(n_roi)

    def generate(self, traits: SubjectTraits) -> np.ndarray:
        """Generate a symmetric FC matrix for one subject.

        Args:
            traits: Subject's trait parameters.

        Returns:
            np.ndarray of shape (n_roi, n_roi), symmetric, diagonal=1,
            values approximately in [-1, 1].
        """
        # Normalized trait vector
        t_norm = traits.to_normalized_vector()  # (n_traits,)

        # ROI activation vector
        a = self.W_roi @ t_norm  # (n_roi,)

        # Center and scale activations
        a = (a - a.mean()) / (a.std() + 1e-10)

        # Raw FC via outer product
        fc_raw = np.outer(a, a)  # (n_roi, n_roi)

        # Add symmetric noise
        noise = self.rng.standard_normal((self.n_roi, self.n_roi))
        noise = (noise + noise.T) / 2.0  # symmetrize
        fc = fc_raw + self.noise_scale * noise

        # Apply condition-specific FC perturbations
        fc_perturb = get_fc_modifiers(traits.condition, self.n_roi)
        if fc_perturb is not None:
            fc += fc_perturb

        # Normalize to correlation-like range via Fisher-z scaling
        # Scale so off-diagonal values are mostly in [-0.8, 0.8]
        fc_max = np.abs(fc).max()
        if fc_max > 0:
            fc = fc / fc_max * 0.8

        # Force diagonal to 1.0
        np.fill_diagonal(fc, 1.0)

        # Ensure symmetry
        fc = (fc + fc.T) / 2.0

        return fc

    def generate_batch(self, subjects: list) -> np.ndarray:
        """Generate FC matrices for multiple subjects.

        Args:
            subjects: List of SubjectTraits.

        Returns:
            np.ndarray of shape (n_subjects, n_roi, n_roi).
        """
        return np.stack([self.generate(s) for s in subjects])
