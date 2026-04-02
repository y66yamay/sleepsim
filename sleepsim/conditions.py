"""Clinical condition profiles for simulating sleep disorders.

Defines how each condition modifies:
1. Trait parameter distributions (systematic shifts from healthy)
2. Sleep stage transition probabilities
3. PSG signal characteristics
4. FC matrix patterns

Supported conditions:
- "healthy" (default): Normal sleep
- "rbd": REM Sleep Behavior Disorder
- "osa": Obstructive Sleep Apnea
- "insomnia": Primary Insomnia
"""

from typing import Dict, Optional
import numpy as np

# Valid condition identifiers
VALID_CONDITIONS = ("healthy", "rbd", "osa", "insomnia")

CONDITION_DESCRIPTIONS = {
    "healthy":  "Healthy control",
    "rbd":      "REM Sleep Behavior Disorder",
    "osa":      "Obstructive Sleep Apnea",
    "insomnia": "Primary Insomnia",
}


# ============================================================
# 1. Trait modification profiles
# ============================================================
# Each condition defines additive shifts and multiplicative scales
# applied to the healthy trait distributions.
# Format: {trait_name: (additive_shift, multiplicative_scale)}

TRAIT_MODIFIERS: Dict[str, Dict[str, tuple]] = {
    "healthy": {},  # no modification

    "rbd": {
        # Key feature: loss of REM atonia
        "muscle_atonia_depth": (-0.35, 1.0),   # much less atonia (lower value)
        "rem_density":         (0.1, 1.0),      # slightly more REM activity
        # RBD often associated with reduced delta in some studies
        "delta_power":         (-0.15, 1.0),
        # Sleep architecture relatively preserved but slight N3 reduction
        "n3_fraction":         (-0.02, 1.0),
    },

    "osa": {
        # Fragmented sleep: reduced efficiency, reduced N3, more arousals
        "sleep_efficiency":    (-0.12, 1.0),    # much lower efficiency
        "n3_fraction":         (-0.08, 1.0),    # significant N3 reduction
        "delta_power":         (-0.3, 1.0),     # reduced slow-wave activity
        # Autonomic effects: higher HR, lower HRV
        "heart_rate_mean":     (8.0, 1.0),      # elevated heart rate
        "hrv_amplitude":       (-0.15, 1.0),    # reduced vagal tone
        # Sleep architecture disruption
        "spindle_density":     (-0.2, 1.0),     # reduced spindle density
        "rem_latency":         (15.0, 1.0),     # delayed REM onset
    },

    "insomnia": {
        # Core: difficulty initiating/maintaining sleep
        "sleep_efficiency":    (-0.18, 1.0),    # significantly reduced
        # Hyperarousal: increased high-frequency EEG
        "alpha_power":         (0.3, 1.0),      # elevated alpha (cortical hyperarousal)
        # Sleep architecture changes
        "n3_fraction":         (-0.04, 1.0),    # mild N3 reduction
        "delta_power":         (-0.2, 1.0),     # reduced delta power
        "spindle_density":     (-0.1, 1.0),     # slightly reduced spindles
        # Autonomic hyperarousal
        "heart_rate_mean":     (5.0, 1.0),      # mildly elevated HR
        "hrv_amplitude":       (-0.1, 1.0),     # reduced HRV
    },
}


# ============================================================
# 2. Stage transition modifiers
# ============================================================
# Functions that modify the base transition matrix per condition.

def get_stage_modifiers(condition: str) -> Dict:
    """Return stage transition modification parameters for a condition.

    Returns dict with keys used by SleepStageSequence to adjust behavior.
    """
    if condition == "healthy":
        return {}

    if condition == "rbd":
        return {
            # RBD: sleep architecture mostly preserved
            # Slightly more REM fragmentation
            "rem_fragmentation": 0.15,  # probability of REM->W micro-arousal
        }

    if condition == "osa":
        return {
            # OSA: periodic arousals (~20-60/hour in severe cases)
            "arousal_rate_per_hour": 30.0,  # apnea-related arousals
            "arousal_from_stages": [1, 2, 3, 4],  # can occur from any sleep stage
            "n3_suppression": 0.5,  # reduce N3 transition probability by this factor
            # Apnea events cluster in REM and supine N1/N2
            "rem_arousal_boost": 1.5,  # more arousals during REM
        }

    if condition == "insomnia":
        return {
            # Insomnia: prolonged onset, frequent WASO, early morning awakening
            "sleep_onset_delay_factor": 2.5,  # multiply sleep onset latency
            "waso_boost": 3.0,  # multiply wake transition probabilities during sleep
            "early_morning_wake_hour": 5.5,  # hour at which wake probability spikes
            "n3_suppression": 0.7,  # mild N3 suppression
        }

    return {}


# ============================================================
# 3. Signal characteristic modifiers
# ============================================================

def get_signal_modifiers(condition: str) -> Dict:
    """Return signal generation modification parameters for a condition.

    These are used by PSGChannelGenerator to add condition-specific features.
    """
    if condition == "healthy":
        return {}

    if condition == "rbd":
        return {
            # REM without atonia: EMG stays elevated during REM
            "rem_emg_tonic_floor": 0.5,   # minimum EMG amplitude in REM
            "rem_emg_phasic_rate": 8.0,   # many more phasic bursts per epoch
            "rem_emg_phasic_amplitude": 2.0,  # larger bursts (limb movements)
            # Possible dream enactment: large EMG bursts
            "rem_movement_artifacts": True,
        }

    if condition == "osa":
        return {
            # Respiratory events
            "apnea_rate_per_hour": 30.0,   # AHI (apnea-hypopnea index)
            "apnea_duration_range": (10.0, 40.0),  # seconds
            "desaturation_depth": 0.08,    # SpO2 drop magnitude (e.g., 0.08 = 8%)
            "desaturation_baseline": 0.95, # baseline SpO2
            # Post-apnea arousal EEG burst
            "arousal_eeg_burst": True,
            # Snoring-related vibration in EMG
            "snoring_artifact": True,
        }

    if condition == "insomnia":
        return {
            # Cortical hyperarousal
            "beta_power_boost": 1.8,       # elevated beta during NREM
            "alpha_intrusion_nrem": True,  # alpha waves intruding into NREM
            "alpha_intrusion_amplitude": 0.4,
        }

    return {}


# ============================================================
# 4. FC matrix modifiers
# ============================================================

def get_fc_modifiers(condition: str, n_roi: int = 20) -> Optional[np.ndarray]:
    """Return an additive FC perturbation matrix for a condition.

    Returns None for healthy, or (n_roi, n_roi) perturbation matrix.
    These perturbations reflect known FC abnormalities in each condition.
    """
    if condition == "healthy":
        return None

    P = np.zeros((n_roi, n_roi))

    # Compute ROI group boundaries proportional to n_roi
    # Groups: Frontal, Central/Parietal, Temporal, Occipital, Subcortical
    group_size = n_roi // 5
    remainder = n_roi % 5
    boundaries = []
    start = 0
    for g in range(5):
        end = start + group_size + (1 if g < remainder else 0)
        boundaries.append((start, end))
        start = end
    frontal = range(boundaries[0][0], boundaries[0][1])
    central = range(boundaries[1][0], boundaries[1][1])
    temporal = range(boundaries[2][0], boundaries[2][1])
    occipital = range(boundaries[3][0], boundaries[3][1])
    subcortical = range(boundaries[4][0], boundaries[4][1])
    cortical = range(0, boundaries[4][0])

    # Insular ROIs: second half of temporal group
    temporal_list = list(temporal)
    insular = temporal_list[len(temporal_list) // 2:]

    if condition == "rbd":
        # RBD: reduced brainstem-cortical connectivity,
        # reduced basal ganglia (subcortical) connectivity
        for i in subcortical:
            for j in cortical:
                P[i, j] = -0.15
                P[j, i] = -0.15
        # Reduced connectivity within subcortical network
        sub_list = list(subcortical)
        for ii, i in enumerate(sub_list):
            for j in sub_list[ii + 1:]:
                P[i, j] = -0.10
                P[j, i] = -0.10

    elif condition == "osa":
        # OSA: reduced default mode network connectivity (frontal-parietal)
        for i in frontal:
            for j in central:
                P[i, j] = -0.12
                P[j, i] = -0.12
        # Reduced insular connectivity
        for i in insular:
            for j in range(n_roi):
                if j != i:
                    P[i, j] -= 0.08
                    P[j, i] -= 0.08
        # Increased subcortical-frontal (compensatory)
        for i in subcortical:
            for j in frontal:
                P[i, j] = 0.08
                P[j, i] = 0.08

    elif condition == "insomnia":
        # Insomnia: hyperconnectivity in arousal networks
        # Increased frontal connectivity (hyperarousal)
        frontal_list = list(frontal)
        for ii, i in enumerate(frontal_list):
            for j in frontal_list[ii + 1:]:
                P[i, j] = 0.12
                P[j, i] = 0.12
        # Increased frontal-subcortical (salience network overactivity)
        for i in frontal:
            for j in subcortical:
                P[i, j] = 0.10
                P[j, i] = 0.10
        # Reduced occipital-frontal (disrupted DMN)
        for i in occipital:
            for j in frontal:
                P[i, j] = -0.08
                P[j, i] = -0.08

    return P


def validate_condition(condition: str) -> str:
    """Validate and normalize condition string."""
    condition = condition.lower().strip()
    if condition not in VALID_CONDITIONS:
        raise ValueError(
            f"Unknown condition '{condition}'. "
            f"Valid conditions: {VALID_CONDITIONS}"
        )
    return condition
