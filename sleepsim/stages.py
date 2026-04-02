"""Sleep stage sequence generation via time-inhomogeneous Markov chain.

Produces realistic hypnograms with ultradian cycling, appropriate
stage distributions, and trait-dependent characteristics.
"""

from typing import Optional, Tuple, List

import numpy as np

from .traits import SubjectTraits
from .conditions import get_stage_modifiers

# Stage encoding
W, N1, N2, N3, REM = 0, 1, 2, 3, 4
STAGE_NAMES = {W: "Wake", N1: "N1", N2: "N2", N3: "N3", REM: "REM"}
N_STAGES = 5


class SleepStageSequence:
    """Generate a full-night hypnogram based on subject traits.

    Uses a time-inhomogeneous Markov chain on 30-second epochs, with
    transition probabilities modulated by ultradian cycle phase, cycle
    index, individual trait parameters, and clinical condition.
    """

    def __init__(self, traits: SubjectTraits, duration_hours: float = 8.0,
                 epoch_sec: float = 30.0,
                 rng: Optional[np.random.Generator] = None):
        self.traits = traits
        self.duration_hours = duration_hours
        self.epoch_sec = epoch_sec
        self.n_epochs = int(duration_hours * 3600 / epoch_sec)
        self.rng = rng or np.random.default_rng(traits.subject_id)
        self.condition_mods = get_stage_modifiers(traits.condition)

    def generate(self) -> np.ndarray:
        """Generate sleep stage sequence.

        Returns:
            np.ndarray of shape (n_epochs,) with stage labels (0-4).
        """
        stages = np.zeros(self.n_epochs, dtype=np.int8)
        stages[0] = W

        cycle_dur_epochs = int(self.traits.sleep_cycle_duration * 60 / self.epoch_sec)
        rem_latency_epochs = int(self.traits.rem_latency * 60 / self.epoch_sec)

        # Sleep onset: first few epochs are wake
        sleep_onset_epoch = self._compute_sleep_onset()

        for i in range(1, self.n_epochs):
            t_min = i * self.epoch_sec / 60.0  # time in minutes
            cycle_idx = max(0, (i - sleep_onset_epoch)) // cycle_dur_epochs
            cycle_phase = ((i - sleep_onset_epoch) % cycle_dur_epochs) / max(cycle_dur_epochs, 1)

            prev = stages[i - 1]
            T = self._transition_matrix(
                prev_stage=prev,
                epoch_idx=i,
                cycle_idx=cycle_idx,
                cycle_phase=cycle_phase,
                sleep_onset_epoch=sleep_onset_epoch,
                rem_latency_epochs=rem_latency_epochs,
            )

            # Sample next stage
            stages[i] = self.rng.choice(N_STAGES, p=T[prev])

        # Post-process: enforce minimum stage durations and remove impossibilities
        stages = self._postprocess(stages)
        return stages

    def _compute_sleep_onset(self) -> int:
        """Determine the epoch at which sleep onset occurs."""
        # Higher sleep efficiency -> faster onset
        onset_min = 5.0 + (1.0 - self.traits.sleep_efficiency) * 40.0
        onset_min += self.rng.normal(0, 2)
        # Insomnia: prolonged sleep onset
        delay_factor = self.condition_mods.get("sleep_onset_delay_factor", 1.0)
        onset_min *= delay_factor
        onset_min = max(2, onset_min)
        return int(onset_min * 60 / self.epoch_sec)

    def _transition_matrix(self, prev_stage: int, epoch_idx: int,
                           cycle_idx: int, cycle_phase: float,
                           sleep_onset_epoch: int,
                           rem_latency_epochs: int) -> np.ndarray:
        """Compute 5x5 transition matrix for the current epoch."""
        T = np.zeros((N_STAGES, N_STAGES))

        # Pre-sleep-onset: stay in Wake
        if epoch_idx < sleep_onset_epoch:
            T[W, W] = 0.9
            T[W, N1] = 0.1
            T[N1, W] = 0.5
            T[N1, N1] = 0.4
            T[N1, N2] = 0.1
            # If somehow in deeper stages, drift back
            for s in [N2, N3, REM]:
                T[s, N1] = 0.5
                T[s, s] = 0.5
            return self._normalize(T)

        # --- Base transition probabilities ---
        # Wake row
        T[W, W] = 0.3 * (1.0 - self.traits.sleep_efficiency)
        T[W, N1] = 0.6
        T[W, N2] = 0.1

        # N1 row
        T[N1, W] = 0.05 * (1.0 - self.traits.sleep_efficiency)
        T[N1, N1] = 0.3
        T[N1, N2] = 0.6
        T[N1, REM] = 0.05

        # N2 row
        T[N2, W] = 0.02
        T[N2, N1] = 0.05
        T[N2, N2] = 0.70
        T[N2, N3] = 0.18
        T[N2, REM] = 0.05

        # N3 row
        T[N3, N2] = 0.15
        T[N3, N3] = 0.83
        T[N3, W] = 0.02

        # REM row
        T[REM, W] = 0.08
        T[REM, N1] = 0.07
        T[REM, N2] = 0.15
        T[REM, REM] = 0.70

        # --- Modulate by cycle phase ---
        # Early in cycle: favor deeper stages
        # Late in cycle: favor REM
        if cycle_phase < 0.5:
            # Descending phase: bias toward N3
            n3_boost = self.traits.n3_fraction * 2.0 * (1.0 - cycle_phase * 2)
            T[N2, N3] += n3_boost * 0.3
            T[N3, N3] += n3_boost * 0.1
        else:
            # Ascending phase: bias toward REM
            rem_boost = self.traits.rem_density * (cycle_phase - 0.5) * 2
            T[N2, REM] += rem_boost * 0.3
            T[N1, REM] += rem_boost * 0.2
            T[REM, REM] += rem_boost * 0.15

        # --- Modulate by cycle index ---
        # N3 decreases across the night; REM increases
        n3_decay = max(0.0, 1.0 - cycle_idx * 0.25)
        rem_growth = min(1.5, 1.0 + cycle_idx * 0.2)

        T[N2, N3] *= n3_decay
        T[N3, N3] *= n3_decay
        T[N2, REM] *= rem_growth
        T[N1, REM] *= rem_growth
        T[REM, REM] *= rem_growth

        # --- Enforce REM latency ---
        epochs_since_onset = epoch_idx - sleep_onset_epoch
        if epochs_since_onset < rem_latency_epochs:
            # Suppress REM transitions
            for s in range(N_STAGES):
                T[s, REM] = 0.0

        # --- Condition-specific modifications ---
        mods = self.condition_mods

        # RBD: slight REM fragmentation
        rem_frag = mods.get("rem_fragmentation", 0.0)
        if rem_frag > 0:
            T[REM, W] += rem_frag
            T[REM, REM] -= rem_frag * 0.5

        # OSA: periodic arousals from any sleep stage
        arousal_rate = mods.get("arousal_rate_per_hour", 0.0)
        if arousal_rate > 0:
            # Convert arousals/hour to probability per epoch
            p_arousal = arousal_rate * (self.epoch_sec / 3600.0)
            p_arousal = min(p_arousal, 0.5)
            rem_boost = mods.get("rem_arousal_boost", 1.0)
            for s in mods.get("arousal_from_stages", []):
                if s == REM:
                    T[s, W] += p_arousal * rem_boost
                else:
                    T[s, W] += p_arousal
            # N3 suppression
            n3_supp = mods.get("n3_suppression", 1.0)
            T[N2, N3] *= n3_supp
            T[N3, N3] *= n3_supp

        # Insomnia: increased WASO, early morning awakening
        waso_boost = mods.get("waso_boost", 1.0)
        if waso_boost > 1.0:
            for s in [N1, N2, N3, REM]:
                # Scale up wake transition while proportionally reducing others
                old_wake = T[s, W]
                new_wake = min(old_wake * waso_boost, 0.6)
                if old_wake < new_wake:
                    scale = (1.0 - new_wake) / max(1.0 - old_wake, 1e-10)
                    for t_stage in range(N_STAGES):
                        if t_stage != W:
                            T[s, t_stage] *= scale
                    T[s, W] = new_wake
            # N3 suppression
            n3_supp = mods.get("n3_suppression", 1.0)
            T[N2, N3] *= n3_supp
            T[N3, N3] *= n3_supp
        # Early morning awakening
        em_hour = mods.get("early_morning_wake_hour", None)
        if em_hour is not None:
            current_hour = epoch_idx * self.epoch_sec / 3600.0
            if current_hour > em_hour:
                hours_past = current_hour - em_hour
                wake_surge = min(0.4, 0.12 * hours_past)
                for s in [N1, N2, REM]:
                    # Proportionally increase wake at the expense of other transitions
                    old_wake = T[s, W]
                    new_wake = min(old_wake + wake_surge, 0.7)
                    if old_wake < new_wake:
                        scale = (1.0 - new_wake) / max(1.0 - old_wake, 1e-10)
                        for t_stage in range(N_STAGES):
                            if t_stage != W:
                                T[s, t_stage] *= scale
                        T[s, W] = new_wake

        # --- Forbidden transitions ---
        T[W, N3] = 0.0   # No direct Wake -> N3
        T[W, REM] = 0.0  # No direct Wake -> REM
        T[N3, REM] = 0.0 # No direct N3 -> REM (must go through N2)

        return self._normalize(T)

    def _normalize(self, T: np.ndarray) -> np.ndarray:
        """Normalize each row of transition matrix to sum to 1."""
        row_sums = T.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        return T / row_sums

    def _postprocess(self, stages: np.ndarray) -> np.ndarray:
        """Clean up the generated stage sequence.

        - Merge isolated single-epoch stages into neighbors.
        - Merge isolated 2-epoch segments surrounded by the same stage.
        - Ensure no physiologically impossible sequences remain.
        """
        stages = stages.copy()
        n = len(stages)

        # Pass 1: Remove isolated single-epoch stages (except Wake)
        for i in range(1, n - 1):
            if stages[i] == W:
                continue
            prev_same = (stages[i - 1] == stages[i])
            next_same = (stages[i + 1] == stages[i])
            if not prev_same and not next_same:
                stages[i] = stages[i - 1]

        # Pass 2: Merge isolated 2-epoch segments surrounded by the same stage
        for i in range(1, n - 2):
            if stages[i] == W:
                continue
            if (stages[i] == stages[i + 1] and
                    stages[i] != stages[i - 1] and
                    i + 2 < n and stages[i + 2] == stages[i - 1]):
                stages[i] = stages[i - 1]
                stages[i + 1] = stages[i - 1]

        return stages

    def to_events(self) -> List[Tuple[float, float, int]]:
        """Convert stage sequence to (onset_sec, duration_sec, stage) events.

        Must call generate() first and pass the result, or this generates anew.
        """
        stages = self.generate()
        events = []
        current_stage = stages[0]
        onset = 0.0
        for i in range(1, len(stages)):
            if stages[i] != current_stage:
                events.append((onset, i * self.epoch_sec - onset, int(current_stage)))
                current_stage = stages[i]
                onset = i * self.epoch_sec
        events.append((onset, len(stages) * self.epoch_sec - onset, int(current_stage)))
        return events
