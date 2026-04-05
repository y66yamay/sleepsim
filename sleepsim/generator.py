"""Top-level orchestrator for synthetic PSG dataset generation.

Wires together subject trait generation, sleep stage sequencing,
PSG signal synthesis, and FC matrix embedding into a single pipeline.
"""

from typing import Dict, Iterator, List, Optional
import warnings
from pathlib import Path

import numpy as np

from .traits import SubjectTraits, generate_subjects
from .stages import SleepStageSequence, STAGE_NAMES
from .channels import PSGChannelGenerator, CHANNEL_NAMES, N_CHANNELS
from .fc_matrix import FCMatrixGenerator
from .conditions import validate_condition, VALID_CONDITIONS


class SleepDataGenerator:
    """Generate a complete synthetic sleep PSG dataset.

    Produces, for each virtual subject:
    - Individual trait parameters
    - A virtual rsfMRI FC matrix (embedding of traits)
    - A hypnogram (sleep stage sequence)
    - Multi-channel PSG time series data

    Supports both full-dataset generation and memory-efficient iteration.
    Supports clinical conditions: "healthy", "rbd", "osa", "insomnia".
    """

    def __init__(self, n_subjects: int = 50, sampling_rate: int = 256,
                 duration_hours: float = 8.0, epoch_sec: float = 30.0,
                 n_roi: int = 20, seed: int = 42,
                 downsample_factor: int = 1,
                 condition: str = "healthy"):
        """
        Args:
            n_subjects: Number of virtual subjects to generate.
            sampling_rate: PSG sampling rate in Hz.
            duration_hours: Recording duration in hours.
            epoch_sec: Epoch length in seconds (standard: 30).
            n_roi: Number of ROIs for FC matrix.
            seed: Master random seed for reproducibility.
            downsample_factor: Factor to downsample PSG signals (e.g., 4 -> fs/4).
            condition: Clinical condition ("healthy", "rbd", "osa", "insomnia").
        """
        self.condition = validate_condition(condition)
        self.n_subjects = n_subjects
        self.sampling_rate = sampling_rate
        self.duration_hours = duration_hours
        self.epoch_sec = epoch_sec
        self.n_roi = n_roi
        self.seed = seed
        self.downsample_factor = downsample_factor
        self.n_epochs = int(duration_hours * 3600 / epoch_sec)

        # Pre-generate subjects and FC matrices (lightweight)
        self.rng = np.random.default_rng(seed)
        self.subjects = generate_subjects(
            n_subjects, condition=self.condition, rng=self.rng)
        self.fc_gen = FCMatrixGenerator(n_roi=n_roi, rng=np.random.default_rng(seed + 1))

    def generate_subject(self, traits: SubjectTraits) -> Dict:
        """Generate all data for a single subject.

        Returns:
            dict with keys: 'traits', 'fc_matrix', 'hypnogram', 'psg_data',
            'channel_names', 'sampling_rate'.
        """
        # Use SeedSequence to avoid collisions with large subject IDs
        ss = np.random.SeedSequence([self.seed, traits.subject_id])
        rng_stage, rng_signal = [np.random.default_rng(s) for s in ss.spawn(2)]

        # Generate hypnogram
        stage_gen = SleepStageSequence(
            traits, duration_hours=self.duration_hours,
            epoch_sec=self.epoch_sec, rng=rng_stage
        )
        hypnogram = stage_gen.generate()

        # Generate FC matrix
        fc_matrix = self.fc_gen.generate(traits)

        # Generate PSG signals
        channel_gen = PSGChannelGenerator(
            traits, sampling_rate=self.sampling_rate,
            epoch_sec=self.epoch_sec, rng=rng_signal
        )
        psg_data = channel_gen.generate_all(hypnogram)

        # Downsample if requested
        if self.downsample_factor > 1:
            psg_data = psg_data[:, ::self.downsample_factor]

        effective_fs = self.sampling_rate // self.downsample_factor

        return {
            "traits": traits,
            "fc_matrix": fc_matrix,
            "hypnogram": hypnogram,
            "psg_data": psg_data,
            "channel_names": CHANNEL_NAMES,
            "sampling_rate": effective_fs,
        }

    def generate_subject_iter(self) -> Iterator[Dict]:
        """Lazily yield data for each subject (memory-efficient).

        Yields:
            dict for each subject (same format as generate_subject).
        """
        for traits in self.subjects:
            yield self.generate_subject(traits)

    def generate_dataset(self) -> Dict:
        """Generate the complete dataset for all subjects.

        Warning: This loads all PSG data into memory. For large datasets,
        prefer generate_subject_iter().

        Returns:
            dict with keys:
            - 'subjects': List[SubjectTraits]
            - 'fc_matrices': np.ndarray (n_subjects, n_roi, n_roi)
            - 'hypnograms': np.ndarray (n_subjects, n_epochs)
            - 'psg_data': List[np.ndarray], each (n_channels, total_samples)
            - 'channel_names': List[str]
            - 'sampling_rate': int
            - 'metadata': dict
        """
        # Warn about memory usage for large datasets
        samples_per_subject = int(self.duration_hours * 3600 * self.sampling_rate
                                  / self.downsample_factor)
        est_bytes = self.n_subjects * N_CHANNELS * samples_per_subject * 4  # float32
        est_gb = est_bytes / (1024 ** 3)
        if est_gb > 1.0:
            warnings.warn(
                f"generate_dataset() will load ~{est_gb:.1f} GB of PSG data into "
                f"memory. Consider using generate_subject_iter() for large datasets.")

        fc_matrices = self.fc_gen.generate_batch(self.subjects)
        hypnograms = []
        psg_list = []

        for subject_data in self.generate_subject_iter():
            hypnograms.append(subject_data["hypnogram"])
            psg_list.append(subject_data["psg_data"])

        return {
            "subjects": self.subjects,
            "fc_matrices": fc_matrices,
            "hypnograms": np.stack(hypnograms),
            "psg_data": psg_list,
            "channel_names": CHANNEL_NAMES,
            "sampling_rate": self.sampling_rate // self.downsample_factor,
            "epoch_sec": self.epoch_sec,
            "metadata": {
                "n_subjects": self.n_subjects,
                "duration_hours": self.duration_hours,
                "n_roi": self.n_roi,
                "seed": self.seed,
                "downsample_factor": self.downsample_factor,
                "condition": self.condition,
            },
        }

    def save_to_disk(self, output_dir, fmt: str = "npz",
                     compress: bool = True) -> Dict:
        """Generate and save the full dataset to disk.

        Writes a directory structure containing:
            output_dir/
            ├── metadata.json
            ├── traits.csv
            └── subjects/
                ├── subject_0000.npz (or .edf)
                ├── subject_0000_hypnogram.csv
                └── ...

        Uses memory-efficient iteration: only one subject is held in memory
        at a time.

        Args:
            output_dir: Directory path to write files into.
            fmt: Per-subject PSG format ("npz" or "edf").
                 EDF requires `pyedflib` (pip install pyedflib).
            compress: Use compressed NPZ (ignored for EDF).

        Returns:
            Dict with summary info ({"n_subjects_saved", "output_dir"}).
        """
        from .io import save_dataset

        meta = {
            "n_subjects": self.n_subjects,
            "sampling_rate": self.sampling_rate // self.downsample_factor,
            "duration_hours": self.duration_hours,
            "epoch_sec": self.epoch_sec,
            "n_roi": self.n_roi,
            "seed": self.seed,
            "downsample_factor": self.downsample_factor,
            "condition": self.condition,
        }
        return save_dataset(
            self.generate_subject_iter(),
            output_dir=output_dir,
            metadata=meta,
            fmt=fmt,
            epoch_sec=self.epoch_sec,
            compress=compress,
        )

    def generate_epoch_batch(self, subject_indices: List[int],
                             epoch_indices: List[int]) -> Dict:
        """Generate specific epochs for specific subjects (for RNN training).

        Args:
            subject_indices: Which subjects to include.
            epoch_indices: Which epoch positions to generate.

        Returns:
            dict with 'psg_epochs' (n_subjects, n_epochs, n_channels, epoch_samples),
            'stages' (n_subjects, n_epochs), 'fc_matrices' (n_subjects, n_roi, n_roi).
        """
        epoch_samples = int(self.sampling_rate * self.epoch_sec)
        if self.downsample_factor > 1:
            epoch_samples //= self.downsample_factor

        n_sub = len(subject_indices)
        n_ep = len(epoch_indices)

        psg_epochs = np.zeros((n_sub, n_ep, N_CHANNELS, epoch_samples), dtype=np.float32)
        stages = np.zeros((n_sub, n_ep), dtype=np.int8)
        fc_matrices = np.zeros((n_sub, self.n_roi, self.n_roi), dtype=np.float32)

        for si, subj_idx in enumerate(subject_indices):
            traits = self.subjects[subj_idx]
            ss = np.random.SeedSequence([self.seed, traits.subject_id])
            rng_stage, rng_signal = [np.random.default_rng(s) for s in ss.spawn(2)]

            # Full hypnogram (needed for stage lookup)
            stage_gen = SleepStageSequence(
                traits, duration_hours=self.duration_hours,
                epoch_sec=self.epoch_sec, rng=rng_stage
            )
            hypnogram = stage_gen.generate()

            # FC matrix
            fc_matrices[si] = self.fc_gen.generate(traits)

            # Generate only requested epochs
            channel_gen = PSGChannelGenerator(
                traits, sampling_rate=self.sampling_rate,
                epoch_sec=self.epoch_sec, rng=rng_signal
            )

            for ei, ep_idx in enumerate(epoch_indices):
                if ep_idx < len(hypnogram):
                    stage = int(hypnogram[ep_idx])
                    stages[si, ei] = stage
                    epoch_data = channel_gen.generate_epoch(stage, ep_idx)
                    if self.downsample_factor > 1:
                        epoch_data = epoch_data[:, ::self.downsample_factor]
                    psg_epochs[si, ei] = epoch_data[:, :epoch_samples]

        return {
            "psg_epochs": psg_epochs,
            "stages": stages,
            "fc_matrices": fc_matrices,
        }
