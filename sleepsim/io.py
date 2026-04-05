"""Data export and loading utilities for synthetic PSG datasets.

Supports multiple output formats:
- NPZ: Compressed numpy format (default, no extra dependencies)
- CSV: Human-readable format for traits, hypnograms (as events), and metadata
- JSON: Metadata and per-subject summary information
- EDF: PSG standard format (requires `pyedflib`, optional)
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .stages import STAGE_NAMES
from .traits import SubjectTraits, TRAIT_NAMES


# ============================================================
# Single-subject exports
# ============================================================

def save_subject_npz(subject_data: Dict, path: Union[str, Path],
                     compress: bool = True) -> None:
    """Save one subject's data to a single NPZ file.

    Args:
        subject_data: Dict returned by SleepDataGenerator.generate_subject()
            or yielded by generate_subject_iter().
        path: Output file path (extension .npz will be added if missing).
        compress: Use compressed NPZ format (smaller files, slower I/O).
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    traits = subject_data["traits"]
    arrays = {
        "psg_data": subject_data["psg_data"].astype(np.float32),
        "hypnogram": subject_data["hypnogram"].astype(np.int8),
        "fc_matrix": subject_data["fc_matrix"].astype(np.float32),
        "trait_vector": traits.to_vector(),
        "trait_names": np.array(TRAIT_NAMES),
        "channel_names": np.array(subject_data["channel_names"]),
        "sampling_rate": np.array(subject_data["sampling_rate"]),
        "subject_id": np.array(traits.subject_id),
        "condition": np.array(traits.condition),
    }
    saver = np.savez_compressed if compress else np.savez
    saver(path, **arrays)


def load_subject_npz(path: Union[str, Path]) -> Dict:
    """Load a subject's data previously saved by save_subject_npz.

    Returns:
        Dict with keys mirroring subject_data (traits as SubjectTraits).
    """
    path = Path(path)
    npz = np.load(path, allow_pickle=False)
    trait_vec = npz["trait_vector"]
    traits = SubjectTraits.from_vector(
        subject_id=int(npz["subject_id"]),
        values=trait_vec,
        condition=str(npz["condition"]),
    )
    return {
        "traits": traits,
        "psg_data": npz["psg_data"],
        "hypnogram": npz["hypnogram"],
        "fc_matrix": npz["fc_matrix"],
        "channel_names": [str(c) for c in npz["channel_names"]],
        "sampling_rate": int(npz["sampling_rate"]),
    }


# ============================================================
# Hypnogram export (CSV as events)
# ============================================================

def save_hypnogram_csv(hypnogram: np.ndarray, path: Union[str, Path],
                       epoch_sec: float = 30.0) -> None:
    """Save hypnogram as CSV in events format (onset, duration, stage).

    Output columns: onset_sec, duration_sec, stage_code, stage_name

    Args:
        hypnogram: 1-D array of stage labels (one per epoch).
        path: Output file path.
        epoch_sec: Epoch duration in seconds.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Collapse to event segments
    events = []
    current = int(hypnogram[0])
    onset = 0.0
    for i in range(1, len(hypnogram)):
        if int(hypnogram[i]) != current:
            events.append((onset, i * epoch_sec - onset, current))
            current = int(hypnogram[i])
            onset = i * epoch_sec
    events.append((onset, len(hypnogram) * epoch_sec - onset, current))

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["onset_sec", "duration_sec", "stage_code", "stage_name"])
        for onset_sec, dur_sec, code in events:
            writer.writerow([f"{onset_sec:.3f}", f"{dur_sec:.3f}",
                             code, STAGE_NAMES[code]])


def save_hypnogram_epochs_csv(hypnogram: np.ndarray, path: Union[str, Path],
                              epoch_sec: float = 30.0) -> None:
    """Save hypnogram as CSV with one row per epoch.

    Output columns: epoch_index, onset_sec, stage_code, stage_name
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch_index", "onset_sec", "stage_code", "stage_name"])
        for i, s in enumerate(hypnogram):
            code = int(s)
            writer.writerow([i, f"{i * epoch_sec:.3f}", code, STAGE_NAMES[code]])


# ============================================================
# Traits export
# ============================================================

def save_traits_csv(subjects: List[SubjectTraits],
                    path: Union[str, Path]) -> None:
    """Save a list of SubjectTraits to CSV (one subject per row).

    Output columns: subject_id, condition, + one column per trait.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "condition"] + list(TRAIT_NAMES))
        for s in subjects:
            vec = s.to_vector()
            writer.writerow([s.subject_id, s.condition] +
                            [f"{v:.6f}" for v in vec])


# ============================================================
# Metadata (JSON)
# ============================================================

def save_metadata_json(metadata: Dict, path: Union[str, Path]) -> None:
    """Save dataset metadata to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_safe(metadata), f, indent=2, ensure_ascii=False)


def _json_safe(obj):
    """Convert numpy types to plain Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if dataclasses.is_dataclass(obj):
        return _json_safe(dataclasses.asdict(obj))
    return obj


# ============================================================
# Dataset-level export
# ============================================================

def save_dataset(subject_iter, output_dir: Union[str, Path],
                 metadata: Optional[Dict] = None,
                 fmt: str = "npz",
                 epoch_sec: float = 30.0,
                 compress: bool = True) -> Dict:
    """Save a full dataset to a directory.

    Creates the following structure:
        output_dir/
        ├── metadata.json
        ├── traits.csv
        └── subjects/
            ├── subject_0000.npz   (or .edf depending on fmt)
            ├── subject_0000_hypnogram.csv
            ├── subject_0001.npz
            └── ...

    Args:
        subject_iter: Iterable yielding subject_data dicts (from
            SleepDataGenerator.generate_subject_iter()).
        output_dir: Directory to write files into.
        metadata: Optional metadata dict to save as metadata.json.
        fmt: Per-subject PSG format: "npz" or "edf".
        epoch_sec: Epoch duration for hypnogram CSV export.
        compress: Use compressed NPZ (ignored for EDF).

    Returns:
        Dict with summary information (n_subjects_saved, output_dir).
    """
    fmt = fmt.lower()
    if fmt not in ("npz", "edf"):
        raise ValueError(f"Unsupported format '{fmt}'. Use 'npz' or 'edf'.")

    output_dir = Path(output_dir)
    subjects_dir = output_dir / "subjects"
    subjects_dir.mkdir(parents=True, exist_ok=True)

    all_traits = []
    n_saved = 0

    for subject_data in subject_iter:
        traits = subject_data["traits"]
        all_traits.append(traits)
        base = subjects_dir / f"subject_{traits.subject_id:04d}"

        if fmt == "npz":
            save_subject_npz(subject_data, base.with_suffix(".npz"),
                             compress=compress)
        elif fmt == "edf":
            save_subject_edf(subject_data, base.with_suffix(".edf"))

        # Hypnogram CSV is always written alongside
        save_hypnogram_csv(
            subject_data["hypnogram"],
            subjects_dir / f"subject_{traits.subject_id:04d}_hypnogram.csv",
            epoch_sec=epoch_sec,
        )
        n_saved += 1

    # Write traits.csv
    save_traits_csv(all_traits, output_dir / "traits.csv")

    # Write metadata.json
    meta = dict(metadata or {})
    meta.update({
        "n_subjects_saved": n_saved,
        "format": fmt,
        "epoch_sec": epoch_sec,
        "channel_names": subject_data["channel_names"] if n_saved > 0 else [],
        "sampling_rate": subject_data["sampling_rate"] if n_saved > 0 else None,
    })
    save_metadata_json(meta, output_dir / "metadata.json")

    return {"n_subjects_saved": n_saved, "output_dir": str(output_dir)}


# ============================================================
# EDF export (optional; requires pyedflib)
# ============================================================

def _require_pyedflib():
    try:
        import pyedflib  # type: ignore
        return pyedflib
    except ImportError as e:
        raise ImportError(
            "EDF export requires pyedflib. Install with: pip install pyedflib"
        ) from e


# Physical value ranges per channel (used for EDF digital->physical scaling)
# Values are approximate and chosen to cover the synthetic signal range.
_EDF_PHYS_RANGES = {
    "EOG_L":       ("uV",      -300.0, 300.0),
    "EOG_R":       ("uV",      -300.0, 300.0),
    "EMG_chin":    ("uV",      -150.0, 150.0),
    "ECG":         ("mV",      -3.0,   3.0),
    "Resp_effort": ("a.u.",    -3.0,   3.0),
    "SpO2":        ("%",       0.0,    100.0),
}

# Default range for any EEG_* channel
_EDF_EEG_RANGE = ("uV", -200.0, 200.0)


def _get_edf_range(channel_name: str):
    """Return (unit, phys_min, phys_max) for a channel name."""
    if channel_name in _EDF_PHYS_RANGES:
        return _EDF_PHYS_RANGES[channel_name]
    if channel_name.startswith("EEG_"):
        return _EDF_EEG_RANGE
    return ("a.u.", -1.0, 1.0)


def save_subject_edf(subject_data: Dict, path: Union[str, Path]) -> None:
    """Save one subject's PSG data to EDF (European Data Format).

    Requires the `pyedflib` package (``pip install pyedflib``). The hypnogram
    is written as EDF annotations at each stage transition.

    Args:
        subject_data: Dict returned by SleepDataGenerator.generate_subject().
        path: Output EDF file path.
    """
    pyedflib = _require_pyedflib()

    path = Path(path)
    if path.suffix.lower() != ".edf":
        path = path.with_suffix(".edf")
    path.parent.mkdir(parents=True, exist_ok=True)

    psg = subject_data["psg_data"]  # (n_channels, n_samples)
    ch_names = subject_data["channel_names"]
    fs = int(subject_data["sampling_rate"])
    traits = subject_data["traits"]
    n_channels, n_samples = psg.shape

    # Build signal headers with appropriate physical ranges
    signal_headers = []
    scaled_data = []
    for i, name in enumerate(ch_names):
        unit, pmin, pmax = _get_edf_range(name)
        # SpO2 signals are stored as fraction (0-1); rescale to percent
        data = psg[i].astype(np.float64)
        if name == "SpO2":
            data = data * 100.0
        # Clip to physical range to avoid EDF digital overflow
        data = np.clip(data, pmin, pmax)
        scaled_data.append(data)
        signal_headers.append({
            "label": name,
            "dimension": unit,
            "sample_frequency": fs,
            "physical_min": pmin,
            "physical_max": pmax,
            "digital_min": -32768,
            "digital_max": 32767,
            "transducer": "synthetic",
            "prefilter": "",
        })

    writer = pyedflib.EdfWriter(
        str(path), n_channels, file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        writer.setSignalHeaders(signal_headers)
        writer.setPatientCode(f"synthetic_{traits.subject_id:04d}")
        writer.setPatientName(f"sleepsim_subject_{traits.subject_id}")
        writer.setEquipment("sleepsim")
        writer.setAdmincode(traits.condition)
        writer.writeSamples(scaled_data)

        # Write hypnogram as annotations
        hyp = subject_data["hypnogram"]
        epoch_sec = n_samples / fs / len(hyp)
        prev = int(hyp[0])
        onset = 0.0
        for i in range(1, len(hyp)):
            if int(hyp[i]) != prev:
                dur = i * epoch_sec - onset
                writer.writeAnnotation(
                    onset, dur, f"Sleep stage {STAGE_NAMES[prev]}")
                prev = int(hyp[i])
                onset = i * epoch_sec
        writer.writeAnnotation(
            onset, len(hyp) * epoch_sec - onset,
            f"Sleep stage {STAGE_NAMES[prev]}")
    finally:
        writer.close()
