"""sleepsim - Synthetic PSG data generator for sleep digital twin model testing."""

__version__ = "0.2.0"

from .traits import SubjectTraits, generate_subjects
from .stages import SleepStageSequence, STAGE_NAMES
from .channels import (
    PSGChannelGenerator,
    EEG_TOPOGRAPHY, AVAILABLE_EEG_CHANNELS,
    DEFAULT_EEG_CHANNELS, NON_EEG_CHANNELS,
)
from .fc_matrix import FCMatrixGenerator
from .generator import SleepDataGenerator
from .conditions import VALID_CONDITIONS, CONDITION_DESCRIPTIONS
from .io import (
    save_subject_npz, load_subject_npz,
    save_hypnogram_csv, save_hypnogram_epochs_csv,
    save_traits_csv, save_metadata_json,
    save_dataset, save_subject_edf,
)
