"""sleepsim - Synthetic PSG data generator for sleep digital twin model testing."""

__version__ = "0.2.0"

from .traits import SubjectTraits, generate_subjects
from .stages import SleepStageSequence, STAGE_NAMES
from .channels import PSGChannelGenerator
from .fc_matrix import FCMatrixGenerator
from .generator import SleepDataGenerator
from .conditions import VALID_CONDITIONS, CONDITION_DESCRIPTIONS
