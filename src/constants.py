from enum import Enum
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIGS = PROJECT_ROOT / 'configs'
TRAIN_CFG = CONFIGS / 'train_eval.yaml'
DATA_CFG = CONFIGS / 'data.yaml'

DATA_DIR = PROJECT_ROOT / 'datasets'
EXPERIMENTS_DIR = PROJECT_ROOT / 'experiments'


class StorageEnum(str, Enum):  # noqa: WPS600 str is required here
    clearml = 'clearml'
    local = 'local'


FILENAME_COL = 'filename'
PREP_LABELS_FILENAME = 'annotations.csv'
PREP_IMAGES_FOLDER = 'data'
GT_COLUMN = 'code'

MONITOR_METRIC = 'string_match/valid'
PRED_TEXT_SIZE = 30
