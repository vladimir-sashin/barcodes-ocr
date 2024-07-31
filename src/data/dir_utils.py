import shutil
from pathlib import Path

from src.data.constants import (
    PREPARED_DATASET_PREFIX,
    RAW_DATASET_PREFIX,
    RAW_DATASETS,
    TMP_RAW_DATASETS,
)


def handle_tmp_dir(dir_path: Path, keep: bool) -> None:
    if not keep:
        shutil.rmtree(str(dir_path))


def get_raw_ds_name(dataset_name: str) -> str:
    return f'{RAW_DATASET_PREFIX}{dataset_name}'


def get_raw_data_dir(dataset_name: str, tmp: bool = False) -> Path:
    if tmp:
        return TMP_RAW_DATASETS / dataset_name
    return RAW_DATASETS / dataset_name


def get_prepared_ds_name(dataset_name: str) -> str:
    return f'{PREPARED_DATASET_PREFIX}{dataset_name}'
