from pathlib import Path

from src.data.constants import PREPARED_DATASETS, TMP_PREPARED_DATASETS


def get_prepared_data_dir(dataset_name: str, tmp: bool) -> Path:
    if tmp:
        return TMP_PREPARED_DATASETS / dataset_name
    return PREPARED_DATASETS / dataset_name
