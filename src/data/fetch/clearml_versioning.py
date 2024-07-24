from pathlib import Path

from clearml import Dataset

from src.data.clearml_utils import get_ds_if_exists
from src.data.logger import LOGGER


def upload_raw_dataset(
    dataset_name: str,
    project_name: str,
    dataset_path: Path,
) -> None:
    if get_ds_if_exists(dataset_name, project_name, alias='raw_dataset'):
        LOGGER.info('Skipped uploading of dataset to ClearML.')
        return
    _upload_raw_ds(project_name, dataset_path)
    LOGGER.info('Dataset is successfully uploaded to ClearML.')


def _upload_raw_ds(project_name: str, dataset_path: Path) -> Dataset:
    dataset = Dataset.create(
        dataset_project=project_name,
        use_current_task=True,
    )
    dataset.add_files(dataset_path)
    dataset.tags = ['raw']
    dataset.finalize(auto_upload=True)
    return dataset
