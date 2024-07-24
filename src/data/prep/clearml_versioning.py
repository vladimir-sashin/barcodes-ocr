from pathlib import Path
from typing import Sequence

from clearml import Dataset


def upload_processed_ds(
    processed_dir: Path,
    raw_dataset: Dataset,
    project_name: str,
    tags: Sequence[str] = ('preprocessed',),
) -> None:
    processed_ds = Dataset.create(
        use_current_task=True,
        dataset_project=project_name,
        parent_datasets=[raw_dataset],
        dataset_tags=tags,
    )
    processed_ds.sync_folder(local_path=processed_dir)
    processed_ds.finalize(auto_upload=True)


def get_ds_local_copy(dataset: Dataset) -> Path:
    return Path(dataset.get_local_copy())
