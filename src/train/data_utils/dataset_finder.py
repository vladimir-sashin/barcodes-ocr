from functools import partial
from pathlib import Path

from clearml import Dataset

from src.base_config import BaseValidatedConfig
from src.constants import PREP_IMAGES_FOLDER, PREP_LABELS_FILENAME, StorageEnum
from src.data.constants import SPLITS
from src.dir_utils import get_prepared_data_dir
from src.train.config.datamodule_cfg import DataSourceConfig


class SplitLocation(BaseValidatedConfig):
    img_folder: Path
    ann_file: Path

    @property
    def coco_dir(self) -> Path:
        return self.ann_file.parent


class DatasetSplits(BaseValidatedConfig):
    train: SplitLocation
    valid: SplitLocation
    test: SplitLocation


def _get_split_location(dataset_dir: Path, split: str) -> SplitLocation:
    return SplitLocation(
        img_folder=dataset_dir / split / PREP_IMAGES_FOLDER,
        ann_file=dataset_dir / split / PREP_LABELS_FILENAME,
    )


def get_data_splits_locations(coco_dir: Path) -> DatasetSplits:
    get_split_path = partial(_get_split_location, coco_dir)
    train, valid, test = [get_split_path(split) for split in SPLITS]

    return DatasetSplits(train=train, valid=valid, test=test)


def get_ds_from_cml(data_source_cfg: DataSourceConfig) -> Path:
    return Path(
        Dataset.get(
            dataset_name=data_source_cfg.dataset_name,
            dataset_project=data_source_cfg.clearml_storage_cfg.project_name,
            dataset_version=data_source_cfg.clearml_storage_cfg.dataset_version,
            alias='train_dataset',
        ).get_local_copy(),
    )


def find_dataset(data_source_cfg: DataSourceConfig) -> DatasetSplits:
    if data_source_cfg.storage == StorageEnum.local:
        dataset_dir = get_prepared_data_dir(data_source_cfg.dataset_name, tmp=False)
    elif data_source_cfg.storage == StorageEnum.clearml:
        dataset_dir = get_ds_from_cml(data_source_cfg)

    return get_data_splits_locations(dataset_dir)
