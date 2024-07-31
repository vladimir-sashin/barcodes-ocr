from pathlib import Path

from src.data.clearml_utils import get_ds_if_exists
from src.data.config import BarcodesDataConfig
from src.data.dir_utils import handle_tmp_dir
from src.data.prep.clearml_versioning import (
    get_ds_local_copy,
    upload_processed_ds,
)
from src.data.prep.core.export_crops import crop_splits
from src.data.prep.core.split import split_raw_dataset


def prepare_data(cfg: BarcodesDataConfig, raw_dataset_dir: Path, output_dir: Path) -> None:
    # Split to train/val/test
    splits = split_raw_dataset(raw_dataset_dir, cfg.split_cfg.split_ratios, cfg.split_cfg.seed)
    # Crop barcodes from images and save
    crop_splits(splits, raw_dataset_dir, output_dir)


def prepare_data_from_cml(
    cfg: BarcodesDataConfig,
    raw_dataset_name: str,
    processed_dataset_dir: Path,
) -> None:
    if raw_dataset := get_ds_if_exists(raw_dataset_name, cfg.clearml_cfg.project_name, alias='raw_dataset'):
        raw_dataset_dir = get_ds_local_copy(raw_dataset)
        prepare_data(cfg, raw_dataset_dir, processed_dataset_dir)
        upload_processed_ds(processed_dataset_dir, raw_dataset, cfg.clearml_cfg.project_name)
        handle_tmp_dir(processed_dataset_dir, cfg.clearml_cfg.keep_local_copy)
