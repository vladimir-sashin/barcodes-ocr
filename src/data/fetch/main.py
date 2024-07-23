from src.constants import StorageEnum
from src.data.clearml_utils import create_dataset_task
from src.data.config import BarcodesDataConfig, get_data_config
from src.data.constants import TMP_RAW_DATASETS
from src.data.dir_utils import (  # noqa: I001 # flake8 FP
    get_raw_data_dir,
    get_raw_ds_name,
    handle_tmp_dir,
)
from src.data.fetch.clearml_versioning import (  # noqa: I005 # flake8 FP
    upload_raw_dataset,
)
from src.data.fetch.core import download_and_unzip


def fetch_raw_dataset(cfg: BarcodesDataConfig) -> None:
    dataset_zip_path = TMP_RAW_DATASETS / f'{cfg.dataset_name}.zip'

    if cfg.storage == StorageEnum.clearml:
        cml_dataset_name = get_raw_ds_name(cfg.dataset_name)
        cfg = create_dataset_task(cfg, cml_dataset_name)
        unzipped_data_dir = get_raw_data_dir(cfg.dataset_name, tmp=True)

        download_and_unzip(cfg.url, dataset_zip_path, unzipped_data_dir)
        upload_raw_dataset(
            cml_dataset_name,
            cfg.clearml_cfg.project_name,
            unzipped_data_dir,
            cfg.clearml_cfg.description,
        )
        handle_tmp_dir(unzipped_data_dir, cfg.clearml_cfg.keep_local_copy)

    elif cfg.storage == StorageEnum.local:
        unzipped_data_dir = get_raw_data_dir(cfg.dataset_name, tmp=False)
        download_and_unzip(cfg.url, dataset_zip_path, unzipped_data_dir)


if __name__ == '__main__':
    cfg = get_data_config()
    fetch_raw_dataset(cfg)
