from src.constants import StorageEnum
from src.data.clearml_utils import create_dataset_task
from src.data.config import BarcodesDataConfig, get_data_config
from src.data.dir_utils import get_prepared_ds_name, get_raw_ds_name
from src.data.prep.core.pipe import prepare_data, prepare_data_from_cml
from src.data.prep.dir_utils import get_local_raw_dataset
from src.dir_utils import get_prepared_data_dir


def preprocess_data(cfg: BarcodesDataConfig) -> None:
    if cfg.storage == StorageEnum.clearml:
        # Find out names and paths, create dataset task in ClearML
        cml_dataset_name = get_prepared_ds_name(cfg.dataset_name)
        cfg = create_dataset_task(cfg, cml_dataset_name)

        # Find out parent dataset name (raw version) and output path
        cml_raw_ds_name = get_raw_ds_name(cfg.dataset_name)
        prepared_dataset_dir = get_prepared_data_dir(cfg.dataset_name, tmp=True)

        # Run preprocessing and upload to ClearML as a new dataset version
        prepare_data_from_cml(
            cfg=cfg,
            raw_dataset_name=cml_raw_ds_name,
            processed_dataset_dir=prepared_dataset_dir,
        )

    elif cfg.storage == StorageEnum.local:
        # Find out output path
        prepared_dataset_dir = get_prepared_data_dir(cfg.dataset_name, tmp=False)
        # Run preprocessing and save output dataset locally
        prepare_data(
            cfg=cfg,
            raw_dataset_dir=get_local_raw_dataset(cfg.dataset_name),
            output_dir=prepared_dataset_dir,
        )


if __name__ == '__main__':
    cfg = get_data_config()
    preprocess_data(cfg)
