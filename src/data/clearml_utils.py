from typing import Optional

from clearml import Dataset, Task

from src.data.config import BarcodesDataConfig
from src.data.logger import LOGGER


def get_ds_if_exists(dataset_name: str, project_name: str, alias: str = 'dataset') -> Optional[Dataset]:
    existing_ds_names = {ds['name'] for ds in Dataset.list_datasets(dataset_project=project_name)}
    if dataset_name in existing_ds_names:
        LOGGER.info("'%s' dataset has been found in ClearML's '%s' project.", dataset_name, project_name)
        return Dataset.get(
            dataset_project=project_name,
            dataset_name=dataset_name,
            alias=alias,
        )
    return None


def create_dataset_task(cfg: BarcodesDataConfig, dataset_name: str) -> BarcodesDataConfig:
    Task.force_requirements_env_freeze()
    task = Task.init(
        project_name=cfg.clearml_cfg.project_name,
        task_name=dataset_name,
        output_uri=True,  # If `output_uri=True` uses default ClearML output URI
        reuse_last_task_id=False,
    )

    cfg_dump = cfg.model_dump()
    task.connect_configuration(configuration=cfg_dump)
    return BarcodesDataConfig.model_validate(cfg_dump)  # To enable config overriding in ClearML
