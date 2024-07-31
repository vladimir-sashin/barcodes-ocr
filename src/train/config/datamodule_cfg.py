from typing import Optional

from pydantic import Field

from src.base_config import BaseValidatedConfig
from src.constants import StorageEnum


class ClearMLStorageConfig(BaseValidatedConfig):
    project_name: Optional[str] = None
    dataset_version: Optional[str] = None


class DataSourceConfig(BaseValidatedConfig):
    storage: StorageEnum = StorageEnum.local
    dataset_name: str = 'barcodes'
    clearml_storage_cfg: ClearMLStorageConfig = Field(default=ClearMLStorageConfig())


class ClearMLConfig(BaseValidatedConfig):
    project_name: str = 'Barcode OCR'
    experiment_name: str = 'OCR Training'
    track_in_clearml: bool = True


class DataConfig(BaseValidatedConfig):
    batch_size: int
    num_iterations: int
    n_workers: int
    width: int
    height: int
    vocab: str
    text_size: int


class DataModuleConfig(BaseValidatedConfig):
    data_source_cfg: DataSourceConfig
    data_cfg: DataConfig
