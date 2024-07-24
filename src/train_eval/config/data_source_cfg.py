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
