from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, RandomSampler

from src.train.config.datamodule_cfg import DataModuleConfig
from src.train.data_utils.dataset_finder import find_dataset
from src.train.data_utils.transforms import get_transforms
from src.train.dataset import BarcodeDataset


class OCRDataModule(LightningDataModule):
    def __init__(self, datamodule_cfg: DataModuleConfig):
        super().__init__()
        data_cfg, data_source_cfg = datamodule_cfg.data_cfg, datamodule_cfg.data_source_cfg
        self.data_cfg = data_cfg
        self._dataset_splits = find_dataset(data_source_cfg)
        self._train_transforms = get_transforms(
            data_cfg=data_cfg,
        )
        self._valid_transforms = get_transforms(
            data_cfg=data_cfg,
            augmentations=False,
        )

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.train_sampler: Optional[RandomSampler] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            train_location = self._dataset_splits.train
            valid_location = self._dataset_splits.valid

            self.train_dataset = BarcodeDataset(
                anns_path=train_location.ann_file,
                data_folder=train_location.img_folder,
                transforms=self._train_transforms,
            )
            self.valid_dataset = BarcodeDataset(
                anns_path=valid_location.ann_file,
                data_folder=valid_location.img_folder,
                transforms=self._valid_transforms,
            )

            if self.data_cfg.num_iterations != -1:
                self.train_sampler = RandomSampler(
                    data_source=self.train_dataset,
                    num_samples=self.data_cfg.num_iterations * self.data_cfg.batch_size,
                )

        if stage == 'test':
            test_location = self._dataset_splits.test
            self.test_dataset = BarcodeDataset(
                anns_path=test_location.ann_file,
                data_folder=test_location.img_folder,
                transforms=self._valid_transforms,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.data_cfg.batch_size,
            num_workers=self.data_cfg.n_workers,
            sampler=self.train_sampler,
            shuffle=False if self.train_sampler else True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.data_cfg.batch_size,
            num_workers=self.data_cfg.n_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.data_cfg.batch_size,
            num_workers=self.data_cfg.n_workers,
            shuffle=False,
            pin_memory=True,
        )
