from pathlib import Path
from typing import Optional

import cv2
import pandas as pd
from numpy.typing import NDArray
from torch.utils.data import Dataset

from src.constants import FILENAME_COL, GT_COLUMN
from src.train.data_utils.types import TRANSFORM_TYPE


class BarcodeDataset(Dataset):
    def __init__(
        self,
        anns_path: Path,
        data_folder: Path,
        transforms: Optional[TRANSFORM_TYPE] = None,
    ):
        self.anns = pd.read_csv(anns_path)
        self.data_folder = data_folder
        self.transforms = transforms

    def __getitem__(self, idx: int) -> tuple[NDArray, str, int]:
        image_path = str(self.data_folder / self.anns[FILENAME_COL].iloc[idx])
        image = cv2.imread(image_path)[..., ::-1]
        if image.shape[0] > image.shape[1]:
            image = cv2.rotate(image, 2)
        text = str(self.anns[GT_COLUMN].iloc[idx])

        data_item = {
            'image': image,
            'text': text,
            'text_length': len(text),
        }

        if self.transforms:
            data_item = self.transforms(**data_item)

        return data_item['image'], data_item['text'], data_item['text_length']

    def __len__(self) -> int:
        return len(self.anns)
