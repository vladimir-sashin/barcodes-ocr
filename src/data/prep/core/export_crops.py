import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data.constants import FILENAME_COL
from src.data.logger import LOGGER


def crop_splits(splits_df: pd.DataFrame, input_data_dir: Path, output_dir: Path) -> None:
    LOGGER.info('Cropping barcodes from images...')
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in splits_df['split'].unique():
        split_df = splits_df.loc[splits_df['split'] == split]
        _crop_barcodes_in_split(split_df, input_data_dir, output_dir, split)
    LOGGER.info('Cropped barcodes and annotations are successfully saved to %s', output_dir)


def _crop_barcodes_in_split(split_df: pd.DataFrame, input_data_dir: Path, output_dir: Path, split: str) -> None:
    output_split_dir = output_dir / split
    output_images_dir = output_split_dir / 'data'

    output_split_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)

    for _, row in split_df.iterrows():
        crop = _crop_image(row, input_data_dir)
        _save_crop(crop, row[FILENAME_COL], output_images_dir)

    output_df = split_df.assign(filename=split_df[FILENAME_COL].apply(_get_file_name))
    output_df = output_df.drop_duplicates(subset=[FILENAME_COL])
    output_df.to_csv(output_split_dir / 'annotations.csv')


def _get_file_name(filepath: str) -> str:
    return Path(filepath).name


def _crop_image(sample_row: pd.Series, input_data_dir: Path) -> np.ndarray:
    image = cv2.imread(input_data_dir / sample_row[FILENAME_COL])
    image = image[..., ::-1]
    x1 = int(sample_row['x_from'])
    y1 = int(sample_row['y_from'])
    x2 = int(sample_row['x_from']) + int(sample_row['width'])
    y2 = int(sample_row['y_from']) + int(sample_row['height'])
    crop = image[y1:y2, x1:x2]

    if crop.shape[0] > crop.shape[1]:
        crop = cv2.rotate(crop, 2)

    return crop


def _save_crop(crop: np.ndarray, old_filename: str, output_dir: Path) -> None:
    output_path = str(output_dir / Path(old_filename).name)
    cv2.imwrite(output_path, crop)
