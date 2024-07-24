from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.config import DEFAULT_SPLIT_RATIOS, SplitRatios
from src.data.constants import SPLITS
from src.data.logger import LOGGER


def read_split_df(tsv_path: Path, ratios: SplitRatios, random_state: int) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep='\t')

    train, remaining = train_test_split(df, train_size=ratios.train, random_state=random_state)
    valid, test = train_test_split(
        remaining,
        test_size=ratios.test / (ratios.test + ratios.valid),
        random_state=random_state,
    )

    train['split'] = SPLITS.train
    valid['split'] = SPLITS.valid
    test['split'] = SPLITS.test

    return pd.concat([train, valid, test], axis=0)


def _get_tsv_path(raw_dataset_path: Path) -> Path:
    try:
        tsv_path = next(raw_dataset_path.rglob('*.tsv'))
    except StopIteration:
        raise ValueError(f'annotations.tsv file is not found in {raw_dataset_path}.')
    return tsv_path


def split_raw_dataset(
    raw_dataset_path: Path,
    ratios: SplitRatios = DEFAULT_SPLIT_RATIOS,
    random_state: int = 42,
) -> pd.DataFrame:
    tsv_path = _get_tsv_path(raw_dataset_path)
    df = read_split_df(tsv_path, ratios, random_state)
    LOGGER.info('Barcodes dataset is successfully split into train/val/test sets')

    return df
