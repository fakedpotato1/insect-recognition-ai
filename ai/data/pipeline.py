from dataclasses import dataclass

from ai.data.csv_dataset import CsvDataset, load_csv_dataset
from ai.data.splitting import DatasetSplits, train_val_test_split


@dataclass(frozen=True)
class CsvTrainValTestData:
    dataset: CsvDataset
    splits: DatasetSplits


def load_csv_train_val_test(
    csv_path,
    split_ratios=(0.7, 0.2, 0.1),
    random_state=None,
    shuffle=True,
    stratify=True,
    **csv_kwargs,
):
    """
    Load feature-extraction CSV output and split it into train/val/test sets.

    csv_kwargs are forwarded to load_csv_dataset, so callers can set
    label_column, feature_columns, delimiter, encoding, or label_mapping.
    """
    dataset = load_csv_dataset(csv_path, **csv_kwargs)
    splits = train_val_test_split(
        dataset.X,
        dataset.y,
        ratios=split_ratios,
        shuffle=shuffle,
        random_state=random_state,
        stratify=stratify,
    )
    return CsvTrainValTestData(dataset=dataset, splits=splits)
