from pathlib import Path
from typing import List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "ml" / "data" / "raw"


def get_csv_files(data_dir: Path = RAW_DATA_DIR) -> List[Path]:
    """
    Return all CSV files available in the raw dataset directory.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {data_dir}"
        )

    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in: {data_dir}"
        )

    return csv_files


def load_dataset(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Load and combine all CSV files from the raw dataset directory.
    """
    csv_files = get_csv_files(data_dir)

    dataframes = []

    for csv_file in csv_files:
        print(f"Loading: {csv_file.name}")

        df = pd.read_csv(
            csv_file,
            low_memory=False
        )

        dataframes.append(df)

    dataset = pd.concat(
        dataframes,
        ignore_index=True
    )

    return dataset


if __name__ == "__main__":
    try:
        dataset = load_dataset()

        print("\nDataset loaded successfully!")
        print(f"Rows    : {dataset.shape[0]:,}")
        print(f"Columns : {dataset.shape[1]:,}")

        print("\nColumn names:")
        print(dataset.columns.tolist())

    except FileNotFoundError as error:
        print(f"\nERROR: {error}")