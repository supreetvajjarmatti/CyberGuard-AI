from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
    / "cic_ids2017_cleaned.csv"
)


def analyze_dataset():
    """Perform initial quality analysis of CIC-IDS2017."""

    print("Loading processed dataset...")

    df = pd.read_csv(
        DATASET_PATH,
        low_memory=False
    )

    print("\n========== DATASET OVERVIEW ==========")

    print(f"Rows       : {df.shape[0]:,}")
    print(f"Columns    : {df.shape[1]:,}")
    print(f"Memory     : {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")

    print("\n========== DATA TYPES ==========")

    print(df.dtypes.value_counts())

    print("\n========== MISSING VALUES ==========")

    missing = df.isnull().sum()

    missing = missing[missing > 0].sort_values(
        ascending=False
    )

    if missing.empty:
        print("No missing values found.")
    else:
        print(missing)

    print("\n========== INFINITE VALUES ==========")

    numeric_df = df.select_dtypes(include=np.number)

    infinite_count = np.isinf(
        numeric_df.to_numpy()
    ).sum()

    print(f"Infinite values: {infinite_count:,}")

    print("\n========== DUPLICATES ==========")

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate rows: {duplicate_count:,}")

    print("\n========== LABEL DISTRIBUTION ==========")

    label_counts = (
        df["Label"]
        .value_counts()
    )

    print(label_counts)

    print("\n========== UNIQUE LABELS ==========")

    print(f"Number of unique labels: {df['Label'].nunique()}")

    print("\nLabels:")

    for label in sorted(df["Label"].dropna().unique()):
        print(f" - {label}")


if __name__ == "__main__":

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    analyze_dataset()