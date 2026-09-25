from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
)


def get_processed_files():
    """Return all cleaned CIC-IDS2017 CSV files."""

    files = sorted(
        PROCESSED_DATA_DIR.glob("*_cleaned.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No processed CSV files found in: "
            f"{PROCESSED_DATA_DIR}"
        )

    return files


def analyze_file(file_path):
    """Analyze one processed dataset file."""

    print("\n" + "=" * 70)
    print(f"FILE: {file_path.name}")
    print("=" * 70)

    df = pd.read_csv(
        file_path,
        low_memory=False
    )

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    # Column validation
    if len(df.columns) != 79:
        print("WARNING: Expected 79 columns.")

    if "Label" not in df.columns:
        raise ValueError(
            f"'Label' column missing in {file_path.name}"
        )

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    print("\nMissing values:")

    if missing.empty:
        print("None")
    else:
        print(missing)

    # Infinite values
    numeric_df = df.select_dtypes(
        include=np.number
    )

    infinite_count = np.isinf(
        numeric_df.to_numpy()
    ).sum()

    print(
        f"\nInfinite values: "
        f"{infinite_count:,}"
    )

    # Duplicate rows
    duplicate_count = df.duplicated().sum()

    print(
        f"Duplicate rows: "
        f"{duplicate_count:,}"
    )

    # Labels
    print("\nLabel distribution:")

    label_counts = (
        df["Label"]
        .value_counts()
    )

    print(label_counts)

    return df


def analyze_dataset():

    files = get_processed_files()

    print("\n" + "=" * 70)
    print("CYBERGUARD-AI DATASET QUALITY ANALYSIS")
    print("=" * 70)

    print(
        f"\nProcessed files found: "
        f"{len(files)}"
    )

    total_rows = 0
    all_labels = set()

    for file_path in files:

        df = analyze_file(file_path)

        total_rows += len(df)

        all_labels.update(
            df["Label"].dropna().unique()
        )

    print("\n" + "=" * 70)
    print("FINAL DATASET SUMMARY")
    print("=" * 70)

    print(
        f"\nTotal processed rows: "
        f"{total_rows:,}"
    )

    print(
        f"Total unique labels: "
        f"{len(all_labels)}"
    )

    print("\nAll labels:")

    for label in sorted(all_labels):
        print(f" - {label}")

    expected_rows = 2_830_743

    print("\nRow-count validation:")

    if total_rows == expected_rows:
        print(
            "PASS: All 2,830,743 records "
            "are present."
        )
    else:
        print(
            "WARNING: Row count does not "
            "match the raw dataset."
        )


if __name__ == "__main__":
    analyze_dataset()