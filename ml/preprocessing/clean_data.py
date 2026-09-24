from pathlib import Path

import numpy as np
import pandas as pd

from ml.preprocessing.data_loader import get_csv_files


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "ml" / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "ml" / "data" / "processed"


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove leading and trailing whitespace from column names.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform initial cleaning without removing valid records.
    """
    df = df.copy()

    # Clean column names
    df = clean_column_names(df)

    # Replace infinite values with NaN
    numeric_columns = df.select_dtypes(include=np.number).columns

    if len(numeric_columns) > 0:
        df[numeric_columns] = df[numeric_columns].replace(
            [np.inf, -np.inf],
            np.nan
        )

    return df


def process_file(input_file: Path) -> Path:
    """
    Clean one raw CSV file and save it separately.
    """

    print("\n" + "=" * 70)
    print(f"Processing: {input_file.name}")
    print("=" * 70)

    # Load one file
    df = pd.read_csv(
        input_file,
        low_memory=False
    )

    original_rows = len(df)

    print(f"Original rows : {original_rows:,}")

    # Clean
    df = clean_dataset(df)

    cleaned_rows = len(df)

    print(f"Cleaned rows  : {cleaned_rows:,}")

    # Safety check
    if original_rows != cleaned_rows:
        raise RuntimeError(
            f"ROW COUNT CHANGED!\n"
            f"File: {input_file.name}\n"
            f"Original: {original_rows:,}\n"
            f"Cleaned : {cleaned_rows:,}"
        )

    # Create processed directory
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        PROCESSED_DATA_DIR
        / f"{input_file.stem}_cleaned.csv"
    )

    # Save
    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )

    print(f"Saved to      : {output_file.name}")

    return output_file


def main():
    """
    Process every raw CIC-IDS2017 CSV separately.
    """

    csv_files = get_csv_files(RAW_DATA_DIR)

    print(f"\nFound {len(csv_files)} raw CSV files.")

    total_original = 0
    total_cleaned = 0

    for csv_file in csv_files:

        output_file = process_file(csv_file)

        # Verify saved file by reading it again
        saved_df = pd.read_csv(
            output_file,
            low_memory=False
        )

        saved_rows = len(saved_df)

        original_rows = len(
            pd.read_csv(
                csv_file,
                low_memory=False
            )
        )

        if saved_rows != original_rows:
            raise RuntimeError(
                f"SAVED ROW COUNT MISMATCH!\n"
                f"File: {csv_file.name}\n"
                f"Original: {original_rows:,}\n"
                f"Saved: {saved_rows:,}"
            )

        total_original += original_rows
        total_cleaned += saved_rows

        print(f"Verified rows : {saved_rows:,}")

    print("\n" + "=" * 70)
    print("FINAL VERIFICATION")
    print("=" * 70)

    print(f"Total raw rows       : {total_original:,}")
    print(f"Total processed rows : {total_cleaned:,}")

    if total_original == total_cleaned:
        print("\nSUCCESS: No rows were lost.")
    else:
        print("\nERROR: Row count mismatch detected.")


if __name__ == "__main__":
    main()