from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
)


def get_processed_files():
    """Get all cleaned CIC-IDS2017 files."""

    files = sorted(
        PROCESSED_DATA_DIR.glob("*_cleaned.csv")
    )

    if not files:
        raise FileNotFoundError(
            "No processed dataset files found."
        )

    return files


def create_binary_target(df):
    """
    Convert CIC-IDS2017 labels into binary classification.

    BENIGN = 0
    Attack = 1
    """

    df = df.copy()

    df["Target"] = (
        df["Label"]
        .str.strip()
        .str.upper()
        .ne("BENIGN")
        .astype(int)
    )

    return df


def process_files():

    files = get_processed_files()

    total_rows = 0

    output_files = []

    for file_path in files:

        print("\n" + "=" * 70)
        print(f"Processing: {file_path.name}")
        print("=" * 70)

        df = pd.read_csv(
            file_path,
            low_memory=False
        )

        print(
            f"Original rows: "
            f"{len(df):,}"
        )

        # Create binary target
        df = create_binary_target(df)

        print("\nTarget distribution:")

        print(
            df["Target"]
            .value_counts()
            .sort_index()
        )

        # Remove original text label
        # after target creation
        df = df.drop(
            columns=["Label"]
        )

        output_file = (
            OUTPUT_DIR
            / file_path.name.replace(
                "_cleaned.csv",
                "_features.csv"
            )
        )

        df.to_csv(
            output_file,
            index=False
        )

        # Verify
        saved_df = pd.read_csv(
            output_file,
            low_memory=False
        )

        if len(saved_df) != len(df):
            raise RuntimeError(
                f"Row count mismatch in "
                f"{output_file.name}"
            )

        print(
            f"Saved: {output_file.name}"
        )

        print(
            f"Verified rows: "
            f"{len(saved_df):,}"
        )

        total_rows += len(saved_df)

        output_files.append(output_file)

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)

    print(
        f"\nTotal rows processed: "
        f"{total_rows:,}"
    )

    print(
        f"Files created: "
        f"{len(output_files)}"
    )


if __name__ == "__main__":
    process_files()