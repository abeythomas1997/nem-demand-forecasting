from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RAW_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "aemo"
    / "operational_demand"
)


def parse_aemo_file(file_path: Path) -> pd.DataFrame:
    """
    Parse one AEMO CSV file.
        C rows: control/report metadata
        I rows: column headers
        D rows: actual data records

    This function uses the I row as the column names and returns
    only the D rows.

    Parameters
    ----------
    file_path:
        Path to an AEMO CSV file.

    Returns
    -------
    pd.DataFrame
        Parsed AEMO data rows.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    ValueError
        If an I header row or D data rows cannot be found.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"AEMO file not found: {file_path}")

    header_row: list[str] | None = None
    data_rows: list[list[str]] = []

    with file_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.reader(csv_file)

        for row in reader:
            if not row:
                continue

            record_type = row[0].strip()

            if record_type == "I":
                header_row = row.copy()
                header_row[0] = "RECORD_TYPE"
                header_row[1] = "DATASET"
                header_row[2] = "DATA_TYPE"
                header_row[3] = "VERSION"

            elif record_type == "D":
                data_rows.append(row)

    if header_row is None:
        raise ValueError(
            f"No AEMO item/header row ('I') found in: {file_path.name}"
        )

    if not data_rows:
        raise ValueError(
            f"No AEMO data rows ('D') found in: {file_path.name}"
        )

    expected_columns = len(header_row)

    invalid_rows = [
        row
        for row in data_rows
        if len(row) != expected_columns
    ]

    if invalid_rows:
        raise ValueError(
            f"{file_path.name} contains {len(invalid_rows)} data rows "
            f"whose column count does not match the header."
        )

    dataframe = pd.DataFrame(
        data_rows,
        columns=header_row,
    )

    dataframe["SOURCE_FILE"] = file_path.name

    return dataframe


def load_aemo_directory(
    directory: Path = DEFAULT_RAW_DIRECTORY,
) -> pd.DataFrame:
    """
    Parse and concatenate all AEMO CSV files in a directory.

    Parameters
    ----------
    directory:
        Directory containing raw AEMO CSV files.

    Returns
    -------
    pd.DataFrame
        Combined data from every CSV file.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist or contains no CSV files.
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"AEMO raw-data directory not found: {directory}"
        )

    csv_files = sorted(directory.glob("*.csv"))

    if not csv_files:
        csv_files = sorted(directory.glob("*.CSV"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in: {directory}"
        )

    parsed_dataframes: list[pd.DataFrame] = []

    for csv_file in csv_files:
        dataframe = parse_aemo_file(csv_file)
        parsed_dataframes.append(dataframe)

    return pd.concat(
        parsed_dataframes,
        ignore_index=True,
    )


def main() -> None:
    """Run a basic inspection of the raw AEMO files."""
    dataframe = load_aemo_directory()

    print(f"Files parsed: {dataframe['SOURCE_FILE'].nunique()}")
    print(f"Rows parsed: {len(dataframe):,}")
    print(f"Columns: {len(dataframe.columns)}")

    print("\nColumn names:")
    print(dataframe.columns.tolist())

    print("\nFirst five rows:")
    print(dataframe.head())

if __name__ == "__main__":
   main()



