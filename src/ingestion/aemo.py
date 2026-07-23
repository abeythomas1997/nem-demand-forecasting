from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RAW_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "aemo"
    / "operational_demand"
    / "sample"
)


def _parse_aemo_rows(
    rows: Iterable[list[str]],
    source_file: str,
) -> pd.DataFrame:
    """
    Parse AEMO CSV rows containing C, I and D record types.

    C rows contain report metadata.
    I rows contain column headers.
    D rows contain data records.
    """

    header_row: list[str] | None = None
    data_rows: list[list[str]] = []

    for row in rows:
        if not row:
            continue

        record_type = row[0].strip()

        if record_type == "I":
            header_row = row.copy()

            if len(header_row) < 4:
                raise ValueError(
                    f"Invalid AEMO header row in: {source_file}"
                )

            header_row[0] = "RECORD_TYPE"
            header_row[1] = "DATASET"
            header_row[2] = "DATA_TYPE"
            header_row[3] = "VERSION"

        elif record_type == "D":
            data_rows.append(row)

    if header_row is None:
        raise ValueError(
            f"No AEMO item/header row ('I') found in: {source_file}"
        )

    if not data_rows:
        raise ValueError(
            f"No AEMO data rows ('D') found in: {source_file}"
        )

    expected_columns = len(header_row)

    invalid_rows = [
        row
        for row in data_rows
        if len(row) != expected_columns
    ]

    if invalid_rows:
        raise ValueError(
            f"{source_file} contains {len(invalid_rows)} data rows "
            f"whose column count does not match the header."
        )

    dataframe = pd.DataFrame(
        data_rows,
        columns=header_row,
    )

    dataframe["SOURCE_FILE"] = source_file

    return dataframe


def parse_aemo_file(file_path: str | Path) -> pd.DataFrame:
    """
    Parse one AEMO CSV file stored on disk.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"AEMO file not found: {file_path}"
        )

    with file_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.reader(csv_file)

        return _parse_aemo_rows(
            rows=reader,
            source_file=file_path.name,
        )


def parse_aemo_bytes(
    csv_bytes: bytes,
    source_file: str,
) -> pd.DataFrame:
    """
    Parse an AEMO CSV held in memory as bytes.

    This is used when CSV files are read directly from ZIP archives.
    """

    csv_text = csv_bytes.decode(
        "utf-8-sig",
        errors="replace",
    )

    reader = csv.reader(
        StringIO(csv_text)
    )

    return _parse_aemo_rows(
        rows=reader,
        source_file=source_file,
    )


def load_aemo_directory(
    directory: str | Path = DEFAULT_RAW_DIRECTORY,
) -> pd.DataFrame:
    """
    Parse and concatenate all standalone AEMO CSV files in a directory.
    """

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"AEMO raw-data directory not found: {directory}"
        )

    csv_files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".csv"
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in: {directory}"
        )

    parsed_dataframes = [
        parse_aemo_file(csv_file)
        for csv_file in csv_files
    ]

    return pd.concat(
        parsed_dataframes,
        ignore_index=True,
    )


def main() -> None:
    """Run a basic inspection of standalone raw AEMO CSV files."""

    dataframe = load_aemo_directory()

    print(
        f"Files parsed: "
        f"{dataframe['SOURCE_FILE'].nunique()}"
    )
    print(f"Rows parsed: {len(dataframe):,}")
    print(f"Columns: {len(dataframe.columns)}")

    print("\nColumn names:")
    print(dataframe.columns.tolist())

    print("\nFirst five rows:")
    print(dataframe.head())


if __name__ == "__main__":
    main()
    