from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pandas as pd

from src.ingestion.aemo import parse_aemo_bytes


def _read_daily_zip(
    daily_zip_bytes: bytes,
    monthly_zip_name: str,
    daily_zip_name: str,
) -> list[pd.DataFrame]:
    """
    Read and parse every CSV inside one daily ZIP archive.
    """

    dataframes: list[pd.DataFrame] = []

    try:
        with ZipFile(BytesIO(daily_zip_bytes)) as daily_zip:
            csv_names = sorted(
                name
                for name in daily_zip.namelist()
                if not name.endswith("/")
                and not name.startswith("__MACOSX")
                and name.lower().endswith(".csv")
            )

            for csv_name in csv_names:
                csv_bytes = daily_zip.read(csv_name)

                source_file = (
                    f"{monthly_zip_name}/"
                    f"{daily_zip_name}/"
                    f"{csv_name}"
                )

                dataframe = parse_aemo_bytes(
                    csv_bytes=csv_bytes,
                    source_file=source_file,
                )

                dataframes.append(dataframe)

    except BadZipFile as error:
        raise ValueError(
            f"Invalid daily ZIP file: "
            f"{monthly_zip_name}/{daily_zip_name}"
        ) from error

    return dataframes


def load_historical_operational_demand(
    archive_directory: str | Path,
) -> pd.DataFrame:
    """
    Load historical AEMO operational-demand data from nested ZIP files.

    Expected structure:

    archive_directory/
        monthly_archive.zip
            daily_archive.zip
                operational_demand.csv
    """

    archive_path = Path(archive_directory)

    if not archive_path.exists():
        raise FileNotFoundError(
            f"Archive directory does not exist: {archive_path}"
        )

    if not archive_path.is_dir():
        raise NotADirectoryError(
            f"Archive path is not a directory: {archive_path}"
        )

    monthly_zip_paths = sorted(
        path
        for path in archive_path.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".zip"
    )

    if not monthly_zip_paths:
        raise FileNotFoundError(
            f"No monthly ZIP files found in: {archive_path}"
        )

    all_dataframes: list[pd.DataFrame] = []
    daily_zip_count = 0
    csv_count = 0

    for monthly_zip_path in monthly_zip_paths:
        print(f"Reading {monthly_zip_path.name}")

        try:
            with ZipFile(monthly_zip_path) as monthly_zip:
                member_names = sorted(
                    name
                    for name in monthly_zip.namelist()
                    if not name.endswith("/")
                    and not name.startswith("__MACOSX")
                )

                for member_name in member_names:
                    member_bytes = monthly_zip.read(member_name)

                    if member_name.lower().endswith(".zip"):
                        daily_dataframes = _read_daily_zip(
                            daily_zip_bytes=member_bytes,
                            monthly_zip_name=monthly_zip_path.name,
                            daily_zip_name=member_name,
                        )

                        daily_zip_count += 1
                        csv_count += len(daily_dataframes)
                        all_dataframes.extend(daily_dataframes)

                    elif member_name.lower().endswith(".csv"):
                        source_file = (
                            f"{monthly_zip_path.name}/"
                            f"{member_name}"
                        )

                        dataframe = parse_aemo_bytes(
                            csv_bytes=member_bytes,
                            source_file=source_file,
                        )

                        csv_count += 1
                        all_dataframes.append(dataframe)

        except BadZipFile as error:
            raise ValueError(
                f"Invalid monthly ZIP file: {monthly_zip_path}"
            ) from error

    if not all_dataframes:
        raise ValueError(
            "No operational-demand CSV data was found "
            "inside the monthly archives."
        )

    combined_dataframe = pd.concat(
        all_dataframes,
        ignore_index=True,
    )

    print(
        f"\nLoaded {len(monthly_zip_paths)} monthly archives, "
        f"{daily_zip_count} daily archives, "
        f"{csv_count} CSV files and "
        f"{len(combined_dataframe):,} data rows."
    )

    return combined_dataframe