"""Utilities for loading and preparing CICIDS2017 network-flow CSV files."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def load_flow_data(data_dir: str | Path) -> pd.DataFrame:
    """Load and concatenate CSV files from a directory."""
    folder = Path(data_dir)
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {folder.resolve()}. "
            "Download CICIDS2017 and place its CSV files in the data directory."
        )

    frames: list[pd.DataFrame] = []
    for file in files:
        frame = pd.read_csv(file, encoding="latin1", low_memory=False)
        frame.columns = frame.columns.astype(str).str.strip()
        if "Label" not in frame.columns:
            raise ValueError(f"{file.name} does not contain the required 'Label' column.")
        frames.append(frame)

    return pd.concat(frames, ignore_index=True, sort=False)


def clean_flow_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean a flow table while retaining the original attack label."""
    if "Label" not in frame.columns:
        raise ValueError("Expected a 'Label' target column.")

    data = frame.copy()
    data.columns = data.columns.astype(str).str.strip()
    data = data.loc[:, ~data.columns.duplicated()]
    data["Label"] = data["Label"].astype("string").str.strip()
    before = len(data)

    data = data.dropna(axis=1, how="all")
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data.loc[:, numeric_columns] = data.loc[:, numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    invalid_target = data["Label"].isna() | data["Label"].isin(["", "nan", "None"])
    data = data.loc[~invalid_target].drop_duplicates().reset_index(drop=True)

    return data, {
        "rows_before": before,
        "rows_after": len(data),
        "rows_removed": before - len(data),
    }


def binary_target(labels: Iterable[object]) -> np.ndarray:
    """Map BENIGN to 0 and all other non-empty labels to 1."""
    series = pd.Series(labels).astype("string").str.strip()
    if series.isna().any() or series.empty:
        raise ValueError("Target labels contain missing values or are empty.")
    return (series.str.upper() != "BENIGN").astype(int).to_numpy()
