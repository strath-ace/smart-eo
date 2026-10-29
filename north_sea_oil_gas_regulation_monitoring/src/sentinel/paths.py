"""Repository-relative paths for Sentinel raw and processed data."""

from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_SENTINEL_DIRECTORY = PROJECT_ROOT / "RAW" / "SENTINEL"
PROCESSED_SENTINEL_DIRECTORY = PROJECT_ROOT / "PROCESSED" / "SENTINEL"
MAP_DIRECTORY = PROCESSED_SENTINEL_DIRECTORY / "maps"


def normalize_chemical(chemical: str) -> str:
    value = chemical.strip().upper()
    if not value or re.fullmatch(r"[A-Z0-9]+", value) is None:
        raise ValueError(f"Invalid chemical name: {chemical!r}")
    return value


def raw_directory(chemical: str) -> Path:
    return RAW_SENTINEL_DIRECTORY / f"raw_data_{normalize_chemical(chemical)}"


def ensure_raw_directory(chemical: str) -> Path:
    """Create and return the repository's raw cache for one chemical."""
    directory = raw_directory(chemical)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def numpy_directory(chemical: str) -> Path:
    return PROCESSED_SENTINEL_DIRECTORY / f"numpy_{normalize_chemical(chemical)}"


def png_directory(chemical: str) -> Path:
    return PROCESSED_SENTINEL_DIRECTORY / f"png_{normalize_chemical(chemical)}"


def aggregate_path(chemical: str) -> Path:
    return PROCESSED_SENTINEL_DIRECTORY / f"all_days_{normalize_chemical(chemical)}.npz"


def viewable_path(chemical: str) -> Path:
    return PROCESSED_SENTINEL_DIRECTORY / f"viewable_{normalize_chemical(chemical)}.npy"


def map_path(chemical: str, *, custom: bool = False) -> Path:
    prefix = "custom_map" if custom else "map"
    return MAP_DIRECTORY / f"{prefix}_{normalize_chemical(chemical)}.png"


def facility_output_path(filename: str) -> Path:
    if Path(filename).name != filename:
        raise ValueError(f"Facility output must be a filename: {filename!r}")
    return PROCESSED_SENTINEL_DIRECTORY / filename
