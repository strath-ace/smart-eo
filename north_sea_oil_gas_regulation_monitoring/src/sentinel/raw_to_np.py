#!/usr/bin/env python3
"""Convert Sentinel Hub TIFF caches into daily NumPy arrays and PNGs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm

try:
    from .paths import PROCESSED_SENTINEL_DIRECTORY, RAW_SENTINEL_DIRECTORY, normalize_chemical
except ImportError:  # Direct execution: python src/sentinel/raw_to_np.py
    from paths import (  # type: ignore[no-redef]
        PROCESSED_SENTINEL_DIRECTORY,
        RAW_SENTINEL_DIRECTORY,
        normalize_chemical,
    )


@dataclass(frozen=True)
class RawObservation:
    folder: Path
    tiff_file: Path
    date: str


@dataclass(frozen=True)
class ProcessingSummary:
    chemical: str
    valid_raw_files: int
    created_numpy: int
    created_png: int
    skipped: int
    invalid_tiffs: int
    incomplete_folders: int
    unreadable_requests: int
    failures: int


def file_is_complete(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def is_valid_tiff(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 8:
        return False
    try:
        with path.open("rb") as handle:
            signature = handle.read(4)
    except OSError:
        return False
    return signature in {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"}


def extract_date(request_file: Path) -> str:
    with request_file.open("r", encoding="utf-8") as handle:
        request_data = json.load(handle)
    request_section = request_data.get("request", request_data)
    payload = request_section.get("payload") or request_section.get("post_values")
    if payload is None:
        raise KeyError("Could not find payload or post_values in request.json")
    return payload["input"]["data"][0]["dataFilter"]["timeRange"]["from"][:10]


def create_png(value: np.ndarray, mask: np.ndarray, output_file: Path) -> None:
    valid_pixels = (mask != 0) & np.isfinite(value)
    display_value = np.zeros(value.shape, dtype=np.uint8)
    if np.any(valid_pixels):
        valid_values = value[valid_pixels]
        minimum = np.nanmin(valid_values)
        maximum = np.nanmax(valid_values)
        value_range = maximum - minimum
        if np.isfinite(value_range) and value_range > 0:
            scaled = 255.0 * (value[valid_pixels] - minimum) / value_range
            display_value[valid_pixels] = np.clip(scaled, 0, 255).astype(np.uint8)
    alpha = np.where(valid_pixels, 255, 0).astype(np.uint8)
    Image.fromarray(np.stack([display_value, alpha], axis=-1)).save(output_file)


def discover_chemicals(raw_root: Path = RAW_SENTINEL_DIRECTORY) -> list[str]:
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Sentinel raw-data directory does not exist: {raw_root}")
    chemicals = sorted(
        normalize_chemical(path.name.removeprefix("raw_data_"))
        for path in raw_root.iterdir()
        if path.is_dir() and path.name.casefold().startswith("raw_data_")
    )
    if not chemicals:
        raise FileNotFoundError(f"No raw_data_* directories found in {raw_root}")
    return chemicals


def scan_raw_directory(raw_directory: Path) -> tuple[list[RawObservation], int, int, int]:
    valid: list[RawObservation] = []
    invalid_tiffs = incomplete = unreadable = 0
    for folder in sorted(path for path in raw_directory.iterdir() if path.is_dir()):
        request_file = folder / "request.json"
        tiff_file = folder / "response.tiff"
        if not file_is_complete(request_file) or not tiff_file.is_file():
            incomplete += 1
            continue
        if not is_valid_tiff(tiff_file):
            invalid_tiffs += 1
            continue
        try:
            date = extract_date(request_file)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            unreadable += 1
            continue
        valid.append(RawObservation(folder, tiff_file, date))
    return valid, invalid_tiffs, incomplete, unreadable


def process_chemical(
    chemical: str,
    *,
    raw_root: Path = RAW_SENTINEL_DIRECTORY,
    processed_root: Path = PROCESSED_SENTINEL_DIRECTORY,
) -> ProcessingSummary:
    chemical = normalize_chemical(chemical)
    raw_directory = raw_root / f"raw_data_{chemical}"
    if not raw_directory.is_dir():
        raise FileNotFoundError(f"Raw {chemical} directory does not exist: {raw_directory}")
    numpy_directory = processed_root / f"numpy_{chemical}"
    png_directory = processed_root / f"png_{chemical}"
    numpy_directory.mkdir(parents=True, exist_ok=True)
    png_directory.mkdir(parents=True, exist_ok=True)

    observations, invalid_tiffs, incomplete, unreadable = scan_raw_directory(raw_directory)
    print(f"{chemical}: {len(observations)} valid raw TIFFs")
    created_numpy = created_png = skipped = failures = 0
    for observation in tqdm(observations, desc=chemical):
        numpy_file = numpy_directory / f"{observation.date}.npy"
        png_file = png_directory / f"{observation.date}.png"
        numpy_exists = file_is_complete(numpy_file)
        png_exists = file_is_complete(png_file)
        if numpy_exists and png_exists:
            skipped += 1
            continue
        try:
            with rasterio.open(observation.tiff_file) as source:
                if source.count < 2:
                    raise ValueError("TIFF does not contain value and dataMask bands")
                value = source.read(1).astype(np.float32, copy=False)
                mask = source.read(2)
            valid_pixels = (mask != 0) & np.isfinite(value)
            masked_value = value.copy()
            masked_value[~valid_pixels] = np.nan
            if not numpy_exists:
                np.save(numpy_file, masked_value)
                if not file_is_complete(numpy_file):
                    raise RuntimeError(f"NumPy output was not saved: {numpy_file}")
                created_numpy += 1
            if not png_exists:
                create_png(masked_value, mask, png_file)
                if not file_is_complete(png_file):
                    raise RuntimeError(f"PNG output was not saved: {png_file}")
                created_png += 1
        except Exception as error:
            tqdm.write(f"Failed to process {observation.folder.name}: {error}")
            failures += 1

    summary = ProcessingSummary(
        chemical, len(observations), created_numpy, created_png, skipped,
        invalid_tiffs, incomplete, unreadable, failures,
    )
    print(
        f"{chemical} finished: numpy={created_numpy}, png={created_png}, "
        f"complete={skipped}, invalid={invalid_tiffs}, incomplete={incomplete}, "
        f"unreadable={unreadable}, failures={failures}"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "chemicals", nargs="*",
        help="Chemicals to process (for example NO2 CO). Defaults to all raw_data_* folders.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        chemicals = (
            [normalize_chemical(value) for value in args.chemicals]
            if args.chemicals else discover_chemicals()
        )
        for chemical in chemicals:
            process_chemical(chemical)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
