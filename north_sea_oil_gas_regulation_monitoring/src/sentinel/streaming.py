"""Memory-bounded readers for processed daily Sentinel arrays."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    from .paths import numpy_directory
except ImportError:
    from paths import numpy_directory


def dated_array_files(
    chemical: str,
    *,
    directory: Path | None = None,
    start: date = date(2018, 1, 1),
    end: date | None = None,
) -> list[tuple[date, Path]]:
    """Return daily arrays in date order, restricted to real calendar dates."""
    directory = directory or numpy_directory(chemical)
    end = end or date.today()
    if not directory.is_dir():
        raise FileNotFoundError(f"Processed Sentinel directory does not exist: {directory}")

    files: list[tuple[date, Path]] = []
    for path in directory.glob("*.npy"):
        try:
            observation_date = datetime.strptime(path.stem, "%Y-%m-%d").date()
        except ValueError:
            continue
        if start <= observation_date <= end:
            files.append((observation_date, path))
    files.sort(key=lambda item: item[0])
    if not files:
        raise FileNotFoundError(
            f"No daily {chemical.upper()} arrays dated {start} through {end} in {directory}"
        )
    return files


def iter_valid_arrays(
    chemical: str,
    *,
    directory: Path | None = None,
    start: date = date(2018, 1, 1),
    end: date | None = None,
) -> Iterator[tuple[date, np.ndarray]]:
    """Yield finite, two-dimensional daily arrays and skip empty observations."""
    expected_shape: tuple[int, int] | None = None
    for observation_date, path in dated_array_files(
        chemical, directory=directory, start=start, end=end
    ):
        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            print(f"Skipping unreadable Sentinel array {path}: {error}")
            continue
        if array.ndim != 2:
            print(f"Skipping Sentinel array with shape {array.shape}: {path}")
            continue
        if expected_shape is None:
            expected_shape = array.shape
        elif array.shape != expected_shape:
            print(
                f"Skipping Sentinel array with inconsistent shape {array.shape}; "
                f"expected {expected_shape}: {path}"
            )
            continue
        if not np.isfinite(array).any():
            continue
        yield observation_date, array


def streamed_filtered_mean(
    chemical: str,
    *,
    directory: Path | None = None,
    start: date = date(2018, 1, 1),
    end: date | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Reproduce the maps' global normalization/outlier filter without a 3-D stack."""
    total = total_squared = 0.0
    count = 0
    minimum = np.inf
    maximum = -np.inf
    valid_dates: list[str] = []

    for observation_date, array in iter_valid_arrays(
        chemical, directory=directory, start=start, end=end
    ):
        values = array[np.isfinite(array)].astype(np.float64, copy=False)
        minimum = min(minimum, float(values.min()))
        maximum = max(maximum, float(values.max()))
        total += float(values.sum())
        total_squared += float(np.square(values).sum())
        count += values.size
        valid_dates.append(observation_date.isoformat())

    if count == 0 or not valid_dates:
        raise ValueError(f"No finite {chemical.upper()} observations were found")
    value_range = maximum - minimum
    if not np.isfinite(value_range) or value_range == 0:
        raise ValueError(f"{chemical.upper()} observations have no finite value range")

    raw_mean = total / count
    raw_variance = max(total_squared / count - raw_mean * raw_mean, 0.0)
    normalized_mean = (raw_mean - minimum) / value_range
    normalized_std = np.sqrt(raw_variance) / value_range
    lower = normalized_mean - 3 * normalized_std
    upper = normalized_mean + 3 * normalized_std

    sum_image: np.ndarray | None = None
    count_image: np.ndarray | None = None
    for _, array in iter_valid_arrays(
        chemical, directory=directory, start=start, end=end
    ):
        normalized = (array.astype(np.float64, copy=False) - minimum) / value_range
        valid = np.isfinite(normalized) & (normalized >= lower) & (normalized <= upper)
        if sum_image is None:
            sum_image = np.zeros(array.shape, dtype=np.float64)
            count_image = np.zeros(array.shape, dtype=np.uint32)
        sum_image[valid] += normalized[valid]
        count_image[valid] += 1

    assert sum_image is not None and count_image is not None
    mean = np.full(sum_image.shape, np.nan, dtype=np.float64)
    np.divide(sum_image, count_image, out=mean, where=count_image > 0)
    return mean, valid_dates
