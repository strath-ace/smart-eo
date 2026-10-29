"""Build and validate time-aware monthly facility NO2 estimates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_NSTA = PROJECT_ROOT / "PROCESSED" / "NSTA"
PROCESSED_SENTINEL = PROJECT_ROOT / "PROCESSED" / "SENTINEL"
DEFAULT_MAPPING = PROCESSED_NSTA / "mapped_surface_points.csv"
DEFAULT_SOURCE_MANIFEST = PROCESSED_NSTA / "mapped_surface_points_manifest.json"
DEFAULT_DAILY = PROCESSED_SENTINEL / "numpy_NO2"
DEFAULT_MONTHLY = PROCESSED_SENTINEL / "facility_no2_monthly.csv"
DEFAULT_MANIFEST = PROCESSED_SENTINEL / "facility_no2_monthly_manifest.json"
ALGORITHM_VERSION = 4
BBOX = (-2.0, 52.0, 4.0, 62.0)
RADIUS_DEGREES = 0.1155

MONTHLY_COLUMNS = [
    "facility_row_index", "feature_id", "Match", "facility_name",
    "analysis_longitude", "analysis_latitude", "coordinate_source",
    "analysis_status", "area_pixel_count", "year", "month",
    "mean_no2_mol_m2", "valid_pixel_observations", "valid_days",
    "available_raster_days", "coverage_pct",
]


@dataclass(frozen=True)
class DailyRaster:
    path: Path
    day: date
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class SnapshotSource:
    label: str
    period: str


@dataclass
class SnapshotAssignment:
    source: SnapshotSource
    facilities: pd.DataFrame
    assignment: np.ndarray
    area_pixel_count: np.ndarray
    inside_bbox: np.ndarray


@dataclass(frozen=True)
class CachePaths:
    mapped_csv: Path
    source_manifest_json: Path
    daily_dir: Path
    monthly_csv: Path
    manifest_json: Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_manifest(path: Path, mapped_csv: Path) -> tuple[dict, list[SnapshotSource]]:
    if not path.is_file():
        raise FileNotFoundError(f"Stage 1 source manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"Stage 1 manifest has no sources: {path}")
    if payload.get("output_sha256") != sha256_file(mapped_csv):
        raise ValueError(
            "Stage 1 mapping checksum does not match its manifest; rerun "
            "python src/1_join_nsta/run.py"
        )
    result: list[SnapshotSource] = []
    for source in sources:
        label = str(source.get("label", "")).strip()
        period = str(source.get("period", "")).strip()
        if not label or not re.fullmatch(r"\d{4}-(?:0[1-9]|1[0-2])", period):
            raise ValueError(f"Invalid source entry in {path}: {source!r}")
        result.append(SnapshotSource(label, period))
    if result != sorted(result, key=lambda item: item.period):
        raise ValueError("Stage 1 manifest sources are not chronological")
    if len({source.period for source in result}) != len(result):
        raise ValueError("Stage 1 manifest contains duplicate snapshot periods")
    return payload, result


def discover_daily_rasters(daily_dir: Path, through: date | None = None) -> list[DailyRaster]:
    through = through or date.today()
    if not daily_dir.is_dir():
        raise FileNotFoundError(f"NO2 daily directory does not exist: {daily_dir}")
    rasters: list[DailyRaster] = []
    for path in sorted(daily_dir.glob("*.npy")):
        try:
            day = datetime.strptime(path.stem, "%Y-%m-%d").date()
        except ValueError:
            continue
        if day > through:
            continue
        stat = path.stat()
        rasters.append(DailyRaster(path, day, stat.st_size, stat.st_mtime_ns))
    if not rasters:
        raise FileNotFoundError(
            f"No YYYY-MM-DD.npy daily NO2 files dated through {through} were found in {daily_dir}"
        )
    return rasters


def inventory_fingerprint(rasters: Sequence[DailyRaster]) -> str:
    digest = hashlib.sha256()
    for raster in rasters:
        digest.update(f"{raster.path.name}\0{raster.size}\0{raster.mtime_ns}\n".encode())
    return digest.hexdigest()


def month_range(first: str, last: str) -> list[str]:
    year, month = map(int, first.split("-"))
    end_year, end_month = map(int, last.split("-"))
    result: list[str] = []
    while (year, month) <= (end_year, end_month):
        result.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return result


def selected_source(month: str, sources: Sequence[SnapshotSource]) -> SnapshotSource:
    return next((source for source in sources if source.period >= month), sources[-1])


def load_mapping(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Stage 1 mapping does not exist: {path}")
    mapped = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted({"feature_id", "Match"}.difference(mapped.columns))
    if missing:
        raise ValueError(f"Mapped facility CSV is missing columns: {', '.join(missing)}")
    if mapped["feature_id"].str.strip().eq("").any():
        raise ValueError("Every mapped facility row must have a Feature ID")
    if mapped["feature_id"].str.casefold().duplicated().any():
        raise ValueError("Mapped facility Feature IDs must be unique")
    return mapped.reset_index(drop=True)


def facilities_for_source(mapped: pd.DataFrame, source: SnapshotSource) -> pd.DataFrame:
    prefix = source.label
    required = [f"{prefix}_infrastructure_name", f"{prefix}_x", f"{prefix}_y"]
    missing = [column for column in required if column not in mapped.columns]
    if missing:
        raise ValueError(
            f"Mapping is missing columns for snapshot {source.period}: {', '.join(missing)}"
        )
    longitude = pd.to_numeric(mapped[f"{prefix}_x"], errors="coerce")
    latitude = pd.to_numeric(mapped[f"{prefix}_y"], errors="coerce")
    present = longitude.notna() & latitude.notna()
    facilities = pd.DataFrame(
        {
            "facility_row_index": mapped.index[present].astype(np.int64),
            "feature_id": mapped.loc[present, "feature_id"].str.strip().to_numpy(),
            "Match": mapped.loc[present, "Match"].str.strip().to_numpy(),
            "facility_name": mapped.loc[present, f"{prefix}_infrastructure_name"].str.strip().to_numpy(),
            "analysis_longitude": longitude[present].to_numpy(dtype=np.float64),
            "analysis_latitude": latitude[present].to_numpy(dtype=np.float64),
            "coordinate_source": source.label,
        }
    )
    if facilities.empty:
        raise ValueError(f"Snapshot {source.period} contains no usable facilities")
    return facilities.reset_index(drop=True)


def build_assignment(coordinates: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = shape
    west, south, east, north = BBOX
    longitude, latitude = np.meshgrid(
        np.linspace(west, east, width), np.linspace(north, south, height)
    )
    pixels = np.column_stack((longitude.ravel(), latitude.ravel()))
    inside = (
        (coordinates[:, 0] > west) & (coordinates[:, 0] < east)
        & (coordinates[:, 1] > south) & (coordinates[:, 1] < north)
    )
    source_indices = np.flatnonzero(inside)
    labels = np.full(len(pixels), -1, dtype=np.int32)
    if source_indices.size:
        candidates = coordinates[source_indices]
        radius_squared = RADIUS_DEGREES**2
        for start in range(0, len(pixels), 4096):
            stop = min(start + 4096, len(pixels))
            delta = pixels[start:stop, None, :] - candidates[None, :, :]
            distances = np.einsum("ijk,ijk->ij", delta, delta)
            nearest = np.argmin(distances, axis=1)
            accepted = distances[np.arange(stop - start), nearest] <= radius_squared
            chunk = labels[start:stop]
            chunk[accepted] = source_indices[nearest[accepted]]
    counts = np.bincount(labels[labels >= 0], minlength=len(coordinates)).astype(np.int64)
    return labels.reshape(shape), counts, inside


def prepare_assignments(mapped: pd.DataFrame, sources: Sequence[SnapshotSource], shape: tuple[int, int]) -> dict[str, SnapshotAssignment]:
    result: dict[str, SnapshotAssignment] = {}
    for source in sources:
        facilities = facilities_for_source(mapped, source)
        coordinates = facilities[["analysis_longitude", "analysis_latitude"]].to_numpy(float)
        assignment, counts, inside = build_assignment(coordinates, shape)
        result[source.label] = SnapshotAssignment(source, facilities, assignment, counts, inside)
        print(f"Prepared {source.period} ({source.label}): {len(facilities)} facilities")
    return result


def aggregate_month(month: str, rasters: Sequence[DailyRaster], snapshot: SnapshotAssignment) -> list[dict[str, object]]:
    count = len(snapshot.facilities)
    sums = np.zeros(count)
    observations = np.zeros(count, dtype=np.int64)
    valid_days = np.zeros(count, dtype=np.int64)
    flat_assignment = snapshot.assignment.ravel()
    for raster in rasters:
        values = np.load(raster.path, mmap_mode="r")
        if values.shape != snapshot.assignment.shape:
            raise ValueError(f"Unexpected shape {values.shape} for {raster.path}; expected {snapshot.assignment.shape}")
        flat_values = np.asarray(values).ravel()
        valid = (flat_assignment >= 0) & np.isfinite(flat_values)
        if not np.any(valid):
            continue
        labels = flat_assignment[valid]
        day_values = flat_values[valid].astype(np.float64, copy=False)
        day_sums = np.bincount(labels, weights=day_values, minlength=count)
        day_counts = np.bincount(labels, minlength=count)
        sums += day_sums
        observations += day_counts
        valid_days += day_counts > 0

    year, month_number = map(int, month.split("-"))
    records: list[dict[str, object]] = []
    for index, facility in snapshot.facilities.iterrows():
        if not rasters:
            status = "no_daily_rasters"
        elif not snapshot.inside_bbox[index]:
            status = "outside_no2_bbox"
        elif snapshot.area_pixel_count[index] == 0:
            status = "no_assigned_pixels"
        else:
            status = "available"
        observation_count = int(observations[index])
        available = len(rasters)
        days = int(valid_days[index])
        records.append(
            {
                **facility.to_dict(),
                "analysis_status": status,
                "area_pixel_count": int(snapshot.area_pixel_count[index]),
                "year": year,
                "month": month_number,
                "mean_no2_mol_m2": float(sums[index] / observation_count) if observation_count else np.nan,
                "valid_pixel_observations": observation_count,
                "valid_days": days,
                "available_raster_days": available,
                "coverage_pct": 100.0 * days / available if available else np.nan,
            }
        )
    return records


def aggregate_monthly_to_yearly(monthly: pd.DataFrame) -> pd.DataFrame:
    """Return observation-weighted yearly rows for display-only consumers."""
    data = monthly.copy()
    for column in (
        "year", "month", "mean_no2_mol_m2", "valid_pixel_observations",
        "valid_days", "available_raster_days",
    ):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    rows: list[dict[str, object]] = []
    for (_, year), group in data.groupby(["feature_id", "year"], sort=False):
        group = group.sort_values("month")
        latest = group.iloc[-1].to_dict()
        observations = int(group["valid_pixel_observations"].fillna(0).sum())
        valid_days = int(group["valid_days"].fillna(0).sum())
        available = int(group["available_raster_days"].fillna(0).sum())
        weighted = (
            group["mean_no2_mol_m2"].fillna(0)
            * group["valid_pixel_observations"].fillna(0)
        ).sum()
        latest.update(
            year=int(year),
            mean_no2_mol_m2=(float(weighted / observations) if observations else np.nan),
            valid_pixel_observations=observations,
            valid_days=valid_days,
            available_raster_days=available,
            coverage_pct=(100.0 * valid_days / available if available else np.nan),
            analysis_status=("available" if observations else latest["analysis_status"]),
        )
        latest.pop("month", None)
        rows.append(latest)
    return pd.DataFrame(rows)


def expected_manifest(paths: CachePaths, rasters: Sequence[DailyRaster], months: Sequence[str]) -> dict[str, object]:
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "cutoff_date": date.today().isoformat(),
        "mapped_csv_sha256": sha256_file(paths.mapped_csv),
        "source_manifest_sha256": sha256_file(paths.source_manifest_json),
        "daily_inventory_sha256": inventory_fingerprint(rasters),
        "daily_file_count": len(rasters),
        "first_month": months[0], "last_month": months[-1],
    }


def validate_cache(paths: CachePaths, expected: dict[str, object]) -> tuple[bool, str]:
    if not paths.monthly_csv.is_file() or not paths.manifest_json.is_file():
        return False, "cache files are missing"
    try:
        manifest = json.loads(paths.manifest_json.read_text(encoding="utf-8"))
    except Exception as error:
        return False, f"manifest cannot be read: {error}"
    for key, value in expected.items():
        if manifest.get(key) != value:
            return False, f"manifest field {key!r} is stale"
    if manifest.get("monthly_csv_sha256") != sha256_file(paths.monthly_csv):
        return False, "monthly CSV checksum does not match"
    try:
        monthly = pd.read_csv(paths.monthly_csv, dtype={"feature_id": str, "Match": str})
    except Exception as error:
        return False, f"monthly CSV cannot be read: {error}"
    if list(monthly.columns) != MONTHLY_COLUMNS:
        return False, "monthly CSV schema is incorrect"
    if monthly.duplicated(["feature_id", "year", "month"]).any():
        return False, "monthly CSV contains duplicate facility/month rows"
    return True, "cache is complete and current"


def atomic_write(frame: pd.DataFrame, manifest: dict[str, object], paths: CachePaths) -> None:
    paths.monthly_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_temp: Path | None = None
    manifest_temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv.tmp", dir=paths.monthly_csv.parent, delete=False) as handle:
            csv_temp = Path(handle.name)
        frame.to_csv(csv_temp, index=False)
        manifest["monthly_csv_sha256"] = sha256_file(csv_temp)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json.tmp", dir=paths.manifest_json.parent, encoding="utf-8", delete=False) as handle:
            manifest_temp = Path(handle.name)
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(csv_temp, paths.monthly_csv)
        csv_temp = None
        os.replace(manifest_temp, paths.manifest_json)
        manifest_temp = None
    finally:
        for temporary in (csv_temp, manifest_temp):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def ensure_cache(paths: CachePaths) -> tuple[Path, bool]:
    mapped = load_mapping(paths.mapped_csv)
    _, sources = load_source_manifest(paths.source_manifest_json, paths.mapped_csv)
    rasters = discover_daily_rasters(paths.daily_dir)
    months = month_range(rasters[0].day.strftime("%Y-%m"), date.today().strftime("%Y-%m"))
    expected = expected_manifest(paths, rasters, months)
    valid, reason = validate_cache(paths, expected)
    if valid:
        print(f"Reusing {paths.monthly_csv} ({reason}).")
        return paths.monthly_csv, False
    print(f"Rebuilding monthly NO2 cache because {reason}.")
    first = np.load(rasters[0].path, mmap_mode="r")
    if first.ndim != 2:
        raise ValueError(f"Daily NO2 arrays must be two-dimensional, got {first.shape}")
    assignments = prepare_assignments(mapped, sources, tuple(map(int, first.shape)))
    rasters_by_month: dict[str, list[DailyRaster]] = defaultdict(list)
    for raster in rasters:
        rasters_by_month[raster.day.strftime("%Y-%m")].append(raster)
    records: list[dict[str, object]] = []
    for index, month in enumerate(months, start=1):
        source = selected_source(month, sources)
        records.extend(aggregate_month(month, rasters_by_month[month], assignments[source.label]))
        if index == 1 or index == len(months) or index % 12 == 0:
            print(f"Aggregated {index}/{len(months)} months through {month}")
    monthly = pd.DataFrame.from_records(records, columns=MONTHLY_COLUMNS)
    if monthly.duplicated(["feature_id", "year", "month"]).any():
        raise ValueError("Generated duplicate facility/month records")
    manifest = {
        **expected,
        "created_at": datetime.now().astimezone().isoformat(),
        "row_count": len(monthly),
        "facility_count": int(monthly["feature_id"].nunique()),
        "month_count": len(months),
        "snapshot_policy": "earliest_snapshot_on_or_after_month_else_latest",
    }
    atomic_write(monthly, manifest, paths)
    print(f"Created {paths.monthly_csv} with {len(monthly)} rows.")
    return paths.monthly_csv, True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--daily", type=Path, default=DEFAULT_DAILY)
    parser.add_argument("--monthly", type=Path, default=DEFAULT_MONTHLY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = CachePaths(
        args.mapping.expanduser().resolve(), args.source_manifest.expanduser().resolve(),
        args.daily.expanduser().resolve(), args.monthly.expanduser().resolve(),
        args.manifest.expanduser().resolve(),
    )
    try:
        ensure_cache(paths)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
