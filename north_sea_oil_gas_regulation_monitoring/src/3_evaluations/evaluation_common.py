"""Shared Stage 3 artifact validation and monthly aggregation helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAPPING = PROJECT_ROOT / "PROCESSED" / "NSTA" / "mapped_surface_points.csv"
SOURCE_MANIFEST = PROJECT_ROOT / "PROCESSED" / "NSTA" / "mapped_surface_points_manifest.json"
MONTHLY = PROJECT_ROOT / "PROCESSED" / "SENTINEL" / "facility_no2_monthly.csv"
MONTHLY_MANIFEST = PROJECT_ROOT / "PROCESSED" / "SENTINEL" / "facility_no2_monthly_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Required predecessor artifact is missing: {path}. Run the earlier stage first."
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def validate_stage_inputs(
    mapping: Path = MAPPING,
    source_manifest: Path = SOURCE_MANIFEST,
    monthly: Path = MONTHLY,
    monthly_manifest: Path = MONTHLY_MANIFEST,
) -> tuple[dict, dict]:
    source = read_json(source_manifest)
    estimates = read_json(monthly_manifest)
    for path in (mapping, monthly):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required predecessor artifact is missing: {path}. Run the earlier stage first."
            )
    mapping_hash = sha256_file(mapping)
    source_hash = sha256_file(source_manifest)
    monthly_hash = sha256_file(monthly)
    if source.get("output_sha256") != mapping_hash:
        raise ValueError("Stage 1 mapping is stale relative to its manifest")
    if estimates.get("mapped_csv_sha256") != mapping_hash:
        raise ValueError("Stage 2 estimates were built from a different Stage 1 mapping")
    if estimates.get("source_manifest_sha256") != source_hash:
        raise ValueError("Stage 2 estimates were built from a different source manifest")
    if estimates.get("monthly_csv_sha256") != monthly_hash:
        raise ValueError("Stage 2 monthly estimate checksum does not match its manifest")
    return source, estimates


def latest_snapshot_label(source_manifest: dict, year: int) -> str:
    candidates = [
        source for source in source_manifest.get("sources", [])
        if str(source.get("period", "")).startswith(f"{year:04d}-")
    ]
    if not candidates:
        raise ValueError(f"No NSTA snapshot is available for {year}")
    return str(max(candidates, key=lambda source: source["period"])["label"])


def comparison_labels(source_manifest: dict) -> tuple[str, str]:
    return latest_snapshot_label(source_manifest, 2025), latest_snapshot_label(source_manifest, 2026)


def newest_snapshot_label(source_manifest: dict) -> str:
    sources = source_manifest.get("sources", [])
    if not sources:
        raise ValueError("Stage 1 source manifest has no snapshots")
    return str(max(sources, key=lambda source: source["period"])["label"])


def first_nonempty(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    result = pd.Series("", index=frame.index, dtype=object)
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column].fillna("").astype(str).str.strip()
        result = result.where(result.ne(""), values)
    return result


YEARLY_COLUMNS = [
    "facility_row_index", "feature_id", "Match", "facility_name",
    "analysis_longitude", "analysis_latitude", "coordinate_source",
    "analysis_status", "area_pixel_count", "year", "mean_no2_mol_m2",
    "valid_pixel_observations", "valid_days", "available_raster_days",
    "coverage_pct",
]


def aggregate_monthly_to_yearly(monthly: pd.DataFrame) -> pd.DataFrame:
    """Create observation-weighted yearly rows from authoritative monthly rows."""
    if monthly.empty:
        return pd.DataFrame(columns=YEARLY_COLUMNS)
    data = monthly.copy()
    numeric = [
        "year", "month", "mean_no2_mol_m2", "valid_pixel_observations",
        "valid_days", "available_raster_days", "area_pixel_count",
        "analysis_longitude", "analysis_latitude",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    rows: list[dict[str, object]] = []
    for (_, year), group in data.groupby(["feature_id", "year"], sort=False):
        group = group.sort_values("month")
        latest = group.iloc[-1]
        observations = int(group["valid_pixel_observations"].fillna(0).sum())
        valid_days = int(group["valid_days"].fillna(0).sum())
        available_days = int(group["available_raster_days"].fillna(0).sum())
        weighted = (
            group["mean_no2_mol_m2"].fillna(0)
            * group["valid_pixel_observations"].fillna(0)
        ).sum()
        mean = float(weighted / observations) if observations else np.nan
        status = "available" if observations else str(latest["analysis_status"])
        rows.append(
            {
                "facility_row_index": int(latest["facility_row_index"]),
                "feature_id": str(latest["feature_id"]),
                "Match": str(latest["Match"]),
                "facility_name": str(latest["facility_name"]),
                "analysis_longitude": latest["analysis_longitude"],
                "analysis_latitude": latest["analysis_latitude"],
                "coordinate_source": str(latest["coordinate_source"]),
                "analysis_status": status,
                "area_pixel_count": int(latest["area_pixel_count"]),
                "year": int(year),
                "mean_no2_mol_m2": mean,
                "valid_pixel_observations": observations,
                "valid_days": valid_days,
                "available_raster_days": available_days,
                "coverage_pct": (
                    100.0 * valid_days / available_days if available_days else np.nan
                ),
            }
        )
    return pd.DataFrame.from_records(rows, columns=YEARLY_COLUMNS)


def copy_provenance(
    output_source: Path,
    mapping: Path = MAPPING,
    source_manifest: Path = SOURCE_MANIFEST,
) -> tuple[Path, Path]:
    output_source.mkdir(parents=True, exist_ok=True)
    mapping_copy = output_source / mapping.name
    manifest_copy = output_source / source_manifest.name
    shutil.copy2(mapping, mapping_copy)
    shutil.copy2(source_manifest, manifest_copy)
    return mapping_copy, manifest_copy
