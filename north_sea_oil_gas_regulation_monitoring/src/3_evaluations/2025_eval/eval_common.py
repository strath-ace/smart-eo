"""Shared inputs, paths, plotting helpers, and streaming for evaluation scripts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Sequence

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DATASETS = ("2025", "2026", "joined")
RAW_NSTA = PROJECT_ROOT / "RAW" / "NSTA"
RAW_FIRMS = PROJECT_ROOT / "RAW" / "FIRMS"
RAW_OIL_PRICE = PROJECT_ROOT / "RAW" / "OIL_PRICE"
PROCESSED_NSTA = PROJECT_ROOT / "PROCESSED" / "NSTA"
RESULTS_ROOT = PROJECT_ROOT / "results"
OWNERSHIP_2025 = RAW_NSTA / "mar-2025" / "who_owns_the_north_sea.json"
JOINED_NSTA = PROCESSED_NSTA / "mapped_surface_points.csv"
SOURCE_MANIFEST = PROCESSED_NSTA / "mapped_surface_points_manifest.json"
FACILITY_MONTHLY = PROJECT_ROOT / "PROCESSED" / "SENTINEL" / "facility_no2_monthly.csv"
OIL_PRICE_CSV = RAW_OIL_PRICE / "Brent_oil_monthly_spot.csv"

FIRMS_NOAA20 = (
    RAW_FIRMS / "DL_FIRE_J1V-C2_587751" / "fire_archive_J1V-C2_587751.json"
)
FIRMS_SNPP = (
    RAW_FIRMS / "DL_FIRE_SV-C2_587753" / "fire_archive_SV-C2_587753.json"
)

OWNER_REMAP = {
    "REPSOL RESOURCES UK": "REPSOL SINOPEC RESOURCES",
    "CALENERGY GAS (HOLDINGS)": "CALENERGY GAS HOLDINGS",
    "INEOS FPS LTD": "INEOS INDUSTRIES",
    "WINTERSHALL NOORDZEE": "WINTERSHALL B.V.",
    "KELLAS MIDSTREAM": "HARBOUR ENERGY PLC",
    "ANASURIA OPERATING COMPANY LIMITED": "PING PETROLEUM LIMITED",
}
ADURA_DETAILS = {
    "parent_company": "Adura Energy",
    "private_equity_backing": "No",
    "parent_company_hq": "United Kingdom",
    "parent_company_type": "LTD",
    "number_of_licenses": "",
    "total_north_sea_stake": "",
    "equity_ranking": "",
}

JOINED_FIELDS = {
    "OBJECTID": "object_id",
    "NAME": "infrastructure_name",
    "INF_TYPE": "type_of_infrastructure",
    "REP_GROUP": "reporting_group_name",
    "DESCRIPTIO": "description",
    "PLAT_FLAG": "platform_flag",
    "PIPE_SYS": "pipeline_system_name",
    "POINT_DESC": "description_of_point",
    "CRS_CODE": "original_crs_code",
    "CRS_NAME": "original_crs_name",
    "ORIG_X_LON": "original_x_longitude",
    "ORIG_Y_LAT": "original_y_latitude",
    "STATUS": "status",
    "COMMENTS": "comments",
    "START_DATE": "start_date",
    "END_DATE": "end_date",
    "END_REAS": "reason_for_end_date",
    "UPD_DATE": "date_updated",
    "UPD_TYPE": "update_type",
    "UPD_REAS": "reason_for_update",
    "LEGACY_ID": "legacy_id",
}


def processed_directory(dataset: str) -> Path:
    validate_dataset(dataset)
    return PROCESSED_NSTA / f"{dataset}_eval"


def results_directory(dataset: str) -> Path:
    validate_dataset(dataset)
    return RESULTS_ROOT / f"{dataset}_results"


def prepare_directories(dataset: str) -> tuple[Path, Path]:
    processed = processed_directory(dataset)
    results = results_directory(dataset)
    for directory in (processed, results, results / "FIRMS", results / "rigs"):
        directory.mkdir(parents=True, exist_ok=True)
    return processed, results


def validate_dataset(dataset: str) -> str:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {', '.join(DATASETS)}")
    return dataset


def parse_datasets(argv: Sequence[str] | None = None, description: str = "") -> list[str]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset", choices=(*DATASETS, "all"), default="2025",
        help="NSTA dataset to evaluate (default: 2025)",
    )
    selected = parser.parse_args(argv).dataset
    return list(DATASETS) if selected == "all" else [selected]


def load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)


def csv_rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.reader(handle))


def write_csv(path: Path, rows: Iterable[Iterable[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def _source_labels() -> tuple[str, str, list[str]]:
    manifest = load_json(SOURCE_MANIFEST)
    sources = manifest.get("sources", [])
    labels = [str(source["label"]) for source in sources]
    selected = {}
    for year in (2025, 2026):
        candidates = [source for source in sources if str(source["period"]).startswith(f"{year}-")]
        if not candidates:
            raise ValueError(f"No NSTA source is available for {year}")
        selected[year] = str(max(candidates, key=lambda source: source["period"])["label"])
    return selected[2025], selected[2026], labels


def _mapped_features(dataset: str, path: Path = JOINED_NSTA) -> list[dict]:
    label_2025, label_2026, labels = _source_labels()
    features: list[dict] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            if dataset == "2025":
                label = label_2025
            elif dataset == "2026":
                label = label_2026
            else:
                label = next(
                    (
                        candidate for candidate in reversed(labels)
                        if (row.get(f"{candidate}_infrastructure_name") or "").strip()
                    ),
                    "",
                )
            if not label or not (row.get(f"{label}_infrastructure_name") or "").strip():
                continue
            properties = {
                target: (row.get(f"{label}_{source}") or "").strip()
                for target, source in JOINED_FIELDS.items()
            }
            properties["FEATURE_ID"] = (row.get("feature_id") or "").strip()
            try:
                coordinates = [float(row[f"{label}_x"]), float(row[f"{label}_y"])]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid {label} coordinates in {path} row {row_number}"
                ) from error
            features.append(
                {"type": "Feature", "properties": properties,
                 "geometry": {"type": "Point", "coordinates": coordinates}}
            )
    return features


def source_features(dataset: str) -> list[dict]:
    validate_dataset(dataset)
    return _mapped_features(dataset)


def clean_platform_records(dataset: str) -> list[dict]:
    ownership = load_json(OWNERSHIP_2025)
    if not isinstance(ownership, dict):
        raise ValueError(f"Expected ownership object in {OWNERSHIP_2025}")
    ownership = dict(ownership)
    ownership["ADURA ENERGY"] = ADURA_DETAILS

    clean: list[dict] = []
    id_counts: Counter[str] = Counter()
    for feature in source_features(dataset):
        properties = dict(feature.get("properties") or {})
        status = str(properties.get("STATUS") or "").strip().upper()
        infrastructure_type = str(properties.get("INF_TYPE") or "").strip().upper()
        if status != "ACTIVE" or infrastructure_type not in {"PLATFORM", "FPSO"}:
            continue
        feature_id = str(properties.get("FEATURE_ID") or "").strip()
        if not feature_id:
            raise ValueError(f"Active platform has no FEATURE_ID: {properties.get('NAME')!r}")
        normalized_id = feature_id.casefold()
        id_counts[normalized_id] += 1
        if id_counts[normalized_id] > 1:
            feature_id = f"{feature_id}_{id_counts[normalized_id]}"

        source_owner = str(properties.get("REP_GROUP") or "").strip()
        normalized_owner = OWNER_REMAP.get(source_owner, source_owner)
        if normalized_owner not in ownership:
            raise KeyError(
                f"No ownership classification for {source_owner!r} "
                f"(normalized as {normalized_owner!r}) in {dataset}"
            )
        owner_details = dict(ownership[normalized_owner])
        owner_details["asset_owner"] = normalized_owner
        coordinates = feature.get("geometry", {}).get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            raise ValueError(f"Invalid coordinates for {feature_id}")
        properties["REP_GROUP"] = normalized_owner
        clean.append(
            {
                "feature_id": feature_id,
                "name": str(properties.get("NAME") or "").strip(),
                "asset_owner": normalized_owner,
                "parent_company": owner_details["parent_company"],
                "parent_company_type": owner_details["parent_company_type"],
                "coordinates": [float(coordinates[0]), float(coordinates[1])],
                "ownership_details": owner_details,
                "NSTA_details": {
                    key: value for key, value in properties.items()
                    if key not in {"NAME", "coordinates"}
                },
            }
        )
    return clean


def clean_platform_path(dataset: str) -> Path:
    return processed_directory(dataset) / "surface_points_clean.json"


def firms_platform_path(dataset: str) -> Path:
    return processed_directory(dataset) / "surface_points_plus_FIRMS.json"


def point_in_bounds(
    coordinates: Sequence[float], lon_min: float = -2, lon_max: float = 4,
    lat_min: float = 52, lat_max: float = 62,
) -> bool:
    longitude, latitude = coordinates[:2]
    return lon_min < longitude < lon_max and lat_min < latitude < lat_max


def bounded_rigs(rigs: Sequence[dict]) -> list[dict]:
    return [rig for rig in rigs if point_in_bounds(rig["coordinates"])]


def firms_observations(path: Path = FIRMS_SNPP) -> list[dict]:
    observations = load_json(path)
    if not isinstance(observations, list):
        raise ValueError(f"Expected a FIRMS observation list: {path}")
    return [item for item in observations if str(item.get("type")) == "3"]


def attach_firms(rigs: Sequence[dict], observations: Sequence[dict]) -> list[dict]:
    enriched = [dict(rig, FIRMS_observations=[]) for rig in rigs]
    if not enriched:
        return enriched
    points = np.asarray([rig["coordinates"] for rig in enriched], dtype=float)
    tree = cKDTree(points)
    coordinates = np.asarray(
        [[float(item["longitude"]), float(item["latitude"])] for item in observations]
    )
    distances, indexes = tree.query(coordinates)
    for observation, distance, index in zip(observations, distances, indexes):
        if distance <= 0.1155:
            enriched[int(index)]["FIRMS_observations"].append(observation)
    return enriched


def normalize(array: np.ndarray) -> np.ndarray:
    if array.size == 0 or not np.isfinite(array).any():
        return np.asarray(array)
    minimum = np.nanmin(array)
    value_range = np.nanmax(array) - minimum
    if value_range == 0:
        return np.zeros_like(array, dtype=float)
    return (array - minimum) / value_range


def trim_nan_edges(array: np.ndarray) -> np.ndarray:
    mask = np.isfinite(array)
    if not mask.any():
        return np.empty((0, 0))
    rows, columns = np.where(mask)
    return array[rows.min():rows.max() + 1, columns.min():columns.max() + 1]


def plot_many_firms(path: Path, rigs: Sequence[dict]) -> None:
    columns = 10
    rows = max(1, int(np.ceil(len(rigs) / columns)))
    figure, axes = plt.subplots(
        rows, columns, figsize=(columns * 2, rows * 2), squeeze=False,
        subplot_kw={"projection": ccrs.PlateCarree()}, layout="constrained",
    )
    for axis, rig in zip(axes.flat, rigs):
        observations = rig.get("FIRMS_observations", [])
        axis.scatter(
            [item["longitude"] for item in observations],
            [item["latitude"] for item in observations], c="red", alpha=0.2,
        )
        longitude, latitude = rig["coordinates"]
        axis.scatter(longitude, latitude, marker="x", c="blue")
        extra = 0.1
        aspect = 1 / np.cos(np.deg2rad(latitude))
        axis.set_extent(
            [longitude - extra, longitude + extra,
             latitude - extra / aspect, latitude + extra / aspect]
        )
        axis.set_aspect(aspect)
        axis.set_title(rig["name"])
    for axis in list(axes.flat)[len(rigs):]:
        axis.set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def plot_combined_firms(path: Path, groups: Sequence[Sequence[dict]]) -> None:
    names = ("PLC Rigs", "SOE Rigs", "LTD Rigs")
    colors = ("skyblue", "olivedrab", "sienna")
    months = [f"{year}-{month:02d}" for year in range(2012, 2025) for month in range(1, 13)]
    figure = plt.figure(figsize=(7, 7), layout="constrained")
    for rigs, name, color in zip(groups, names, colors):
        counts = Counter(
            item["acq_date"][:7]
            for rig in rigs for item in rig.get("FIRMS_observations", [])
        )
        plt.bar(months, [counts.get(month, 0) for month in months],
                label=name, color=color)
    ticks = months[::12] + ["2024-12"]
    labels = [value[:4] for value in ticks]
    plt.xticks(ticks, labels)
    plt.legend()
    plt.xlabel("Time (Months)")
    plt.ylabel("Summed number of FIRMS observations")
    plt.title("Time series of summed FIRMS observations segregated by month")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def _month_labels(end_date: date) -> list[str]:
    labels: list[str] = []
    year, month = 2018, 1
    while (year, month) <= (end_date.year, end_date.month):
        labels.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return labels


def _sentinel_signature(rigs: Sequence[dict], files: Sequence[tuple[date, Path]]) -> str:
    digest = hashlib.sha256()
    for rig in rigs:
        digest.update(
            f"{rig['feature_id']}|{rig['coordinates'][0]}|{rig['coordinates'][1]}\n".encode()
        )
    for observation_date, path in files:
        stat = path.stat()
        digest.update(
            f"{observation_date.isoformat()}|{path.name}|{stat.st_size}|{stat.st_mtime_ns}\n".encode()
        )
    return digest.hexdigest()


def stream_monthly_no2(
    dataset: str, rigs: Sequence[dict], *, daily_directory: Path | None = None,
    cache_path: Path | None = None, monthly_path: Path | None = None,
) -> dict[str, np.ndarray]:
    """Select Stage 2 monthly estimates for the requested facilities."""
    del daily_directory, cache_path
    monthly_path = monthly_path or FACILITY_MONTHLY
    monthly = pd.read_csv(monthly_path, dtype={"feature_id": str, "Match": str})
    monthly["period"] = (
        monthly["year"].astype(int).astype(str).str.zfill(4)
        + "-" + monthly["month"].astype(int).astype(str).str.zfill(2)
    )
    months = sorted(monthly["period"].unique())
    month_index = {month: index for index, month in enumerate(months)}
    rig_index = {
        str(rig["feature_id"]).strip().casefold(): index
        for index, rig in enumerate(rigs)
    }
    values = np.full((len(rigs), len(months)), np.nan, dtype=np.float64)
    for row in monthly.itertuples(index=False):
        index = rig_index.get(str(row.feature_id).strip().casefold())
        if index is None:
            continue
        mean = pd.to_numeric(pd.Series([row.mean_no2_mol_m2]), errors="coerce").iloc[0]
        if np.isfinite(mean):
            values[index, month_index[row.period]] = float(mean)
    valid_months = [
        month for column, month in enumerate(months)
        if np.isfinite(values[:, column]).any()
    ]
    if not valid_months:
        raise ValueError(f"No Stage 2 monthly estimates matched facilities for {dataset}")
    return {
        "monthly_mean": values,
        "months": np.asarray(months),
        "valid_dates": np.asarray(valid_months),
    }
