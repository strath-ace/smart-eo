#!/usr/bin/env python3
"""Combine all available NSTA surface-point exports into one facility CSV.

Dated source folders are discovered automatically. Rows are matched by Feature
ID first, then by optional aliases from ``feature_id_manual_maps.csv``. Any rows
left unmatched are compared by geographic distance and printed for review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANUAL_MAPS = Path(__file__).resolve().with_name(
    "feature_id_manual_maps.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "PROCESSED" / "NSTA" / "mapped_surface_points.csv"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "PROCESSED" / "NSTA" / "mapped_surface_points_manifest.json"
)
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "RAW" / "NSTA"
EARTH_RADIUS_KM = 6371.0088


# (output label, legacy column name, modern column name)
FIELD_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("object_id", "OBJECTID", "OBJECTID"),
    ("infrastructure_name", "NAME", "Infrastructure name"),
    ("type_of_infrastructure", "INF_TYPE", "Type of infrastructure"),
    ("reporting_group_name", "REP_GROUP", "Reporting group name"),
    ("description", "DESCRIPTIO", "Description"),
    ("platform_flag", "PLAT_FLAG", "Platform flag"),
    ("pipeline_system_name", "PIPE_SYS", "Pipeline system name"),
    ("description_of_point", "POINT_DESC", "Description of point"),
    ("original_crs_code", "CRS_CODE", "Original CRS code"),
    ("original_crs_name", "CRS_NAME", "Original CRS name"),
    ("original_x_longitude", "ORIG_X_LON", "Original X longitude"),
    ("original_y_latitude", "ORIG_Y_LAT", "Original Y latitude"),
    ("status", "STATUS", "Status"),
    ("comments", "COMMENTS", "Comments"),
    ("start_date", "START_DATE", "Start date"),
    ("end_date", "END_DATE", "End date"),
    ("reason_for_end_date", "END_REAS", "Reason for end date"),
    ("date_updated", "UPD_DATE", "Date updated"),
    ("update_type", "UPD_TYPE", "Update type"),
    ("reason_for_update", "UPD_REAS", "Reason for update"),
    ("legacy_id", "LEGACY_ID", "Legacy ID"),
    ("x", "X", "x"),
    ("y", "Y", "y"),
)


def feature_key(value: object) -> str:
    return str(value or "").strip().casefold()


def distances(
    current: tuple[float, float], previous: tuple[float, float]
) -> tuple[float, float]:
    """Return Euclidean degree distance and great-circle distance in km."""
    current_lon, current_lat = current
    previous_lon, previous_lat = previous
    degree_distance = math.hypot(
        current_lon - previous_lon, current_lat - previous_lat
    )

    lon_delta = math.radians(current_lon - previous_lon)
    lat_delta = math.radians(current_lat - previous_lat)
    current_lat_rad = math.radians(current_lat)
    previous_lat_rad = math.radians(previous_lat)
    haversine = (
        math.sin(lat_delta / 2) ** 2
        + math.cos(current_lat_rad)
        * math.cos(previous_lat_rad)
        * math.sin(lon_delta / 2) ** 2
    )
    haversine = min(1.0, max(0.0, haversine))
    km_distance = 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(haversine))
    return degree_distance, km_distance


def _looks_like_mapping_header(row: list[str]) -> bool:
    header_terms = (
        "feature",
        "alias",
        "canonical",
        "current",
        "previous",
        "new_key",
        "new key",
    )
    return any(
        any(term in cell.strip().casefold() for term in header_terms)
        for cell in row
    )


def read_manual_maps(path: Path) -> tuple[dict[str, str], bool]:
    """Read alias-to-canonical mappings; return mappings and existence flag."""
    if not path.is_file():
        return {}, False

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            first_row = next(reader)
        except StopIteration:
            raise ValueError(f"Manual mapping file is empty: {path}") from None
        if len(first_row) < 2:
            raise ValueError(
                f"Manual mapping file must contain at least two columns: {path}"
            )

        mappings: dict[str, str] = {}
        origins: dict[str, int] = {}
        has_header = _looks_like_mapping_header(first_row)
        rows = reader if has_header else iter([first_row, *reader])
        starting_row = 2 if has_header else 1
        for source_row, row in enumerate(rows, start=starting_row):
            if not row or not any(cell.strip() for cell in row):
                continue
            if len(row) < 2:
                raise ValueError(
                    f"Manual mapping row {source_row} must contain at least two columns"
                )
            canonical = row[-1].strip()
            if not canonical:
                raise ValueError(
                    f"Manual mapping row {source_row} has an empty canonical "
                    "Feature ID in its final column"
                )
            aliases = [cell.strip() for cell in row if cell.strip()]
            if len(aliases) < 2:
                raise ValueError(
                    f"Manual mapping row {source_row} must contain at least one "
                    "alias and a canonical Feature ID"
                )
            for alias in aliases:
                key = feature_key(alias)
                existing = mappings.get(key)
                if existing is not None and feature_key(existing) != feature_key(
                    canonical
                ):
                    raise ValueError(
                        f"Manual mapping alias {alias!r} on row {source_row} "
                        f"conflicts with canonical ID {existing!r} from row "
                        f"{origins[key]}"
                    )
                mappings[key] = canonical
                origins[key] = source_row
    return mappings, True


MONTH_NUMBERS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
DATED_FOLDER = re.compile(
    r"^(?P<month>jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"(?:uary|ruary|ch|il|e|y|ust|tember|ober|ember)?[-_ ]+"
    r"(?P<year>\d{4})$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceSpec:
    period: str
    path: Path


@dataclass(frozen=True)
class SourceDataset:
    label: str
    period: str
    path: Path
    rows: list[dict[str, str]]
    columns: dict[str, str]


@dataclass
class FacilityEntity:
    feature_id: str
    records: dict[int, int]


def discover_source_csvs(source_root: Path = DEFAULT_SOURCE_ROOT) -> list[Path]:
    """Find one surface-points CSV in every month-year child directory."""
    if not source_root.is_dir():
        raise ValueError(f"NSTA source directory does not exist: {source_root}")
    discovered: list[tuple[int, int, str, Path]] = []
    for directory in source_root.iterdir():
        if not directory.is_dir():
            continue
        match = DATED_FOLDER.match(directory.name)
        if match is None:
            continue
        candidates = [
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.casefold() == ".csv"
            and "surface" in path.name.casefold()
            and "point" in path.name.casefold()
            and "wgs84" in path.name.casefold()
        ]
        if not candidates:
            print(
                f"Dated source folder has no surface-points WGS84 CSV; skipping: "
                f"{directory.resolve()}"
            )
            continue
        if len(candidates) > 1:
            names = ", ".join(sorted(path.name for path in candidates))
            raise ValueError(
                f"Multiple surface-points CSVs found in {directory}: {names}"
            )
        year = int(match.group("year"))
        month = MONTH_NUMBERS[match.group("month")[:3].casefold()]
        discovered.append((year, month, directory.name.casefold(), candidates[0]))
    discovered.sort(key=lambda item: (item[0], item[1], item[2]))
    if len(discovered) < 2:
        raise ValueError(
            f"At least two dated NSTA surface-point CSVs are required under {source_root}"
        )
    return [item[3] for item in discovered]


def snapshot_period(path: Path) -> str:
    match = DATED_FOLDER.match(path.parent.name)
    if match is None:
        raise ValueError(
            f"Cannot infer a YYYY-MM snapshot period from parent folder "
            f"{path.parent.name!r}; use --source YYYY-MM=PATH"
        )
    month = MONTH_NUMBERS[match.group("month")[:3].casefold()]
    return f"{int(match.group('year')):04d}-{month:02d}"


def discover_source_specs(source_root: Path = DEFAULT_SOURCE_ROOT) -> list[SourceSpec]:
    return [SourceSpec(snapshot_period(path), path) for path in discover_source_csvs(source_root)]


def parse_explicit_source(value: str) -> SourceSpec:
    period, separator, raw_path = value.partition("=")
    if not separator or not re.fullmatch(r"\d{4}-(?:0[1-9]|1[0-2])", period):
        raise argparse.ArgumentTypeError(
            "--source must have the form YYYY-MM=PATH"
        )
    if not raw_path.strip():
        raise argparse.ArgumentTypeError("--source PATH must not be empty")
    return SourceSpec(period, Path(raw_path).expanduser())


def source_labels(specs: list[SourceSpec]) -> list[str]:
    paths = [spec.path for spec in specs]
    years = [spec.period[:4] for spec in specs]
    counts = Counter(years)
    labels: list[str] = []
    for path, year in zip(paths, years, strict=True):
        if year and counts[year] == 1:
            label = year
        else:
            label = re.sub(r"[^0-9A-Za-z]+", "_", path.parent.name).strip("_")
        if not label:
            label = re.sub(r"[^0-9A-Za-z]+", "_", path.stem).strip("_")
        labels.append(label)
    duplicates = [label for label, count in Counter(labels).items() if count > 1]
    if duplicates:
        raise ValueError(f"Source labels are not unique: {', '.join(duplicates)}")
    return labels


def resolve_source_columns(fields: list[str], path: Path) -> dict[str, str]:
    available = {field.strip().casefold(): field for field in fields}
    candidates: dict[str, tuple[str, ...]] = {
        "feature_id": ("FEATURE_ID", "Feature ID"),
    }
    candidates.update(
        {label: (legacy, modern) for label, legacy, modern in FIELD_COLUMNS}
    )
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for logical_name, names in candidates.items():
        actual = next(
            (available[name.casefold()] for name in names if name.casefold() in available),
            None,
        )
        if actual is None:
            missing.append(f"{logical_name} ({' or '.join(names)})")
        else:
            resolved[logical_name] = actual
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    return resolved


def read_source_dataset(path: Path, label: str, period: str) -> SourceDataset:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        columns = resolve_source_columns(fields, path)
        rows = list(reader)
    blank_rows = [
        index + 2
        for index, row in enumerate(rows)
        if not feature_key(row.get(columns["feature_id"]))
    ]
    if blank_rows:
        raise ValueError(
            f"{path} has blank Feature IDs on source rows: "
            + ", ".join(map(str, blank_rows))
        )
    return SourceDataset(label, period, path, rows, columns)


def dataset_value(dataset: SourceDataset, row_index: int, logical_name: str) -> str:
    return dataset.rows[row_index].get(dataset.columns[logical_name], "")


def dataset_coordinates(dataset: SourceDataset, row_index: int) -> tuple[float, float]:
    longitude = dataset_value(dataset, row_index, "x").strip()
    latitude = dataset_value(dataset, row_index, "y").strip()
    try:
        result = float(longitude), float(latitude)
    except ValueError as error:
        raise ValueError(
            f"Invalid coordinates in {dataset.path} row {row_index + 2}: "
            f"x={longitude!r}, y={latitude!r}"
        ) from error
    if not all(math.isfinite(value) for value in result):
        raise ValueError(
            f"Non-finite coordinates in {dataset.path} row {row_index + 2}"
        )
    return result


def entity_reference_coordinates(
    entity: FacilityEntity, datasets: list[SourceDataset]
) -> tuple[float, float]:
    dataset_index = max(entity.records)
    return dataset_coordinates(datasets[dataset_index], entity.records[dataset_index])


def add_dataset_to_entities(
    entities: list[FacilityEntity],
    dataset_index: int,
    datasets: list[SourceDataset],
) -> None:
    dataset = datasets[dataset_index]
    entities_by_key: dict[str, list[int]] = defaultdict(list)
    for entity_index, entity in enumerate(entities):
        if dataset_index not in entity.records:
            entities_by_key[feature_key(entity.feature_id)].append(entity_index)
    rows_by_key: dict[str, list[int]] = defaultdict(list)
    for row_index in range(len(dataset.rows)):
        rows_by_key[
            feature_key(dataset_value(dataset, row_index, "feature_id"))
        ].append(row_index)

    consumed_rows: set[int] = set()
    for key in sorted(entities_by_key.keys() & rows_by_key.keys()):
        candidates: list[tuple[float, float, int, int]] = []
        for entity_index in entities_by_key[key]:
            entity_xy = entity_reference_coordinates(entities[entity_index], datasets)
            for row_index in rows_by_key[key]:
                degree_distance, km_distance = distances(
                    dataset_coordinates(dataset, row_index), entity_xy
                )
                candidates.append(
                    (km_distance, degree_distance, entity_index, row_index)
                )
        candidates.sort()
        used_entities: set[int] = set()
        for _, _, entity_index, row_index in candidates:
            if entity_index in used_entities or row_index in consumed_rows:
                continue
            entities[entity_index].records[dataset_index] = row_index
            used_entities.add(entity_index)
            consumed_rows.add(row_index)

    for row_index in range(len(dataset.rows)):
        if row_index not in consumed_rows:
            entities.append(
                FacilityEntity(
                    dataset_value(dataset, row_index, "feature_id").strip(),
                    {dataset_index: row_index},
                )
            )


def dataset_aliases(dataset: SourceDataset) -> dict[int, tuple[str, ...]]:
    raw_ids = [
        dataset_value(dataset, index, "feature_id").strip()
        for index in range(len(dataset.rows))
    ]
    counts = Counter(feature_key(value) for value in raw_ids)
    occurrences: Counter[str] = Counter()
    result: dict[int, tuple[str, ...]] = {}
    for index, raw_id in enumerate(raw_ids):
        key = feature_key(raw_id)
        if counts[key] > 1:
            occurrences[key] += 1
            result[index] = (f"{raw_id}_{occurrences[key]}", raw_id)
        else:
            result[index] = (raw_id,)
    return result


def mapped_entity_key(
    entity: FacilityEntity,
    datasets: list[SourceDataset],
    aliases: list[dict[int, tuple[str, ...]]],
    mappings: dict[str, str],
) -> str | None:
    canonicals: dict[str, str] = {}
    for dataset_index, row_index in entity.records.items():
        for alias in aliases[dataset_index][row_index]:
            canonical = mappings.get(feature_key(alias))
            if canonical is not None:
                canonicals[feature_key(canonical)] = canonical
                break
    if len(canonicals) > 1:
        details = ", ".join(canonicals.values())
        raise ValueError(
            f"Records already matched by exact Feature ID map to conflicting "
            f"canonical IDs: {details}"
        )
    return next(iter(canonicals.values()), None)


def entity_distance(
    left: FacilityEntity,
    right: FacilityEntity,
    datasets: list[SourceDataset],
) -> tuple[float, float]:
    candidates: list[tuple[float, float]] = []
    for left_dataset, left_row in left.records.items():
        for right_dataset, right_row in right.records.items():
            degree_distance, km_distance = distances(
                dataset_coordinates(datasets[left_dataset], left_row),
                dataset_coordinates(datasets[right_dataset], right_row),
            )
            candidates.append((km_distance, degree_distance))
    km_distance, degree_distance = min(candidates)
    return degree_distance, km_distance


def apply_manual_maps_to_entities(
    entities: list[FacilityEntity],
    datasets: list[SourceDataset],
    mappings: dict[str, str],
) -> int:
    aliases = [dataset_aliases(dataset) for dataset in datasets]
    grouped: dict[str, list[FacilityEntity]] = defaultdict(list)
    for entity in entities:
        canonical = mapped_entity_key(entity, datasets, aliases, mappings)
        if canonical is not None:
            entity.feature_id = canonical
            grouped[feature_key(canonical)].append(entity)

    merge_count = 0
    for group in grouped.values():
        while True:
            candidates: list[tuple[float, float, int, int]] = []
            for left_index, left in enumerate(group):
                for right_index in range(left_index + 1, len(group)):
                    right = group[right_index]
                    if left.records.keys() & right.records.keys():
                        continue
                    degree_distance, km_distance = entity_distance(
                        left, right, datasets
                    )
                    candidates.append(
                        (km_distance, degree_distance, left_index, right_index)
                    )
            if not candidates:
                break
            _, _, left_index, right_index = min(candidates)
            group[left_index].records.update(group[right_index].records)
            removed = group.pop(right_index)
            entities.remove(removed)
            merge_count += 1
    return merge_count


def uniquify_entity_ids(entities: list[FacilityEntity]) -> None:
    counts = Counter(feature_key(entity.feature_id) for entity in entities)
    occurrences: Counter[str] = Counter()
    for entity in entities:
        key = feature_key(entity.feature_id)
        if counts[key] > 1:
            occurrences[key] += 1
            entity.feature_id = f"{entity.feature_id}_{occurrences[key]}"


def many_source_header(datasets: list[SourceDataset]) -> list[str]:
    header = ["feature_id", "Match"]
    for position, (logical_name, _, _) in enumerate(FIELD_COLUMNS):
        if position:
            header.append("")
        header.extend(
            f"{dataset.label}_{logical_name}" for dataset in datasets
        )
        header.append(f"changed_{logical_name}")
    return header


def entity_match_label(entity: FacilityEntity, datasets: list[SourceDataset]) -> str:
    if len(datasets) == 2 and len(entity.records) == 2:
        return "Both"
    if len(entity.records) == len(datasets):
        return "All"
    return ",".join(
        datasets[index].label for index in sorted(entity.records)
    )


def write_many_source_output(
    path: Path, entities: list[FacilityEntity], datasets: list[SourceDataset]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(many_source_header(datasets))
        for entity in entities:
            output = [entity.feature_id, entity_match_label(entity, datasets)]
            for position, (logical_name, _, _) in enumerate(FIELD_COLUMNS):
                if position:
                    output.append("")
                values = [
                    dataset_value(dataset, entity.records[index], logical_name)
                    if index in entity.records
                    else ""
                    for index, dataset in enumerate(datasets)
                ]
                changed = len({value.casefold() for value in values}) > 1
                output.extend(values)
                output.append("Y" if changed else "")
            writer.writerow(output)


def print_many_source_unmatched_distances(
    entities: list[FacilityEntity], datasets: list[SourceDataset]
) -> None:
    partial = [entity for entity in entities if len(entity.records) < len(datasets)]
    pairs: list[tuple[float, float, int, int, int, int]] = []
    for left_index, left in enumerate(partial):
        for right_index in range(left_index + 1, len(partial)):
            right = partial[right_index]
            for left_dataset, left_row in left.records.items():
                for right_dataset, right_row in right.records.items():
                    if left_dataset == right_dataset:
                        continue
                    degree_distance, km_distance = distances(
                        dataset_coordinates(datasets[left_dataset], left_row),
                        dataset_coordinates(datasets[right_dataset], right_row),
                    )
                    pairs.append(
                        (
                            km_distance,
                            degree_distance,
                            left_dataset,
                            left_row,
                            right_dataset,
                            right_row,
                        )
                    )
    pairs.sort()
    print("\nUNMATCHED CROSS-SOURCE DISTANCES")
    print("--------------------------------")
    if not pairs:
        print("No cross-source unmatched record pairs.")
        return
    for km_distance, degree_distance, left_dataset, left_row, right_dataset, right_row in pairs:
        left = datasets[left_dataset]
        right = datasets[right_dataset]
        left_xy = dataset_coordinates(left, left_row)
        right_xy = dataset_coordinates(right, right_row)
        print(
            f"{km_distance:.6f} km | {degree_distance:.9f} degrees | "
            f"{left.label} row={left_row + 2} "
            f"id={dataset_value(left, left_row, 'feature_id')} "
            f"name={dataset_value(left, left_row, 'infrastructure_name')} "
            f"coord=({left_xy[0]:.9f}, {left_xy[1]:.9f}) | "
            f"{right.label} row={right_row + 2} "
            f"id={dataset_value(right, right_row, 'feature_id')} "
            f"name={dataset_value(right, right_row, 'infrastructure_name')} "
            f"coord=({right_xy[0]:.9f}, {right_xy[1]:.9f})"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def write_manifest(
    path: Path,
    datasets: list[SourceDataset],
    manual_maps_path: Path,
    output_path: Path,
    entity_count: int,
) -> None:
    payload = {
        "algorithm_version": 1,
        "output": display_path(output_path),
        "output_sha256": sha256_file(output_path),
        "record_count": entity_count,
        "manual_maps": display_path(manual_maps_path),
        "manual_maps_sha256": (
            sha256_file(manual_maps_path) if manual_maps_path.is_file() else None
        ),
        "sources": [
            {
                "label": dataset.label,
                "period": dataset.period,
                "path": display_path(dataset.path),
                "sha256": sha256_file(dataset.path),
                "record_count": len(dataset.rows),
            }
            for dataset in datasets
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_many(
    source_specs: list[SourceSpec],
    manual_maps_path: Path,
    output_path: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> None:
    source_specs = sorted(source_specs, key=lambda spec: (spec.period, str(spec.path)))
    periods = [spec.period for spec in source_specs]
    if len(periods) != len(set(periods)):
        duplicates = sorted(period for period, count in Counter(periods).items() if count > 1)
        raise ValueError(f"Snapshot periods must be unique: {', '.join(duplicates)}")
    labels = source_labels(source_specs)
    datasets = [
        read_source_dataset(spec.path, label, spec.period)
        for spec, label in zip(source_specs, labels, strict=True)
    ]
    entities: list[FacilityEntity] = []
    for dataset_index in range(len(datasets)):
        add_dataset_to_entities(entities, dataset_index, datasets)

    mappings, maps_exist = read_manual_maps(manual_maps_path)
    manual_merges = apply_manual_maps_to_entities(entities, datasets, mappings)
    uniquify_entity_ids(entities)
    entities.sort(
        key=lambda entity: (
            min(entity.records),
            min(entity.records.values()),
            feature_key(entity.feature_id),
        )
    )
    write_many_source_output(output_path, entities, datasets)
    write_manifest(manifest_path, datasets, manual_maps_path, output_path, len(entities))

    print(f"Discovered source CSVs: {len(datasets)}")
    for dataset in datasets:
        print(f"  {dataset.label}: {len(dataset.rows)} records | {dataset.path.resolve()}")
        duplicates = Counter(
            feature_key(dataset_value(dataset, index, "feature_id"))
            for index in range(len(dataset.rows))
        )
        for key, count in duplicates.items():
            if count > 1:
                print(f"    duplicate Feature ID: {key} ({count} records)")
    if maps_exist:
        print(f"Manual Feature ID aliases: {len(mappings)}")
    else:
        print(
            f"Manual mapping file not found; skipping manual mappings: "
            f"{manual_maps_path.resolve()}"
        )
    print(f"Manual entity merges: {manual_merges}")
    print(f"Mapped output rows: {len(entities)}")
    print(f"Output: {output_path.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")
    print_many_source_unmatched_distances(entities, datasets)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_csvs",
        nargs="*",
        type=Path,
        help=(
            "NSTA surface-point CSVs in chronological order. If omitted, files "
            "are discovered in month-year folders under RAW/NSTA."
        ),
    )
    parser.add_argument(
        "--source-root", type=Path, default=DEFAULT_SOURCE_ROOT
    )
    parser.add_argument(
        "--source",
        action="append",
        type=parse_explicit_source,
        default=[],
        help="Explicit dated source in YYYY-MM=PATH form (repeatable).",
    )
    parser.add_argument(
        "--manual-maps", type=Path, default=DEFAULT_MANUAL_MAPS
    )
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.source and args.source_csvs:
            raise ValueError("Use either positional source CSVs or --source, not both")
        if args.source:
            source_specs = args.source
        elif args.source_csvs:
            source_specs = [SourceSpec(snapshot_period(path), path) for path in args.source_csvs]
        else:
            source_specs = discover_source_specs(args.source_root)
        if len(source_specs) < 2:
            raise ValueError("At least two source CSVs are required")
        run_many(source_specs, args.manual_maps, args.output, args.manifest)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
