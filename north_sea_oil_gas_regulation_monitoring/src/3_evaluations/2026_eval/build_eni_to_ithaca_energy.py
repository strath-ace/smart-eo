from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "eo-fril-matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

try:
    from .eval_paths import FACILITY_MONTHLY, MAPPING, RESULTS_DIRECTORY, SOURCE_MANIFEST
    from .suite_common import comparison_prefixes, yearly_rows
except ImportError:
    from eval_paths import FACILITY_MONTHLY, MAPPING, RESULTS_DIRECTORY, SOURCE_MANIFEST
    from suite_common import comparison_prefixes, yearly_rows


SOURCE_OWNER = "ENI UK LIMITED"
TARGET_OWNER = "ITHACA ENERGY"
SPLIT_YEAR = 2025
SOLID_INFRASTRUCTURE_TYPES = {"FPSO", "PLATFORM"}

CHANGED_GROUP = "eni_changed_to_ithaca_energy"
CONTROL_GROUP = "eni_kept_eni"
ITHACA_GROUP = "ithaca_kept_ithaca"

COLORS = {
    CHANGED_GROUP: "#E67E22",
    CONTROL_GROUP: "#2C7FB8",
    ITHACA_GROUP: "#2CA25F",
}

LABELS = {
    CHANGED_GROUP: "ENI to Ithaca Energy",
    CONTROL_GROUP: "Remained ENI",
    ITHACA_GROUP: "Remained Ithaca Energy",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the ENI to Ithaca Energy NO2 comparison case."
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=MAPPING,
    )
    parser.add_argument(
        "--monthly",
        type=Path,
        default=FACILITY_MONTHLY,
    )
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIRECTORY / "eni_to_ithaca_energy",
    )
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def normalize(value: str | None) -> str:
    return (value or "").strip().upper()


def normalize_id(value: str | None) -> str:
    return (value or "").strip().lower()


def has_usable_no2(rows: list[dict[str, str]]) -> bool:
    return any(
        normalize(row.get("analysis_status")) == "AVAILABLE"
        and (row.get("mean_no2_mol_m2") or "").strip()
        for row in rows
    )


def export_facility_csvs(
    output_dir: Path,
    header: list[str],
    selected_ids: list[str],
    yearly_by_id: dict[str, list[dict[str, str]]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for feature_id in selected_ids:
        rows = yearly_by_id[feature_id]
        output_path = output_dir / f"{feature_id}_yearly_no2.csv"
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def series_for_ids(
    selected_ids: list[str],
    yearly_by_id: dict[str, list[dict[str, str]]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for feature_id in selected_ids:
        points: list[tuple[int, float]] = []
        for row in yearly_by_id[feature_id]:
            raw_mean = (row.get("mean_no2_mol_m2") or "").strip()
            raw_year = (row.get("year") or "").strip()
            if not raw_mean or not raw_year:
                continue
            try:
                mean = float(raw_mean)
                year = int(raw_year)
            except ValueError:
                continue
            if np.isfinite(mean):
                points.append((year, mean))
        points.sort()
        if points:
            result[feature_id] = (
                np.array([point[0] for point in points], dtype=int),
                np.array([point[1] for point in points], dtype=float),
            )
    return result


def yearly_group_means(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[int, float]:
    values: dict[int, list[float]] = defaultdict(list)
    for years, means in series.values():
        for year, mean in zip(years, means, strict=True):
            values[int(year)].append(float(mean))
    return {
        year: float(np.mean(means))
        for year, means in sorted(values.items())
        if means
    }


def style_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean NO₂ column density (mol/m²)")
    ax.grid(alpha=0.25)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


def facility_line_style(infrastructure_type: str) -> str:
    return "-" if infrastructure_type in SOLID_INFRASTRUCTURE_TYPES else "--"


def add_infrastructure_style_key(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(
        [
            Line2D([0], [0], color="#555555", linewidth=1.5, linestyle="-"),
            Line2D([0], [0], color="#555555", linewidth=1.5, linestyle="--"),
        ]
    )
    labels.extend(
        [
            "FPSO / platform facility",
            "Other infrastructure facility",
        ]
    )
    ax.legend(handles, labels, frameon=True)


def plot_pre_post_split(
    changed: dict[str, tuple[np.ndarray, np.ndarray]],
    control: dict[str, tuple[np.ndarray, np.ndarray]],
    ithaca: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    combined_values: dict[int, list[float]] = defaultdict(list)
    for group in (changed, control):
        for years, means in group.values():
            for year, mean in zip(years, means, strict=True):
                if int(year) <= SPLIT_YEAR:
                    combined_values[int(year)].append(float(mean))

    combined_means = {
        year: float(np.mean(values))
        for year, values in sorted(combined_values.items())
        if values
    }
    changed_means = yearly_group_means(changed)
    control_means = yearly_group_means(control)
    ithaca_means = yearly_group_means(ithaca)

    figure, ax = plt.subplots(figsize=(11, 6.5))
    combined_years = sorted(combined_means)
    ax.plot(
        combined_years,
        [combined_means[year] for year in combined_years],
        color="#222222",
        marker="o",
        linewidth=2.6,
        label="Combined ENI facilities before transfer",
    )

    for group_name, means in (
        (CHANGED_GROUP, changed_means),
        (CONTROL_GROUP, control_means),
    ):
        post_years = sorted(year for year in means if year > SPLIT_YEAR)
        if not post_years:
            continue
        plot_years = post_years
        plot_values = [means[year] for year in post_years]
        if SPLIT_YEAR in combined_means:
            plot_years = [SPLIT_YEAR, *plot_years]
            plot_values = [combined_means[SPLIT_YEAR], *plot_values]
        ax.plot(
            plot_years,
            plot_values,
            color=COLORS[group_name],
            marker="o",
            linewidth=2.3,
            label=LABELS[group_name],
        )

    ithaca_years = sorted(ithaca_means)
    ax.plot(
        ithaca_years,
        [ithaca_means[year] for year in ithaca_years],
        color=COLORS[ITHACA_GROUP],
        marker="o",
        linewidth=2.3,
        label=LABELS[ITHACA_GROUP],
    )

    ax.axvline(SPLIT_YEAR, color="#777777", linestyle="--", linewidth=1.2)
    ax.set_title(
        "ENI and Ithaca Energy facilities: mean NO₂ around the 2025 ownership split"
    )
    style_axis(ax)
    ax.legend(frameon=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_facility_series(
    changed: dict[str, tuple[np.ndarray, np.ndarray]],
    control: dict[str, tuple[np.ndarray, np.ndarray]],
    ithaca: dict[str, tuple[np.ndarray, np.ndarray]],
    facility_types: dict[str, str],
    output_path: Path,
) -> None:
    figure, ax = plt.subplots(figsize=(11, 7))
    for group_name, group in (
        (CHANGED_GROUP, changed),
        (CONTROL_GROUP, control),
        (ITHACA_GROUP, ithaca),
    ):
        color = COLORS[group_name]
        for feature_id, (years, means) in group.items():
            ax.plot(
                years,
                means,
                color=color,
                linewidth=1.1,
                linestyle=facility_line_style(facility_types[feature_id]),
                alpha=0.35,
            )

        group_means = yearly_group_means(group)
        years = sorted(group_means)
        ax.plot(
            years,
            [group_means[year] for year in years],
            color=color,
            linewidth=3,
            label=f"{LABELS[group_name]} average (n={len(group)})",
        )

    ax.set_title(
        "All ENI and Ithaca Energy facilities: individual and mean NO₂ time series"
    )
    style_axis(ax)
    add_infrastructure_style_key(ax)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    previous_label, current_label = comparison_prefixes(args.source_manifest)
    _, mapping_rows = read_csv(args.mapping)
    yearly_header, annual_rows = yearly_rows(args.monthly)

    yearly_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in annual_rows:
        yearly_by_id[normalize_id(row.get("feature_id"))].append(row)

    changed_ids: list[str] = []
    retained_ids: list[str] = []
    excluded_retained_ids: list[str] = []
    ithaca_ids: list[str] = []
    excluded_ithaca_ids: list[str] = []
    facility_types: dict[str, str] = {}

    for row in mapping_rows:
        feature_id = normalize_id(row.get("feature_id"))
        facility_types[feature_id] = normalize(
            row.get(f"{current_label}_type_of_infrastructure")
            or row.get(f"{previous_label}_type_of_infrastructure")
        )
        old_owner = normalize(row.get(f"{previous_label}_reporting_group_name"))
        new_owner = normalize(row.get(f"{current_label}_reporting_group_name"))
        if old_owner == SOURCE_OWNER and new_owner == TARGET_OWNER:
            changed_ids.append(feature_id)
        elif old_owner == SOURCE_OWNER and new_owner == SOURCE_OWNER:
            if has_usable_no2(yearly_by_id[feature_id]):
                retained_ids.append(feature_id)
            else:
                excluded_retained_ids.append(feature_id)
        elif old_owner == TARGET_OWNER and new_owner == TARGET_OWNER:
            if has_usable_no2(yearly_by_id[feature_id]):
                ithaca_ids.append(feature_id)
            else:
                excluded_ithaca_ids.append(feature_id)

    if not changed_ids:
        raise ValueError(
            f"No ENI to Ithaca transfers were found between {previous_label} and {current_label}"
        )

    data_dir = args.output / "data"
    figures_dir = args.output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    export_facility_csvs(
        data_dir / CHANGED_GROUP,
        yearly_header,
        changed_ids,
        yearly_by_id,
    )
    export_facility_csvs(
        data_dir / CONTROL_GROUP,
        yearly_header,
        retained_ids,
        yearly_by_id,
    )
    export_facility_csvs(
        data_dir / ITHACA_GROUP,
        yearly_header,
        ithaca_ids,
        yearly_by_id,
    )

    changed_series = series_for_ids(changed_ids, yearly_by_id)
    control_series = series_for_ids(retained_ids, yearly_by_id)
    ithaca_series = series_for_ids(ithaca_ids, yearly_by_id)
    plot_pre_post_split(
        changed_series,
        control_series,
        ithaca_series,
        figures_dir / "pre_post_2025_split.png",
    )
    plot_facility_series(
        changed_series,
        control_series,
        ithaca_series,
        facility_types,
        figures_dir / "facility_no2_timeseries.png",
    )

    print(f"Changed facilities exported: {len(changed_ids)}")
    print(f"Retained ENI facilities exported: {len(retained_ids)}")
    print(f"Retained Ithaca Energy facilities exported: {len(ithaca_ids)}")
    print(f"Retained facilities excluded (outside NO2 coverage): {len(excluded_retained_ids)}")
    print("Changed IDs:")
    for feature_id in changed_ids:
        print(f"  {feature_id}")
    print("Excluded retained IDs:")
    for feature_id in excluded_retained_ids:
        print(f"  {feature_id}")


if __name__ == "__main__":
    main()
