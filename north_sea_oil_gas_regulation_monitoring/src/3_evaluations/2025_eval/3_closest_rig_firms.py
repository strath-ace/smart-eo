#!/usr/bin/env python3
"""Combine NSTA, FIRMS, Sentinel NO2, and oil-price evaluation outputs."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import cKDTree
from scipy.stats import median_abs_deviation

from eval_common import (
    OIL_PRICE_CSV,
    attach_firms,
    bounded_rigs,
    clean_platform_path,
    csv_rows,
    firms_observations,
    firms_platform_path,
    load_json,
    parse_datasets,
    plot_combined_firms,
    plot_many_firms,
    prepare_directories,
    results_directory,
    save_json,
    stream_monthly_no2,
    write_csv,
)


COLORS = {"PLC": "skyblue", "LTD": "sienna", "SOE": "olivedrab"}
GROUPS = tuple(COLORS)
METADATA_COLUMNS = [
    "asset_name", "asset_owner", "parent_company", "ownership_type",
    "longitude", "latitude",
]


def finite_mean(values: np.ndarray, axis=None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = np.nansum(values, axis=axis)
    count = np.sum(np.isfinite(values), axis=axis)
    output = np.full(np.shape(total), np.nan, dtype=float)
    return np.divide(total, count, out=output, where=count > 0)


def finite_std(values: np.ndarray, axis=None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = finite_mean(values, axis=axis)
    expanded = np.expand_dims(mean, axis) if axis is not None else mean
    finite = np.isfinite(values)
    squares = np.where(finite, np.square(values - expanded), 0.0)
    count = np.sum(finite, axis=axis)
    variance = np.full(np.shape(mean), np.nan, dtype=float)
    np.divide(np.sum(squares, axis=axis), count, out=variance, where=count > 0)
    return np.sqrt(variance)


def rolling_average(values: np.ndarray, window: int = 12) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < window:
        return np.asarray([], dtype=float)
    output = np.full(values.size - window + 1, np.nan)
    for index in range(output.size):
        window_values = values[index:index + window]
        if np.isfinite(window_values).any():
                output[index] = finite_mean(window_values)
    return output


def normalize_by_neighbours(raw: np.ndarray, rigs: Sequence[dict]) -> np.ndarray:
    points = np.asarray([rig["coordinates"] for rig in rigs], dtype=float)
    tree = cKDTree(points)
    neighbour_count = min(11, len(rigs))
    _, indexes = tree.query(points, k=neighbour_count)
    if neighbour_count == 1:
        return raw.copy()
    indexes = np.atleast_2d(indexes)[:, 1:]
    # Preserve the original evaluation's normalization: every facility is
    # compared with one common, neighbour-weighted monthly reference formed
    # from the ten nearest facilities of every facility in the dataset.
    reference = raw[indexes].reshape(-1, raw.shape[1])
    reference_mean = finite_mean(reference, axis=0)
    reference_std = finite_std(reference, axis=0)
    overall_mean = finite_mean(raw)
    overall_std = finite_std(raw)
    normalized = np.full(raw.shape, np.nan)
    for index in range(len(rigs)):
        z_score = np.full(raw.shape[1], np.nan)
        np.divide(
            raw[index] - reference_mean, reference_std,
            out=z_score, where=reference_std > 0,
        )
        z_score[raw[index] > 3] = np.nan
        z_score[raw[index] < -3] = np.nan
        normalized[index] = z_score * overall_std + overall_mean
    return normalized


def group_array(values: np.ndarray, rigs: Sequence[dict], ownership_type: str) -> np.ndarray:
    mask = np.asarray(
        [rig["parent_company_type"] == ownership_type for rig in rigs], dtype=bool
    )
    return values[mask]


def plot_emission_summaries(
    results, months: Sequence[str], raw: np.ndarray, normalized: np.ndarray,
    rigs: Sequence[dict],
) -> None:
    x = np.arange(len(months))
    tick_indexes = np.arange(0, len(months), 12)
    tick_labels = [months[index][:4] for index in tick_indexes]
    group_means = {
        group: finite_mean(group_array(normalized, rigs, group), axis=0)
        for group in GROUPS
    }

    figure = plt.figure(layout="constrained")
    bottom = np.zeros(len(months))
    for group in GROUPS:
        values = np.nan_to_num(group_means[group], nan=0.0)
        plt.bar(x, values, bottom=bottom, label=f"{group} Rigs", color=COLORS[group])
        bottom += values
    plt.xticks(tick_indexes, tick_labels)
    plt.xlabel("Time (Months)")
    plt.ylabel("NO2 Emissions (molecules/cm²)")
    plt.title("Total NO2 Emissions per month")
    plt.legend()
    figure.savefig(results / "emissions_summed.png")
    plt.close(figure)

    figure = plt.figure(layout="constrained")
    for group in GROUPS:
        rolled = rolling_average(group_means[group])
        plt.plot(np.arange(6, 6 + len(rolled)), rolled, color=COLORS[group], label=group)
    all_rolled = rolling_average(finite_mean(normalized, axis=0))
    plt.plot(np.arange(6, 6 + len(all_rolled)), all_rolled, color="black", label="Average")
    plt.xticks(tick_indexes, tick_labels)
    plt.xlabel("Time (Months)")
    plt.ylabel("NO2 Emissions (molecules/cm²)")
    plt.title("Rolling average NO2 Emissions for each ownership type")
    plt.legend()
    figure.savefig(results / "emissions_averaged.png")
    plt.close(figure)

    figure, axis = plt.subplots(
        figsize=(7, 7), subplot_kw={"projection": "polar"}, layout="constrained"
    )
    rolled_length = len(rolling_average(normalized[0]))
    # A 12-month rolling value is centred roughly six months after January,
    # so start at pi (July) and make one complete revolution per year.
    theta = np.pi + np.arange(rolled_length) * (2 * np.pi / 12)
    ownership_counts = Counter(rig["parent_company_type"] for rig in rigs)
    for rig_values, rig in zip(normalized, rigs):
        group = rig["parent_company_type"]
        axis.plot(
            theta, rolling_average(rig_values), color=COLORS[group],
            alpha=ownership_counts[group] / 2000,
        )
    for group in GROUPS:
        axis.plot(
            theta, rolling_average(group_means[group]), color=COLORS[group],
            label=f"{group} Ownership",
        )
    axis.plot(
        theta, rolling_average(finite_mean(normalized, axis=0)),
        color="black", label="All Rigs",
    )
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    axis.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
    axis.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    axis.set_yticks([])
    axis.set_yticklabels([])
    axis.grid(False)
    axis.plot(
        [0, 0], [0, 0.00006], color="black", linewidth=1.5,
        linestyle="--", label="y-values x10^-5",
    )
    label_radii = [0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.000058]
    label_angles = [0.1, 0.05, 0.1 / 3, 0.025, 0.02, 0.1 / 6]
    for angle, radius in zip(label_angles, label_radii):
        axis.text(
            angle, radius, str(int(round(radius * 100000))),
            ha="center", va="bottom", fontsize=15, color="black",
        )
    axis.legend(loc="upper left", bbox_to_anchor=(0.8, 1.05))
    axis.set_title(
        "Rolling average NO2 Emissions for each rig averaged by ownership per month"
    )
    figure.savefig(results / "NO2_emissions_averaged_with_assumptions.png", dpi=300)
    plt.close(figure)

    recent = normalized[:, -12:]
    distributions: list[np.ndarray] = []
    labels: list[str] = []
    colors: list[str] = []
    for group in GROUPS:
        values = finite_mean(group_array(recent, rigs, group), axis=1)
        values = values[np.isfinite(values)]
        if values.size:
            distributions.append(values)
            labels.append(group)
            colors.append(COLORS[group])
    figure = plt.figure(figsize=(7, 7), layout="constrained")
    parts = plt.violinplot(distributions, showmeans=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(1)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.title(f"Distribution of Oil Rigs with emissions ({months[-1][:4]} rolling year)")
    plt.xlabel("Ownership Type")
    plt.ylabel("NO2 Emissions (molecules/cm²)")
    figure.savefig(results / "NO2_emissions_violin.png", dpi=300)
    plt.close(figure)


def firms_monthly(rigs: Sequence[dict], months: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    month_index = {month: index for index, month in enumerate(months)}
    counts = np.zeros((len(rigs), len(months)), dtype=float)
    brightness_sum = np.zeros_like(counts)
    brightness_count = np.zeros_like(counts)
    for rig_index, rig in enumerate(rigs):
        for observation in rig.get("FIRMS_observations", []):
            index = month_index.get(str(observation.get("acq_date", ""))[:7])
            if index is None:
                continue
            counts[rig_index, index] += 1
            try:
                brightness = float(observation["brightness"])
            except (KeyError, TypeError, ValueError):
                continue
            brightness_sum[rig_index, index] += brightness
            brightness_count[rig_index, index] += 1
    brightness = np.full_like(brightness_sum, np.nan)
    np.divide(
        brightness_sum, brightness_count, out=brightness, where=brightness_count > 0
    )
    return counts, brightness


def oil_prices(months: Sequence[str]) -> np.ndarray:
    by_month: dict[str, float] = {}
    for row in csv_rows(OIL_PRICE_CSV):
        if len(row) < 2:
            continue
        try:
            parsed = pd.to_datetime(row[0], format="%d/%m/%Y")
            by_month[parsed.strftime("%Y-%m")] = float(row[1])
        except (ValueError, TypeError):
            continue
    return np.asarray([by_month.get(month, np.nan) for month in months])


def table_frame(values: np.ndarray, rigs: Sequence[dict], months: Sequence[str]) -> pd.DataFrame:
    rows = []
    for rig, observations in zip(rigs, values):
        rows.append(
            [rig["name"], rig["asset_owner"], rig["parent_company"],
             rig["parent_company_type"], *rig["coordinates"], *observations]
        )
    return pd.DataFrame(rows, columns=[*METADATA_COLUMNS, *months])


def write_tables(
    results, months: Sequence[str], raw: np.ndarray, normalized: np.ndarray,
    counts: np.ndarray, brightness: np.ndarray, rigs: Sequence[dict],
) -> None:
    frames = {
        "SENTINEL_raw_NO2": table_frame(raw, rigs, months),
        "SENTINEL_normalised_NO2": table_frame(normalized, rigs, months),
        "FIRMS_counts": table_frame(counts, rigs, months),
        "FIRMS_average_brightness": table_frame(brightness, rigs, months),
    }
    oil = oil_prices(months)
    frames["OIL_price_monthly"] = pd.DataFrame(
        [["OIL SPOT PRICE", "", "", "", "", "", *oil]],
        columns=[*METADATA_COLUMNS, *months],
    )
    with pd.ExcelWriter(results / "output.xlsx", engine="openpyxl") as writer:
        for sheet_name, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)

    no2_rows = [[
        "Pre/Post Normalisation", "Ownership Type", "Number of oil rigs in group",
        "Missing datapoints", "Mean", "Standard deviation", "Median Absolute Deviation",
        "Minimum", "Maximum", "Interquartile Range",
    ]]
    for label, values in (("Raw", raw), ("Normalised", normalized)):
        for group in GROUPS:
            grouped = group_array(values, rigs, group)
            finite = grouped[np.isfinite(grouped)]
            no2_rows.append([
                label, group, grouped.shape[0], int(np.isnan(grouped).sum()),
                float(np.nanmean(grouped)), float(np.nanstd(grouped)),
                float(median_abs_deviation(finite)) if finite.size else np.nan,
                float(np.nanmin(grouped)), float(np.nanmax(grouped)),
                float(np.percentile(finite, 75) - np.percentile(finite, 25))
                if finite.size else np.nan,
            ])
    write_csv(results / "table_for_paper_no2.csv", no2_rows)

    firms_rows = [[
        "Ownership Type", "Number of oil rigs in group",
        "Percentage of each rig where no flaring detected", "Mean", "Standard deviation",
        "Median Absolute Deviation", "Minimum", "Maximum", "Interquartile Range",
    ]]
    for group in GROUPS:
        grouped = group_array(counts, rigs, group)
        flat = grouped[np.isfinite(grouped)]
        firms_rows.append([
            group, grouped.shape[0], 100 * float(np.mean(grouped == 0)),
            float(np.mean(flat)), float(np.std(flat)),
            float(median_abs_deviation(flat)), float(np.min(flat)), float(np.max(flat)),
            float(np.percentile(flat, 75) - np.percentile(flat, 25)),
        ])
    write_csv(results / "table_for_paper_viirs.csv", firms_rows)


def plot_facility_areas(results, raw: np.ndarray, rigs: Sequence[dict]) -> None:
    """Plot Stage 2 facility summaries without rereading daily rasters."""
    overall = finite_mean(raw, axis=1)
    sizes = np.sum(np.isfinite(raw), axis=1).astype(int).tolist()
    order = [index for index in np.argsort(sizes)[::-1] if sizes[index] > 0]

    figure, axis = plt.subplots(figsize=(8, 9))
    image = axis.scatter(
        [rig["coordinates"][0] for rig in rigs],
        [rig["coordinates"][1] for rig in rigs],
        c=overall, cmap="Reds", s=24, edgecolor="black", linewidth=0.2,
    )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title("Mean monthly Sentinel-5P NO₂ by facility")
    figure.colorbar(image, ax=axis, label="Mean NO₂ (mol/m²)")
    figure.savefig(results / "SENTINEL_all.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    maximum = max(sizes) if sizes else 1
    figure = plt.figure(figsize=(7, 7))
    for group in GROUPS:
        values = [
            size / maximum for size, rig in zip(sizes, rigs)
            if rig["parent_company_type"] == group
        ]
        plt.hist(values, bins=np.linspace(0, 1, 10), label=group,
                 color=COLORS[group], alpha=0.5)
    plt.legend()
    plt.xlabel("Portion of largest valid-month coverage")
    plt.ylabel("Number of oil rigs")
    figure.savefig(results / "area_count_histogram.png", dpi=300)
    plt.close(figure)

    if order:
        if len(order) >= 9:
            positions = np.linspace(0, len(order) - 1, 9, dtype=int)
            selected_indexes = [order[position] for position in positions]
        else:
            selected_indexes = order
        figure, axes = plt.subplots(1, len(selected_indexes), figsize=(12, 3), squeeze=False)
        for axis, index in zip(axes.flat, selected_indexes):
            axis.plot(raw[index], linewidth=0.8, color=COLORS[rigs[index]["parent_company_type"]])
            axis.set_title(rigs[index]["name"], fontsize=8)
            axis.set_xticks([])
        figure.suptitle("Examples of monthly facility NO₂ estimates")
        figure.tight_layout()
        figure.savefig(results / "SENTINEL_selected.png", dpi=300)
        plt.close(figure)


def run(dataset: str) -> None:
    prepare_directories(dataset)
    clean_path = clean_platform_path(dataset)
    if not clean_path.is_file():
        raise FileNotFoundError(f"Run 1_clean_NSTA_data.py first; missing {clean_path}")
    rigs = bounded_rigs(load_json(clean_path))
    rigs = attach_firms(rigs, firms_observations())
    save_json(firms_platform_path(dataset), rigs)
    results = results_directory(dataset)

    plot_many_firms(results / "FIRMS" / "FIRMS_all.png", rigs)
    grouped_rigs = {
        group: [rig for rig in rigs if rig["parent_company_type"] == group]
        for group in GROUPS
    }
    for group in GROUPS:
        plot_many_firms(results / "FIRMS" / f"FIRMS_{group.lower()}.png", grouped_rigs[group])
    plot_combined_firms(
        results / "FIRMS" / "FIRMS_combined_time_series.png",
        [grouped_rigs["PLC"], grouped_rigs["SOE"], grouped_rigs["LTD"]],
    )

    sentinel = stream_monthly_no2(dataset, rigs)
    raw = sentinel["monthly_mean"]
    months = sentinel["months"].astype(str).tolist()
    normalized = normalize_by_neighbours(raw, rigs)
    plot_emission_summaries(results, months, raw, normalized, rigs)
    counts, brightness = firms_monthly(rigs, months)

    for rig in rigs:
        safe_name = "".join(character if character.isalnum() else "_" for character in rig["name"])
        save_json(results / "rigs" / safe_name / "rig_data.json", rig)
    write_tables(results, months, raw, normalized, counts, brightness, rigs)
    plot_facility_areas(results, raw, rigs)
    print(
        f"{dataset}: evaluated {len(rigs)} facilities through "
        f"{sentinel['valid_dates'][-1]}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    sns.set_theme(style="whitegrid")
    for dataset in parse_datasets(argv, __doc__ or ""):
        run(dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
