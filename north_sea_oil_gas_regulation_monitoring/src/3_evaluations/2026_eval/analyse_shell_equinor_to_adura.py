import argparse
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "eo-fril-matplotlib-cache"),
)

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from .eval_paths import FACILITY_MONTHLY, MAPPING, RESULTS_DIRECTORY, SOURCE_MANIFEST
    from .suite_common import comparison_prefixes, export_rows, yearly_rows
except ImportError:
    from eval_paths import FACILITY_MONTHLY, MAPPING, RESULTS_DIRECTORY, SOURCE_MANIFEST
    from suite_common import comparison_prefixes, export_rows, yearly_rows

# ============================================================
# Configuration
# ============================================================

parser = argparse.ArgumentParser(
    description="Plot the Shell/Equinor to Adura Energy NO2 comparison."
)
parser.add_argument("--mapping", type=Path, default=MAPPING)
parser.add_argument("--monthly", type=Path, default=FACILITY_MONTHLY)
parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
parser.add_argument(
    "--output", type=Path,
    default=RESULTS_DIRECTORY / "shell_equinor_to_adura",
)
args = parser.parse_args()

CASE_DIR = args.output
DATA_DIR = CASE_DIR / "data"
FIGURES_DIR = CASE_DIR / "figures"
COORDINATE_FILE = args.mapping

FOLDERS = [
    "shell_changed_to_adura",
    "shell_kept_shell",
    "equinor_changed_to_adura",
]
COLORS = {
    "shell_changed_to_adura": "tab:orange",
    "shell_kept_shell": "tab:blue",
    "equinor_changed_to_adura": "tab:green",
}

NORTH_SOUTH_BOUNDARY = 56.0
SPLIT_YEAR = 2025

REGIONS = ["Northern North Sea", "Southern North Sea"]
SOLID_INFRASTRUCTURE_TYPES = {"FPSO", "PLATFORM"}


# ============================================================
# Load facility coordinates using feature_id
# ============================================================

previous_label, current_label = comparison_prefixes(args.source_manifest)
facility_locations = {}
facility_groups = {}

with open(COORDINATE_FILE, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for row in reader:
        feature_id = row["feature_id"].strip().lower()

        lat_str = row.get(f"{current_label}_y") or row.get(f"{previous_label}_y")
        lon_str = row.get(f"{current_label}_x") or row.get(f"{previous_label}_x")

        try:
            latitude = float(lat_str)
            longitude = float(lon_str)
        except (ValueError, TypeError):
            continue

        if not (-90 <= latitude <= 90):
            continue

        region = (
            "Northern North Sea"
            if latitude >= NORTH_SOUTH_BOUNDARY
            else "Southern North Sea"
        )

        infrastructure_type = (
            row.get(f"{current_label}_type_of_infrastructure")
            or row.get(f"{previous_label}_type_of_infrastructure")
            or ""
        ).strip().upper()

        old_owner = (row.get(f"{previous_label}_reporting_group_name") or "").strip().upper()
        new_owner = (row.get(f"{current_label}_reporting_group_name") or "").strip().upper()
        if old_owner == "SHELL PLC" and new_owner == "ADURA ENERGY":
            facility_groups[feature_id] = "shell_changed_to_adura"
        elif new_owner == "SHELL PLC":
            facility_groups[feature_id] = "shell_kept_shell"
        elif old_owner == "EQUINOR ASA" and new_owner == "ADURA ENERGY":
            facility_groups[feature_id] = "equinor_changed_to_adura"

        facility_locations[feature_id] = {
            "latitude": latitude,
            "longitude": longitude,
            "region": region,
            "infrastructure_type": infrastructure_type,
        }


# ============================================================
# Load individual CSV files and organise by region
# ============================================================

individual_csvs = {
    region: {folder: {} for folder in FOLDERS}
    for region in REGIONS
}

yearly_values = {
    region: {folder: {} for folder in FOLDERS}
    for region in REGIONS
}

annual_header, annual_records = yearly_rows(args.monthly)
annual_by_id = {}
for record in annual_records:
    annual_by_id.setdefault(record["feature_id"].strip().lower(), []).append(record)

for feature_id, folder in facility_groups.items():
    rows = annual_by_id.get(feature_id, [])
    if not rows or feature_id not in facility_locations:
        continue
    export_rows(DATA_DIR / folder / f"{feature_id}_yearly_no2.csv", annual_header, rows)
    years_and_means = []
    for row in rows:
        try:
            mean = float(row["mean_no2_mol_m2"])
            year = int(float(row["year"]))
        except (TypeError, ValueError):
            continue
        if np.isfinite(mean):
            years_and_means.append((year, mean))
    if not years_and_means:
        continue
    years_and_means.sort()
    years = np.array([item[0] for item in years_and_means], dtype=int)
    means = np.array([item[1] for item in years_and_means], dtype=float)
    region = facility_locations[feature_id]["region"]
    individual_csvs[region][folder][feature_id] = (years, means)
    for year, mean in years_and_means:
        yearly_values[region][folder].setdefault(year, []).append(mean)


# ============================================================
# Calculate yearly averages from existing CSV mean values
# ============================================================

yearly_means = {
    region: {
        folder: {
            year: np.mean(values)
            for year, values in sorted(
                yearly_values[region][folder].items()
            )
        }
        for folder in FOLDERS
    }
    for region in REGIONS
}


# ============================================================
# Helper: plot formatting
# ============================================================

def format_axis(ax, title, ylabel="Mean NO₂ (mol/m²)"):
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()


def facility_line_style(feature_id):
    infrastructure_type = facility_locations[feature_id]["infrastructure_type"]
    return "-" if infrastructure_type in SOLID_INFRASTRUCTURE_TYPES else "--"


def add_infrastructure_style_key(ax):
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
    ax.legend(handles, labels)


# ============================================================
# Figure 1: Yearly mean NO2 for North and South
# ============================================================

fig1, axes = plt.subplots(
    2, 1, figsize=(11, 9), sharex=True
)

for ax, region in zip(axes, REGIONS):

    for folder in FOLDERS:
        means = yearly_means[region][folder]

        if not means:
            continue

        years = sorted(means)
        values = [means[y] for y in years]

        ax.plot(
            years,
            values,
            marker="o",
            linewidth=2,
            color=COLORS[folder],
            label=folder
        )

    format_axis(ax, region)

fig1.suptitle("Yearly Mean NO₂ by North Sea Region")
fig1.tight_layout()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / "regional_group_means.png")


# ============================================================
# Figure 2: Year-on-year percentage change
# ============================================================

fig2, axes = plt.subplots(
    2, 1, figsize=(11, 9), sharex=True
)

for ax, region in zip(axes, REGIONS):

    for folder in FOLDERS:
        means = yearly_means[region][folder]

        years = np.array(sorted(means))
        values = np.array(
            [means[y] for y in years],
            dtype=float
        )

        if len(years) < 2:
            continue

        # Only compare consecutive calendar years
        consecutive = np.diff(years) == 1
        valid = consecutive & (values[:-1] != 0)

        change_years = years[1:][valid]

        pct_change = (
            (values[1:][valid] - values[:-1][valid])
            / values[:-1][valid] * 100
        )

        ax.plot(
            change_years,
            pct_change,
            marker="o",
            linewidth=2,
            color=COLORS[folder],
            label=folder
        )

    ax.axhline(
        0, color="black", linestyle="--", linewidth=1
    )

    format_axis(
        ax,
        region,
        "Change from Previous Year (%)"
    )

fig2.suptitle("Year-on-Year NO₂ Change by North Sea Region")
fig2.tight_layout()
plt.savefig(FIGURES_DIR / "regional_year_on_year_change.png")


# ============================================================
# Figure 3: Combined history with post-2025 split
# Only folders 1 and 2
# ============================================================

fig3, axes = plt.subplots(
    2, 1, figsize=(11, 9), sharex=True
)

for ax, region in zip(axes, REGIONS):

    combined_values = {}

    # Combine individual CSV means from folders 1 and 2
    for folder in FOLDERS[:2]:
        for year, values in yearly_values[region][folder].items():
            if year <= SPLIT_YEAR:
                combined_values.setdefault(
                    year, []
                ).extend(values)

    combined_means = {
        year: np.mean(values)
        for year, values in sorted(combined_values.items())
    }

    if combined_means:
        shared_years = sorted(combined_means)
        shared_values = [
            combined_means[y] for y in shared_years
        ]

        ax.plot(
            shared_years,
            shared_values,
            color="black",
            marker="o",
            linewidth=2.5,
            label="Combined"
        )

    # Branch into folder-specific values after 2025
    for folder in FOLDERS[:2]:
        means = yearly_means[region][folder]

        split_years = sorted(
            y for y in means if y > SPLIT_YEAR
        )

        if not split_years:
            continue

        if SPLIT_YEAR in combined_means:
            plot_years = [SPLIT_YEAR] + split_years
            plot_values = [
                combined_means[SPLIT_YEAR]
            ] + [means[y] for y in split_years]
        else:
            plot_years = split_years
            plot_values = [
                means[y] for y in split_years
            ]

        ax.plot(
            plot_years,
            plot_values,
            marker="o",
            linewidth=2,
            color=COLORS[folder],
            label=folder
        )

    ax.axvline(
        SPLIT_YEAR,
        color="gray",
        linestyle="--",
        alpha=0.6
    )

    format_axis(ax, region)

fig3.suptitle(
    "Combined Historical NO₂ with Post-2025 Split"
)
fig3.tight_layout()
plt.savefig(FIGURES_DIR / "pre_post_2025_split.png")


# ============================================================
# Figure 4: Individual CSV lines by North/South region
# ============================================================

fig4, axes = plt.subplots(
    2, 1, figsize=(12, 10), sharex=True
)

for ax, region in zip(axes, REGIONS):

    # Individual facility lines
    for folder in FOLDERS:
        for feature_id, (years, means) in (
            individual_csvs[region][folder].items()
        ):
            ax.plot(
                years,
                means,
                color=COLORS[folder],
                linewidth=1.1,
                linestyle=facility_line_style(feature_id),
                alpha=0.35
            )

    # Thick folder-average lines
    for folder in FOLDERS:
        means = yearly_means[region][folder]

        if not means:
            continue

        years = sorted(means)
        values = [means[y] for y in years]

        ax.plot(
            years,
            values,
            color=COLORS[folder],
            linewidth=3,
            label=f"{folder} Average"
        )

    format_axis(ax, region)
    add_infrastructure_style_key(ax)

fig4.suptitle(
    "Individual Facility NO₂ Measurements by North Sea Region"
)
fig4.tight_layout()
plt.savefig(FIGURES_DIR / "facility_no2_timeseries.png")


# ============================================================
# Print number of facilities per region and folder
# ============================================================

print("\nFacility counts by North Sea region")
print("-" * 55)

for region in REGIONS:
    print(f"\n{region}")

    for folder in FOLDERS:
        count = len(individual_csvs[region][folder])
        print(f"  {folder}: {count} facilities")


# ============================================================
# Print yearly averages by region
# ============================================================

for region in REGIONS:

    print(f"\n{region} - Yearly NO₂ means")
    print("-" * 75)

    all_years = sorted(
        set().union(
            *[
                set(yearly_means[region][folder])
                for folder in FOLDERS
            ]
        )
    )

    print(
        f"{'Year':<8}"
        + "".join(f"{folder:>20}" for folder in FOLDERS)
    )

    for year in all_years:
        line = f"{year:<8}"

        for folder in FOLDERS:
            value = yearly_means[region][folder].get(
                year, np.nan
            )

            line += f"{value:>20.6e}"

        print(line)


plt.close("all")
