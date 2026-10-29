#!/usr/bin/env python3
"""Create a multi-page PDF with one monthly NO2 figure per facility.

Each page combines the raw monthly Sentinel-5P NO2 estimate with the number
of valid observation days for the same facility and month. The script reads
the two wide CSV exports produced for this project and does not recalculate or
normalise the underlying estimates.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOT_CACHE = PROJECT_ROOT / "PROCESSED" / "SENTINEL" / ".matplotlib-cache"
PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402


MONTH_COLUMN = re.compile(r"^\d{4}-\d{2}$")


def parse_args() -> argparse.Namespace:
    processed = PROJECT_ROOT / "PROCESSED" / "SENTINEL"

    parser = argparse.ArgumentParser(
        description=(
            "Create one facility figure per page from the wide monthly NO2 "
            "estimate and valid-day CSV files."
        )
    )
    parser.add_argument(
        "--estimates",
        type=Path,
        default=processed / "facility_no2_monthly_estimates_wide.csv",
        help="Wide CSV containing monthly mean NO2 estimates.",
    )
    parser.add_argument(
        "--valid-days",
        type=Path,
        default=processed / "facility_no2_monthly_valid_days_wide.csv",
        help="Wide CSV containing monthly valid-day counts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=processed / "facility_no2_monthly_figures.pdf",
        help="Destination multi-page PDF.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Render only the first N facilities (useful for previewing).",
    )
    return parser.parse_args()


def month_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if MONTH_COLUMN.fullmatch(column)]


def load_inputs(
    estimates_path: Path, valid_days_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DatetimeIndex]:
    if not estimates_path.is_file():
        raise FileNotFoundError(f"Estimate CSV does not exist: {estimates_path}")
    if not valid_days_path.is_file():
        raise FileNotFoundError(f"Valid-day CSV does not exist: {valid_days_path}")

    estimates = pd.read_csv(estimates_path, low_memory=False)
    valid_days = pd.read_csv(valid_days_path, low_memory=False)

    estimate_months = month_columns(estimates)
    valid_day_months = month_columns(valid_days)
    if not estimate_months:
        raise ValueError("No YYYY-MM estimate columns were found")
    if estimate_months != valid_day_months:
        raise ValueError("Estimate and valid-day files do not have identical months")
    if estimate_months != sorted(estimate_months):
        raise ValueError("Monthly columns are not in chronological order")

    metadata_columns = [
        column for column in estimates.columns if column not in estimate_months
    ]
    valid_metadata_columns = [
        column for column in valid_days.columns if column not in valid_day_months
    ]
    if metadata_columns != valid_metadata_columns:
        raise ValueError("Estimate and valid-day files have different metadata columns")

    required = {
        "facility_row_index",
        "feature_id",
        "facility_name",
        "analysis_longitude",
        "analysis_latitude",
        "coordinate_source",
        "analysis_status",
        "area_pixel_count",
    }
    missing = sorted(required.difference(metadata_columns))
    if missing:
        raise ValueError(f"Required metadata columns are missing: {', '.join(missing)}")

    if len(estimates) != len(valid_days):
        raise ValueError("Estimate and valid-day files have different row counts")
    pd.testing.assert_frame_equal(
        estimates[metadata_columns],
        valid_days[metadata_columns],
        check_dtype=False,
        check_names=True,
    )

    if estimates["facility_row_index"].duplicated().any():
        raise ValueError("facility_row_index must uniquely identify each row")

    dates = pd.to_datetime(estimate_months, format="%Y-%m")
    estimate_values = estimates[estimate_months].apply(pd.to_numeric, errors="coerce")
    valid_day_values = valid_days[valid_day_months].apply(
        pd.to_numeric, errors="raise"
    )
    if valid_day_values.isna().any().any():
        raise ValueError("Valid-day data contains blank or non-numeric cells")
    if ((valid_day_values < 0) | (valid_day_values > 31)).any().any():
        raise ValueError("Valid-day counts must be between 0 and 31")

    estimates.loc[:, estimate_months] = estimate_values
    valid_days.loc[:, valid_day_months] = valid_day_values
    return estimates, valid_days, estimate_months, dates


def text_value(row: pd.Series, column: str, fallback: str = "n/a") -> str:
    value = row.get(column)
    if pd.isna(value) or str(value).strip() == "":
        return fallback
    return str(value).strip()


def coordinate_text(row: pd.Series) -> str:
    longitude = pd.to_numeric(row.get("analysis_longitude"), errors="coerce")
    latitude = pd.to_numeric(row.get("analysis_latitude"), errors="coerce")
    if not np.isfinite(longitude) or not np.isfinite(latitude):
        return "Coordinates unavailable"
    return f"{latitude:.5f}, {longitude:.5f}"


def render_facility_page(
    estimate_row: pd.Series,
    valid_day_row: pd.Series,
    months: list[str],
    dates: pd.DatetimeIndex,
    page_number: int,
    page_count: int,
) -> plt.Figure:
    estimates = pd.to_numeric(estimate_row[months], errors="coerce").to_numpy(
        dtype=float
    )
    valid_days = pd.to_numeric(valid_day_row[months], errors="raise").to_numpy(
        dtype=float
    )

    feature_id = text_value(estimate_row, "feature_id")
    facility_name = text_value(estimate_row, "facility_name", feature_id)
    facility_index = int(estimate_row["facility_row_index"])
    coordinate_source = text_value(estimate_row, "coordinate_source")
    analysis_status = text_value(estimate_row, "analysis_status")
    area_pixel_count = int(pd.to_numeric(estimate_row["area_pixel_count"]))
    valid_estimate_count = int(np.count_nonzero(np.isfinite(estimates)))

    figure, (estimate_axis, days_axis) = plt.subplots(
        2,
        1,
        figsize=(11.69, 8.27),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.14},
    )
    figure.subplots_adjust(left=0.09, right=0.965, top=0.82, bottom=0.12)
    figure.patch.set_facecolor("white")

    figure.suptitle(
        facility_name,
        x=0.09,
        y=0.955,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#172033",
    )
    figure.text(
        0.09,
        0.91,
        f"Feature ID: {feature_id}   |   Facility row: {facility_index}   |   "
        f"Coordinates: {coordinate_text(estimate_row)} ({coordinate_source})",
        ha="left",
        fontsize=9.5,
        color="#465268",
    )
    figure.text(
        0.09,
        0.875,
        f"Analysis status: {analysis_status}   |   Assigned pixels: "
        f"{area_pixel_count:,}   |   Months with estimates: "
        f"{valid_estimate_count}/{len(months)}",
        ha="left",
        fontsize=9.5,
        color="#465268",
    )

    estimate_axis.plot(
        dates,
        estimates,
        color="#176B87",
        linewidth=1.8,
        marker="o",
        markersize=3.2,
        markeredgewidth=0,
    )
    estimate_axis.set_title(
        "Monthly raw Sentinel-5P NO$_2$ estimate",
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=9,
    )
    estimate_axis.set_ylabel("Mean NO$_2$ (mol/m$^2$)")
    estimate_axis.grid(axis="both", color="#D7DDE5", linewidth=0.7, alpha=0.75)
    estimate_axis.spines[["top", "right"]].set_visible(False)
    estimate_axis.tick_params(axis="both", labelsize=9, colors="#313A4A")
    estimate_axis.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    estimate_axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    if valid_estimate_count == 0:
        estimate_axis.text(
            0.5,
            0.5,
            "No valid monthly NO$_2$ estimates",
            transform=estimate_axis.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="#6B7280",
        )
        estimate_axis.set_ylim(0, 1)

    days_axis.bar(
        dates,
        valid_days,
        width=24,
        color="#E09F3E",
        edgecolor="#B7791F",
        linewidth=0.35,
    )
    days_axis.set_title(
        "Valid observation days per month",
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    days_axis.set_ylabel("Days")
    days_axis.set_xlabel("Year")
    days_axis.set_ylim(0, max(31, float(np.nanmax(valid_days)) + 2))
    days_axis.set_yticks([0, 10, 20, 30])
    days_axis.grid(axis="y", color="#D7DDE5", linewidth=0.7, alpha=0.75)
    days_axis.spines[["top", "right"]].set_visible(False)
    days_axis.tick_params(axis="both", labelsize=9, colors="#313A4A")
    days_axis.xaxis.set_major_locator(mdates.YearLocator())
    days_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    days_axis.set_xlim(
        dates[0] - pd.Timedelta(days=20), dates[-1] + pd.Timedelta(days=20)
    )

    figure.text(
        0.09,
        0.045,
        "Blank estimate months are shown as gaps. A valid day has at least one "
        "finite observation in the facility's assigned pixels.",
        ha="left",
        fontsize=8.5,
        color="#667085",
    )
    figure.text(
        0.965,
        0.045,
        f"Facility {page_number} of {page_count}",
        ha="right",
        fontsize=8.5,
        color="#667085",
    )
    return figure


def build_pdf(
    estimates: pd.DataFrame,
    valid_days: pd.DataFrame,
    months: list[str],
    dates: pd.DatetimeIndex,
    output_path: Path,
    limit: int | None,
) -> int:
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be greater than zero")
        estimates = estimates.iloc[:limit].copy()
        valid_days = valid_days.iloc[:limit].copy()

    page_count = len(estimates)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.pdf")

    try:
        with PdfPages(
            temporary_path,
            metadata={
                "Title": "Monthly facility NO2 estimates and valid-day coverage",
                "Subject": "One facility per page",
                "Author": "FRIL Oil and Gas Regulation Monitoring project",
                "Creator": "plot_facility_monthly_figures.py",
                "Keywords": "Sentinel-5P, NO2, facilities, monthly estimates",
            },
        ) as pdf:
            for page_index, (_, estimate_row) in enumerate(
                estimates.iterrows(), start=1
            ):
                valid_day_row = valid_days.iloc[page_index - 1]
                figure = render_facility_page(
                    estimate_row,
                    valid_day_row,
                    months,
                    dates,
                    page_index,
                    page_count,
                )
                pdf.savefig(figure, facecolor="white")
                plt.close(figure)

                if page_index == 1 or page_index % 25 == 0 or page_index == page_count:
                    print(f"Rendered facility {page_index} of {page_count}", flush=True)

        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return page_count


def main() -> int:
    args = parse_args()
    estimates, valid_days, months, dates = load_inputs(
        args.estimates.resolve(), args.valid_days.resolve()
    )
    page_count = build_pdf(
        estimates,
        valid_days,
        months,
        dates,
        args.output.resolve(),
        args.limit,
    )
    print(f"Created {args.output.resolve()} with {page_count} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
