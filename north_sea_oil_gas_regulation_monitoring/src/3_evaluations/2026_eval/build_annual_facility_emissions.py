"""Build annual facility NO2 tables and year-on-year change metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .eval_paths import (
        FACILITY_MANIFEST,
        FACILITY_MONTHLY,
        MAPPING,
        RESULTS_DIRECTORY,
        SOURCE_MANIFEST,
    )
except ImportError:
    from eval_paths import (  # type: ignore[no-redef]
        FACILITY_MANIFEST,
        FACILITY_MONTHLY,
        MAPPING,
        RESULTS_DIRECTORY,
        SOURCE_MANIFEST,
    )

STAGE_DIRECTORY = Path(__file__).resolve().parents[1]
if str(STAGE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STAGE_DIRECTORY))

from evaluation_common import validate_stage_inputs  # noqa: E402


REQUIRED_COLUMNS = [
    "facility_row_index",
    "feature_id",
    "Match",
    "facility_name",
    "analysis_longitude",
    "analysis_latitude",
    "coordinate_source",
    "analysis_status",
    "area_pixel_count",
    "year",
    "month",
    "mean_no2_mol_m2",
    "valid_pixel_observations",
    "valid_days",
    "available_raster_days",
    "coverage_pct",
]

ANNUAL_COLUMNS = [
    "facility_row_index",
    "feature_id",
    "Match",
    "facility_name",
    "analysis_longitude",
    "analysis_latitude",
    "coordinate_source",
    "analysis_status",
    "area_pixel_count",
    "year",
    "mean_no2_mol_m2",
    "valid_pixel_observations",
    "valid_days",
    "available_raster_days",
    "coverage_pct",
    "previous_year",
    "comparison_months",
    "comparison_previous_mean_no2_mol_m2",
    "comparison_current_mean_no2_mol_m2",
    "percentage_change_from_previous_year",
    "change_direction",
    "comparison_status",
]

SUMMARY_COLUMNS = [
    "year",
    "previous_year",
    "comparison_months",
    "facility_count",
    "matched_facility_count",
    "previous_overall_mean_no2_mol_m2",
    "current_overall_mean_no2_mol_m2",
    "percentage_change_from_previous_year",
    "change_direction",
    "comparison_status",
]

NUMERIC_COLUMNS = [
    "facility_row_index",
    "analysis_longitude",
    "analysis_latitude",
    "area_pixel_count",
    "year",
    "month",
    "mean_no2_mol_m2",
    "valid_pixel_observations",
    "valid_days",
    "available_raster_days",
    "coverage_pct",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=MAPPING)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--monthly", type=Path, default=FACILITY_MONTHLY)
    parser.add_argument("--monthly-manifest", type=Path, default=FACILITY_MANIFEST)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIRECTORY / "yearly_facility_emissions",
    )
    return parser.parse_args(argv)


def load_monthly(path: Path) -> pd.DataFrame:
    monthly = pd.read_csv(path, dtype={"feature_id": str, "Match": str})
    missing = [column for column in REQUIRED_COLUMNS if column not in monthly.columns]
    if missing:
        raise ValueError(f"Monthly facility CSV is missing columns: {', '.join(missing)}")
    monthly = monthly[REQUIRED_COLUMNS].copy()
    if monthly.empty:
        raise ValueError("Monthly facility CSV contains no records")
    if monthly["feature_id"].fillna("").str.strip().eq("").any():
        raise ValueError("Monthly facility CSV contains a blank feature_id")
    for column in NUMERIC_COLUMNS:
        monthly[column] = pd.to_numeric(monthly[column], errors="coerce")
    if monthly[["year", "month"]].isna().any().any():
        raise ValueError("Monthly facility CSV contains an invalid year or month")
    monthly["year"] = monthly["year"].astype(int)
    monthly["month"] = monthly["month"].astype(int)
    if not monthly["month"].between(1, 12).all():
        raise ValueError("Monthly facility CSV contains a month outside 1..12")
    if monthly.duplicated(["feature_id", "year", "month"]).any():
        raise ValueError("Monthly facility CSV contains duplicate facility/month rows")
    invalid_means = (
        monthly["valid_pixel_observations"].fillna(0).gt(0)
        & monthly["mean_no2_mol_m2"].isna()
    )
    if invalid_means.any():
        raise ValueError("Monthly rows with observations must contain a mean NO2 value")
    return monthly.sort_values(["year", "feature_id", "month"], kind="mergesort")


def weighted_mean(group: pd.DataFrame) -> tuple[float, int]:
    observations = group["valid_pixel_observations"].fillna(0).clip(lower=0)
    total = int(observations.sum())
    if total == 0:
        return np.nan, 0
    value = float((group["mean_no2_mol_m2"].fillna(0) * observations).sum() / total)
    return value, total


def aggregate_annual(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (_, year), group in monthly.groupby(["feature_id", "year"], sort=True):
        group = group.sort_values("month", kind="mergesort")
        latest = group.iloc[-1]
        mean, observations = weighted_mean(group)
        valid_days = int(group["valid_days"].fillna(0).sum())
        available_days = int(group["available_raster_days"].fillna(0).sum())
        rows.append(
            {
                "facility_row_index": int(latest["facility_row_index"]),
                "feature_id": str(latest["feature_id"]),
                "Match": str(latest["Match"]),
                "facility_name": str(latest["facility_name"]),
                "analysis_longitude": latest["analysis_longitude"],
                "analysis_latitude": latest["analysis_latitude"],
                "coordinate_source": str(latest["coordinate_source"]),
                "analysis_status": (
                    "available" if observations else str(latest["analysis_status"])
                ),
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
    return pd.DataFrame.from_records(rows)


def comparison_means(
    monthly: pd.DataFrame, year: int, months: list[int]
) -> dict[str, float]:
    selected = monthly[
        monthly["year"].eq(year) & monthly["month"].isin(months)
    ]
    means: dict[str, float] = {}
    for feature_id, group in selected.groupby("feature_id", sort=False):
        mean, observations = weighted_mean(group)
        if observations:
            means[str(feature_id)] = mean
    return means


def direction(percentage: float) -> str:
    if not np.isfinite(percentage):
        return "not_comparable"
    if np.isclose(percentage, 0.0, rtol=0.0, atol=1e-12):
        return "no_change"
    return "increase" if percentage > 0 else "decrease"


def add_comparisons(
    monthly: pd.DataFrame, annual: pd.DataFrame
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    years = sorted(int(year) for year in annual["year"].unique())
    year_set = set(years)
    outputs: dict[int, pd.DataFrame] = {}
    summaries: list[dict[str, object]] = []

    for year in years:
        current = annual[annual["year"].eq(year)].copy()
        previous_year = year - 1
        months: list[int] = []
        previous_means: dict[str, float] = {}
        current_means: dict[str, float] = {}
        base_status = "comparable"
        if previous_year not in year_set:
            base_status = "no_previous_year" if year == years[0] else "no_consecutive_previous_year"
        else:
            current_months = set(monthly.loc[monthly["year"].eq(year), "month"].astype(int))
            previous_months = set(
                monthly.loc[monthly["year"].eq(previous_year), "month"].astype(int)
            )
            months = sorted(current_months & previous_months)
            if not months:
                base_status = "no_common_months"
            else:
                previous_means = comparison_means(monthly, previous_year, months)
                current_means = comparison_means(monthly, year, months)

        month_text = ";".join(f"{month:02d}" for month in months)
        comparison_rows: list[dict[str, object]] = []
        matched_previous: list[float] = []
        matched_current: list[float] = []
        for row in current.to_dict(orient="records"):
            feature_id = str(row["feature_id"])
            previous_mean = previous_means.get(feature_id, np.nan)
            current_mean = current_means.get(feature_id, np.nan)
            status = base_status
            percentage = np.nan
            if base_status == "comparable":
                if feature_id not in previous_means:
                    status = "missing_previous_observations"
                elif feature_id not in current_means:
                    status = "missing_current_observations"
                elif np.isclose(previous_mean, 0.0, rtol=0.0, atol=0.0):
                    status = "zero_previous_mean"
                else:
                    percentage = 100.0 * (current_mean - previous_mean) / previous_mean
                    matched_previous.append(previous_mean)
                    matched_current.append(current_mean)
            row.update(
                previous_year=(previous_year if previous_year in year_set else np.nan),
                comparison_months=month_text,
                comparison_previous_mean_no2_mol_m2=previous_mean,
                comparison_current_mean_no2_mol_m2=current_mean,
                percentage_change_from_previous_year=percentage,
                change_direction=direction(percentage),
                comparison_status=status,
            )
            comparison_rows.append(row)

        output = pd.DataFrame.from_records(comparison_rows, columns=ANNUAL_COLUMNS)
        output = output.sort_values(["facility_name", "feature_id"], kind="mergesort")
        outputs[year] = output

        previous_overall = (
            float(np.mean(matched_previous)) if matched_previous else np.nan
        )
        current_overall = float(np.mean(matched_current)) if matched_current else np.nan
        overall_percentage = np.nan
        summary_status = base_status
        if matched_previous:
            if np.isclose(previous_overall, 0.0, rtol=0.0, atol=0.0):
                summary_status = "zero_previous_mean"
            else:
                overall_percentage = (
                    100.0 * (current_overall - previous_overall) / previous_overall
                )
                summary_status = "comparable"
        elif base_status == "comparable":
            summary_status = "no_matched_facilities"
        summaries.append(
            {
                "year": year,
                "previous_year": previous_year if previous_year in year_set else np.nan,
                "comparison_months": month_text,
                "facility_count": len(output),
                "matched_facility_count": len(matched_previous),
                "previous_overall_mean_no2_mol_m2": previous_overall,
                "current_overall_mean_no2_mol_m2": current_overall,
                "percentage_change_from_previous_year": overall_percentage,
                "change_direction": direction(overall_percentage),
                "comparison_status": summary_status,
            }
        )
    return outputs, pd.DataFrame.from_records(summaries, columns=SUMMARY_COLUMNS)


def write_outputs(
    annual_outputs: dict[int, pd.DataFrame], summary: pd.DataFrame, output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for year, frame in sorted(annual_outputs.items()):
        frame.to_csv(output / f"facility_emissions_{year}.csv", index=False, na_rep="")
    summary.to_csv(output / "annual_summary.csv", index=False, na_rep="")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mapping = args.mapping.expanduser().resolve()
    source_manifest = args.source_manifest.expanduser().resolve()
    monthly_path = args.monthly.expanduser().resolve()
    monthly_manifest = args.monthly_manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    validate_stage_inputs(mapping, source_manifest, monthly_path, monthly_manifest)
    monthly = load_monthly(monthly_path)
    annual = aggregate_annual(monthly)
    annual_outputs, summary = add_comparisons(monthly, annual)
    write_outputs(annual_outputs, summary, output)
    print(f"Annual facility files: {len(annual_outputs)}")
    print(f"Annual summary: {output / 'annual_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
