from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "eo-fril-matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .eval_paths import FACILITY_MONTHLY, MAPPING, RESULTS_DIRECTORY, SOURCE_MANIFEST
    from .suite_common import comparison_prefixes
except ImportError:
    from eval_paths import FACILITY_MONTHLY, MAPPING, RESULTS_DIRECTORY, SOURCE_MANIFEST
    from suite_common import comparison_prefixes


INCLUDED_TYPES = {"PLATFORM", "FPSO"}
NORTH_SOUTH_BOUNDARY = 56.0
REGIONS = ("Northern North Sea", "Southern North Sea")
REGION_DIRECTORIES = {
    "Northern North Sea": "northern_north_sea",
    "Southern North Sea": "southern_north_sea",
}
MONTH_LABELS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
DEFAULT_THROUGH = "2026-09"
DEFAULT_MIN_VALID_DAYS = 5
DEFAULT_MIN_COVERAGE_PCT = 20.0
DEFAULT_MIN_BASELINE_YEARS = 4
DEFAULT_BASELINE_EPSILON = 1e-12
DEFAULT_TOP_SUMMARY = 25

SCORE_COLUMNS = [
    "rank",
    "region",
    "feature_id",
    "facility_name",
    "infrastructure_type",
    "owner_2025",
    "owner_2026",
    "longitude",
    "latitude",
    "year",
    "month",
    "period",
    "observed_mean_no2_mol_m2",
    "seasonal_baseline_median_mol_m2",
    "signed_percentage_deviation",
    "absolute_percentage_deviation",
    "direction",
    "valid_days",
    "coverage_pct",
    "valid_pixel_observations",
    "baseline_year_count",
]

UNSCORED_COLUMNS = [
    "region",
    "feature_id",
    "facility_name",
    "infrastructure_type",
    "year",
    "month",
    "period",
    "mean_no2_mol_m2",
    "valid_days",
    "coverage_pct",
    "reason",
    "baseline_year_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build monthly PLATFORM/FPSO NO2 anomaly plots and rankings."
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=MAPPING,
    )
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument(
        "--monthly",
        type=Path,
        default=FACILITY_MONTHLY,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIRECTORY / "anomaly_detection",
    )
    parser.add_argument("--through", default=DEFAULT_THROUGH, metavar="YYYY-MM")
    parser.add_argument("--min-valid-days", type=int, default=DEFAULT_MIN_VALID_DAYS)
    parser.add_argument(
        "--min-coverage-pct", type=float, default=DEFAULT_MIN_COVERAGE_PCT
    )
    parser.add_argument(
        "--min-baseline-years", type=int, default=DEFAULT_MIN_BASELINE_YEARS
    )
    parser.add_argument(
        "--baseline-epsilon", type=float, default=DEFAULT_BASELINE_EPSILON
    )
    parser.add_argument("--top-summary", type=int, default=DEFAULT_TOP_SUMMARY)
    return parser.parse_args()


def parse_period(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d{4})-(\d{2})", value.strip())
    if not match:
        raise ValueError(f"Expected YYYY-MM, received {value!r}")
    year, month = int(match.group(1)), int(match.group(2))
    if not 1 <= month <= 12:
        raise ValueError(f"Month must be between 01 and 12, received {value!r}")
    return year, month


def first_nonempty(frame: pd.DataFrame, preferred: str, fallback: str) -> pd.Series:
    preferred_values = frame[preferred].fillna("").astype(str).str.strip()
    fallback_values = frame[fallback].fillna("").astype(str).str.strip()
    return preferred_values.where(preferred_values.ne(""), fallback_values)


def classify_region(latitude: float) -> str:
    return "Northern North Sea" if latitude >= NORTH_SOUTH_BOUNDARY else "Southern North Sea"


def is_included_type(infrastructure_type: str) -> bool:
    return str(infrastructure_type).strip().upper() in INCLUDED_TYPES


def build_facility_metadata(
    mapping: pd.DataFrame, previous_label: str, current_label: str
) -> pd.DataFrame:
    mapping = mapping.copy()
    mapping["_source_order"] = np.arange(len(mapping))
    mapping["_has_current_record"] = (
        mapping[f"{current_label}_type_of_infrastructure"].fillna("").astype(str).str.strip().ne("")
    )
    mapping["_is_both_years"] = mapping["Match"].fillna("").astype(str).str.strip().eq("Both")
    mapping["_normalised_feature_id"] = (
        mapping["feature_id"].fillna("").astype(str).str.strip().str.lower()
    )
    # One mapping ID is attached to both a matched current record and a separate
    # 2025-only legacy point. Prefer the matched/current row so each satellite
    # feature remains one facility, as intended by the feature ID.
    mapping = (
        mapping.sort_values(
            ["_normalised_feature_id", "_is_both_years", "_has_current_record", "_source_order"],
            ascending=[True, False, False, True],
            kind="mergesort",
        )
        .drop_duplicates("_normalised_feature_id", keep="first")
        .copy()
    )
    infrastructure_type = first_nonempty(
        mapping, f"{current_label}_type_of_infrastructure", f"{previous_label}_type_of_infrastructure"
    ).str.upper()
    longitude = pd.to_numeric(first_nonempty(mapping, f"{current_label}_x", f"{previous_label}_x"), errors="coerce")
    latitude = pd.to_numeric(first_nonempty(mapping, f"{current_label}_y", f"{previous_label}_y"), errors="coerce")
    metadata = pd.DataFrame(
        {
            "feature_id": mapping["_normalised_feature_id"],
            "facility_name": first_nonempty(
                mapping, f"{current_label}_infrastructure_name", f"{previous_label}_infrastructure_name"
            ),
            "infrastructure_type": infrastructure_type,
            "owner_2025": mapping[f"{previous_label}_reporting_group_name"].fillna("").astype(str).str.strip(),
            "owner_2026": mapping[f"{current_label}_reporting_group_name"].fillna("").astype(str).str.strip(),
            "longitude": longitude,
            "latitude": latitude,
        }
    )
    metadata = metadata[metadata["infrastructure_type"].isin(INCLUDED_TYPES)].copy()
    if metadata["feature_id"].eq("").any():
        raise ValueError("Eligible mapping rows contain blank feature IDs")
    if metadata["feature_id"].duplicated().any():
        duplicates = metadata.loc[metadata["feature_id"].duplicated(), "feature_id"].tolist()
        raise ValueError(f"Eligible mapping rows contain duplicate feature IDs: {duplicates[:5]}")
    if metadata[["longitude", "latitude"]].isna().any(axis=None):
        invalid = metadata.loc[
            metadata[["longitude", "latitude"]].isna().any(axis=1), "feature_id"
        ].tolist()
        raise ValueError(f"Eligible facilities contain invalid coordinates: {invalid[:5]}")
    metadata["region"] = metadata["latitude"].map(classify_region)
    return metadata.sort_values(["region", "facility_name", "feature_id"]).reset_index(drop=True)


def prepare_monthly_data(
    monthly: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    through: tuple[int, int],
    min_valid_days: int,
    min_coverage_pct: float,
) -> pd.DataFrame:
    data = monthly.copy()
    data["feature_id"] = data["feature_id"].fillna("").astype(str).str.strip().str.lower()
    # Mapping metadata is authoritative for the displayed current/fallback name.
    # The monthly extract also carries a source name, including one legacy alias.
    data = data.drop(columns=["facility_name"], errors="ignore")
    numeric_columns = [
        "year",
        "month",
        "mean_no2_mol_m2",
        "valid_days",
        "coverage_pct",
        "valid_pixel_observations",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.merge(metadata, on="feature_id", how="inner", validate="many_to_one")
    cutoff_year, cutoff_month = through
    at_or_before_cutoff = (data["year"] < cutoff_year) | (
        (data["year"] == cutoff_year) & (data["month"] <= cutoff_month)
    )
    qualified = data[
        data["analysis_status"].eq("available")
        & data["mean_no2_mol_m2"].notna()
        & data["valid_days"].ge(min_valid_days)
        & data["coverage_pct"].ge(min_coverage_pct)
        & at_or_before_cutoff
    ].copy()
    qualified["year"] = qualified["year"].astype(int)
    qualified["month"] = qualified["month"].astype(int)
    qualified["valid_days"] = qualified["valid_days"].astype(int)
    qualified["valid_pixel_observations"] = qualified["valid_pixel_observations"].astype(int)
    qualified["period"] = qualified.apply(
        lambda row: f"{int(row['year']):04d}-{int(row['month']):02d}", axis=1
    )
    if qualified.duplicated(["feature_id", "year", "month"]).any():
        raise ValueError("Qualified monthly data contain duplicate facility/year/month rows")
    return qualified.sort_values(["feature_id", "year", "month"]).reset_index(drop=True)


def calculate_scores(
    qualified: pd.DataFrame,
    *,
    min_baseline_years: int,
    baseline_epsilon: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores: list[dict[str, object]] = []
    unscored: list[dict[str, object]] = []

    for (_, _), group in qualified.groupby(["feature_id", "month"], sort=True):
        for index, observation in group.iterrows():
            baseline_values = group.loc[
                group.index != index, "mean_no2_mol_m2"
            ].to_numpy(dtype=float)
            baseline_count = len(baseline_values)
            base_record = {
                "region": observation["region"],
                "feature_id": observation["feature_id"],
                "facility_name": observation["facility_name"],
                "infrastructure_type": observation["infrastructure_type"],
                "year": int(observation["year"]),
                "month": int(observation["month"]),
                "period": observation["period"],
                "mean_no2_mol_m2": float(observation["mean_no2_mol_m2"]),
                "valid_days": int(observation["valid_days"]),
                "coverage_pct": float(observation["coverage_pct"]),
                "baseline_year_count": baseline_count,
            }
            if baseline_count < min_baseline_years:
                unscored.append({**base_record, "reason": "fewer_than_minimum_baseline_years"})
                continue

            baseline = float(np.median(baseline_values))
            if not math.isfinite(baseline) or abs(baseline) < baseline_epsilon:
                unscored.append({**base_record, "reason": "near_zero_or_invalid_baseline"})
                continue

            observed = float(observation["mean_no2_mol_m2"])
            signed_percentage = 100.0 * (observed - baseline) / abs(baseline)
            if signed_percentage > 0:
                direction = "increase"
            elif signed_percentage < 0:
                direction = "decrease"
            else:
                direction = "unchanged"

            scores.append(
                {
                    "region": observation["region"],
                    "feature_id": observation["feature_id"],
                    "facility_name": observation["facility_name"],
                    "infrastructure_type": observation["infrastructure_type"],
                    "owner_2025": observation["owner_2025"],
                    "owner_2026": observation["owner_2026"],
                    "longitude": float(observation["longitude"]),
                    "latitude": float(observation["latitude"]),
                    "year": int(observation["year"]),
                    "month": int(observation["month"]),
                    "period": observation["period"],
                    "observed_mean_no2_mol_m2": observed,
                    "seasonal_baseline_median_mol_m2": baseline,
                    "signed_percentage_deviation": signed_percentage,
                    "absolute_percentage_deviation": abs(signed_percentage),
                    "direction": direction,
                    "valid_days": int(observation["valid_days"]),
                    "coverage_pct": float(observation["coverage_pct"]),
                    "valid_pixel_observations": int(observation["valid_pixel_observations"]),
                    "baseline_year_count": baseline_count,
                }
            )

    score_frame = pd.DataFrame.from_records(scores)
    if score_frame.empty:
        score_frame = pd.DataFrame(columns=[column for column in SCORE_COLUMNS if column != "rank"])
    unscored_frame = pd.DataFrame.from_records(unscored, columns=UNSCORED_COLUMNS)
    return score_frame, unscored_frame


def rank_scores(scores: pd.DataFrame) -> pd.DataFrame:
    ranked = scores.sort_values(
        [
            "absolute_percentage_deviation",
            "signed_percentage_deviation",
            "facility_name",
            "feature_id",
            "year",
            "month",
        ],
        ascending=[False, False, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1, dtype=int))
    return ranked.reindex(columns=SCORE_COLUMNS)


def make_exclusions(
    metadata: pd.DataFrame,
    monthly: pd.DataFrame,
    qualified: pd.DataFrame,
) -> pd.DataFrame:
    status = (
        monthly.assign(
            feature_id=monthly["feature_id"].fillna("").astype(str).str.strip().str.lower()
        )
        .sort_values(["feature_id", "year", "month"])
        .drop_duplicates("feature_id")[["feature_id", "analysis_status"]]
    )
    status_map = dict(zip(status["feature_id"], status["analysis_status"], strict=False))
    plotted_ids = set(qualified["feature_id"])
    records: list[dict[str, object]] = []
    for row in metadata.itertuples(index=False):
        if row.feature_id in plotted_ids:
            continue
        analysis_status = status_map.get(row.feature_id, "missing_monthly_record")
        if analysis_status in {"outside_no2_bbox", "no_assigned_pixels"}:
            reason = analysis_status
        elif analysis_status == "missing_monthly_record":
            reason = analysis_status
        else:
            reason = "no_month_meets_quality_and_cutoff_filters"
        records.append(
            {
                "feature_id": row.feature_id,
                "facility_name": row.facility_name,
                "infrastructure_type": row.infrastructure_type,
                "region": row.region,
                "owner_2025": row.owner_2025,
                "owner_2026": row.owner_2026,
                "longitude": row.longitude,
                "latitude": row.latitude,
                "analysis_status": analysis_status,
                "exclusion_reason": reason,
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["exclusion_reason", "region", "facility_name", "feature_id"]
    )


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "unnamed-facility"


def plot_facility(
    facility_data: pd.DataFrame,
    facility_scores: pd.DataFrame,
    output_path: Path,
    *,
    through_label: str,
    min_valid_days: int,
    min_coverage_pct: float,
) -> None:
    first = facility_data.iloc[0]
    years = sorted(facility_data["year"].unique())
    color_map = plt.colormaps.get_cmap("viridis").resampled(len(years))
    figure, ax = plt.subplots(figsize=(13, 7.2))

    for color_index, year in enumerate(years):
        year_data = (
            facility_data[facility_data["year"] == year]
            .set_index("month")
            .reindex(range(1, 13))
        )
        label = str(year)
        if year == 2026:
            label = f"2026 (through {through_label})"
        ax.plot(
            year_data.index,
            year_data["mean_no2_mol_m2"],
            color=color_map(color_index),
            marker="o",
            markersize=4,
            linewidth=1.6 if year != 2026 else 2.5,
            alpha=0.9,
            label=label,
        )

    seasonal_median = facility_data.groupby("month")["mean_no2_mol_m2"].median()
    ax.plot(
        seasonal_median.index,
        seasonal_median.values,
        color="#222222",
        linestyle=":",
        marker="s",
        markersize=4,
        linewidth=2.8,
        label="Seasonal month median",
        zorder=5,
    )

    top_anomalies = facility_scores.nlargest(3, "absolute_percentage_deviation")
    value_min = float(facility_data["mean_no2_mol_m2"].min())
    value_max = float(facility_data["mean_no2_mol_m2"].max())
    high_value_threshold = value_min + 0.82 * (value_max - value_min)
    annotations_by_month: dict[int, int] = {}
    for anomaly in top_anomalies.itertuples(index=False):
        ax.scatter(
            anomaly.month,
            anomaly.observed_mean_no2_mol_m2,
            s=95,
            facecolors="none",
            edgecolors="#D62728",
            linewidths=2,
            zorder=8,
        )
        same_month_index = annotations_by_month.get(anomaly.month, 0)
        annotations_by_month[anomaly.month] = same_month_index + 1
        is_near_top = anomaly.observed_mean_no2_mol_m2 >= high_value_threshold
        vertical_offset = 14 + 36 * same_month_index
        if is_near_top:
            vertical_offset = -vertical_offset
        ax.annotate(
            f"{anomaly.period}\n{anomaly.signed_percentage_deviation:+.1f}%",
            (anomaly.month, anomaly.observed_mean_no2_mol_m2),
            xytext=(7, vertical_offset),
            textcoords="offset points",
            fontsize=8,
            color="#9C1C1C",
            va="top" if is_near_top else "bottom",
            arrowprops={"arrowstyle": "-", "color": "#9C1C1C", "alpha": 0.65},
        )

    ax.set_xlim(0.6, 12.4)
    ax.set_xticks(range(1, 13), MONTH_LABELS)
    ax.set_xlabel("Calendar month")
    ax.set_ylabel("Mean tropospheric NO₂ column density (mol/m²)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(alpha=0.25)
    ax.set_title(
        f"{first['facility_name']} — monthly NO₂ profiles by year\n"
        f"{first['infrastructure_type']} | {first['region']} | {first['feature_id']}"
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    figure.text(
        0.01,
        0.01,
        (
            f"Shown/ranked months require ≥{min_valid_days} valid days and "
            f"≥{min_coverage_pct:g}% coverage. 2026 ends at {through_label}. "
            "Red rings mark this facility's three largest absolute seasonal deviations."
        ),
        fontsize=8.5,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.04, 0.84, 1))
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_top_summary(
    ranked: pd.DataFrame,
    output_path: Path,
    *,
    region: str,
    top_n: int,
) -> None:
    top = ranked.head(top_n)
    lines = [
        f"# {region}: largest monthly NO₂ anomalies",
        "",
        (
            "Ranked by absolute percentage deviation from the leave-one-out median "
            "for the same facility and calendar month. The signed percentage shows "
            "whether the observation increased or decreased."
        ),
        "",
        "| Rank | Facility | Type | Period | Observed | Baseline | Change | Direction | Coverage | Valid days |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            "| {rank} | {name} | {kind} | {period} | {observed:.6e} | "
            "{baseline:.6e} | {change:+.1f}% | {direction} | {coverage:.1f}% | "
            "{days} |".format(
                rank=row.rank,
                name=str(row.facility_name).replace("|", "\\|"),
                kind=row.infrastructure_type,
                period=row.period,
                observed=row.observed_mean_no2_mol_m2,
                baseline=row.seasonal_baseline_median_mol_m2,
                change=row.signed_percentage_deviation,
                direction=row.direction,
                coverage=row.coverage_pct,
                days=row.valid_days,
            )
        )
    lines.extend(
        [
            "",
            "The complete ordered list is in `anomalies_ranked.csv`.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_readme(
    output_dir: Path,
    summary: dict[str, object],
) -> None:
    lines = [
        "# Platform/FPSO monthly NO₂ anomaly analysis",
        "",
        (
            "This analysis overlays the qualified monthly Sentinel-5P NO₂ profiles "
            "for every mapped `PLATFORM` and `FPSO`, then ranks monthly departures "
            "from each facility's usual seasonal pattern."
        ),
        "",
        "## Method",
        "",
        (
            "- Infrastructure type comes from the 2026 mapping, falling back to 2025; "
            "only `PLATFORM` and `FPSO` records are included."
        ),
        (
            f"- Regions use the existing {NORTH_SOUTH_BOUNDARY:g}° latitude boundary: "
            "facilities at or north of the boundary are Northern North Sea."
        ),
        (
            "- Monthly rows must have `analysis_status == available` and a non-missing "
            f"mean, at least {summary['min_valid_days']} valid satellite days, and "
            f"{summary['min_coverage_pct']:g}% coverage."
        ),
        f"- The analysis cutoff is `{summary['through']}`; later 2026 months are excluded.",
        (
            "- For each facility/month/year, the baseline is the median for that same "
            "calendar month in the other years. At least four baseline years are required."
        ),
        (
            f"- Baselines with absolute magnitude below {summary['baseline_epsilon']:.0e} "
            "mol/m² are treated as effectively zero and recorded as unscored."
        ),
        (
            "- Percentage deviation is `100 × (observed − baseline) / abs(baseline)`. "
            "Rankings use the absolute magnitude but retain the signed direction."
        ),
        "- No anomaly threshold is applied: every qualified, scoreable observation is ranked.",
        "",
        "## Outputs",
        "",
        (
            f"- Northern North Sea: {summary['regions']['Northern North Sea']['plots']} "
            "facility plots and "
            f"{summary['regions']['Northern North Sea']['ranked_observations']:,} ranked observations."
        ),
        (
            f"- Southern North Sea: {summary['regions']['Southern North Sea']['plots']} "
            "facility plots and "
            f"{summary['regions']['Southern North Sea']['ranked_observations']:,} ranked observations."
        ),
        (
            f"- `excluded_facilities.csv`: {summary['excluded_facilities']} eligible "
            "facilities that could not be plotted: "
            f"{summary['exclusion_counts'].get('outside_no2_bbox', 0)} outside the NO₂ "
            "bounding box and "
            f"{summary['exclusion_counts'].get('no_assigned_pixels', 0)} without assigned pixels."
        ),
        (
            f"- `unscored_observations.csv`: {summary['unscored_observations']} qualified "
            "observations lacking enough baseline years or a usable baseline."
        ),
        "- `analysis_summary.json`: machine-readable parameters and reconciliation counts.",
        "",
        "Each regional folder contains `anomalies_ranked.csv`, `top_anomalies.md`, and `plots/`.",
        "",
        "## Interpretation limits",
        "",
        (
            "These are descriptive anomalies in raw satellite tropospheric NO₂ columns. "
            "They are not adjusted for regional weather or background pollution and do not "
            "establish a facility emissions change, ownership effect, or statistical significance."
        ),
        (
            "TROPOMI pixels are spatially broad and the repository assigns resampled pixels "
            "to the nearest facility, so nearby facilities can share strongly correlated signals."
        ),
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    through = parse_period(args.through)
    if args.min_valid_days < 1:
        raise ValueError("--min-valid-days must be positive")
    if not 0 <= args.min_coverage_pct <= 100:
        raise ValueError("--min-coverage-pct must be between 0 and 100")
    if args.min_baseline_years < 1:
        raise ValueError("--min-baseline-years must be positive")
    if args.baseline_epsilon <= 0:
        raise ValueError("--baseline-epsilon must be positive")
    if args.top_summary < 1:
        raise ValueError("--top-summary must be positive")

    mapping = pd.read_csv(args.mapping, dtype=str, keep_default_na=False, low_memory=False)
    monthly = pd.read_csv(
        args.monthly,
        dtype={"feature_id": str, "Match": str},
        low_memory=False,
    )
    previous_label, current_label = comparison_prefixes(args.source_manifest)
    metadata = build_facility_metadata(mapping, previous_label, current_label)
    qualified = prepare_monthly_data(
        monthly,
        metadata,
        through=through,
        min_valid_days=args.min_valid_days,
        min_coverage_pct=args.min_coverage_pct,
    )
    scores, unscored = calculate_scores(
        qualified,
        min_baseline_years=args.min_baseline_years,
        baseline_epsilon=args.baseline_epsilon,
    )
    exclusions = make_exclusions(metadata, monthly, qualified)

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    exclusions.to_csv(args.output / "excluded_facilities.csv", index=False)
    unscored.to_csv(args.output / "unscored_observations.csv", index=False)

    region_summary: dict[str, dict[str, int]] = {}
    through_label = f"{MONTH_LABELS[through[1] - 1]} {through[0]}"
    for region in REGIONS:
        region_directory = args.output / REGION_DIRECTORIES[region]
        plots_directory = region_directory / "plots"
        plots_directory.mkdir(parents=True)

        region_scores = rank_scores(scores[scores["region"] == region].copy())
        region_scores.to_csv(
            region_directory / "anomalies_ranked.csv",
            index=False,
            float_format="%.12g",
        )
        write_top_summary(
            region_scores,
            region_directory / "top_anomalies.md",
            region=region,
            top_n=args.top_summary,
        )

        region_data = qualified[qualified["region"] == region]
        plot_count = 0
        for feature_id, facility_data in region_data.groupby("feature_id", sort=True):
            first = facility_data.iloc[0]
            filename = f"{slugify(str(first['facility_name']))}_{feature_id[:8]}.png"
            plot_facility(
                facility_data,
                region_scores[region_scores["feature_id"] == feature_id],
                plots_directory / filename,
                through_label=through_label,
                min_valid_days=args.min_valid_days,
                min_coverage_pct=args.min_coverage_pct,
            )
            plot_count += 1

        region_summary[region] = {
            "plots": plot_count,
            "ranked_observations": len(region_scores),
        }

    exclusion_counts = exclusions["exclusion_reason"].value_counts().to_dict()
    summary: dict[str, object] = {
        "mapping": str(args.mapping),
        "monthly_source": str(args.monthly),
        "through": args.through,
        "min_valid_days": args.min_valid_days,
        "min_coverage_pct": args.min_coverage_pct,
        "min_baseline_years": args.min_baseline_years,
        "baseline_epsilon": args.baseline_epsilon,
        "eligible_mapped_facilities": len(metadata),
        "plotted_facilities": qualified["feature_id"].nunique(),
        "excluded_facilities": len(exclusions),
        "exclusion_counts": {str(key): int(value) for key, value in exclusion_counts.items()},
        "unscored_observations": len(unscored),
        "regions": region_summary,
    }
    (args.output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_readme(args.output, summary)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
