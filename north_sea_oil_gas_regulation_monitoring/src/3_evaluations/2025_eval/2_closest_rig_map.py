#!/usr/bin/env python3
"""Plot facility locations and ownership summaries."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

from eval_common import (
    bounded_rigs,
    clean_platform_path,
    load_json,
    parse_datasets,
    prepare_directories,
    results_directory,
)


COLORS = {"PLC": "skyblue", "LTD": "sienna", "SOE": "olivedrab"}


def run(dataset: str) -> None:
    prepare_directories(dataset)
    source = clean_platform_path(dataset)
    if not source.is_file():
        raise FileNotFoundError(f"Run 1_clean_NSTA_data.py first; missing {source}")
    rigs = bounded_rigs(load_json(source))
    results = results_directory(dataset)

    aspect = 1 / np.cos(np.deg2rad(57))
    figure, axis = plt.subplots(
        figsize=(6, aspect * 7), subplot_kw={"projection": ccrs.PlateCarree()},
        layout="constrained",
    )
    axis.coastlines()
    axis.gridlines(draw_labels=True, linestyle="--", alpha=0)
    axis.add_feature(cfeature.LAND, facecolor="lightgray")
    for ownership_type, color in COLORS.items():
        points = np.asarray(
            [rig["coordinates"] for rig in rigs if rig["parent_company_type"] == ownership_type]
        )
        if points.size:
            axis.scatter(points[:, 0], points[:, 1], s=100, c=color, label=ownership_type)
    axis.set_extent([-2.5, 4.5, 52 - 0.5 / aspect, 62 + 0.5 / aspect])
    axis.set_aspect(aspect)
    axis.legend(title="Ownership Types")
    axis.set_title(
        "Oil Platforms and FPSO in UK Waters subdivided by Parent Company Ownership Type",
        wrap=True,
    )
    figure.savefig(results / "oil_rig_location_map.png", dpi=300)
    plt.close(figure)

    counts = Counter(rig["parent_company_type"] for rig in rigs)
    figure = plt.figure(figsize=(10, 10))
    plt.pie(
        [counts.get(key, 0) for key in COLORS],
        labels=["PLC Owned", "LTD Owned", "SOE Owned"],
        colors=list(COLORS.values()),
        wedgeprops={"edgecolor": "black", "linewidth": 1.5},
    )
    figure.savefig(results / "pie_number_of_ownership_types.png", dpi=300)
    plt.close(figure)

    company_counts = Counter(rig["parent_company"] for rig in rigs)
    company_types = {rig["parent_company"]: rig["parent_company_type"] for rig in rigs}
    retained = sorted(
        ((name, count) for name, count in company_counts.items() if count > 1),
        key=lambda item: (-item[1], item[0]),
    )
    other = sum(count for count in company_counts.values() if count <= 1)
    names = [name for name, _ in retained] + (["Other"] if other else [])
    values = [count for _, count in retained] + ([other] if other else [])
    colors = [COLORS.get(company_types[name], "grey") for name, _ in retained]
    if other:
        colors.append("grey")
    figure = plt.figure(figsize=(8, 7), layout="constrained")
    plt.pie(
        values, labels=names, colors=colors,
        wedgeprops={"edgecolor": "black", "linewidth": 1.5},
        startangle=-8, labeldistance=1.5,
    )
    plt.title(f"Division of oil rigs by parent company of {len(rigs)} total", pad=20)
    figure.savefig(results / "pie_number_of_rigs_per_company.png", dpi=300)
    plt.close(figure)
    print(f"{dataset}: plotted ownership summaries for {len(rigs)} facilities")


def main(argv: Sequence[str] | None = None) -> int:
    for dataset in parse_datasets(argv, __doc__ or ""):
        run(dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
