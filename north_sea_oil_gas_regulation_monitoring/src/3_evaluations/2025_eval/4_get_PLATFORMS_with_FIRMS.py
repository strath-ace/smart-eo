#!/usr/bin/env python3
"""Plot S-NPP FIRMS observations alongside the selected NSTA dataset."""

from __future__ import annotations

from typing import Sequence

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

from eval_common import (
    FIRMS_SNPP,
    firms_observations,
    parse_datasets,
    prepare_directories,
    results_directory,
    source_features,
)


def run(dataset: str) -> None:
    prepare_directories(dataset)
    observations = firms_observations(FIRMS_SNPP)
    platforms = [
        feature for feature in source_features(dataset)
        if str(feature.get("properties", {}).get("INF_TYPE") or "").upper()
        in {"PLATFORM", "FPSO"}
    ]
    lats = np.asarray([float(item["latitude"]) for item in observations])
    lons = np.asarray([float(item["longitude"]) for item in observations])
    platform_points = np.asarray(
        [feature["geometry"]["coordinates"][:2] for feature in platforms], dtype=float
    )
    extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]
    aspect = 1 / np.cos(np.deg2rad((extent[2] + extent[3]) / 2))
    figure, axis = plt.subplots(
        figsize=(7, 7), subplot_kw={"projection": ccrs.PlateCarree()},
        layout="constrained",
    )
    hexbin = axis.hexbin(
        lons, lats, gridsize=150, cmap="autumn_r", bins="log", mincnt=1,
        linewidths=0.1, extent=extent, transform=ccrs.PlateCarree(),
    )
    if platform_points.size:
        axis.scatter(
            platform_points[:, 0], platform_points[:, 1], marker="x", s=8,
            c="black", alpha=0.35, transform=ccrs.PlateCarree(), label="Platforms/FPSOs",
        )
    axis.coastlines()
    axis.add_feature(cfeature.LAND, facecolor="lightgray")
    axis.gridlines(draw_labels=True, linestyle="--", alpha=0)
    axis.set_extent(extent)
    axis.set_aspect(aspect)
    axis.set_title("Satellite dataset S-NPP")
    axis.legend(loc="lower left")
    plt.colorbar(hexbin, orientation="vertical", label="Log of Number of observations")
    figure.savefig(results_directory(dataset) / "FIRMS_S-NPP_data.png", dpi=300)
    plt.close(figure)
    print(f"{dataset}: plotted {len(observations)} S-NPP observations")


def main(argv: Sequence[str] | None = None) -> int:
    for dataset in parse_datasets(argv, __doc__ or ""):
        run(dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
