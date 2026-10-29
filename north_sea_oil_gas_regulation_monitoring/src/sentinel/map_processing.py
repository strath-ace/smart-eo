"""Shared streaming and rendering for full-scale Sentinel maps."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from cartopy.io import shapereader
from rasterio.features import rasterize
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter

try:
    from .paths import PROJECT_ROOT, viewable_path
    from .streaming import streamed_filtered_mean
except ImportError:
    from paths import PROJECT_ROOT, viewable_path
    from streaming import streamed_filtered_mean


FULL_SCALE_RESULTS = PROJECT_ROOT / "results" / "full_scale_maps"
JOINED_NSTA = PROJECT_ROOT / "PROCESSED" / "NSTA" / "mapped_surface_points.csv"
NSTA_MANIFEST = PROJECT_ROOT / "PROCESSED" / "NSTA" / "mapped_surface_points_manifest.json"
LON_MIN, LON_MAX = -2, 4
LAT_MIN, LAT_MAX = 52, 62


def normalize(array: np.ndarray) -> np.ndarray:
    finite = np.isfinite(array)
    if not finite.any():
        return np.asarray(array)
    minimum = np.nanmin(array)
    value_range = np.nanmax(array) - minimum
    if value_range == 0:
        return np.where(finite, 0.0, np.nan)
    return (array - minimum) / value_range


def mask_land(array: np.ndarray) -> np.ndarray:
    lons = np.linspace(LON_MIN, LON_MAX, array.shape[1])
    lats = np.linspace(LAT_MIN, LAT_MAX, array.shape[0])
    land_path = shapereader.natural_earth(
        resolution="10m", category="physical", name="land"
    )
    polygons = list(shapereader.Reader(land_path).geometries())
    land = rasterize(
        [(polygon, 1) for polygon in polygons], out_shape=array.shape,
        transform=[
            (LON_MAX - LON_MIN) / len(lons), 0, LON_MIN,
            0, (LAT_MIN - LAT_MAX) / len(lats), LAT_MAX,
        ],
    )
    output = array.copy()
    output[land == 1] = np.nan
    return output


def spatial_contrast(array: np.ndarray, radius: int, *, local_zscore: bool) -> np.ndarray:
    size = radius * 2 + 1
    finite = np.isfinite(array)
    if local_zscore:
        values = np.where(finite, array, 0.0)
        count = uniform_filter(finite.astype(float), size=size, mode="constant")
        mean = uniform_filter(values, size=size, mode="constant")
        square_mean = uniform_filter(values * values, size=size, mode="constant")
        mean = np.divide(mean, count, out=np.zeros_like(mean), where=count > 0)
        square_mean = np.divide(
            square_mean, count, out=np.zeros_like(square_mean), where=count > 0
        )
        std = np.sqrt(np.maximum(square_mean - mean * mean, 0))
        output = np.full(array.shape, np.nan)
        np.divide(array - mean, std, out=output, where=finite & (std > 0))
        return output

    local_minimum = minimum_filter(
        np.where(finite, array, np.inf), size=size, mode="constant", cval=np.inf
    )
    local_maximum = maximum_filter(
        np.where(finite, array, -np.inf), size=size, mode="constant", cval=-np.inf
    )
    value_range = local_maximum - local_minimum
    output = np.full(array.shape, np.nan)
    np.divide(
        array - local_minimum, value_range, out=output,
        where=finite & np.isfinite(value_range) & (value_range > 0),
    )
    return output


def joined_platform_coordinates(
    path: Path = JOINED_NSTA, manifest_path: Path = NSTA_MANIFEST
) -> np.ndarray:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels = [source["label"] for source in manifest["sources"]]
    coordinates: list[list[float]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            label = next(
                (
                    candidate for candidate in reversed(labels)
                    if (row.get(f"{candidate}_infrastructure_name") or "").strip()
                ),
                "",
            )
            if not label:
                continue
            status = (row.get(f"{label}_status") or "").strip().upper()
            infrastructure_type = (
                row.get(f"{label}_type_of_infrastructure") or ""
            ).strip().upper()
            if status != "ACTIVE" or infrastructure_type not in {"PLATFORM", "FPSO"}:
                continue
            try:
                coordinates.append([float(row[f"{label}_x"]), float(row[f"{label}_y"])])
            except (KeyError, TypeError, ValueError):
                continue
    return np.asarray(coordinates, dtype=float)


def render_full_scale_map(chemical: str, *, custom: bool = False) -> Path:
    print(f"Streaming {chemical} observations")
    derived, valid_dates = streamed_filtered_mean(chemical)
    derived = mask_land(derived)
    derived = spatial_contrast(
        derived, 30 if chemical == "NO2" or not custom else 60,
        local_zscore=custom,
    )
    derived = np.power(10, normalize(derived))
    derived = normalize(derived)
    viewable = viewable_path(chemical)
    viewable.parent.mkdir(parents=True, exist_ok=True)
    np.save(viewable, derived)

    aspect = 1 / np.cos(np.deg2rad((LAT_MAX + LAT_MIN) / 2))
    figure, axis = plt.subplots(
        figsize=((10 if custom else 7), aspect * (10 if custom else 7)),
        subplot_kw={"projection": ccrs.PlateCarree()}, layout="constrained",
    )
    image = axis.imshow(
        derived, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], origin="upper",
        transform=ccrs.PlateCarree(), cmap="Reds",
    )
    axis.coastlines()
    axis.gridlines(draw_labels=True, linestyle="--", alpha=0)
    if not custom:
        axis.add_feature(cfeature.LAND, facecolor="lightgray")
    else:
        platforms = joined_platform_coordinates()
        if platforms.size:
            axis.scatter(
                platforms[:, 0], platforms[:, 1], alpha=0.3, marker="x",
                transform=ccrs.PlateCarree(),
            )
    axis.set_aspect(aspect)
    extra = 0.5
    axis.set_extent(
        [LON_MIN - extra, LON_MAX + extra,
         LAT_MIN - extra / aspect, LAT_MAX + extra / aspect]
    )
    plt.colorbar(image, orientation="vertical")
    axis.set_title(
        f"{chemical} Sentinel-5P overview through {valid_dates[-1]}"
        if not custom else chemical
    )
    FULL_SCALE_RESULTS.mkdir(parents=True, exist_ok=True)
    prefix = "custom_map" if custom else "map"
    output = FULL_SCALE_RESULTS / f"{prefix}_{chemical.upper()}.png"
    figure.savefig(output, dpi=500)
    plt.close(figure)
    print(f"Wrote {output}")
    return output
