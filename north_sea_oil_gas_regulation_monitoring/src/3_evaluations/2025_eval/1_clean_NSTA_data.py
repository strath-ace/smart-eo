#!/usr/bin/env python3
"""Clean NSTA platform data and plot infrastructure summaries."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from eval_common import (
    clean_platform_path,
    clean_platform_records,
    parse_datasets,
    prepare_directories,
    results_directory,
    save_json,
    source_features,
)


STATUS_COLORS = {
    "ACTIVE": "#FFA07A",
    "NOT IN USE": "#FF8C00",
    "ABANDONED": "#D2691E",
    "REMOVED": "#D2691E",
}


def run(dataset: str) -> None:
    prepare_directories(dataset)
    features = source_features(dataset)
    clean = clean_platform_records(dataset)
    save_json(clean_platform_path(dataset), clean)
    results = results_directory(dataset)

    type_counts = Counter(
        str(feature.get("properties", {}).get("INF_TYPE") or "UNKNOWN")
        for feature in features
    )
    ordered_types = sorted(type_counts, key=lambda key: (-type_counts[key], key))
    status_values: list[int] = []
    status_colors: list[str] = []
    for infrastructure_type in ordered_types:
        counts = Counter(
            str(feature.get("properties", {}).get("STATUS") or "UNKNOWN").upper()
            for feature in features
            if str(feature.get("properties", {}).get("INF_TYPE") or "UNKNOWN")
            == infrastructure_type
        )
        for status, count in counts.items():
            status_values.append(count)
            status_colors.append(STATUS_COLORS.get(status, "#999999"))

    significant = [(name, count) for name, count in type_counts.items() if count > 2]
    other = sum(count for count in type_counts.values() if count <= 2)
    labels = [item[0] for item in significant] + (["Other"] if other else [])
    values = [item[1] for item in significant] + ([other] if other else [])
    figure = plt.figure(figsize=(7, 7), layout="constrained")
    plt.pie(status_values, colors=status_colors, startangle=0)
    plt.pie(
        values, labels=labels, colors=[(1, 1, 1, 0.1)] * len(values),
        wedgeprops={"edgecolor": "black", "linewidth": 1.5}, labeldistance=1.3,
    )
    plt.title("Surface infrastructure in North Sea recorded by the NSTA")
    figure.savefig(results / "pie_number_of_nsta_items.png", dpi=300)
    plt.close(figure)

    years: list[int] = []
    for rig in clean:
        start_date = rig["NSTA_details"].get("START_DATE")
        if start_date:
            try:
                years.append(int(str(start_date)[:4]))
            except ValueError:
                pass
    figure = plt.figure(layout="constrained")
    if years:
        plt.hist(years, bins=range(min(years), max(years) + 2))
    plt.xlabel("Year")
    plt.ylabel("Number of Oil Rigs")
    plt.title("Oil Rig Creation Over Time")
    plt.xticks(rotation=45)
    figure.savefig(results / "oil_rig_creation_histogram.png")
    plt.close(figure)
    print(f"{dataset}: wrote {len(clean)} cleaned active platforms")


def main(argv: Sequence[str] | None = None) -> int:
    for dataset in parse_datasets(argv, __doc__ or ""):
        run(dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
