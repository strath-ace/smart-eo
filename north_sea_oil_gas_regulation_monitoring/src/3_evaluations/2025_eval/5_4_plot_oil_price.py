#!/usr/bin/env python3
"""Plot the Brent monthly spot price used by the evaluation."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import matplotlib.pyplot as plt

from eval_common import OIL_PRICE_CSV, csv_rows, parse_datasets, prepare_directories, results_directory


def run(dataset: str) -> None:
    prepare_directories(dataset)
    observations: list[tuple[datetime, float]] = []
    for row in csv_rows(OIL_PRICE_CSV):
        if len(row) < 2:
            continue
        try:
            observation_date = datetime.strptime(row[0], "%d/%m/%Y")
            value = float(row[1])
        except ValueError:
            continue
        if datetime(2018, 1, 1) <= observation_date <= datetime.today():
            observations.append((observation_date, value))
    observations.sort()
    if not observations:
        raise ValueError(f"No oil prices from 2018 onward in {OIL_PRICE_CSV}")
    figure = plt.figure(figsize=(9, 4), layout="constrained")
    plt.plot([item[0] for item in observations], [item[1] for item in observations])
    plt.title("Brent Oil Spot Price Monthly")
    plt.xlabel("Time (monthly)")
    plt.ylabel("Brent Oil Spot Price ($ per barrel)")
    figure.savefig(results_directory(dataset) / "oil_price_simple.png", dpi=300)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    for dataset in parse_datasets(argv, __doc__ or ""):
        run(dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
