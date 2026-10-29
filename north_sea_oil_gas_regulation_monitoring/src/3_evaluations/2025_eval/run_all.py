#!/usr/bin/env python3
"""Run evaluation steps 1 through 5 for one dataset or all datasets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from eval_common import DATASETS


SCRIPTS = (
    "1_clean_NSTA_data.py",
    "2_closest_rig_map.py",
    "3_closest_rig_firms.py",
    "4_get_PLATFORMS_with_FIRMS.py",
    "5_4_plot_oil_price.py",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=(*DATASETS, "all"), default="all",
        help="Dataset to evaluate (default: all)",
    )
    args = parser.parse_args(argv)
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    directory = Path(__file__).resolve().parent
    for dataset in datasets:
        for script in SCRIPTS:
            command = [sys.executable, str(directory / script), "--dataset", dataset]
            print(f"Running {script} for {dataset}", flush=True)
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
