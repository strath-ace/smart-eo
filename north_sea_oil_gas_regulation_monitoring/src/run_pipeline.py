#!/usr/bin/env python3
"""Run the three-stage NSTA, facility-estimate, and evaluation pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().parent
STAGES = {
    1: SOURCE / "1_join_nsta" / "run.py",
    2: SOURCE / "2_facility_estimates" / "run.py",
    3: SOURCE / "3_evaluations" / "run.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-at", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--stop-after", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.stop_after < args.start_at:
        parser.error("--stop-after cannot be earlier than --start-at")
    commands = [
        [sys.executable, str(STAGES[number])]
        for number in range(args.start_at, args.stop_after + 1)
    ]
    if args.dry_run:
        for index, command in enumerate(commands, args.start_at):
            print(f"Stage {index}: {' '.join(command)}")
        return 0
    for index, command in enumerate(commands, args.start_at):
        print(f"\n=== Stage {index} ===", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print("\nThree-stage pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
