#!/usr/bin/env python3
"""Run Stage 3: the complete 2025 and 2026 evaluation suites."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from evaluation_common import PROJECT_ROOT, validate_stage_inputs


DIRECTORY = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("2025", "2026", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    commands: list[list[str]] = []
    if args.suite in {"2025", "all"}:
        commands.append([
            sys.executable, str(DIRECTORY / "2025_eval" / "run_all.py"),
            "--dataset", "all",
        ])
    if args.suite in {"2026", "all"}:
        commands.append([
            sys.executable, str(DIRECTORY / "2026_eval" / "run_all.py")
        ])
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return 0
    validate_stage_inputs()
    for command in commands:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
