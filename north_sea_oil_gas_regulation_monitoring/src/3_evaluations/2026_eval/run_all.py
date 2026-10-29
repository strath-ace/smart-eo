"""Run the 2026 ownership and anomaly evaluations from Stage 1/2 artifacts."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from .eval_paths import (
        FACILITY_MANIFEST, FACILITY_MONTHLY, MAPPING, PROJECT_ROOT,
        RESULTS_DIRECTORY, SOURCE_MANIFEST,
    )
except ImportError:
    from eval_paths import (
        FACILITY_MANIFEST, FACILITY_MONTHLY, MAPPING, PROJECT_ROOT,
        RESULTS_DIRECTORY, SOURCE_MANIFEST,
    )

STAGE_DIRECTORY = Path(__file__).resolve().parents[1]
if str(STAGE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STAGE_DIRECTORY))

from evaluation_common import copy_provenance, validate_stage_inputs  # noqa: E402

SCRIPT_DIRECTORY = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=MAPPING)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--monthly", type=Path, default=FACILITY_MONTHLY)
    parser.add_argument("--monthly-manifest", type=Path, default=FACILITY_MANIFEST)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIRECTORY)
    parser.add_argument("--through", metavar="YYYY-MM")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(value) for value in command)


def main() -> int:
    args = parse_args()
    mapping = args.mapping.expanduser().resolve()
    source_manifest = args.source_manifest.expanduser().resolve()
    monthly = args.monthly.expanduser().resolve()
    monthly_manifest = args.monthly_manifest.expanduser().resolve()
    results = args.results_dir.expanduser().resolve()
    provenance_mapping = results / "source" / mapping.name
    provenance_manifest = results / "source" / source_manifest.name
    node = shutil.which("node") or "node"

    commands: list[tuple[str, list[str]]] = [
        (
            "annual facility emissions evaluation",
            [sys.executable, str(SCRIPT_DIRECTORY / "build_annual_facility_emissions.py"),
             "--mapping", str(mapping), "--source-manifest", str(source_manifest),
             "--monthly", str(monthly), "--monthly-manifest", str(monthly_manifest),
             "--output", str(results / "yearly_facility_emissions")],
        ),
        (
            "ENI to Ithaca Energy analysis",
            [sys.executable, str(SCRIPT_DIRECTORY / "build_eni_to_ithaca_energy.py"),
             "--mapping", str(provenance_mapping), "--source-manifest", str(provenance_manifest),
             "--monthly", str(monthly), "--output", str(results / "eni_to_ithaca_energy")],
        ),
        (
            "Shell/Equinor to Adura Energy analysis",
            [sys.executable, str(SCRIPT_DIRECTORY / "analyse_shell_equinor_to_adura.py"),
             "--mapping", str(provenance_mapping), "--source-manifest", str(provenance_manifest),
             "--monthly", str(monthly), "--output", str(results / "shell_equinor_to_adura")],
        ),
        (
            "platform and FPSO anomaly analysis",
            [sys.executable, str(SCRIPT_DIRECTORY / "build_platform_fpso_anomalies.py"),
             "--mapping", str(provenance_mapping), "--source-manifest", str(provenance_manifest),
             "--monthly", str(monthly), "--output", str(results / "anomaly_detection")],
        ),
        (
            "highlighted ENI workbook",
            [node, str(SCRIPT_DIRECTORY / "build_eni_to_ithaca_energy_workbook.mjs"),
             str(provenance_mapping), str(results / "eni_to_ithaca_energy" / "highlighted_facilities.xlsx"),
             str(provenance_manifest)],
        ),
    ]
    if args.through:
        commands[3][1].extend(["--through", args.through])

    if args.dry_run:
        for index, (label, command) in enumerate(commands, 1):
            print(f"{index}. {label}: {command_text(command)}")
        return 0

    validate_stage_inputs(mapping, source_manifest, monthly, monthly_manifest)
    if shutil.which("node") is None:
        raise SystemExit("Node.js is required for the workbook builder but was not found")
    results.mkdir(parents=True, exist_ok=True)
    copy_provenance(results / "source", mapping, source_manifest)
    for index, (label, command) in enumerate(commands, 1):
        print(f"\n[{index}/{len(commands)}] Running {label}", flush=True)
        print(f"$ {command_text(command)}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print(f"\n2026 evaluations completed: {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
