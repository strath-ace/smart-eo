"""Shared loading helpers for the 2026 evaluation suite."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from evaluation_common import (  # noqa: E402
    aggregate_monthly_to_yearly,
    comparison_labels,
)


def comparison_prefixes(source_manifest: Path) -> tuple[str, str]:
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    return comparison_labels(payload)


def yearly_rows(monthly_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    monthly = pd.read_csv(monthly_path, dtype={"feature_id": str, "Match": str})
    yearly = aggregate_monthly_to_yearly(monthly)
    text = yearly.fillna("").astype(str)
    return list(text.columns), text.to_dict(orient="records")


def export_rows(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
