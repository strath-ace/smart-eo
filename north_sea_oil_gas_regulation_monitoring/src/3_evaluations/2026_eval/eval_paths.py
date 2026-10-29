"""Repository-relative paths for the Stage 3 2026 evaluation suite."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "new_eval_2026_results"
MAPPING = PROJECT_ROOT / "PROCESSED" / "NSTA" / "mapped_surface_points.csv"
SOURCE_MANIFEST = (
    PROJECT_ROOT / "PROCESSED" / "NSTA" / "mapped_surface_points_manifest.json"
)
FACILITY_MONTHLY = (
    PROJECT_ROOT / "PROCESSED" / "SENTINEL" / "facility_no2_monthly.csv"
)
FACILITY_MANIFEST = (
    PROJECT_ROOT / "PROCESSED" / "SENTINEL" / "facility_no2_monthly_manifest.json"
)
