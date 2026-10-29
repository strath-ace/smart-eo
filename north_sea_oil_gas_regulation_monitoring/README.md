# FRIL

Satellite data work for FRIL Oil and Gas Regulation Monitoring project


## Setup

The current environment uses Python 3.12. Create and populate a virtual
environment from the repository root:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Organize source data beneath `RAW/`. NSTA surface-point exports must be placed
in month-year folders such as `RAW/NSTA/mar-2025/` and
`RAW/NSTA/oct-2026/`. Sentinel Hub caches belong in
`RAW/SENTINEL/raw_data_<CHEMICAL>/`.

Convert every available Sentinel chemical into daily NumPy arrays and PNGs:

```bash
python src/sentinel/raw_to_np.py
```

To process selected chemicals only:

```bash
python src/sentinel/raw_to_np.py NO2 CO
```

Processed Sentinel products are written under `PROCESSED/SENTINEL/`.

To collect more Sentinel data, copy `.env_template` to `.env` in the project
root and provide the Copernicus Data Space OAuth credentials.


## Run the three-stage pipeline

Run NSTA joining, monthly facility estimation, and both evaluation suites:

```bash
python src/run_pipeline.py
```

The stages can also be run independently. Stage 2 requires Stage 1 outputs,
and Stage 3 requires both Stage 1 and Stage 2 outputs.

### Stage 1: join dated NSTA snapshots

```bash
python src/1_join_nsta/run.py
```

This discovers the surface-point CSV in every dated `RAW/NSTA/<month>-<year>/`
folder, performs exact and manual Feature ID matching, and writes:

- `PROCESSED/NSTA/mapped_surface_points.csv`
- `PROCESSED/NSTA/mapped_surface_points_manifest.json`

### Stage 2: calculate monthly facility values

```bash
python src/2_facility_estimates/run.py
```

This processes the daily arrays under `PROCESSED/SENTINEL/numpy_NO2/` and
writes the authoritative facility-by-month CSV:

`PROCESSED/SENTINEL/facility_no2_monthly.csv`

Each row represents one mapped facility in one calendar month. The CSV contains
the Feature ID, facility name, year, month, selected NSTA coordinate source,
longitude and latitude, monthly mean NO₂ value, valid pixel observations,
valid days, available raster days, coverage percentage, and analysis status.
Facilities absent from the NSTA snapshot selected for a month do not receive a
row for that month. Cache provenance is stored in
`PROCESSED/SENTINEL/facility_no2_monthly_manifest.json`.

### Stage 3: run the evaluations

```bash
python src/3_evaluations/run.py
```

This consumes the joined mapping and monthly facility CSV, runs the 2025 suite
for the 2025, 2026, and joined datasets, then runs the newer 2026 ownership and
anomaly suite. It writes figures, tables, workbooks, and per-rig files beneath:

- `results/2025_results/`
- `results/2026_results/`
- `results/joined_results/`

The evaluation also writes the newer ownership/anomaly analysis beneath
`results/new_eval_2026_results/`. The older `results_new/` bundle remains an
untouched legacy reference.

### Annual facility emissions evaluation

The 2026 evaluation suite also combines the authoritative monthly facility
values into annual files and calculates facility-level and overall
year-on-year percentage changes. Run it independently with:

```bash
python src/3_evaluations/2026_eval/build_annual_facility_emissions.py
```

Outputs are written to
`results/new_eval_2026_results/yearly_facility_emissions/`. There is one
`facility_emissions_<YEAR>.csv` file per available year and an
`annual_summary.csv` containing the overall metric for each year. Annual means
are weighted by valid pixel observations. Partial years are compared with the
same calendar months in the preceding year. These values retain the
`mean_no2_mol_m2` name because they are satellite NO₂ column estimates, not
direct mass-emission totals.

## Full-scale Sentinel maps

```bash
python src/get_SENTINEL_maps.py
python src/get_SENTINEL_maps_custom.py
```

These scripts stream the processed daily arrays and write distinct standard and
custom figures to `results/full_scale_maps/`.

## Facility changes and NO₂ dashboard

Install the dependencies, then run the native desktop dashboard from the project root:

```bash
python run_dashboard.py
```

The first run builds or validates the Stage 2 monthly cache from the daily files
in `PROCESSED/SENTINEL/numpy_NO2`. Annual dashboard views are calculated from
the authoritative monthly rows at load time.

To prepare or validate the cache without opening the PySide6 window:

```bash
python run_dashboard.py --build-only
```
