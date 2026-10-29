# New 2026 evaluation scripts

These scripts rebuild the ownership-transfer and monthly anomaly analyses in
`results/new_eval_2026_results/`.

From the repository root:

```bash
python src/3_evaluations/2026_eval/run_all.py
```

This validates the completed Stage 1 and Stage 2 artifacts, snapshots their
mapping provenance into the result bundle, then runs the ENI analysis,
Shell/Equinor analysis, anomaly analysis, and workbook export. Use `--dry-run`
to display the commands or the path options to override defaults.

Each analysis can also be run separately:

```bash
python src/3_evaluations/2026_eval/build_eni_to_ithaca_energy.py
python src/3_evaluations/2026_eval/analyse_shell_equinor_to_adura.py
python src/3_evaluations/2026_eval/build_platform_fpso_anomalies.py
node src/3_evaluations/2026_eval/build_eni_to_ithaca_energy_workbook.mjs
```

The Python builders accept mapping, source-manifest, monthly-cache, and output
overrides as applicable. Defaults resolve from the repository root and use:

- `PROCESSED/NSTA/mapped_surface_points.csv`
- `PROCESSED/NSTA/mapped_surface_points_manifest.json`
- `PROCESSED/SENTINEL/facility_no2_monthly.csv`

The JavaScript workbook builder accepts optional source and output paths as its
first and second positional arguments. It uses `@oai/artifact-tool` from either
the project installation or the bundled Codex workspace runtime.
