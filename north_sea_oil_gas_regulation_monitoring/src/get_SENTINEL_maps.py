#!/usr/bin/env python3
"""Create standard full-scale Sentinel maps from processed daily arrays."""

from sentinel.map_processing import render_full_scale_map


def main() -> int:
    for chemical in ("NO2", "CO", "HCHO", "SO2"):
        render_full_scale_map(chemical)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
