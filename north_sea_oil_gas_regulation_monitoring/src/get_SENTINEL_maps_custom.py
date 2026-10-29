#!/usr/bin/env python3
"""Create the custom full-scale NO2 map with the joined platform overlay."""

from sentinel.map_processing import render_full_scale_map


def main() -> int:
    render_full_scale_map("NO2", custom=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
