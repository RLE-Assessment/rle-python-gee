"""Assign hierarchical ecosystem codes to a vector dataset (CLI).

Reads an input vector file, assigns per-ecosystem codes by appending a
counter to the functional group code (e.g. ``T1.1`` → ``T1.1.1``,
``T1.1.2``), and writes the result to an output vector file. Input and
output formats are auto-detected by file extension (``.shp``,
``.geojson``, ``.gpkg``, ``.fgb``).

The core logic lives in
``rle_python_gee.ecosystem_codes.assign_ecosystem_codes`` and is
importable directly.
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd

from rle_python_gee.ecosystem_codes import (
    SHAPEFILE_VALUE_LIMIT,
    assign_ecosystem_codes,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to input vector file")
    parser.add_argument("output", help="Path to output vector file")
    parser.add_argument(
        "--fg-code-col", required=True,
        help="Existing column holding the functional group code (e.g. 'T1.1')",
    )
    parser.add_argument(
        "--eco-name-col", required=True,
        help="Existing column holding the ecosystem name",
    )
    parser.add_argument(
        "--eco-code-col", required=True,
        help="Name of the new column to create and populate with ecosystem codes",
    )
    args = parser.parse_args()

    # --- Validate output column name against shapefile limits up-front ---
    output_is_shapefile = Path(args.output).suffix.lower() == ".shp"
    if output_is_shapefile and len(args.eco_code_col) > SHAPEFILE_VALUE_LIMIT:
        sys.exit(
            f"Error: --eco-code-col {args.eco_code_col!r} is "
            f"{len(args.eco_code_col)} characters; shapefile fields are "
            f"limited to {SHAPEFILE_VALUE_LIMIT}. Choose a shorter name or "
            f"use a non-shapefile output format (.geojson, .gpkg, .fgb)."
        )

    print(f"Reading {args.input}")
    gdf = gpd.read_file(args.input)

    try:
        result = assign_ecosystem_codes(
            gdf,
            fg_code_col=args.fg_code_col,
            eco_name_col=args.eco_name_col,
            eco_code_col=args.eco_code_col,
        )
    except ValueError as e:
        sys.exit(f"Error: {e}")

    # --- Summary ---
    distinct = (
        result[[args.fg_code_col, args.eco_name_col]]
        .dropna()
        .drop_duplicates()
    )
    n_fg = distinct[args.fg_code_col].nunique()
    n_eco = len(distinct)
    n_null = result[args.eco_code_col].isna().sum()
    print(f"Functional groups: {n_fg}")
    print(f"Distinct ecosystems: {n_eco}")
    if n_null:
        print(f"Rows with null ecosystem code (null fg or eco name): {n_null}")

    # --- Write output ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_file(output_path)
    print(f"Wrote {len(result)} features to {output_path}")


if __name__ == "__main__":
    main()
