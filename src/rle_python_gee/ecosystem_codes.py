"""Assign hierarchical ecosystem codes to a vector dataset.

For each functional group code, the distinct ecosystem names are sorted
alphabetically and assigned a counter starting at 1. The counter is
appended to the functional group code to form the ecosystem code. For
example, two ecosystems in functional group ``T1.1`` become ``T1.1.1``
and ``T1.1.2``.
"""

import warnings

import geopandas as gpd
import pandas as pd

# Shapefile DBF character fields are limited to 10 characters; longer
# values are silently truncated by downstream tools that export to .shp,
# which can collapse distinct ecosystems into the same reported code.
SHAPEFILE_VALUE_LIMIT = 10


def assign_ecosystem_codes(
    gdf: gpd.GeoDataFrame,
    *,
    fg_code_col: str,
    eco_name_col: str,
    eco_code_col: str,
) -> gpd.GeoDataFrame:
    """Return a copy of ``gdf`` with a new column of hierarchical ecosystem codes.

    Within each functional group, distinct ecosystem names are sorted
    alphabetically before numbering, so the result is deterministic
    regardless of input row order.

    Args:
        gdf: Input GeoDataFrame.
        fg_code_col: Existing column holding the functional group code
            (e.g. values like ``"T1.1"``).
        eco_name_col: Existing column holding the ecosystem name.
        eco_code_col: Name of the new column to create and populate.

    Returns:
        A copy of ``gdf`` with ``eco_code_col`` added. Rows whose
        functional group code or ecosystem name is null receive a null
        ecosystem code.

    Raises:
        ValueError: If ``fg_code_col`` or ``eco_name_col`` is missing
            from ``gdf``, or if ``eco_code_col`` already exists.

    Warns:
        UserWarning: If any generated ecosystem code exceeds
            :data:`SHAPEFILE_VALUE_LIMIT` characters. Such codes will be
            silently truncated if later written to a shapefile field,
            which can collapse distinct ecosystems when reporting
            statistics per ecosystem.
    """
    # --- Validate columns ---
    for col in (fg_code_col, eco_name_col):
        if col not in gdf.columns:
            raise ValueError(
                f"column {col!r} not found in input. "
                f"Available columns: {list(gdf.columns)}"
            )
    if eco_code_col in gdf.columns:
        raise ValueError(
            f"column {eco_code_col!r} already exists in input. "
            f"Refusing to overwrite."
        )

    # --- Build (fg, eco_name) -> code mapping ---
    # Distinct (fg, eco_name) pairs, skipping any with nulls, sorted so
    # that within each functional group, ecosystem names are alphabetical.
    pairs = (
        gdf[[fg_code_col, eco_name_col]]
        .dropna()
        .drop_duplicates()
        .sort_values([fg_code_col, eco_name_col])
    )

    code_map: dict[tuple, str] = {}
    for fg, group in pairs.groupby(fg_code_col, sort=True):
        for counter, eco_name in enumerate(group[eco_name_col].tolist(), start=1):
            code_map[(fg, eco_name)] = f"{fg}.{counter}"

    # --- Apply mapping to every row ---
    def lookup(row):
        fg = row[fg_code_col]
        eco = row[eco_name_col]
        if pd.isna(fg) or pd.isna(eco):
            return None
        return code_map.get((fg, eco))

    result = gdf.copy()
    result[eco_code_col] = result.apply(lookup, axis=1)

    # --- Warn about values that exceed the shapefile character limit ---
    long_codes = sorted(
        {c for c in code_map.values() if len(c) > SHAPEFILE_VALUE_LIMIT}
    )
    if long_codes:
        warnings.warn(
            f"{len(long_codes)} ecosystem code(s) exceed "
            f"{SHAPEFILE_VALUE_LIMIT} characters and will be truncated if "
            f"later written to a shapefile field: {long_codes}",
            stacklevel=2,
        )

    return result
