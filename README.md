# rle-python-gee

Python tools for the IUCN Red List of Ecosystems and Species in Google Earth Engine.

Based on JavaScript functions in the upstream repo:
<https://github.com/red-list-ecosystem/gee-redlist>

## Installation

Requirements: **Python 3.11 or 3.12**.

The project is developed with [uv](https://docs.astral.sh/uv/). From a clone
of this repository:

```bash
uv sync
```

This creates `.venv/` and installs `rle-python-gee` with all required
dependencies. Prefix any command with `uv run` to execute it in that
environment (for example `uv run python scripts/assign_ecosystem_codes.py …`).

### Optional extras

Two optional dependency groups are defined in `pyproject.toml`:

- **`viz`** — installs [lonboard](https://developmentseed.org/lonboard/) for
  interactive map visualization of ecosystem polygons and AOO grids.
- **`gcs`** — installs `gcsfs`, needed when uploading large vector datasets
  to Earth Engine via Google Cloud Storage ingestion.

Install one or both:

```bash
uv sync --extra viz
uv sync --extra gcs
uv sync --extra viz --extra gcs
```

### Without uv

If you are not using uv, an editable install with pip works too:

```bash
pip install -e .
# or with extras:
pip install -e ".[viz,gcs]"
```

## Basic usage: assign ecosystem codes to a shapefile

A common first step when preparing ecosystem data for an RLE assessment is
to assign each ecosystem a short, hierarchical code derived from its
functional group. For example, two distinct ecosystems within functional
group `T1.1` become `T1.1.1` and `T1.1.2`.

Within each functional group, distinct ecosystem names are sorted
alphabetically before numbering, so the result is deterministic regardless
of input row order.

Input and output formats are auto-detected by file extension. Supported
formats include shapefile (`.shp`), GeoJSON (`.geojson`), GeoPackage
(`.gpkg`), and FlatGeobuf (`.fgb`).

### Command-line form

```bash
uv run python scripts/assign_ecosystem_codes.py \
    path/to/input.shp \
    path/to/output.shp \
    --fg-code-col FG_CODE \
    --eco-name-col ECOSYSTEM_NAME \
    --eco-code-col eco_code
```

Arguments:

- `input` — path to the input vector file
- `output` — path to the output vector file (the input is never modified)
- `--fg-code-col` — existing column holding the functional group code
  (e.g. values like `T1.1`)
- `--eco-name-col` — existing column holding the ecosystem name
- `--eco-code-col` — name of the new column to create

Notes:

- Shapefile DBF fields are limited to 10 characters. When writing to `.shp`,
  the script refuses to run if `--eco-code-col` is longer than 10 characters
  — pick a shorter name or use a non-shapefile output format.
- The script also warns if any generated ecosystem code value exceeds 10
  characters, since downstream tools that report per-ecosystem statistics in
  shapefile format would silently truncate such codes and collapse distinct
  ecosystems.

### Python library form

The same logic is importable directly — useful in notebooks:

```python
import geopandas as gpd
from rle_python_gee.ecosystem_codes import assign_ecosystem_codes

gdf = gpd.read_file("path/to/input.shp")

gdf_with_codes = assign_ecosystem_codes(
    gdf,
    fg_code_col="FG_CODE",
    eco_name_col="ECOSYSTEM_NAME",
    eco_code_col="eco_code",
)

gdf_with_codes.to_file("path/to/output.geojson", driver="GeoJSON")
```

The function returns a copy (the input `gdf` is not mutated), raises
`ValueError` if the specified columns are missing or if `eco_code_col`
already exists, and emits a `UserWarning` if any generated code exceeds 10
characters.

## What's next

The [`examples/`](examples/) directory contains Jupyter notebooks showing
deeper workflows:

- [`example_local_vector.ipynb`](examples/example_local_vector.ipynb) —
  working with local shapefile and GeoParquet ecosystem data
- [`example_aoo.ipynb`](examples/example_aoo.ipynb) — computing the Area
  of Occupancy (AOO) grid for an ecosystem
- [`example_gee_featurecollection.ipynb`](examples/example_gee_featurecollection.ipynb)
  — using Earth Engine FeatureCollections as ecosystem sources

The Earth Engine–based workflows require a Google Earth Engine account and
authentication; the `assign_ecosystem_codes` workflow above does not.
