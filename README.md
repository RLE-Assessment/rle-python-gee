# rle-python-gee

Google Earth Engine data-access backends for
[`rle-python`](https://github.com/RLE-Assessment/rle-python) — the core
toolkit for IUCN Red List of Ecosystems (RLE) assessments.

`rle-python` provides the RLE data model and business logic with local /
cloud-file access. `rle-python-gee` adds **Earth Engine** read/write backends
(FeatureCollections and Images), country-map rendering, and EE authentication
helpers. It installs into the shared `rle` namespace as **`rle.gee`**.

Based on JavaScript functions in the upstream repo:
<https://github.com/red-list-ecosystem/gee-redlist>

## Installation

Requirements: **Python 3.11 or 3.12**. Installing `rle-python-gee` pulls in
`rle-python` automatically.

```bash
pip install rle-python-gee
```

With [uv](https://docs.astral.sh/uv/) from a clone of this repo:

```bash
uv sync                                   # installs rle-python-gee + rle-python from git/PyPI
```

To develop against a sibling editable checkout of `rle-python` instead, add
the override after sync:

```bash
uv sync
uv pip install -e ../rle-python --reinstall   # swap in editable sibling
```

## Usage

The core API is unchanged — you construct Earth Engine backends explicitly and
use the same `AOOGrid` / `EOO` interfaces as the core package:

```python
import ee
ee.Initialize(project="my-ee-project")

from rle.gee import GeeEcosystems, AOOGridEEFeatureCollection

eco = GeeEcosystems(
    "projects/my-project/assets/ecosystems",
    ecosystem_column="ECO_NAME",
)
aoo = AOOGridEEFeatureCollection(
    eco, gee_asset_path="projects/my-project/assets/run1"
).compute()
print(aoo.cell_count)
```

You can also export **local** ecosystem data to Earth Engine — the core
`Ecosystems` / `AOOGrid` objects gain a working `.to_ee_feature_collection()`
once `rle-python-gee` is installed:

```python
from rle.core import Ecosystems

eco = Ecosystems.from_file("ecosystems.geojson", ecosystem_column="ECO_NAME")
eco.to_ee_feature_collection("projects/my-project/assets/ecosystems")
```

List the backends available in your environment:

```bash
rle backends
```

### Authentication

```python
from rle.gee import initialize_ee, print_authentication_status

initialize_ee(project="my-ee-project")
print_authentication_status()
```

### Country maps

```python
from rle.gee import create_country_map

create_country_map("CO", output_path="images/country_map.png")
```

## What moved to `rle-python`

The RLE data model, AOO/EOO computation for local data, ecosystem-code
assignment (`rle.core.ecosystem_codes.assign_ecosystem_codes`, and the
`assign_ecosystem_codes.py` CLI), and visualization helpers now live in the
core `rle-python` package. Import them from `rle.core` rather than the old
`rle_python_gee` module.
