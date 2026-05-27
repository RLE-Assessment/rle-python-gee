"""RLE Earth Engine backends — Google Earth Engine data access for ``rle.core``.

Installs into the shared ``rle`` namespace as ``rle.gee`` and extends the core
RLE data model with Earth Engine read/write backends. Construct backends
explicitly::

    from rle.gee import GeeEcosystems, AOOGridEEFeatureCollection
    eco = GeeEcosystems("projects/p/assets/ecosystems", ecosystem_column="ECO_NAME")
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("rle-python-gee")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"

from rle.gee.auth import (
    check_authentication,
    initialize_ee,
    is_authenticated,
    print_authentication_status,
)
from rle.gee.ecosystems import (
    EcosystemsEEFeatureCollection,
    EcosystemsEEImage,
    GeeEcosystems,
    GeeEcosystemsImage,
)
from rle.gee.aoo import (
    AOOGridEEImage,
    AOOGridEEFeatureCollection,
    AOOGridPolygonEEFeatureCollection,
    wait_for_task,
)
from rle.gee.ee_rle import (
    area_km2,
    create_asset_folder,
    ensure_asset_folder_exists,
    export_fractional_coverage_on_aoo_grid,
    get_aoo_grid_projection,
)
from rle.gee.upload import upload_gdf_to_ee_asset


def __getattr__(name):
    """Lazily expose map helpers to avoid importing wkls/cartopy at import time."""
    if name in ("create_country_map", "get_utm_epsg"):
        from rle.gee import map as _map
        return getattr(_map, name)
    raise AttributeError(f"module 'rle.gee' has no attribute {name!r}")


__all__ = [
    "__version__",
    # auth
    "check_authentication",
    "initialize_ee",
    "is_authenticated",
    "print_authentication_status",
    # ecosystems backends
    "EcosystemsEEFeatureCollection",
    "EcosystemsEEImage",
    "GeeEcosystems",
    "GeeEcosystemsImage",
    # AOO backends
    "AOOGridEEImage",
    "AOOGridEEFeatureCollection",
    "AOOGridPolygonEEFeatureCollection",
    "wait_for_task",
    # EE helpers
    "area_km2",
    "create_asset_folder",
    "ensure_asset_folder_exists",
    "export_fractional_coverage_on_aoo_grid",
    "get_aoo_grid_projection",
    "upload_gdf_to_ee_asset",
    # maps (lazy)
    "create_country_map",
    "get_utm_epsg",
]
