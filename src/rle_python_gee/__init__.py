"""RLE Python GEE - Tools for IUCN Red List of Ecosystems analysis using Google Earth Engine."""

from importlib.metadata import version, PackageNotFoundError

# Get version from installed package metadata (reads from pyproject.toml)
try:
    __version__ = version("rle-python-gee")
except PackageNotFoundError:
    # Package not installed (development mode)
    __version__ = "0.0.0.dev"

from rle_python_gee.ee_auth import check_authentication, is_authenticated, print_authentication_status
from rle_python_gee.ee_rle import make_eoo, area_km2
from rle_python_gee.ecosystems import Ecosystems, make_ecosystems
from rle_python_gee.aoo import (
    AOOGrid,
    AOOGridNotComputedError,
    AOOGridPolygons,
    AOOGridPolygonsNotComputedError,
    make_aoo,
    make_aoo_polygons,
)


def __getattr__(name):
    """Lazy import for map module to avoid loading wkls (and its S3 connection) at import time."""
    if name in ("create_country_map", "get_utm_epsg"):
        from rle_python_gee import map as _map
        return getattr(_map, name)
    raise AttributeError(f"module 'rle_python_gee' has no attribute {name!r}")


__all__ = [
    "__version__",
    "check_authentication",
    "is_authenticated",
    "print_authentication_status",
    "make_eoo",
    "area_km2",
    "Ecosystems",
    "make_ecosystems",
    "make_aoo",
    "make_aoo_polygons",
    "AOOGrid",
    "AOOGridNotComputedError",
    "AOOGridPolygons",
    "AOOGridPolygonsNotComputedError",
    "create_country_map",
    "get_utm_epsg",
]
