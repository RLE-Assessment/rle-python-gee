"""RLE Python GEE - Tools for IUCN Red List of Ecosystems analysis using Google Earth Engine."""

from importlib.metadata import version, PackageNotFoundError

# Get version from installed package metadata (reads from pyproject.toml)
try:
    __version__ = version("rle-python-gee")
except PackageNotFoundError:
    # Package not installed (development mode)
    __version__ = "0.0.0.dev"

from rle_python_gee.ee_auth import check_authentication, is_authenticated, print_authentication_status
from rle_python_gee.ee_rle import Ecosystems, make_eoo, area_km2
from rle_python_gee.map import create_country_map, get_utm_epsg

__all__ = [
    "__version__",
    "check_authentication",
    "is_authenticated",
    "print_authentication_status",
    "Ecosystems",
    "make_eoo",
    "area_km2",
    "create_country_map",
    "get_utm_epsg",
]
