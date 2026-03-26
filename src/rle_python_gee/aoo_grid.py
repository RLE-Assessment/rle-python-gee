"""Generate AOO grid cells for local (non-Earth Engine) backends."""

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely.geometry import box


# ESRI:54034 World Cylindrical Equal Area — same projection used by
# get_aoo_grid_projection() in ee_rle.py
AOO_CRS = "ESRI:54034"
AOO_CELL_SIZE = 10_000  # 10 km in meters


def generate_aoo_grid(bounds_4326: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Generate a GeoDataFrame of AOO grid cells covering the given bounds.

    Args:
        bounds_4326: (minx, miny, maxx, maxy) in EPSG:4326.

    Returns:
        GeoDataFrame with grid cell polygons in EPSG:4326.
    """
    minx, miny, maxx, maxy = bounds_4326

    # Transform bounds to equal-area projection
    transformer = Transformer.from_crs("EPSG:4326", AOO_CRS, always_xy=True)
    ea_minx, ea_miny = transformer.transform(minx, miny)
    ea_maxx, ea_maxy = transformer.transform(maxx, maxy)

    # Snap to grid origin (0, 0) so cells are globally consistent
    ea_minx = np.floor(ea_minx / AOO_CELL_SIZE) * AOO_CELL_SIZE
    ea_miny = np.floor(ea_miny / AOO_CELL_SIZE) * AOO_CELL_SIZE
    ea_maxx = np.ceil(ea_maxx / AOO_CELL_SIZE) * AOO_CELL_SIZE
    ea_maxy = np.ceil(ea_maxy / AOO_CELL_SIZE) * AOO_CELL_SIZE

    xs = np.arange(ea_minx, ea_maxx, AOO_CELL_SIZE)
    ys = np.arange(ea_miny, ea_maxy, AOO_CELL_SIZE)

    cells = [
        box(x, y, x + AOO_CELL_SIZE, y + AOO_CELL_SIZE)
        for x in xs
        for y in ys
    ]

    grid = gpd.GeoDataFrame(geometry=cells, crs=AOO_CRS)
    return grid.to_crs("EPSG:4326")
