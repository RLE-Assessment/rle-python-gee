"""Area of Occupancy (AOO) grid computation for RLE assessments.

Provides the AOOGrid class hierarchy and make_aoo_grid() factory function
for computing AOO grids from multiple data sources (Earth Engine Images,
Earth Engine FeatureCollections, geoparquet files, GeoJSON files, and COGs).
"""

import logging
import re
from abc import ABC, abstractmethod

import geopandas as gpd

logger = logging.getLogger(__name__)


def slugify_ecosystem_name(name: str) -> str:
    """Sanitize an ecosystem name for use as a DataFrame column name.

    Replaces any character that is not alphanumeric, underscore, or hyphen
    with an underscore.
    """
    return re.sub(r'[^a-zA-Z0-9_-]', '_', str(name))

from rle_python_gee.ecosystems import (
    Ecosystems,
    EcosystemKind,
    EcosystemsFile,
    EcosystemsGeoParquet,
    EcosystemsEEFeatureCollection,
    EcosystemsEEImage,
    EcosystemsCOG,
    make_ecosystems,
)


import time


# AOO grid cell size in meters (10 x 10 km)
AOO_CELL_SIZE_M = 10_000


def wait_for_task(task, *, poll_interval: int = 15) -> None:
    """Poll an EE export task until it completes, fails, or is cancelled.

    Parameters
    ----------
    task : ee.batch.Task
        The task returned by ``ee.batch.Export.table.*``.
    poll_interval : int
        Seconds between status checks (default 60).
    """
    import ee

    if task is None:
        logger.info("No task to wait for — the asset was already cached or compute() has not been called yet.")
        return

    start = time.monotonic()
    while True:
        status = ee.data.getTaskStatus(task.id)[0]
        state = status['state']
        elapsed = int(time.monotonic() - start)
        if state == 'COMPLETED':
            logger.info("Export completed after %d s: %s", elapsed, task.id)
            return
        if state in ('FAILED', 'CANCELLED'):
            raise RuntimeError(
                f"Export {state} after {elapsed} s: "
                f"{status.get('error_message', '')}"
            )
        logger.info(
            "Export state: %s (%d s elapsed) — waiting %d s …",
            state, elapsed, poll_interval,
        )
        time.sleep(poll_interval)



def _remote_file_exists(path: str) -> bool:
    """Check if a file exists, supporting gs:// URIs and local paths."""
    import fsspec
    try:
        fs, fpath = fsspec.core.url_to_fs(path)
        return fs.exists(fpath)
    except Exception:
        return False


def _build_ee_covering_grid(fc, scale: float = 1e4):
    """Build an AOO covering grid with grid_col/grid_row indices.

    Args:
        fc: An ee.FeatureCollection whose bounds define the grid extent.
        scale: Grid cell size in meters (default 10 km).

    Returns:
        An ee.FeatureCollection of grid cells with grid_col and grid_row properties.
    """
    import ee

    from rle_python_gee.ee_rle import get_aoo_grid_projection

    aoo_grid_proj = get_aoo_grid_projection(scale)
    cgrid = fc.bounds().coveringGrid(aoo_grid_proj)

    def _add_indices(f):
        parts = ee.String(f.get("system:index")).split(",")
        return f.set(
            "grid_col", ee.Number.parse(parts.get(0)),
            "grid_row", ee.Number.parse(parts.get(1)),
        )

    return cgrid.map(_add_indices)


class AOOGridNotComputedError(Exception):
    """Raised when accessing grid data before compute() has been called."""

    def __init__(self):
        super().__init__(
            "AOO grid has not been computed yet. "
            "Call .compute() to run the computation."
        )


class AOOGrid(ABC):
    """Base class for Area of Occupancy grid computations.

    Provides derived properties (cell count, AOO) and visualization methods.

    Subclasses implement ``_compute()`` to run the computation and store
    results in the appropriate backend, and ``_load_grid_cells()`` to
    download results on demand.
    """

    def __init__(self, ecosystems: Ecosystems):
        self._ecosystems = ecosystems
        self._computed = False
        self.task = None
        self._grid_cells: gpd.GeoDataFrame | None = None

    # -- classmethods ---------------------------------------------------------

    @classmethod
    def from_gee_image(cls, data, **kwargs) -> "AOOGrid":
        """Create an AOO grid from an Earth Engine Image or asset ID."""
        return AOOGridEEImage(EcosystemsEEImage(data), **kwargs)

    @classmethod
    def from_gee_feature_collection(cls, data, *,
                                    ecosystem_column: str,
                                    gee_asset_path: str | None = None,
                                    gcs_path: str | None = None,
                                    **kwargs) -> "AOOGrid":
        """Create an AOO grid from an Earth Engine FeatureCollection or asset ID."""
        eco = EcosystemsEEFeatureCollection(data, ecosystem_column=ecosystem_column)
        return AOOGridEEFeatureCollection(
            eco, gee_asset_path=gee_asset_path, gcs_path=gcs_path, **kwargs
        )

    @classmethod
    def from_parquet(cls, data, *, ecosystem_column: str, **kwargs) -> "AOOGrid":
        """Create an AOO grid from a GeoParquet file."""
        return AOOGridVectorLocal(
            EcosystemsGeoParquet(data, ecosystem_column=ecosystem_column), **kwargs
        )

    @classmethod
    def from_file(cls, data, *, ecosystem_column: str, **kwargs) -> "AOOGrid":
        """Create an AOO grid from a vector file (Shapefile, GeoJSON, etc.)."""
        return AOOGridVectorLocal(
            EcosystemsFile(data, ecosystem_column=ecosystem_column), **kwargs
        )

    @classmethod
    def from_cog(cls, data, **kwargs) -> "AOOGrid":
        """Create an AOO grid from a Cloud Optimized GeoTIFF."""
        return AOOGridCOG(EcosystemsCOG(data), **kwargs)

    # -- computation ---------------------------------------------------------

    @abstractmethod
    def _compute(self) -> None:
        """Run the AOO grid computation and store results in the backend.

        Must not return the result — store it in the backend (EE asset,
        file, or in-memory cache for local backends).
        """
        ...

    @abstractmethod
    def _load_grid_cells(self) -> gpd.GeoDataFrame:
        """Load grid cells from the backend.

        Returns a GeoDataFrame with geometries in EPSG:4326.
        """
        ...

    def compute(self) -> "AOOGrid":
        """Explicitly run the AOO grid computation.

        Results are stored in the backend. Access them via ``grid_cells``.
        Returns self for method chaining.
        """
        self._compute()
        self._computed = True
        self._grid_cells = None  # clear any stale local cache
        return self

    @property
    def grid_cells(self) -> gpd.GeoDataFrame:
        """GeoDataFrame of AOO grid cells. Raises if compute() not called."""
        if not self._computed:
            raise AOOGridNotComputedError()
        if self._grid_cells is None:
            self._grid_cells = self._load_grid_cells().reset_index(drop=True)
        return self._grid_cells

    # -- derived properties --------------------------------------------------

    @property
    def cell_count(self) -> int:
        """Total number of grid cells that intersect the ecosystem."""
        return len(self.grid_cells)

    @property
    def aoo_km2(self) -> float:
        """AOO in km² (cell count × 100 km² per cell)."""
        return self.cell_count * (AOO_CELL_SIZE_M / 1000) ** 2

    # -- export / write -------------------------------------------------------

    def to_ee_feature_collection(self, asset_id: str, *,
                                  gcs_bucket: str | None = None):
        """Export grid cells to an Earth Engine table asset.

        Returns the task/result, or None if asset already exists.
        """
        from rle_python_gee.ecosystems import _upload_gdf_to_ee_asset

        return _upload_gdf_to_ee_asset(
            self.grid_cells, asset_id,
            gcs_bucket=gcs_bucket, description="aoo_grid_export"
        )

    def to_parquet(self, path) -> None:
        """Write grid cells as a GeoParquet file."""
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.grid_cells.to_parquet(path)

    # -- filtering -----------------------------------------------------------

    def filter_by_ecosystem(self, ecosystem_name: str,
                            threshold: float = 0.0) -> "FilteredAOOGrid":
        """Return a filtered AOOGrid with only cells where the ecosystem
        fraction exceeds *threshold*.

        Args:
            ecosystem_name: Ecosystem name as it appears in the source data.
            threshold: Minimum fractional area (0.0–1.0). Default 0.0 means
                any presence.
        """
        col = slugify_ecosystem_name(ecosystem_name)
        if col not in self.grid_cells.columns:
            skip = {"grid_col", "grid_row", "count_geoms", "count_ecosystems", "geometry"}
            available = [c for c in self.grid_cells.columns if c not in skip]
            raise ValueError(
                f"Ecosystem column '{col}' not found. "
                f"Available: {available}"
            )
        mask = self.grid_cells[col] > threshold
        return FilteredAOOGrid(self, mask)

    # -- visualization -------------------------------------------------------

    def to_layer(self, *, get_fill_color=None, get_line_color=None):
        """Return a lonboard PolygonLayer of AOO grid cells."""
        try:
            from lonboard import PolygonLayer
        except ImportError:
            raise ImportError(
                "lonboard is required for visualization. "
                "Install it with: pip install lonboard"
            ) from None

        if get_fill_color is None:
            get_fill_color = [128, 128, 128, 128]
        if get_line_color is None:
            get_line_color = [0, 0, 0, 255]

        if self.grid_cells.empty:
            return []
        return [PolygonLayer.from_geopandas(
            self.grid_cells,
            get_fill_color=get_fill_color,
            get_line_color=get_line_color,
            line_width_min_pixels=1,
        )]

    def to_map(self, *, get_fill_color=None, get_line_color=None, **kwargs):
        """Return a lonboard Map showing the AOO grid cells."""
        try:
            from lonboard import Map
        except ImportError:
            raise ImportError(
                "lonboard is required for visualization. "
                "Install it with: pip install lonboard"
            ) from None

        layer_kwargs = {}
        if get_fill_color is not None:
            layer_kwargs["get_fill_color"] = get_fill_color
        if get_line_color is not None:
            layer_kwargs["get_line_color"] = get_line_color
        layers = self.to_layer(**layer_kwargs)
        return Map(layers=layers, **kwargs)

    # -- display -------------------------------------------------------------

    def __repr__(self) -> str:
        if not self._computed:
            return f"{type(self).__name__}(not computed)"
        try:
            return (
                f"{type(self).__name__}("
                f"cell_count={self.cell_count}, "
                f"aoo_km2={self.aoo_km2:.0f})"
            )
        except RuntimeError:
            return f"{type(self).__name__}(computed, results pending)"

    def _repr_html_(self) -> str:
        if not self._computed:
            return (
                f"<b>{type(self).__name__}</b><br>"
                f"<i>Not computed — call .compute() to run</i>"
            )
        try:
            return (
                f"<b>{type(self).__name__}</b><br>"
                f"Grid cells: {self.cell_count}<br>"
                f"AOO: {self.aoo_km2:,.0f} km²"
            )
        except RuntimeError:
            return (
                f"<b>{type(self).__name__}</b><br>"
                f"<i>Export task running — check status at "
                f"<a href='https://code.earthengine.google.com/tasks'>EE Tasks</a></i>"
            )


# ---------------------------------------------------------------------------
# Filtered view
# ---------------------------------------------------------------------------


class FilteredAOOGrid(AOOGrid):
    """A filtered view of an AOOGrid, showing only cells matching a predicate."""

    def __init__(self, source: AOOGrid, mask):
        self._source = source
        self._mask = mask
        self._computed = source._computed
        self._grid_cells = None
        self._ecosystems = source._ecosystems

    def _compute(self):
        raise RuntimeError(
            "Cannot compute a filtered grid — compute the source grid first."
        )

    def _load_grid_cells(self):
        return self._source.grid_cells[self._mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Earth Engine Image backend
# ---------------------------------------------------------------------------


class AOOGridEEImage(AOOGrid):
    """AOO grid from an Earth Engine Image (fractional coverage or binary)."""

    def _compute(self) -> None:
        import ee

        from rle_python_gee.ee_rle import get_aoo_grid_projection

        image = self._ecosystems.load()
        aoo_grid_proj = get_aoo_grid_projection()

        fc = image.unmask().reduceRegions(
            collection=image.geometry().coveringGrid(aoo_grid_proj),
            reducer=ee.Reducer.mean(),
        ).filter(ee.Filter.gt("mean", 0))

        # For EE Image, store the result in memory (no persistent backend)
        gdf = ee.data.computeFeatures({
            "expression": fc,
            "fileFormat": "GEOPANDAS_GEODATAFRAME",
        })
        self._computed_gdf = gdf.set_crs("EPSG:4326")

    def _load_grid_cells(self) -> gpd.GeoDataFrame:
        return self._computed_gdf


# ---------------------------------------------------------------------------
# Earth Engine FeatureCollection backend
# ---------------------------------------------------------------------------


class AOOGridEEFeatureCollection(AOOGrid):
    """AOO grid from an Earth Engine FeatureCollection.

    Uses ``ee.Join.saveAll`` to find grid cells that intersect ecosystem
    features.  Results can be exported to an EE asset and/or GCS parquet.
    """

    def __init__(self, ecosystems: EcosystemsEEFeatureCollection, *,
                 gee_asset_path: str,
                 gcs_path: str | None = None):
        super().__init__(ecosystems)
        self._gee_asset_path = gee_asset_path
        self._gcs_path = gcs_path

    @property
    def _intersections_id(self) -> str:
        from pathlib import PurePosixPath
        return str(PurePosixPath(self._gee_asset_path) / 'aoo_grid')

    @property
    def _gcs_parquet_path(self) -> str | None:
        """GCS path for the parquet file (written on first load from EE asset)."""
        if self._gcs_path is None:
            return None
        return self._gcs_path.rstrip('/') + '/aoo_grid.parquet'

    def _compute(self) -> None:
        import ee

        intersections_id = self._intersections_id

        # --- check if already computed (EE asset) ---
        logger.info("Checking for cached asset: %s", intersections_id)
        try:
            ee.data.getAsset(intersections_id)
            logger.info("Found cached asset: %s", intersections_id)
            return
        except ee.EEException:
            logger.info("No cached asset found, computing from scratch")

        # --- resolve input FC ---
        fc = self._ecosystems.load()

        # --- build AOO covering grid ---
        cgrid = _build_ee_covering_grid(fc)

        # Find all grid cells that intersect with the ecosystem features
        spatial_filter = ee.Filter.intersects(leftField='.geo', rightField='.geo')

        matches_key = 'matches'
        join = ee.Join.saveAll(matchesKey=matches_key)
        joined = join.apply(cgrid, fc, spatial_filter)

        ecosystem_column = self._ecosystems.ecosystem_column
        fc_grid_intersects = joined.map(
            lambda join_feat: ee.Feature(join_feat.geometry())
                .copyProperties(join_feat, exclude=['matches'])
                .set('count_geoms', ee.List(join_feat.get(matches_key)).length())
                .set('count_ecosystems', ee.FeatureCollection(
                        ee.List(join_feat.get(matches_key))
                    ).aggregate_count_distinct(ecosystem_column)
                )
        )

        # --- export to EE asset ---
        task = ee.batch.Export.table.toAsset(
            collection=fc_grid_intersects,
            description="AOOGridEEFeatureCollection_grid_intersections",
            assetId=intersections_id,
        )
        task.start()
        self.task = task
        logger.info("EE export task started (task ID: %s)", task.id)

    def _load_grid_cells(self) -> gpd.GeoDataFrame:
        # --- try GCS parquet first ---
        gcs_parquet_path = self._gcs_parquet_path
        if gcs_parquet_path is not None and _remote_file_exists(gcs_parquet_path):
            gdf = gpd.read_parquet(gcs_parquet_path)
            logger.info("Loaded grid cells from parquet: %s", gcs_parquet_path)
            return gdf

        # --- download from EE asset ---
        import ee

        intersections_id = self._intersections_id
        logger.info("Downloading grid cells from: %s", intersections_id)
        try:
            ee.data.getAsset(intersections_id)
        except ee.EEException:
            raise RuntimeError(
                f"Export asset not ready: {intersections_id}\n"
                f"The export task may still be running. "
                f"Check task status at https://code.earthengine.google.com/tasks "
                f"or via ee.data.listOperations(). "
                f"Once complete, access .grid_cells again."
            ) from None
        cached = ee.FeatureCollection(intersections_id)
        gdf = ee.data.computeFeatures({
            "expression": cached,
            "fileFormat": "GEOPANDAS_GEODATAFRAME",
        })
        gdf = gdf.set_crs("EPSG:4326")

        # --- write parquet to GCS if configured ---
        if gcs_parquet_path is not None:
            try:
                gdf.to_parquet(gcs_parquet_path)
                logger.info("GeoParquet written to: %s", gcs_parquet_path)
            except Exception:
                logger.warning("Failed to write GeoParquet to: %s", gcs_parquet_path, exc_info=True)

        return gdf

    def to_layer(self, *, get_fill_color=None, get_line_color=None):
        """Return layers rendering the grid cells.

        Uses EE tiles when a gee_asset_path is available, otherwise falls
        back to the base-class lonboard PolygonLayer from the GeoDataFrame.
        """
        if not self._computed:
            raise AOOGridNotComputedError()

        intersections_id = self._intersections_id
        if intersections_id is None:
            # No EE asset — fall back to base class (GeoDataFrame-based)
            return super().to_layer()

        try:
            from lonboard import BitmapTileLayer
        except ImportError:
            raise ImportError(
                "lonboard is required for visualization. "
                "Install it with: pip install lonboard"
            ) from None

        import ee

        try:
            ee.data.getAsset(intersections_id)
        except ee.EEException:
            raise RuntimeError(
                f"Export asset not ready: {intersections_id}\n"
                f"The export task may still be running. "
                f"Check task status at https://code.earthengine.google.com/tasks "
                f"or via ee.data.listOperations(). "
                f"Once complete, call .to_map() again."
            ) from None
        fc = ee.FeatureCollection(intersections_id)

        # YlOrRd-inspired sequential palette for count_ecosystems
        palette = [
            'FFFFB2',  # 1
            'FECC5C',  # 2
            'FD8D3C',  # 3
            'F03B20',  # 4
            'BD0026',  # 5
            'A00026',  # 6
            '800026',  # 7
            '660020',  # 8
            '4D001A',  # 9
            '330014',  # 10+
        ]
        palette_list = ee.List(palette)
        max_index = palette_list.size().subtract(1)

        def _add_style(feature):
            count = ee.Number(feature.get('count_ecosystems'))
            idx = count.subtract(1).max(0).min(max_index).toInt()
            fill_color = ee.String(palette_list.get(idx)).cat('C0')
            return feature.set('style', {
                'color': '000000',
                'width': 1,
                'fillColor': fill_color,
            })

        styled_fc = fc.map(_add_style)
        styled = styled_fc.style(styleProperty='style')
        map_id = styled.getMapId()
        tile_url = map_id['tile_fetcher'].url_format
        return [BitmapTileLayer(data=tile_url)]

    def to_polygons(self, **kwargs) -> "AOOGridPolygonEEFeatureCollection":
        """Create an AOOGridPolygonEEFeatureCollection from this grid."""
        return AOOGridPolygonEEFeatureCollection(self, **kwargs)


# ---------------------------------------------------------------------------
# Local vector backend (GeoJSON, GeoParquet)
# ---------------------------------------------------------------------------


class AOOGridVectorLocal(AOOGrid):
    """AOO grid from a local vector dataset (GeoJSON or GeoParquet)."""

    def _compute(self) -> None:
        import pandas as pd
        from rle_python_gee.aoo_grid import generate_aoo_grid, AOO_CRS, AOO_CELL_SIZE

        eco = self._ecosystems.load().reset_index(drop=True)
        if eco.crs is not None and not eco.crs.equals("EPSG:4326"):
            eco = eco.to_crs("EPSG:4326")
        grid = generate_aoo_grid(eco.total_bounds)

        eco_col = self._ecosystems.ecosystem_column

        # Spatial join to find (grid_cell, ecosystem) pairs
        joined = gpd.sjoin(grid, eco, how="inner", predicate="intersects")
        if joined.empty:
            self._computed_gdf = gpd.GeoDataFrame(
                columns=["grid_col", "grid_row", "count_geoms",
                          "count_ecosystems", "geometry"]
            )
            return

        # Compute intersection areas in equal-area CRS
        grid_ea = grid.to_crs(AOO_CRS)
        eco_ea = eco.to_crs(AOO_CRS)
        cell_area = AOO_CELL_SIZE * AOO_CELL_SIZE

        rows = []
        for grid_idx, eco_idx in zip(joined.index, joined["index_right"]):
            isect = grid_ea.geometry.iloc[grid_idx].intersection(
                eco_ea.geometry.iloc[eco_idx]
            )
            if isect.is_empty:
                continue
            fraction = isect.area / cell_area
            row = {"grid_idx": grid_idx, "eco_idx": eco_idx, "fraction": fraction}
            if eco_col is not None:
                row["ecosystem"] = eco.iloc[eco_idx][eco_col]
            rows.append(row)

        if not rows:
            self._computed_gdf = gpd.GeoDataFrame(
                columns=["grid_col", "grid_row", "count_geoms",
                          "count_ecosystems", "geometry"]
            )
            return

        fractions = pd.DataFrame(rows)

        # Summary counts per grid cell
        summary = fractions.groupby("grid_idx").agg(
            count_geoms=("eco_idx", "count"),
            count_ecosystems=("ecosystem", "nunique") if eco_col else ("eco_idx", "count"),
        )

        # Build result with grid geometry
        result = grid.loc[summary.index].copy()
        result["count_geoms"] = summary["count_geoms"].values
        result["count_ecosystems"] = summary["count_ecosystems"].values

        # Wide columns: fractional area per ecosystem
        if eco_col is not None:
            pivot = fractions.pivot_table(
                index="grid_idx", columns="ecosystem",
                values="fraction", aggfunc="sum", fill_value=0.0,
            )
            # Sanitize ecosystem names for use as column names
            pivot.columns = [
                slugify_ecosystem_name(c)
                for c in pivot.columns
            ]
            # Ensure all intersecting grid cells have all ecosystem columns (fill 0)
            result = result.join(pivot)
            result[pivot.columns] = result[pivot.columns].fillna(0.0)

        self._computed_gdf = result.reset_index(drop=True)

    def _load_grid_cells(self) -> gpd.GeoDataFrame:
        return self._computed_gdf

    def to_polygons(self, **kwargs) -> "AOOGridPolygonVectorLocal":
        """Create intersection polygons for this local vector grid."""
        return AOOGridPolygonVectorLocal(self, **kwargs)


# Backward-compatibility aliases
AOOGridGeoParquet = AOOGridVectorLocal
AOOGridGeoJSON = AOOGridVectorLocal


# ---------------------------------------------------------------------------
# COG (Cloud Optimized GeoTIFF) backend
# ---------------------------------------------------------------------------


class AOOGridCOG(AOOGrid):
    """AOO grid from a Cloud Optimized GeoTIFF."""

    def _compute(self) -> None:
        from rasterstats import zonal_stats

        from rle_python_gee.aoo_grid import generate_aoo_grid

        rds = self._ecosystems.load()
        # Get bounds in geographic coords
        bounds = rds.rio.transform_bounds("EPSG:4326")
        grid = generate_aoo_grid(bounds)

        # Reproject raster to equal-area for zonal stats
        rds_ea = rds.rio.reproject("ESRI:54034")
        grid_ea = grid.to_crs("ESRI:54034")

        stats = zonal_stats(
            grid_ea.geometry,
            rds_ea.values[0],
            affine=rds_ea.rio.transform(),
            stats=["mean"],
            nodata=rds_ea.rio.nodata,
        )
        # Keep only cells with non-zero values
        has_data = [bool(s["mean"]) for s in stats]
        result = grid_ea[has_data]

        self._computed_gdf = result[["geometry"]].to_crs("EPSG:4326").reset_index(drop=True)

    def _load_grid_cells(self) -> gpd.GeoDataFrame:
        return self._computed_gdf


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def make_aoo_grid(data, **kwargs) -> AOOGrid:
    """Create an AOO grid from the given data source.

    Auto-detects the data type and returns the appropriate AOOGrid subclass.
    Call ``.compute()`` to run the computation before accessing results.

    Args:
        data: Data source. One of:
            - Ecosystems instance
            - Earth Engine asset ID (str starting with 'projects/')
            - ee.Image object
            - ee.FeatureCollection object
            - Path to a GeoParquet file (.parquet)
            - Path to a GeoJSON file (.geojson)
            - Path to a COG file (.tif, .tiff)
        **kwargs: Additional arguments passed to the backend constructor.

    Returns:
        An AOOGrid instance. Call .compute() to run the computation.

    Example:
        >>> aoo = make_aoo_grid("path/to/data.geojson").compute()
        >>> print(aoo.cell_count)
        >>> aoo.to_map()
    """
    # Ecosystems instance — dispatch by kind
    if isinstance(data, Ecosystems):
        kind = data.kind
        if kind == EcosystemKind.VECTOR_LOCAL:
            return AOOGridVectorLocal(data, **kwargs)
        if kind == EcosystemKind.RASTER_LOCAL:
            return AOOGridCOG(data, **kwargs)
        if kind == EcosystemKind.EE_IMAGE:
            return AOOGridEEImage(data, **kwargs)
        if kind == EcosystemKind.EE_FEATURE_COLLECTION:
            return AOOGridEEFeatureCollection(data, **kwargs)

    # Legacy: auto-detect from raw data
    # Split kwargs: ecosystem_column goes to make_ecosystems,
    # asset_path/synchronous go to AOOGrid constructor
    aoo_only_keys = ('gee_asset_path', 'gcs_path')
    eco_kwargs = {k: v for k, v in kwargs.items() if k not in aoo_only_keys}
    aoo_kwargs = {k: v for k, v in kwargs.items() if k not in ('ecosystem_column',)}
    eco = make_ecosystems(data, **eco_kwargs)
    return make_aoo_grid(eco, **aoo_kwargs)


# ---------------------------------------------------------------------------
# AOO Grid Polygons — intersection geometries
# ---------------------------------------------------------------------------


class AOOGridPolygonsNotComputedError(Exception):
    """Raised when accessing polygon data before compute() has been called."""

    def __init__(self):
        super().__init__(
            "AOO grid polygons have not been computed yet. "
            "Call .compute() to run the computation."
        )


class AOOGridPolygons(ABC):
    """Base class for AOO grid × ecosystem intersection polygons.

    Each row represents the geometric intersection of one grid cell with
    one ecosystem polygon.  Subclasses implement ``_compute()`` to produce
    the polygons and ``_load_polygons()`` to retrieve them.
    """

    def __init__(self, aoo_grid: AOOGrid):
        self._aoo_grid = aoo_grid
        self._computed = False
        self._polygons: gpd.GeoDataFrame | None = None
        self.task = None

    # -- abstract interface --------------------------------------------------

    @abstractmethod
    def _compute(self) -> None:
        """Run the intersection computation and store results in backend."""

    @abstractmethod
    def _load_polygons(self) -> gpd.GeoDataFrame:
        """Load the computed intersection polygons as a GeoDataFrame."""

    # -- public API ----------------------------------------------------------

    def compute(self) -> "AOOGridPolygons":
        """Run the intersection computation. Returns *self* for chaining."""
        self._compute()
        self._computed = True
        self._polygons = None  # clear cache
        return self

    @property
    def polygons(self) -> gpd.GeoDataFrame:
        """The intersection polygons as a GeoDataFrame."""
        if not self._computed:
            raise AOOGridPolygonsNotComputedError()
        if self._polygons is None:
            self._polygons = self._load_polygons().reset_index(drop=True)
        return self._polygons

    @property
    def polygon_count(self) -> int:
        """Number of (grid cell × ecosystem) intersection polygons."""
        return len(self.polygons)

    def filter_by_ecosystem(self, ecosystem_name: str) -> "FilteredAOOGridPolygons":
        """Return a filtered view with only polygons for the given ecosystem."""
        eco_col = self._aoo_grid._ecosystems.ecosystem_column
        if eco_col is None:
            raise ValueError("ecosystem_column is not set on the source data")
        mask = self.polygons[eco_col] == ecosystem_name
        return FilteredAOOGridPolygons(self, mask)

    # -- display -------------------------------------------------------------

    def __repr__(self) -> str:
        if not self._computed:
            return f"{type(self).__name__}(not computed)"
        try:
            return f"{type(self).__name__}(polygons={self.polygon_count})"
        except RuntimeError:
            return f"{type(self).__name__}(computed, results pending)"

    def _repr_html_(self) -> str:
        if not self._computed:
            return (
                f"<b>{type(self).__name__}</b><br>"
                f"<i>Not computed — call .compute() to run</i>"
            )
        try:
            count = self.polygon_count
        except Exception:
            return (
                f"<b>{type(self).__name__}</b><br>"
                f"<i>Computed — polygons not yet available (export may be running)</i>"
            )
        return (
            f"<b>{type(self).__name__}</b><br>"
            f"Polygons: {count:,}"
        )

    # -- export / write -------------------------------------------------------

    def to_ee_feature_collection(self, asset_id: str, *,
                                  gcs_bucket: str | None = None):
        """Export intersection polygons to an Earth Engine table asset.

        Returns the task/result, or None if asset already exists.
        """
        from rle_python_gee.ecosystems import _upload_gdf_to_ee_asset

        return _upload_gdf_to_ee_asset(
            self.polygons, asset_id,
            gcs_bucket=gcs_bucket, description="aoo_grid_polygons_export"
        )

    def to_parquet(self, path) -> None:
        """Write intersection polygons as a GeoParquet file."""
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.polygons.to_parquet(path)

    # -- visualization -------------------------------------------------------

    def to_layer(self, *, get_fill_color=None, get_line_color=None):
        """Return lonboard layer(s) for the intersection polygons."""
        if not self._computed:
            raise AOOGridPolygonsNotComputedError()
        try:
            from lonboard import PolygonLayer
        except ImportError:
            raise ImportError(
                "lonboard is required for visualization. "
                "Install it with: pip install lonboard"
            ) from None

        if get_fill_color is None:
            get_fill_color = [0, 128, 255, 128]
        if get_line_color is None:
            get_line_color = [0, 0, 0, 255]

        gdf = self.polygons
        if gdf.empty:
            return []
        if len(gdf) > 1000:
            raise ValueError(
                f"Dataset has {len(gdf):,} polygons, which is too many to "
                f"display interactively. Export to parquet with "
                f".polygons.to_parquet(path) instead."
            )
        return [PolygonLayer.from_geopandas(
            gdf,
            get_fill_color=get_fill_color,
            get_line_color=get_line_color,
            line_width_min_pixels=1,
        )]

    def to_map(self, *, get_fill_color=None, get_line_color=None, **kwargs):
        """Return a lonboard Map of the intersection polygons."""
        try:
            from lonboard import Map
        except ImportError:
            raise ImportError(
                "lonboard is required for visualization. "
                "Install it with: pip install lonboard"
            ) from None

        layer_kwargs = {}
        if get_fill_color is not None:
            layer_kwargs["get_fill_color"] = get_fill_color
        if get_line_color is not None:
            layer_kwargs["get_line_color"] = get_line_color
        try:
            layers = self.to_layer(**layer_kwargs)
        except ValueError as e:
            from IPython.display import HTML, display
            display(HTML(f"<div style='padding:12px;background:#fff3cd;border:1px solid #ffc107;border-radius:4px'>"
                         f"<b>Cannot display map:</b> {e}</div>"))
            return None
        return Map(layers=layers, **kwargs)


class FilteredAOOGridPolygons(AOOGridPolygons):
    """A filtered view of AOOGridPolygons, showing only polygons matching a predicate."""

    def __init__(self, source: AOOGridPolygons, mask):
        self._source = source
        self._mask = mask
        self._computed = source._computed
        self._polygons = None
        self._aoo_grid = source._aoo_grid

    def _compute(self):
        raise RuntimeError(
            "Cannot compute a filtered polygon set — compute the source first."
        )

    def _load_polygons(self):
        return self._source.polygons[self._mask].reset_index(drop=True)


class AOOGridPolygonEEFeatureCollection(AOOGridPolygons):
    """Intersection polygons computed via Earth Engine.

    Uses EE server-side spatial join and ``geometry().intersection()``
    to compute the actual intersection geometry for each
    (grid cell, ecosystem) pair.
    """

    def __init__(self, aoo_grid: AOOGridEEFeatureCollection, *,
                 gee_asset_path: str | None = None,
                 gcs_path: str | None = None):
        if not isinstance(aoo_grid, AOOGridEEFeatureCollection):
            raise TypeError(
                "AOOGridPolygonEEFeatureCollection requires an "
                "AOOGridEEFeatureCollection instance"
            )
        super().__init__(aoo_grid)
        self._gee_asset_path = gee_asset_path or aoo_grid._gee_asset_path
        self._gcs_path = gcs_path if gcs_path is not None else aoo_grid._gcs_path

    @property
    def _polygons_id(self) -> str:
        from pathlib import PurePosixPath
        return str(PurePosixPath(self._gee_asset_path) / 'aoo_grid_polygons')

    @property
    def _gcs_parquet_path(self) -> str | None:
        if self._gcs_path is None:
            return None
        return self._gcs_path.rstrip('/') + '/aoo_grid_polygons.parquet'

    def __repr__(self) -> str:
        if not self._computed:
            return f"{type(self).__name__}(not computed)"
        task = getattr(self, "task", None)
        if task is not None:
            import ee

            state = ee.data.getTaskStatus(task.id)[0]["state"]
            if state != "COMPLETED":
                if state in ("RUNNING", "READY", "PENDING"):
                    return (
                        f"{type(self).__name__}"
                        f"(export task still running, id={task.id})"
                    )
                return (
                    f"{type(self).__name__}"
                    f"(export task {state.lower()}, id={task.id})"
                )
        try:
            return f"{type(self).__name__}(polygons={self.polygon_count})"
        except RuntimeError:
            return f"{type(self).__name__}(computed, results pending)"

    def _compute(self) -> None:
        import ee

        polygons_id = self._polygons_id

        # --- check if already computed (EE asset) ---
        logger.info("Checking for cached polygons asset: %s", polygons_id)
        try:
            ee.data.getAsset(polygons_id)
            logger.info("Found cached polygons asset: %s", polygons_id)
            return
        except ee.EEException:
            logger.info("No cached polygons asset, computing from scratch")

        # --- resolve input FC ---
        fc = self._aoo_grid._ecosystems.load()

        # --- build covering grid and spatial join ---
        cgrid = _build_ee_covering_grid(fc)

        spatial_filter = ee.Filter.intersects(
            leftField='.geo', rightField='.geo'
        )
        matches_key = 'matches'
        join = ee.Join.saveAll(matchesKey=matches_key)
        joined = join.apply(cgrid, fc, spatial_filter)

        ecosystem_column = self._aoo_grid._ecosystems.ecosystem_column

        # --- flatten and compute intersection geometries ---
        def _flatten_and_intersect(grid_feat):
            grid_geom = grid_feat.geometry()
            grid_col = grid_feat.get('grid_col')
            grid_row = grid_feat.get('grid_row')
            matches = ee.List(grid_feat.get(matches_key))

            def _intersect_one(eco_feat_raw):
                eco_feat = ee.Feature(eco_feat_raw)
                intersection = grid_geom.intersection(eco_feat.geometry(), 1)
                return ee.Feature(intersection).set({
                    'grid_col': grid_col,
                    'grid_row': grid_row,
                    ecosystem_column: eco_feat.get(ecosystem_column),
                })

            return ee.FeatureCollection(matches.map(_intersect_one))

        polygons_fc = joined.map(_flatten_and_intersect).flatten()

        # --- export to EE asset ---
        task = ee.batch.Export.table.toAsset(
            collection=polygons_fc,
            description="AOOGridPolygonEEFeatureCollection_grid_polygons",
            assetId=polygons_id,
        )
        task.start()
        self.task = task
        logger.info("Polygons EE export task started (task ID: %s)", task.id)

    def _load_polygons(self) -> gpd.GeoDataFrame:
        # --- try GCS parquet first ---
        gcs_parquet_path = self._gcs_parquet_path
        if gcs_parquet_path is not None and _remote_file_exists(gcs_parquet_path):
            gdf = gpd.read_parquet(gcs_parquet_path)
            logger.info("Loaded polygons from parquet: %s", gcs_parquet_path)
            return gdf

        # --- download from EE asset ---
        import ee

        polygons_id = self._polygons_id
        logger.info("Downloading polygons from: %s", polygons_id)
        try:
            ee.data.getAsset(polygons_id)
        except ee.EEException:
            raise RuntimeError(
                f"Export asset not ready: {polygons_id}\n"
                f"The export task may still be running. "
                f"Check task status at https://code.earthengine.google.com/tasks "
                f"or via ee.data.listOperations(). "
                f"Once complete, access .polygons again."
            ) from None
        cached = ee.FeatureCollection(polygons_id)
        gdf = ee.data.computeFeatures({
            "expression": cached,
            "fileFormat": "GEOPANDAS_GEODATAFRAME",
        })
        gdf = gdf.set_crs("EPSG:4326")

        # --- write parquet to GCS if configured ---
        if gcs_parquet_path is not None:
            try:
                gdf.to_parquet(gcs_parquet_path)
                logger.info("GeoParquet written to: %s", gcs_parquet_path)
            except Exception:
                logger.warning("Failed to write GeoParquet to: %s", gcs_parquet_path, exc_info=True)

        return gdf

    def to_layer(self):
        """Return a BitmapTileLayer rendering intersection polygons via EE tiles."""
        if not self._computed:
            raise AOOGridPolygonsNotComputedError()
        try:
            from lonboard import BitmapTileLayer
        except ImportError:
            raise ImportError(
                "lonboard is required for visualization. "
                "Install it with: pip install lonboard"
            ) from None

        import ee

        polygons_id = self._polygons_id
        try:
            ee.data.getAsset(polygons_id)
        except ee.EEException:
            raise RuntimeError(
                f"Export asset not ready: {polygons_id}\n"
                f"The export task may still be running. "
                f"Check task status at https://code.earthengine.google.com/tasks "
                f"or via ee.data.listOperations(). "
                f"Once complete, call .to_map() again."
            ) from None

        fc = ee.FeatureCollection(polygons_id)
        styled = fc.style(color='000000', fillColor='0080FF80')
        map_id = styled.getMapId()
        tile_url = map_id['tile_fetcher'].url_format
        return [BitmapTileLayer(data=tile_url)]


class AOOGridPolygonVectorLocal(AOOGridPolygons):
    """Intersection polygons computed locally via shapely.

    Intersects each grid cell with ecosystem features one at a time
    to keep memory usage bounded.
    """

    def _compute(self) -> None:
        import math
        import pandas as pd
        from shapely.geometry import box
        from rle_python_gee.aoo_grid import AOO_CRS, AOO_CELL_SIZE

        eco = self._aoo_grid._ecosystems.load()
        if eco.crs is None:
            eco = eco.set_crs("EPSG:4326")
        eco_cea = eco.to_crs(AOO_CRS)

        bounds_cea = eco_cea.total_bounds
        sindex = eco_cea.sindex

        col_min = math.floor(bounds_cea[0] / AOO_CELL_SIZE)
        col_max = math.ceil(bounds_cea[2] / AOO_CELL_SIZE)
        row_min = math.floor(bounds_cea[1] / AOO_CELL_SIZE)
        row_max = math.ceil(bounds_cea[3] / AOO_CELL_SIZE)

        chunks = []
        for col in range(col_min, col_max):
            for row in range(row_min, row_max):
                x0 = col * AOO_CELL_SIZE
                y0 = row * AOO_CELL_SIZE
                cell = box(x0, y0, x0 + AOO_CELL_SIZE, y0 + AOO_CELL_SIZE)

                candidates = list(sindex.query(cell))
                if not candidates:
                    continue

                subset = eco_cea.iloc[candidates]
                intersections = subset.intersection(cell)
                mask = ~intersections.is_empty
                if not mask.any():
                    continue

                result = subset.loc[mask].copy()
                result["geometry"] = intersections[mask]
                result["grid_col"] = col
                result["grid_row"] = row
                chunks.append(result)

        if chunks:
            self._computed_gdf = gpd.GeoDataFrame(
                pd.concat(chunks, ignore_index=True), crs=AOO_CRS
            )
        else:
            self._computed_gdf = gpd.GeoDataFrame(
                columns=["geometry", "grid_col", "grid_row"]
            )

    def _load_polygons(self) -> gpd.GeoDataFrame:
        return self._computed_gdf


def make_aoo_polygons(aoo_grid: AOOGrid, **kwargs) -> AOOGridPolygons:
    """Create AOO grid polygons from an AOOGrid instance.

    Args:
        aoo_grid: A computed AOOGrid instance.
        **kwargs: Additional arguments passed to the backend constructor.

    Returns:
        An AOOGridPolygons instance. Call .compute() to run.
    """
    if isinstance(aoo_grid, AOOGridEEFeatureCollection):
        return AOOGridPolygonEEFeatureCollection(aoo_grid, **kwargs)
    if isinstance(aoo_grid, AOOGridVectorLocal):
        return AOOGridPolygonVectorLocal(aoo_grid, **kwargs)
    raise ValueError(
        f"AOOGridPolygons not supported for {type(aoo_grid).__name__}"
    )
