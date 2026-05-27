"""Earth Engine Area of Occupancy (AOO) backends for RLE assessments.

These subclass the core ``rle.core.aoo`` ABCs and compute AOO grids /
intersection polygons server-side in Earth Engine. Construct them explicitly,
e.g.::

    from rle.gee import GeeEcosystems, AOOGridEEFeatureCollection
    eco = GeeEcosystems(asset_id, ecosystem_column="ECO_NAME")
    aoo = AOOGridEEFeatureCollection(eco, gee_asset_path="projects/p/assets/run1").compute()
"""

import logging
import time

import geopandas as gpd

from rle.core.aoo import (
    AOOGrid,
    AOOGridNotComputedError,
    AOOGridPolygons,
    AOOGridPolygonsNotComputedError,
    _remote_file_exists,
)
from rle.gee.ecosystems import EcosystemsEEFeatureCollection

logger = logging.getLogger(__name__)


def wait_for_task(task, *, poll_interval: int = 15) -> None:
    """Poll an EE export task until it completes, fails, or is cancelled.

    Parameters
    ----------
    task : ee.batch.Task
        The task returned by ``ee.batch.Export.table.*``.
    poll_interval : int
        Seconds between status checks (default 15).
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


def _build_ee_covering_grid(fc, scale: float = 1e4):
    """Build an AOO covering grid with grid_col/grid_row indices.

    Args:
        fc: An ee.FeatureCollection whose bounds define the grid extent.
        scale: Grid cell size in meters (default 10 km).

    Returns:
        An ee.FeatureCollection of grid cells with grid_col and grid_row properties.
    """
    import ee

    from rle.gee.ee_rle import get_aoo_grid_projection

    aoo_grid_proj = get_aoo_grid_projection(scale)
    cgrid = fc.bounds().coveringGrid(aoo_grid_proj)

    def _add_indices(f):
        parts = ee.String(f.get("system:index")).split(",")
        return f.set(
            "grid_col", ee.Number.parse(parts.get(0)),
            "grid_row", ee.Number.parse(parts.get(1)),
        )

    return cgrid.map(_add_indices)


# ---------------------------------------------------------------------------
# Earth Engine Image backend
# ---------------------------------------------------------------------------


class AOOGridEEImage(AOOGrid):
    """AOO grid from an Earth Engine Image (fractional coverage or binary)."""

    def _compute(self) -> None:
        import ee

        from rle.gee.ee_rle import get_aoo_grid_projection

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
# Earth Engine intersection polygons
# ---------------------------------------------------------------------------


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
