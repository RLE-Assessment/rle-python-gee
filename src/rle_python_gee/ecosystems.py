"""Ecosystem distribution data sources for RLE assessments.

Provides the Ecosystems class hierarchy and make_ecosystems() factory
for loading ecosystem data from multiple backends (GeoJSON, GeoParquet,
Earth Engine FeatureCollections, Earth Engine Images, COGs).
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class EcosystemKind(Enum):
    VECTOR_LOCAL = "vector_local"
    RASTER_LOCAL = "raster_local"
    EE_FEATURE_COLLECTION = "ee_fc"
    EE_IMAGE = "ee_image"


def _geodataframe_to_ee_fc(gdf):
    """Convert a GeoDataFrame to an ee.FeatureCollection."""
    import json

    import ee

    geojson = json.loads(gdf.to_json())
    return ee.FeatureCollection(geojson)


class Ecosystems(ABC):
    """Base class for ecosystem distribution datasets."""

    def __init__(self, data):
        self._data = data
        self._cached = None

    @property
    @abstractmethod
    def kind(self) -> EcosystemKind: ...

    @abstractmethod
    def _load(self) -> Any: ...

    def load(self) -> Any:
        """Load and cache the ecosystem data. Returns the native object."""
        if self._cached is None:
            self._cached = self._load()
        return self._cached

    def _feature_count(self) -> int | None:
        """Return the number of features, or None if not applicable."""
        if hasattr(self._cached, '__len__'):
            return len(self._cached)
        return None

    # -- export / write -------------------------------------------------------

    def to_geodataframe(self) -> "gpd.GeoDataFrame":
        """Convert to a GeoDataFrame.

        For vector local backends, returns the loaded GeoDataFrame directly.
        For EE FeatureCollection, downloads via the EE API.
        """
        import geopandas as gpd

        if self.kind == EcosystemKind.VECTOR_LOCAL:
            return self.load()
        raise NotImplementedError(
            f"to_geodataframe not supported for {self.kind.value}"
        )

    def to_parquet(self, path) -> None:
        """Write ecosystem data as a GeoParquet file."""
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        gdf = self.to_geodataframe()
        gdf.to_parquet(path)

    def to_geojson(self, path) -> None:
        """Write ecosystem data as a GeoJSON file."""
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        gdf = self.to_geodataframe()
        gdf.to_file(path, driver="GeoJSON")

    def to_ee_feature_collection(self, asset_id: str):
        """Upload ecosystem data as an Earth Engine asset.

        Converts to GeoDataFrame, then to an in-memory ee.FeatureCollection,
        and exports to the given asset ID.

        Returns the export Task (already started).
        """
        import ee

        gdf = self.to_geodataframe()
        fc = _geodataframe_to_ee_fc(gdf)
        task = ee.batch.Export.table.toAsset(
            collection=fc,
            assetId=asset_id,
            description="ecosystem_export",
        )
        task.start()
        return task

    # -- visualization -------------------------------------------------------

    def to_layer(self, *, get_fill_color=None, get_line_color=None):
        """Return lonboard layer(s) for this ecosystem dataset."""
        if self.kind != EcosystemKind.VECTOR_LOCAL:
            raise NotImplementedError(
                f"Visualization not yet supported for {self.kind.value}"
            )
        try:
            from lonboard import PolygonLayer
        except ImportError:
            raise ImportError(
                "lonboard is required for visualization. "
                "Install it with: pip install lonboard"
            ) from None

        if get_fill_color is None:
            get_fill_color = [0, 255, 0, 128]
        if get_line_color is None:
            get_line_color = [0, 0, 0, 255]

        gdf = self.load()
        if gdf.empty:
            return []
        return [PolygonLayer.from_geopandas(
            gdf,
            get_fill_color=get_fill_color,
            get_line_color=get_line_color,
            line_width_min_pixels=1,
        )]

    def to_map(self, **kwargs):
        """Return a lonboard Map showing the ecosystem polygons."""
        try:
            from lonboard import Map
        except ImportError:
            raise ImportError(
                "lonboard is required for visualization. "
                "Install it with: pip install lonboard"
            ) from None

        layers = self.to_layer()
        return Map(layers=layers, **kwargs)

    # -- display -------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{type(self).__name__}(data={self._data!r})"

    def _repr_html_(self) -> str:
        parts = [
            f"<b>{type(self).__name__}</b>",
            f"Kind: {self.kind.value}",
            f"Source: {self._data!r}",
        ]
        if self._cached is not None:
            count = self._feature_count()
            if count is not None:
                parts.append(f"Features: {count:,}")
        return "<br>".join(parts)

    # -- factory classmethods -------------------------------------------------

    @classmethod
    def from_geojson(cls, path, **kwargs) -> "Ecosystems":
        """Create from a GeoJSON file."""
        return EcosystemsGeoJSON(path, **kwargs)

    @classmethod
    def from_parquet(cls, path, **kwargs) -> "Ecosystems":
        """Create from a GeoParquet file."""
        return EcosystemsGeoParquet(path, **kwargs)

    @classmethod
    def from_gee_feature_collection(cls, data, *,
                                    ecosystem_column: str,
                                    **kwargs) -> "Ecosystems":
        """Create from an Earth Engine FeatureCollection or asset ID."""
        return EcosystemsEEFeatureCollection(
            data, ecosystem_column=ecosystem_column, **kwargs
        )

    @classmethod
    def from_gee_image(cls, data, **kwargs) -> "Ecosystems":
        """Create from an Earth Engine Image or asset ID."""
        return EcosystemsEEImage(data, **kwargs)

    @classmethod
    def from_cog(cls, data, **kwargs) -> "Ecosystems":
        """Create from a Cloud Optimized GeoTIFF."""
        return EcosystemsCOG(data, **kwargs)


# ---------------------------------------------------------------------------
# Vector local backends
# ---------------------------------------------------------------------------


class EcosystemsGeoJSON(Ecosystems):
    """Ecosystem polygons from a GeoJSON file."""

    kind = EcosystemKind.VECTOR_LOCAL

    def _load(self):
        import geopandas as gpd

        return gpd.read_file(self._data)


class EcosystemsGeoParquet(Ecosystems):
    """Ecosystem polygons from a GeoParquet file."""

    kind = EcosystemKind.VECTOR_LOCAL

    def _load(self):
        import geopandas as gpd

        return gpd.read_parquet(self._data)


# ---------------------------------------------------------------------------
# Earth Engine backends
# ---------------------------------------------------------------------------


class EcosystemsEEFeatureCollection(Ecosystems):
    """Ecosystem polygons from an Earth Engine FeatureCollection."""

    kind = EcosystemKind.EE_FEATURE_COLLECTION

    def __init__(self, data, *, ecosystem_column: str):
        super().__init__(data)
        self.ecosystem_column = ecosystem_column

    def _load(self):
        import ee

        if isinstance(self._data, str):
            return ee.FeatureCollection(self._data)
        return self._data

    def _feature_count(self) -> int | None:
        if self._cached is not None:
            return self._cached.size().getInfo()
        return None

    def to_geodataframe(self) -> "gpd.GeoDataFrame":
        """Download the FeatureCollection as a GeoDataFrame."""
        import ee

        fc = self.load()
        gdf = ee.data.computeFeatures({
            "expression": fc,
            "fileFormat": "GEOPANDAS_GEODATAFRAME",
        })
        return gdf.set_crs("EPSG:4326")

    def to_ee_feature_collection(self, asset_id: str):
        """Export to an Earth Engine table asset. Returns the started Task."""
        import ee

        fc = self.load()
        task = ee.batch.Export.table.toAsset(
            collection=fc,
            assetId=asset_id,
            description="ecosystem_export",
        )
        task.start()
        return task

    def to_layer(self):
        """Return a BitmapTileLayer rendering the FeatureCollection via EE tiles."""
        try:
            from lonboard import BitmapTileLayer
        except ImportError:
            raise ImportError(
                "lonboard is required for visualization. "
                "Install it with: pip install lonboard"
            ) from None

        fc = self.load()
        styled = fc.style(color='0080FF', fillColor='0080FF40')
        map_id = styled.getMapId()
        tile_url = map_id['tile_fetcher'].url_format
        return [BitmapTileLayer(data=tile_url)]


class EcosystemsEEImage(Ecosystems):
    """Ecosystem coverage from an Earth Engine Image."""

    kind = EcosystemKind.EE_IMAGE

    def _load(self):
        import ee

        if isinstance(self._data, str):
            return ee.Image(self._data)
        return self._data


# ---------------------------------------------------------------------------
# Raster local backend
# ---------------------------------------------------------------------------


class EcosystemsCOG(Ecosystems):
    """Ecosystem coverage from a Cloud Optimized GeoTIFF."""

    kind = EcosystemKind.RASTER_LOCAL

    def _load(self):
        import rioxarray  # noqa: F401

        return rioxarray.open_rasterio(self._data)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def _is_file_path(data) -> bool:
    """Check if data looks like a file path."""
    if not isinstance(data, str):
        return False
    return (
        data.endswith((".parquet", ".geojson", ".tif", ".tiff"))
        or data.startswith(("gs://", "/", "."))
    )


def make_ecosystems(data, **kwargs) -> Ecosystems:
    """Auto-detect and create an Ecosystems instance.

    Args:
        data: Data source. One of:
            - Path to a GeoJSON file (.geojson)
            - Path to a GeoParquet file (.parquet)
            - Path to a COG file (.tif, .tiff)
            - ee.Image object
            - ee.FeatureCollection object
            - Earth Engine asset ID string
        **kwargs: Additional arguments passed to the backend constructor.

    Returns:
        An Ecosystems instance.
    """
    # File paths
    if isinstance(data, str):
        if data.endswith(".geojson"):
            return EcosystemsGeoJSON(data, **kwargs)
        if data.endswith(".parquet"):
            return EcosystemsGeoParquet(data, **kwargs)
        if data.endswith((".tif", ".tiff")):
            return EcosystemsCOG(data, **kwargs)

    # Earth Engine objects
    try:
        import ee

        if isinstance(data, ee.Image):
            return EcosystemsEEImage(data, **kwargs)
        if isinstance(data, ee.FeatureCollection):
            return EcosystemsEEFeatureCollection(data, **kwargs)
    except ImportError:
        pass

    # String asset IDs — detect via EE API
    if isinstance(data, str) and not _is_file_path(data):
        try:
            import ee

            asset_info = ee.data.getAsset(data)
            asset_type = asset_info.get("type", "")
            if asset_type in ("IMAGE", "IMAGE_COLLECTION"):
                return EcosystemsEEImage(data, **kwargs)
            if asset_type == "TABLE":
                return EcosystemsEEFeatureCollection(data, **kwargs)
        except Exception:
            pass

    raise ValueError(
        f"Cannot determine ecosystem backend for data: {data!r}. "
        f"Supported types: .geojson, .parquet, .tif/.tiff, "
        f"ee.Image, ee.FeatureCollection"
    )
