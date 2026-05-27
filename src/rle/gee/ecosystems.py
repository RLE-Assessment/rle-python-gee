"""Earth Engine ecosystem backends for RLE assessments.

These subclass the core :class:`rle.core.ecosystems.Ecosystems` ABC and add
read access to Earth Engine FeatureCollections and Images. Construct them
explicitly::

    from rle.gee import GeeEcosystems
    eco = GeeEcosystems("projects/my-project/assets/ecosystems",
                        ecosystem_column="ECO_NAME")
"""

from rle.core.ecosystems import Ecosystems, EcosystemKind


class EcosystemsEEFeatureCollection(Ecosystems):
    """Ecosystem polygons from an Earth Engine FeatureCollection."""

    kind = EcosystemKind.EE_FEATURE_COLLECTION

    def __init__(self, data, *, ecosystem_column: str, ecosystem_name_column: str | None = None,
                 functional_group_column: str | None = None):
        super().__init__(data, ecosystem_column=ecosystem_column,
                         ecosystem_name_column=ecosystem_name_column,
                         functional_group_column=functional_group_column)

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


# Friendly aliases for explicit construction
GeeEcosystems = EcosystemsEEFeatureCollection
GeeEcosystemsImage = EcosystemsEEImage
