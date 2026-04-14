"""Tests for the Ecosystems class hierarchy."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import geopandas as gpd

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

GEOJSON_PATH = Path(__file__).parent / "test_data" / "null_island.geojson"


# ---------------------------------------------------------------------------
# Subclass unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEcosystemsFile:
    def test_kind(self):
        eco = EcosystemsFile(GEOJSON_PATH, ecosystem_column='ECO_NAME')
        assert eco.kind == EcosystemKind.VECTOR_LOCAL

    def test_load_returns_geodataframe(self):
        eco = EcosystemsFile(GEOJSON_PATH, ecosystem_column='ECO_NAME')
        gdf = eco.load()
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) > 0

    def test_load_caches(self):
        eco = EcosystemsFile(GEOJSON_PATH, ecosystem_column='ECO_NAME')
        first = eco.load()
        second = eco.load()
        assert first is second


@pytest.mark.unit
class TestEcosystemsGeoParquet:
    def test_kind(self):
        eco = EcosystemsGeoParquet("/fake/path.parquet", ecosystem_column='ECO_NAME')
        assert eco.kind == EcosystemKind.VECTOR_LOCAL


@pytest.mark.unit
class TestEcosystemsEEFeatureCollection:
    def test_kind(self):
        eco = EcosystemsEEFeatureCollection("asset/id", ecosystem_column="ECO_NAME")
        assert eco.kind == EcosystemKind.EE_FEATURE_COLLECTION

    def test_ecosystem_column(self):
        eco = EcosystemsEEFeatureCollection("asset/id", ecosystem_column="MY_COL")
        assert eco.ecosystem_column == "MY_COL"


@pytest.mark.unit
class TestEcosystemsEEImage:
    def test_kind(self):
        eco = EcosystemsEEImage("asset/id")
        assert eco.kind == EcosystemKind.EE_IMAGE


@pytest.mark.unit
class TestEcosystemsCOG:
    def test_kind(self):
        eco = EcosystemsCOG("/fake/path.tif")
        assert eco.kind == EcosystemKind.RASTER_LOCAL


# ---------------------------------------------------------------------------
# Factory classmethod tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEcosystemsClassmethods:
    def test_from_file(self):
        eco = Ecosystems.from_file("/path.geojson", ecosystem_column='ECO_NAME')
        assert isinstance(eco, EcosystemsFile)

    def test_from_parquet(self):
        eco = Ecosystems.from_parquet("/path.parquet", ecosystem_column='ECO_NAME')
        assert isinstance(eco, EcosystemsGeoParquet)

    def test_from_gee_feature_collection(self):
        eco = Ecosystems.from_gee_feature_collection("id", ecosystem_column="COL")
        assert isinstance(eco, EcosystemsEEFeatureCollection)

    def test_from_gee_image(self):
        eco = Ecosystems.from_gee_image("id")
        assert isinstance(eco, EcosystemsEEImage)

    def test_from_cog(self):
        eco = Ecosystems.from_cog("/path.tif")
        assert isinstance(eco, EcosystemsCOG)


# ---------------------------------------------------------------------------
# make_ecosystems factory tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMakeEcosystems:
    def test_geojson_detection(self):
        eco = make_ecosystems("/fake/path.geojson", ecosystem_column='ECO_NAME')
        assert isinstance(eco, EcosystemsFile)

    def test_parquet_detection(self):
        eco = make_ecosystems("/fake/path.parquet", ecosystem_column='ECO_NAME')
        assert isinstance(eco, EcosystemsGeoParquet)

    def test_tif_detection(self):
        eco = make_ecosystems("/fake/path.tif")
        assert isinstance(eco, EcosystemsCOG)

    def test_ee_image_detection(self):
        import ee
        mock_image = MagicMock(spec=ee.Image)
        eco = make_ecosystems(mock_image)
        assert isinstance(eco, EcosystemsEEImage)

    def test_unknown_data_raises(self):
        with pytest.raises(ValueError, match="Cannot determine ecosystem backend"):
            make_ecosystems(12345)


# ---------------------------------------------------------------------------
# Display and visualization tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEcosystemsDisplay:
    def test_repr(self):
        eco = EcosystemsFile("/path/to/file.geojson", ecosystem_column='ECO_NAME')
        r = repr(eco)
        assert "EcosystemsFile" in r
        assert "file.geojson" in r

    def test_repr_html(self):
        eco = EcosystemsFile("/path/to/file.geojson", ecosystem_column='ECO_NAME')
        html = eco._repr_html_()
        assert "EcosystemsFile" in html
        assert "vector_local" in html

    def test_to_layer_geojson(self):
        from lonboard import PolygonLayer

        eco = EcosystemsFile(GEOJSON_PATH, ecosystem_column='ECO_NAME')
        layers = eco.to_layer()
        assert len(layers) == 1
        assert isinstance(layers[0], PolygonLayer)

    def test_to_map_geojson(self):
        from lonboard import Map

        eco = EcosystemsFile(GEOJSON_PATH, ecosystem_column='ECO_NAME')
        m = eco.to_map()
        assert isinstance(m, Map)

    def test_to_layer_ee_image_raises(self):
        eco = EcosystemsEEImage("asset/id")
        with pytest.raises(NotImplementedError, match="not yet supported"):
            eco.to_layer()

    def test_to_layer_ee_feature_collection(self):
        from lonboard import BitmapTileLayer

        mock_fc = MagicMock()
        mock_styled = MagicMock()
        mock_fc.style.return_value = mock_styled
        mock_tile_fetcher = MagicMock()
        mock_tile_fetcher.url_format = "https://earthengine.googleapis.com/v1/tiles/{z}/{x}/{y}"
        mock_styled.getMapId.return_value = {"tile_fetcher": mock_tile_fetcher}

        eco = EcosystemsEEFeatureCollection(mock_fc, ecosystem_column="ECO_NAME")
        eco._cached = mock_fc  # simulate load() having been called
        layers = eco.to_layer()
        assert len(layers) == 1
        assert isinstance(layers[0], BitmapTileLayer)
        mock_fc.style.assert_called_once()


# ---------------------------------------------------------------------------
# Export / write tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEcosystemsExport:
    def test_to_geodataframe_geojson(self):
        eco = EcosystemsFile(GEOJSON_PATH, ecosystem_column='ECO_NAME')
        gdf = eco.to_geodataframe()
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) > 0

    def test_to_parquet(self, tmp_path):
        eco = EcosystemsFile(GEOJSON_PATH, ecosystem_column='ECO_NAME')
        out = tmp_path / "output.parquet"
        eco.to_parquet(out)
        result = gpd.read_parquet(out)
        assert len(result) > 0
        assert result.geometry.is_valid.all()

    def test_to_geojson(self, tmp_path):
        eco = EcosystemsFile(GEOJSON_PATH, ecosystem_column='ECO_NAME')
        out = tmp_path / "output.geojson"
        eco.to_geojson(out)
        result = gpd.read_file(out)
        assert len(result) > 0
        assert result.geometry.is_valid.all()

    def test_to_geodataframe_ee_raises_for_image(self):
        eco = EcosystemsEEImage("asset/id")
        with pytest.raises(NotImplementedError, match="not supported"):
            eco.to_geodataframe()

    def test_to_ee_feature_collection_from_ee_fc(self):
        mock_fc = MagicMock()
        mock_task = MagicMock()

        with patch("ee.batch.Export.table.toAsset", return_value=mock_task):
            eco = EcosystemsEEFeatureCollection(mock_fc, ecosystem_column="ECO_NAME")
            eco._cached = mock_fc
            task = eco.to_ee_feature_collection("projects/test/assets/output")
            assert task is mock_task
            mock_task.start.assert_called_once()

    def test_to_ee_feature_collection_from_file(self):
        mock_task = MagicMock()

        with patch("ee.FeatureCollection") as mock_fc_cls, \
             patch("ee.batch.Export.table.toAsset", return_value=mock_task):
            eco = EcosystemsFile(GEOJSON_PATH, ecosystem_column='ECO_NAME')
            task = eco.to_ee_feature_collection("projects/test/assets/output")
            assert task is mock_task
            mock_task.start.assert_called_once()
            mock_fc_cls.assert_called_once()

    def test_to_ee_feature_collection_raises_for_image(self):
        eco = EcosystemsEEImage("asset/id")
        with pytest.raises(NotImplementedError, match="not supported"):
            eco.to_ee_feature_collection("projects/test/assets/output")


@pytest.mark.unit
class TestEcosystemsToRaster:
    # ---- index mode ----

    def test_index_mode_creates_cog(self, tmp_path):
        import json
        import numpy as np
        import rasterio

        eco = EcosystemsFile(GEOJSON_PATH, ecosystem_column='ECO_CODE')
        out = tmp_path / "eco_index.tif"
        mapping = eco.to_raster(out, crs="ESRI:54034", scale=1000)

        assert out.exists()
        with rasterio.open(out) as src:
            # Rasterio reports driver as GTiff for any TIFF on read; the
            # COG layout shows up in IMAGE_STRUCTURE namespace tags.
            assert src.driver == "GTiff"
            assert src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
            assert src.count == 1
            assert src.dtypes[0] in ("uint8", "uint16", "uint32")
            assert src.nodata == np.iinfo(src.dtypes[0]).max
            arr = src.read(1)
            unique_vals = set(np.unique(arr).tolist())
            assert unique_vals - {src.nodata} <= set(mapping.keys())
            assert len(unique_vals - {src.nodata}) >= 1
            tags = src.tags()
            recovered = {
                int(k): v
                for k, v in json.loads(tags["ECOSYSTEM_INDEX_JSON"]).items()
            }
            assert recovered == mapping
            assert tags["RASTERIZE_MODE"] == "index"

    def test_index_value_at_known_location(self, tmp_path):
        import rasterio

        eco = EcosystemsFile(GEOJSON_PATH, ecosystem_column='ECO_CODE')
        out = tmp_path / "eco_index.tif"
        mapping = eco.to_raster(out, crs="ESRI:54034", scale=1000)

        gdf = eco.to_geodataframe().to_crs("ESRI:54034")
        largest = gdf.geometry.iloc[gdf.geometry.area.argmax()]
        cx, cy = largest.centroid.x, largest.centroid.y
        with rasterio.open(out) as src:
            row, col = src.index(cx, cy)
            val = int(src.read(1)[row, col])
            assert val != src.nodata
        assert mapping[val] in eco.unique_ecosystems()

    def test_index_nodata_collision_rejected(self, tmp_path):
        eco = EcosystemsFile(GEOJSON_PATH, ecosystem_column='ECO_CODE')
        with pytest.raises(ValueError, match="collides"):
            eco.to_raster(tmp_path / "x.tif", crs="ESRI:54034",
                          scale=1000, nodata=1)

    # ---- fraction mode ----

    def test_fraction_mode_creates_multiband_cog(self, tmp_path):
        import math
        import rasterio

        eco = EcosystemsFile(GEOJSON_PATH, ecosystem_column='ECO_CODE')
        codes = eco.unique_ecosystems()
        out = tmp_path / "eco_frac.tif"
        mapping = eco.to_raster(out, crs="ESRI:54034", scale=1000,
                                mode="fraction", oversampling=10)

        assert out.exists()
        assert mapping == {i: c for i, c in enumerate(codes, start=1)}
        with rasterio.open(out) as src:
            assert src.driver == "GTiff"
            assert src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
            assert src.count == len(codes)
            assert src.dtypes[0] == "float32"
            assert math.isnan(src.nodata)
            assert list(src.descriptions) == codes
            arr = src.read()
            assert arr.shape[0] == len(codes)
            assert float(arr.min()) >= 0.0
            assert float(arr.max()) <= 1.0
            assert any(b.sum() > 0 for b in arr)

    def test_fraction_band_sum_in_range(self, tmp_path):
        """For non-overlapping ecosystems, the sum across bands at each
        pixel must be in [0, 1]."""
        import rasterio

        eco = EcosystemsFile(GEOJSON_PATH, ecosystem_column='ECO_CODE')
        out = tmp_path / "eco_frac.tif"
        eco.to_raster(out, crs="ESRI:54034", scale=1000,
                      mode="fraction", oversampling=10)
        with rasterio.open(out) as src:
            arr = src.read()
            band_sum = arr.sum(axis=0)
            assert float(band_sum.max()) <= 1.0 + 1e-6
            assert float(band_sum.max()) > 0.0

    def test_fraction_invalid_oversampling(self, tmp_path):
        eco = EcosystemsFile(GEOJSON_PATH, ecosystem_column='ECO_CODE')
        with pytest.raises(ValueError, match="oversampling"):
            eco.to_raster(tmp_path / "x.tif", crs="ESRI:54034", scale=1000,
                          mode="fraction", oversampling=0)

    # ---- shared ----

    def test_invalid_mode(self, tmp_path):
        eco = EcosystemsFile(GEOJSON_PATH, ecosystem_column='ECO_CODE')
        with pytest.raises(ValueError, match="mode must be"):
            eco.to_raster(tmp_path / "x.tif", crs="ESRI:54034",
                          scale=1000, mode="largest")

    def test_ee_image_raises(self, tmp_path):
        eco = EcosystemsEEImage("asset/id")
        with pytest.raises(NotImplementedError, match="not supported"):
            eco.to_raster(tmp_path / "x.tif", crs="EPSG:4326", scale=1000)


@pytest.mark.unit
class TestEcosystemsFunctionalGroupColumn:
    def test_functional_group_column_stored(self):
        eco = EcosystemsFile(GEOJSON_PATH, ecosystem_column='ECO_CODE',
                             functional_group_column='EFG1')
        assert eco.functional_group_column == 'EFG1'

    def test_functional_group_column_default_none(self):
        eco = EcosystemsFile(GEOJSON_PATH, ecosystem_column='ECO_CODE')
        assert eco.functional_group_column is None

    def test_unique_functional_groups(self):
        eco = EcosystemsFile(GEOJSON_PATH, ecosystem_column='ECO_CODE',
                             functional_group_column='EFG1')
        groups = eco.unique_functional_groups()
        assert isinstance(groups, list)
        assert len(groups) == 3
        # Should be naturally sorted
        assert groups == ['M1.1', 'T1.1', 'T6.5']

    def test_unique_functional_groups_raises_without_column(self):
        eco = EcosystemsFile(GEOJSON_PATH, ecosystem_column='ECO_CODE')
        with pytest.raises(ValueError, match="functional_group_column is not set"):
            eco.unique_functional_groups()

    def test_threaded_through_filter(self):
        eco = EcosystemsFile(GEOJSON_PATH, ecosystem_column='ECO_CODE',
                             functional_group_column='EFG1')
        filtered = eco.filter('T1.1.1')
        assert filtered.functional_group_column == 'EFG1'

    def test_threaded_through_limit(self):
        eco = EcosystemsFile(GEOJSON_PATH, ecosystem_column='ECO_CODE',
                             functional_group_column='EFG1')
        limited = eco.limit(1)
        assert limited.functional_group_column == 'EFG1'
