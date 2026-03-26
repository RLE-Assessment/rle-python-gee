"""Tests for the Ecosystems class hierarchy."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import geopandas as gpd

from rle_python_gee.ecosystems import (
    Ecosystems,
    EcosystemKind,
    EcosystemsGeoJSON,
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
class TestEcosystemsGeoJSON:
    def test_kind(self):
        eco = EcosystemsGeoJSON(GEOJSON_PATH)
        assert eco.kind == EcosystemKind.VECTOR_LOCAL

    def test_load_returns_geodataframe(self):
        eco = EcosystemsGeoJSON(GEOJSON_PATH)
        gdf = eco.load()
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) > 0

    def test_load_caches(self):
        eco = EcosystemsGeoJSON(GEOJSON_PATH)
        first = eco.load()
        second = eco.load()
        assert first is second


@pytest.mark.unit
class TestEcosystemsGeoParquet:
    def test_kind(self):
        eco = EcosystemsGeoParquet("/fake/path.parquet")
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
    def test_from_geojson(self):
        eco = Ecosystems.from_geojson("/path.geojson")
        assert isinstance(eco, EcosystemsGeoJSON)

    def test_from_parquet(self):
        eco = Ecosystems.from_parquet("/path.parquet")
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
        eco = make_ecosystems("/fake/path.geojson")
        assert isinstance(eco, EcosystemsGeoJSON)

    def test_parquet_detection(self):
        eco = make_ecosystems("/fake/path.parquet")
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
        eco = EcosystemsGeoJSON("/path/to/file.geojson")
        r = repr(eco)
        assert "EcosystemsGeoJSON" in r
        assert "file.geojson" in r

    def test_repr_html(self):
        eco = EcosystemsGeoJSON("/path/to/file.geojson")
        html = eco._repr_html_()
        assert "EcosystemsGeoJSON" in html
        assert "vector_local" in html

    def test_to_layer_geojson(self):
        from lonboard import PolygonLayer

        eco = EcosystemsGeoJSON(GEOJSON_PATH)
        layers = eco.to_layer()
        assert len(layers) == 1
        assert isinstance(layers[0], PolygonLayer)

    def test_to_map_geojson(self):
        from lonboard import Map

        eco = EcosystemsGeoJSON(GEOJSON_PATH)
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
        eco = EcosystemsGeoJSON(GEOJSON_PATH)
        gdf = eco.to_geodataframe()
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) > 0

    def test_to_parquet(self, tmp_path):
        eco = EcosystemsGeoJSON(GEOJSON_PATH)
        out = tmp_path / "output.parquet"
        eco.to_parquet(out)
        result = gpd.read_parquet(out)
        assert len(result) > 0
        assert result.geometry.is_valid.all()

    def test_to_geojson(self, tmp_path):
        eco = EcosystemsGeoJSON(GEOJSON_PATH)
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

    def test_to_ee_feature_collection_from_geojson(self):
        mock_task = MagicMock()

        with patch("ee.FeatureCollection") as mock_fc_cls, \
             patch("ee.batch.Export.table.toAsset", return_value=mock_task):
            eco = EcosystemsGeoJSON(GEOJSON_PATH)
            task = eco.to_ee_feature_collection("projects/test/assets/output")
            assert task is mock_task
            mock_task.start.assert_called_once()
            mock_fc_cls.assert_called_once()

    def test_to_ee_feature_collection_raises_for_image(self):
        eco = EcosystemsEEImage("asset/id")
        with pytest.raises(NotImplementedError, match="not supported"):
            eco.to_ee_feature_collection("projects/test/assets/output")
