"""Earth Engine backend tests for rle.gee.

Extracted from the original test_aoo.py / test_ecosystems.py. These cover the
Earth Engine-specific backends:

- EcosystemsEEFeatureCollection / EcosystemsEEImage (+ GeeEcosystems aliases)
- AOOGridEEImage / AOOGridEEFeatureCollection / AOOGridPolygonEEFeatureCollection
- wait_for_task
- EE upload (upload_gdf_to_ee_asset)

Tests that construct/operate only on mocks are plain unit tests. Tests that
require a live Earth Engine connection are marked ``integration``.
"""

from unittest.mock import MagicMock, patch

import ee
import geopandas as gpd
import pytest
from shapely.geometry import box

from rle.core.aoo import AOOGrid
from rle.core.ecosystems import Ecosystems, EcosystemKind
from rle.gee.aoo import (
    AOOGridEEFeatureCollection,
    AOOGridEEImage,
    AOOGridPolygonEEFeatureCollection,
    wait_for_task,
)
from rle.gee.ecosystems import (
    EcosystemsEEFeatureCollection,
    EcosystemsEEImage,
    GeeEcosystems,
    GeeEcosystemsImage,
)
from rle.gee.upload import (
    _INLINE_UPLOAD_LIMIT,
    _geodataframe_to_ee_fc,
    upload_gdf_to_ee_asset,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _FakeEcosystems(Ecosystems):
    """Minimal non-EE concrete Ecosystems subclass for rejection tests."""

    kind = EcosystemKind.VECTOR_LOCAL

    def _load(self):
        return self._data


class _FakeAOOGrid(AOOGrid):
    """Minimal non-EE concrete AOOGrid subclass for rejection tests."""

    def __init__(self, grid_cells_gdf, **kwargs):
        super().__init__(ecosystems=_FakeEcosystems(None, ecosystem_column="eco"), **kwargs)
        self._fake_gdf = grid_cells_gdf

    def _compute(self) -> None:
        self._computed_gdf = self._fake_gdf

    def _load_grid_cells(self) -> gpd.GeoDataFrame:
        return self._computed_gdf


def _make_test_gdf(n: int = 3) -> gpd.GeoDataFrame:
    cells = [box(i, 0, i + 0.1, 0.1) for i in range(n)]
    return gpd.GeoDataFrame({"geometry": cells}, crs="EPSG:4326")


def _make_test_polygons_gdf(n: int = 4) -> gpd.GeoDataFrame:
    polys = [box(i * 0.05, 0, i * 0.05 + 0.03, 0.03) for i in range(n)]
    return gpd.GeoDataFrame(
        {
            "geometry": polys,
            "grid_col": [0, 0, 1, 1][:n],
            "grid_row": [0, 0, 0, 0][:n],
            "ecosystem": ["eco_a", "eco_b", "eco_a", "eco_c"][:n],
        },
        crs="EPSG:4326",
    )


def _make_ee_fc_aoo_grid() -> AOOGridEEFeatureCollection:
    """Build an AOOGridEEFeatureCollection without running compute()."""
    aoo = AOOGridEEFeatureCollection.__new__(AOOGridEEFeatureCollection)
    eco = MagicMock()
    eco.ecosystem_column = "ECO_NAME"
    aoo._ecosystems = eco
    aoo._gee_asset_path = "projects/test/assets/cache"
    aoo._gcs_path = None
    aoo._computed = True
    aoo._grid_cells = None
    return aoo


# ---------------------------------------------------------------------------
# EcosystemsEEFeatureCollection / EcosystemsEEImage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEcosystemsEEFeatureCollection:
    def test_kind(self):
        eco = EcosystemsEEFeatureCollection("asset/id", ecosystem_column="ECO_NAME")
        assert eco.kind == EcosystemKind.EE_FEATURE_COLLECTION

    def test_ecosystem_column(self):
        eco = EcosystemsEEFeatureCollection("asset/id", ecosystem_column="MY_COL")
        assert eco.ecosystem_column == "MY_COL"

    def test_gee_ecosystems_alias(self):
        assert GeeEcosystems is EcosystemsEEFeatureCollection


@pytest.mark.unit
class TestEcosystemsEEImage:
    def test_kind(self):
        eco = EcosystemsEEImage("asset/id")
        assert eco.kind == EcosystemKind.EE_IMAGE

    def test_gee_ecosystems_image_alias(self):
        assert GeeEcosystemsImage is EcosystemsEEImage


# ---------------------------------------------------------------------------
# Ecosystems EE display / export (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEcosystemsEEDisplay:
    def test_to_layer_ee_image_raises(self):
        eco = EcosystemsEEImage("asset/id")
        with pytest.raises(NotImplementedError, match="not yet supported"):
            eco.to_layer()

    def test_to_layer_ee_feature_collection(self):
        BitmapTileLayer = pytest.importorskip("lonboard").BitmapTileLayer

        mock_fc = MagicMock()
        mock_styled = MagicMock()
        mock_fc.style.return_value = mock_styled
        mock_tile_fetcher = MagicMock()
        mock_tile_fetcher.url_format = (
            "https://earthengine.googleapis.com/v1/tiles/{z}/{x}/{y}"
        )
        mock_styled.getMapId.return_value = {"tile_fetcher": mock_tile_fetcher}

        eco = EcosystemsEEFeatureCollection(mock_fc, ecosystem_column="ECO_NAME")
        eco._cached = mock_fc  # simulate load() having been called
        layers = eco.to_layer()
        assert len(layers) == 1
        assert isinstance(layers[0], BitmapTileLayer)
        mock_fc.style.assert_called_once()


@pytest.mark.unit
class TestEcosystemsEEExport:
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

    def test_to_ee_feature_collection_raises_for_image(self):
        eco = EcosystemsEEImage("asset/id")
        with pytest.raises(NotImplementedError, match="not supported"):
            eco.to_ee_feature_collection("projects/test/assets/output")

    def test_to_raster_ee_image_raises(self, tmp_path):
        eco = EcosystemsEEImage("asset/id")
        with pytest.raises(NotImplementedError, match="not supported"):
            eco.to_raster(tmp_path / "x.tif", crs="EPSG:4326", scale=1000)


# ---------------------------------------------------------------------------
# AOOGridPolygonEEFeatureCollection (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAOOGridPolygonEEFC:
    """Tests for AOOGridPolygonEEFeatureCollection constructor and properties."""

    def test_requires_ee_fc_aoo_grid(self):
        """Should reject non-EE AOOGrid instances."""
        aoo = _FakeAOOGrid(_make_test_gdf(2)).compute()
        with pytest.raises(TypeError, match="AOOGridEEFeatureCollection"):
            AOOGridPolygonEEFeatureCollection(aoo)

    def test_polygons_id_derivation(self):
        """_polygons_id should derive from gee_asset_path."""
        aoo = _make_ee_fc_aoo_grid()
        polygons = AOOGridPolygonEEFeatureCollection(aoo)
        assert polygons._polygons_id == "projects/test/assets/cache/aoo_grid_polygons"

    def test_custom_gee_asset_path(self):
        """Should allow overriding gee_asset_path."""
        aoo = _make_ee_fc_aoo_grid()
        polygons = AOOGridPolygonEEFeatureCollection(
            aoo, gee_asset_path="projects/test/assets/custom"
        )
        assert polygons._polygons_id == "projects/test/assets/custom/aoo_grid_polygons"

    def test_to_polygons_returns_ee_polygons(self):
        """AOOGridEEFeatureCollection.to_polygons() returns the EE polygon backend."""
        aoo = _make_ee_fc_aoo_grid()
        polygons = aoo.to_polygons()
        assert isinstance(polygons, AOOGridPolygonEEFeatureCollection)

    def test_repr_export_task_running(self):
        """__repr__ should not load polygons while EE export is running."""
        aoo = _make_ee_fc_aoo_grid()
        polygons = AOOGridPolygonEEFeatureCollection(aoo)
        polygons._computed = True
        polygons.task = MagicMock()
        polygons.task.id = "TASK_RUN_1"
        polygons._load_polygons = MagicMock(
            side_effect=AssertionError("should not load while export running")
        )

        with patch.object(ee.data, "getTaskStatus", return_value=[{"state": "RUNNING"}]):
            r = repr(polygons)
        assert "still running" in r
        assert "TASK_RUN_1" in r
        assert "AOOGridPolygonEEFeatureCollection" in r

    def test_repr_export_task_completed(self):
        aoo = _make_ee_fc_aoo_grid()
        polygons = AOOGridPolygonEEFeatureCollection(aoo)
        polygons._computed = True
        polygons.task = MagicMock()
        polygons.task.id = "TASK_DONE"
        polygons._polygons = _make_test_polygons_gdf(3)

        with patch.object(
            ee.data, "getTaskStatus", return_value=[{"state": "COMPLETED"}]
        ):
            r = repr(polygons)
        assert "polygons=3" in r

    def test_repr_export_task_failed(self):
        aoo = _make_ee_fc_aoo_grid()
        polygons = AOOGridPolygonEEFeatureCollection(aoo)
        polygons._computed = True
        polygons.task = MagicMock()
        polygons.task.id = "TASK_FAIL"

        with patch.object(ee.data, "getTaskStatus", return_value=[{"state": "FAILED"}]):
            r = repr(polygons)
        assert "failed" in r
        assert "TASK_FAIL" in r

    def test_repr_no_task_asset_pending(self):
        """Without a session task, defer to base RuntimeError handling."""
        aoo = _make_ee_fc_aoo_grid()
        polygons = AOOGridPolygonEEFeatureCollection(aoo)
        polygons._computed = True
        polygons._load_polygons = MagicMock(
            side_effect=RuntimeError("Export asset not ready")
        )

        r = repr(polygons)
        assert "results pending" in r


# ---------------------------------------------------------------------------
# wait_for_task (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWaitForTask:
    def test_none_task_returns_immediately(self):
        # Should not raise or poll EE when given no task.
        assert wait_for_task(None) is None

    def test_completed_returns(self):
        task = MagicMock()
        task.id = "TASK_OK"
        with patch.object(
            ee.data, "getTaskStatus", return_value=[{"state": "COMPLETED"}]
        ):
            assert wait_for_task(task, poll_interval=0) is None

    def test_failed_raises(self):
        task = MagicMock()
        task.id = "TASK_BAD"
        with patch.object(
            ee.data,
            "getTaskStatus",
            return_value=[{"state": "FAILED", "error_message": "boom"}],
        ):
            with pytest.raises(RuntimeError, match="FAILED"):
                wait_for_task(task, poll_interval=0)


# ---------------------------------------------------------------------------
# EE upload (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUploadGdfToEEAsset:
    def test_returns_none_when_asset_exists(self):
        gdf = _make_test_gdf(2)
        with patch("ee.data.getAsset", return_value={"type": "TABLE"}):
            result = upload_gdf_to_ee_asset(gdf, "projects/test/assets/exists")
        assert result is None

    def test_inline_upload_starts_task(self):
        gdf = _make_test_gdf(2)
        mock_task = MagicMock()
        with patch("ee.data.getAsset", side_effect=ee.EEException("not found")), patch(
            "rle.gee.upload._geodataframe_to_ee_fc", return_value=MagicMock()
        ), patch("ee.batch.Export.table.toAsset", return_value=mock_task):
            result = upload_gdf_to_ee_asset(gdf, "projects/test/assets/new")
        assert result is mock_task
        mock_task.start.assert_called_once()

    def test_large_upload_requires_gcs_bucket(self):
        gdf = _make_test_gdf(2)
        with patch("ee.data.getAsset", side_effect=ee.EEException("not found")), patch(
            "rle.gee.upload._INLINE_UPLOAD_LIMIT", 1
        ):
            with pytest.raises(ValueError, match="gcs_bucket"):
                upload_gdf_to_ee_asset(gdf, "projects/test/assets/big")

    def test_geodataframe_to_ee_fc_builds_collection(self):
        gdf = _make_test_gdf(2)
        with patch("ee.FeatureCollection", return_value="FC") as mock_fc_cls:
            fc = _geodataframe_to_ee_fc(gdf)
        assert fc == "FC"
        mock_fc_cls.assert_called_once()
