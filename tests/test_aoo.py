"""Tests for the AOO grid module."""

from pathlib import Path

import pytest
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from unittest.mock import Mock, patch, MagicMock

from rle_python_gee.aoo import (
    AOOGrid,
    AOOGridEEImage,
    AOOGridNotComputedError,
    AOOGridPolygons,
    AOOGridPolygonsNotComputedError,
    AOOGridVectorLocal,
    make_aoo,
    make_aoo_polygons,
)
from rle_python_gee.ecosystems import (
    Ecosystems,
    EcosystemsGeoJSON,
    EcosystemsGeoParquet,
)


# ---------------------------------------------------------------------------
# Concrete subclass for testing base class logic
# ---------------------------------------------------------------------------


class FakeEcosystems(Ecosystems):
    """Minimal concrete Ecosystems subclass for testing."""
    from rle_python_gee.ecosystems import EcosystemKind
    kind = EcosystemKind.VECTOR_LOCAL

    def _load(self):
        return self._data


class FakeAOOGrid(AOOGrid):
    """Minimal concrete subclass for testing AOOGrid base class."""

    def __init__(self, grid_cells_gdf, **kwargs):
        super().__init__(ecosystems=FakeEcosystems(None), **kwargs)
        self._fake_gdf = grid_cells_gdf

    def _compute(self) -> None:
        self._computed_gdf = self._fake_gdf

    def _load_grid_cells(self) -> gpd.GeoDataFrame:
        return self._computed_gdf


def _make_test_gdf(n: int = 3) -> gpd.GeoDataFrame:
    """Create a test GeoDataFrame with grid cell geometries."""
    cells = [box(i, 0, i + 0.1, 0.1) for i in range(n)]
    return gpd.GeoDataFrame(
        {"geometry": cells},
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# Base class tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAOOGridBase:
    """Tests for AOOGrid base class properties and methods."""

    def test_cell_count(self):
        gdf = _make_test_gdf(3)
        aoo = FakeAOOGrid(gdf).compute()
        assert aoo.cell_count == 3

    def test_aoo_km2(self):
        """AOO should be cell_count * 100 km²."""
        gdf = _make_test_gdf(1)
        aoo = FakeAOOGrid(gdf).compute()
        assert aoo.aoo_km2 == aoo.cell_count * 100

    def test_repr(self):
        gdf = _make_test_gdf(1)
        aoo = FakeAOOGrid(gdf).compute()
        r = repr(aoo)
        assert "FakeAOOGrid" in r
        assert "cell_count=" in r

    def test_repr_html(self):
        gdf = _make_test_gdf(1)
        aoo = FakeAOOGrid(gdf).compute()
        html = aoo._repr_html_()
        assert "FakeAOOGrid" in html
        assert "km²" in html

    def test_not_computed_raises(self):
        """Accessing grid_cells before compute() should raise."""
        gdf = _make_test_gdf(1)
        aoo = FakeAOOGrid(gdf)
        with pytest.raises(AOOGridNotComputedError, match="Call .compute()"):
            _ = aoo.grid_cells

    def test_cell_count_raises_before_compute(self):
        aoo = FakeAOOGrid(_make_test_gdf(1))
        with pytest.raises(AOOGridNotComputedError):
            _ = aoo.cell_count

    def test_aoo_km2_raises_before_compute(self):
        aoo = FakeAOOGrid(_make_test_gdf(1))
        with pytest.raises(AOOGridNotComputedError):
            _ = aoo.aoo_km2

    def test_repr_before_compute(self):
        aoo = FakeAOOGrid(_make_test_gdf(1))
        r = repr(aoo)
        assert "not computed" in r

    def test_repr_html_before_compute(self):
        aoo = FakeAOOGrid(_make_test_gdf(1))
        html = aoo._repr_html_()
        assert "Not computed" in html

    def test_compute_returns_self(self):
        gdf = _make_test_gdf(1)
        aoo = FakeAOOGrid(gdf)
        result = aoo.compute()
        assert result is aoo


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMakeAooFactory:
    """Tests for the make_aoo factory function."""

    def test_parquet_detection(self):
        aoo = make_aoo("/fake/path.parquet")
        assert isinstance(aoo, AOOGridVectorLocal)

    def test_geojson_detection(self):
        aoo = make_aoo("/fake/path.geojson")
        assert isinstance(aoo, AOOGridVectorLocal)

    def test_tif_detection(self):
        from rle_python_gee.aoo import AOOGridCOG

        aoo = make_aoo("/fake/path.tif")
        assert isinstance(aoo, AOOGridCOG)

    def test_ee_image_detection(self):
        import ee

        mock_image = MagicMock(spec=ee.Image)
        aoo = make_aoo(mock_image)
        assert isinstance(aoo, AOOGridEEImage)

    def test_unknown_data_raises(self):
        with pytest.raises(ValueError, match="Cannot determine ecosystem backend"):
            make_aoo(12345)

    def test_ecosystems_instance(self):
        """make_aoo should accept an Ecosystems instance directly."""
        eco = EcosystemsGeoJSON("/fake/path.geojson")
        aoo = make_aoo(eco)
        assert isinstance(aoo, AOOGridVectorLocal)


# ---------------------------------------------------------------------------
# Classmethod tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAOOGridClassmethods:
    """Tests for AOOGrid.from_*() classmethods."""

    def test_from_parquet(self):
        aoo = AOOGrid.from_parquet("/fake/path.parquet")
        assert isinstance(aoo, AOOGridVectorLocal)

    def test_from_geojson(self):
        aoo = AOOGrid.from_geojson("/fake/path.geojson")
        assert isinstance(aoo, AOOGridVectorLocal)

    def test_from_cog(self):
        from rle_python_gee.aoo import AOOGridCOG

        aoo = AOOGrid.from_cog("/fake/path.tif")
        assert isinstance(aoo, AOOGridCOG)

    def test_from_gee_image(self):
        import ee

        mock_image = MagicMock(spec=ee.Image)
        aoo = AOOGrid.from_gee_image(mock_image)
        assert isinstance(aoo, AOOGridEEImage)

    def test_from_gee_feature_collection(self):
        import ee
        from rle_python_gee.aoo import AOOGridEEFeatureCollection

        mock_fc = MagicMock(spec=ee.FeatureCollection)
        aoo = AOOGrid.from_gee_feature_collection(
            mock_fc, ecosystem_column='ECO_NAME',
            asset_path='projects/test/assets/cache',
        )
        assert isinstance(aoo, AOOGridEEFeatureCollection)


# ---------------------------------------------------------------------------
# GeoJSON backend tests
# ---------------------------------------------------------------------------

GEOJSON_PATH = Path(__file__).parent / "test_data" / "null_island.geojson"


@pytest.mark.unit
class TestAOOGridGeoJSON:
    def test_grid_cells_non_empty(self):
        aoo = AOOGrid.from_geojson(GEOJSON_PATH).compute()
        assert len(aoo.grid_cells) > 0

    def test_cell_count(self):
        aoo = AOOGrid.from_geojson(GEOJSON_PATH).compute()
        assert aoo.cell_count > 0

    def test_aoo_km2(self):
        aoo = AOOGrid.from_geojson(GEOJSON_PATH).compute()
        assert aoo.aoo_km2 > 0

    def test_via_ecosystems(self):
        """Constructing via Ecosystems should produce the same result."""
        eco = Ecosystems.from_geojson(GEOJSON_PATH)
        aoo = make_aoo(eco).compute()
        assert isinstance(aoo, AOOGridVectorLocal)
        assert aoo.cell_count > 0


# ---------------------------------------------------------------------------
# Backward-compatibility alias tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBackwardCompatAliases:
    def test_geojson_alias(self):
        from rle_python_gee.aoo import AOOGridGeoJSON
        assert AOOGridGeoJSON is AOOGridVectorLocal

    def test_geoparquet_alias(self):
        from rle_python_gee.aoo import AOOGridGeoParquet
        assert AOOGridGeoParquet is AOOGridVectorLocal


# ---------------------------------------------------------------------------
# AOOGridPolygons base class tests
# ---------------------------------------------------------------------------


class FakeAOOGridPolygons(AOOGridPolygons):
    """Minimal concrete subclass for testing AOOGridPolygons base class."""

    def __init__(self, aoo_grid, polygons_gdf):
        super().__init__(aoo_grid)
        self._fake_polygons = polygons_gdf

    def _compute(self) -> None:
        self._computed_polygons = self._fake_polygons

    def _load_polygons(self) -> gpd.GeoDataFrame:
        return self._computed_polygons


def _make_test_polygons_gdf(n: int = 4) -> gpd.GeoDataFrame:
    """Create a test GeoDataFrame with intersection polygon geometries."""
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


@pytest.mark.unit
class TestAOOGridPolygonsBase:
    """Tests for AOOGridPolygons base class."""

    def _make(self, n=4):
        grid_gdf = _make_test_gdf(2)
        aoo = FakeAOOGrid(grid_gdf).compute()
        poly_gdf = _make_test_polygons_gdf(n)
        return FakeAOOGridPolygons(aoo, poly_gdf)

    def test_polygon_count(self):
        obj = self._make(4).compute()
        assert obj.polygon_count == 4

    def test_compute_returns_self(self):
        obj = self._make()
        result = obj.compute()
        assert result is obj

    def test_not_computed_raises(self):
        obj = self._make()
        with pytest.raises(AOOGridPolygonsNotComputedError, match="Call .compute()"):
            _ = obj.polygons

    def test_polygon_count_raises_before_compute(self):
        obj = self._make()
        with pytest.raises(AOOGridPolygonsNotComputedError):
            _ = obj.polygon_count

    def test_repr(self):
        obj = self._make().compute()
        r = repr(obj)
        assert "FakeAOOGridPolygons" in r
        assert "polygons=" in r

    def test_repr_before_compute(self):
        obj = self._make()
        r = repr(obj)
        assert "not computed" in r

    def test_repr_html(self):
        obj = self._make().compute()
        html = obj._repr_html_()
        assert "FakeAOOGridPolygons" in html
        assert "Polygons:" in html

    def test_repr_html_before_compute(self):
        obj = self._make()
        html = obj._repr_html_()
        assert "Not computed" in html


# ---------------------------------------------------------------------------
# AOOGridPolygonEEFeatureCollection tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAOOGridPolygonEEFC:
    """Tests for AOOGridPolygonEEFeatureCollection constructor and properties."""

    def test_requires_ee_fc_aoo_grid(self):
        """Should reject non-EE AOOGrid instances."""
        from rle_python_gee.aoo import AOOGridPolygonEEFeatureCollection

        grid_gdf = _make_test_gdf(2)
        aoo = FakeAOOGrid(grid_gdf).compute()
        with pytest.raises(TypeError, match="AOOGridEEFeatureCollection"):
            AOOGridPolygonEEFeatureCollection(aoo)

    def test_polygons_id_derivation(self):
        """_polygons_id should derive from asset_path."""
        import ee
        from rle_python_gee.aoo import (
            AOOGridEEFeatureCollection,
            AOOGridPolygonEEFeatureCollection,
        )

        mock_fc = MagicMock(spec=ee.FeatureCollection)
        eco = MagicMock()
        eco.ecosystem_column = 'ECO_NAME'
        aoo = AOOGridEEFeatureCollection.__new__(AOOGridEEFeatureCollection)
        aoo._ecosystems = eco
        aoo._asset_path = 'projects/test/assets/cache'
        aoo._computed = True
        aoo._grid_cells = None

        polygons = AOOGridPolygonEEFeatureCollection(aoo)
        assert polygons._polygons_id == 'projects/test/assets/cache/aoo_grid_polygons'

    def test_custom_asset_path(self):
        """Should allow overriding asset_path."""
        import ee
        from rle_python_gee.aoo import (
            AOOGridEEFeatureCollection,
            AOOGridPolygonEEFeatureCollection,
        )

        aoo = AOOGridEEFeatureCollection.__new__(AOOGridEEFeatureCollection)
        aoo._ecosystems = MagicMock()
        aoo._ecosystems.ecosystem_column = 'ECO_NAME'
        aoo._asset_path = 'projects/test/assets/default'
        aoo._computed = True
        aoo._grid_cells = None

        polygons = AOOGridPolygonEEFeatureCollection(
            aoo, asset_path='projects/test/assets/custom'
        )
        assert polygons._polygons_id == 'projects/test/assets/custom/aoo_grid_polygons'


# ---------------------------------------------------------------------------
# make_aoo_polygons factory tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMakeAooPolygonsFactory:
    def test_unsupported_type_raises(self):
        grid_gdf = _make_test_gdf(2)
        aoo = FakeAOOGrid(grid_gdf).compute()
        with pytest.raises(ValueError, match="not supported"):
            make_aoo_polygons(aoo)

    def test_ee_fc_returns_correct_type(self):
        import ee
        from rle_python_gee.aoo import (
            AOOGridEEFeatureCollection,
            AOOGridPolygonEEFeatureCollection,
        )

        aoo = AOOGridEEFeatureCollection.__new__(AOOGridEEFeatureCollection)
        aoo._ecosystems = MagicMock()
        aoo._ecosystems.ecosystem_column = 'ECO_NAME'
        aoo._asset_path = 'projects/test/assets/cache'
        aoo._computed = True
        aoo._grid_cells = None

        result = make_aoo_polygons(aoo)
        assert isinstance(result, AOOGridPolygonEEFeatureCollection)
