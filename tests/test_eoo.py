"""Tests for the EOO module."""

import pytest
import geopandas as gpd
from shapely.geometry import box, Polygon

from rle_python_gee.eoo import (
    EOO,
    EOONotComputedError,
    EOOVectorLocal,
    make_eoo,
)
from rle_python_gee.ecosystems import (
    Ecosystems,
    EcosystemKind,
    EcosystemsGeoDataFrame,
)


# ---------------------------------------------------------------------------
# Concrete subclass for testing base class logic
# ---------------------------------------------------------------------------


class FakeEcosystems(Ecosystems):
    """Minimal concrete Ecosystems subclass for testing."""
    kind = EcosystemKind.VECTOR_LOCAL

    def _load(self):
        return self._data


class FakeEOO(EOO):
    """Minimal concrete subclass for testing EOO base class."""

    def __init__(self, geometry, area_km2):
        super().__init__(ecosystems=FakeEcosystems(None, ecosystem_column="eco"))
        self._fake_geometry = geometry
        self._fake_area = area_km2

    def _compute(self) -> None:
        pass

    def _load_geometry(self):
        return self._fake_geometry

    def _load_area_km2(self) -> float:
        return self._fake_area


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _make_test_gdf() -> gpd.GeoDataFrame:
    """Create a test GeoDataFrame with three non-overlapping polygons."""
    polys = [
        box(0, 0, 1, 1),
        box(2, 0, 3, 1),
        box(1, 2, 2, 3),
    ]
    return gpd.GeoDataFrame(
        {"ECO_NAME": ["Forest", "Grassland", "Wetland"], "geometry": polys},
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# Base class tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEOOBase:
    """Tests for EOO base class properties and methods."""

    def test_not_computed_error_geometry(self):
        eoo = FakeEOO(None, 0.0)
        with pytest.raises(EOONotComputedError):
            _ = eoo.geometry

    def test_not_computed_error_area(self):
        eoo = FakeEOO(None, 0.0)
        with pytest.raises(EOONotComputedError):
            _ = eoo.area_km2

    def test_compute_returns_self(self):
        eoo = FakeEOO(box(0, 0, 1, 1), 100.0)
        result = eoo.compute()
        assert result is eoo

    def test_geometry_after_compute(self):
        geom = box(0, 0, 1, 1)
        eoo = FakeEOO(geom, 100.0).compute()
        assert eoo.geometry == geom

    def test_area_km2_after_compute(self):
        eoo = FakeEOO(box(0, 0, 1, 1), 42.5).compute()
        assert eoo.area_km2 == 42.5

    def test_repr_not_computed(self):
        eoo = FakeEOO(None, 0.0)
        assert "not computed" in repr(eoo)

    def test_repr_computed(self):
        eoo = FakeEOO(box(0, 0, 1, 1), 1234.0).compute()
        r = repr(eoo)
        assert "1,234" in r or "1234" in r

    def test_repr_html_not_computed(self):
        eoo = FakeEOO(None, 0.0)
        html = eoo._repr_html_()
        assert "compute()" in html

    def test_repr_html_computed(self):
        eoo = FakeEOO(box(0, 0, 1, 1), 500.0).compute()
        html = eoo._repr_html_()
        assert "500" in html


# ---------------------------------------------------------------------------
# EOOVectorLocal tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEOOVectorLocal:
    """Tests for EOOVectorLocal with in-memory GeoDataFrames."""

    def test_geometry_is_convex_hull(self):
        gdf = _make_test_gdf()
        eco = EcosystemsGeoDataFrame(gdf, ecosystem_column="ECO_NAME")
        eoo = EOOVectorLocal(eco).compute()

        geom = eoo.geometry
        assert isinstance(geom, Polygon)
        # Convex hull should contain all input polygons
        for poly in gdf.geometry:
            assert geom.contains(poly)

    def test_area_km2_positive(self):
        gdf = _make_test_gdf()
        eco = EcosystemsGeoDataFrame(gdf, ecosystem_column="ECO_NAME")
        eoo = EOOVectorLocal(eco).compute()

        assert eoo.area_km2 > 0

    def test_to_geodataframe(self):
        gdf = _make_test_gdf()
        eco = EcosystemsGeoDataFrame(gdf, ecosystem_column="ECO_NAME")
        eoo = EOOVectorLocal(eco).compute()

        result = eoo.to_geodataframe()
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1
        assert "area_km2" in result.columns
        assert result.crs is not None
        assert result.geometry.iloc[0].equals(eoo.geometry)

    def test_to_layer(self):
        gdf = _make_test_gdf()
        eco = EcosystemsGeoDataFrame(gdf, ecosystem_column="ECO_NAME")
        eoo = EOOVectorLocal(eco).compute()

        from lonboard import PolygonLayer

        layers = eoo.to_layer()
        assert len(layers) == 1
        assert isinstance(layers[0], PolygonLayer)

    def test_to_map(self):
        gdf = _make_test_gdf()
        eco = EcosystemsGeoDataFrame(gdf, ecosystem_column="ECO_NAME")
        eoo = EOOVectorLocal(eco).compute()

        from lonboard import Map

        m = eoo.to_map()
        assert isinstance(m, Map)

    def test_single_polygon(self):
        gdf = gpd.GeoDataFrame(
            {"ECO_NAME": ["Forest"], "geometry": [box(0, 0, 1, 1)]},
            crs="EPSG:4326",
        )
        eco = EcosystemsGeoDataFrame(gdf, ecosystem_column="ECO_NAME")
        eoo = EOOVectorLocal(eco).compute()

        assert isinstance(eoo.geometry, Polygon)
        assert eoo.area_km2 > 0


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMakeEooFactory:
    """Tests for the make_eoo factory function."""

    def test_ecosystems_geodataframe(self):
        gdf = _make_test_gdf()
        eco = EcosystemsGeoDataFrame(gdf, ecosystem_column="ECO_NAME")
        eoo = make_eoo(eco)
        assert isinstance(eoo, EOOVectorLocal)

    def test_unsupported_kind_raises(self):
        """Test that unsupported ecosystem kinds raise ValueError."""

        class FakeEEEcosystems(Ecosystems):
            kind = EcosystemKind.EE_IMAGE
            def _load(self):
                return None

        eco = FakeEEEcosystems("fake", ecosystem_column="eco")
        with pytest.raises(ValueError, match="not yet supported"):
            make_eoo(eco)

    def test_from_geojson_path(self):
        """Test factory with a .geojson file path."""
        from pathlib import Path

        test_file = Path(__file__).parent / "test_data" / "TEST_ecosystems_to_geojson_null_island.geojson"
        if not test_file.exists():
            pytest.skip("Test GeoJSON file not available")
        eoo = make_eoo(str(test_file), ecosystem_column="ECO_NAME")
        assert isinstance(eoo, EOOVectorLocal)
