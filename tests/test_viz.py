"""Tests for rle_python_gee.viz.smart_map."""

from pathlib import Path

import numpy as np
import pytest

from rle_python_gee.ecosystems import EcosystemsFile
from rle_python_gee.viz import smart_map

GEOJSON_PATH = Path(__file__).parent / "test_data" / "null_island.geojson"


@pytest.mark.unit
class TestSmartMap:
    def test_returns_lonboard_map_under_cap(self):
        eco = EcosystemsFile(GEOJSON_PATH, ecosystem_column="ECO_NAME")
        result = smart_map([eco])
        from lonboard import Map
        assert isinstance(result, Map)

    def test_static_fallback_on_max_features(self):
        eco = EcosystemsFile(GEOJSON_PATH, ecosystem_column="ECO_NAME")
        result = smart_map([(eco, {"max_features": 0})])
        from IPython.display import Image
        assert isinstance(result, Image)
        assert result.data and len(result.data) > 0

    def test_static_fallback_on_too_many(self):
        class _FakeSource:
            def to_layer(self, **_):
                raise ValueError("Dataset has 2,000 polygons, which is too many to display")
            def to_gdf_for_viz(self, **_):
                import geopandas as gpd
                gdf = gpd.read_file(GEOJSON_PATH)
                return gdf, {"fill": [0, 255, 0, 128], "edge": [0, 0, 0, 255]}
        result = smart_map([_FakeSource()])
        from IPython.display import Image
        assert isinstance(result, Image)

    def test_non_fallback_valueerror_propagates(self):
        class _Boom:
            def to_layer(self, **_):
                raise ValueError("unrelated failure")
        with pytest.raises(ValueError, match="unrelated"):
            smart_map([_Boom()])

    def test_spec_with_kwargs_tuple(self):
        eco = EcosystemsFile(GEOJSON_PATH, ecosystem_column="ECO_NAME")
        result = smart_map([(eco, {"get_fill_color": [255, 0, 0, 200]})])
        from lonboard import Map
        assert isinstance(result, Map)


@pytest.mark.unit
class TestColorConversion:
    def test_scalar_byte_rgba(self):
        from rle_python_gee.viz import _lonboard_color_to_mpl
        c = _lonboard_color_to_mpl([255, 0, 0, 128], n_rows=10)
        assert c == pytest.approx((1.0, 0.0, 0.0, 128 / 255))

    def test_scalar_rgb_defaults_alpha(self):
        from rle_python_gee.viz import _lonboard_color_to_mpl
        c = _lonboard_color_to_mpl([0, 255, 0], n_rows=10)
        assert c == pytest.approx((0.0, 1.0, 0.0, 1.0))

    def test_per_row_colors(self):
        from rle_python_gee.viz import _lonboard_color_to_mpl
        arr = np.array([[255, 0, 0, 255], [0, 255, 0, 128]], dtype=np.uint8)
        out = _lonboard_color_to_mpl(arr, n_rows=2)
        assert out.shape == (2, 4)
        assert out[0][0] == pytest.approx(1.0)
        assert out[1][1] == pytest.approx(1.0)
        assert out[1][3] == pytest.approx(128 / 255)

    def test_per_row_wrong_length_returns_none(self):
        from rle_python_gee.viz import _lonboard_color_to_mpl
        arr = np.zeros((5, 4), dtype=np.uint8)
        assert _lonboard_color_to_mpl(arr, n_rows=2) is None

    def test_none_passthrough(self):
        from rle_python_gee.viz import _lonboard_color_to_mpl
        assert _lonboard_color_to_mpl(None, n_rows=5) is None
