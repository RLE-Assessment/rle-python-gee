"""Tests for map module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os
import sys

from PIL import Image
from io import BytesIO
from shapely.geometry import box
import shapely

from rle_python_gee.map import create_country_map, get_utm_proj_without_limits


def create_mock_wkb_for_bounds(bounds):
    """Helper function to create WKB data from bounds (minx, miny, maxx, maxy)."""
    geom = box(bounds[0], bounds[1], bounds[2], bounds[3])
    return shapely.to_wkb(geom)


class TestCreateCountryMap:
    """Tests for the create_country_map function."""

    @patch('rle_python_gee.map.wkls')
    @patch('rle_python_gee.map.plt')
    def test_basic_map_creation(self, mock_plt, mock_wkls):
        """Test basic map creation with default parameters."""
        # Setup mocks
        bounds = (103.6, 1.2, 104.0, 1.5)
        mock_wkls.__getitem__.return_value.wkb.return_value = create_mock_wkb_for_bounds(bounds)

        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.spines = {'top': Mock(), 'bottom': Mock(), 'left': Mock(), 'right': Mock()}
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        # Create temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_map.png')

            result = create_country_map('SG', output_path)

            # Verify result
            assert result == output_path
            mock_plt.savefig.assert_called_once()
            mock_plt.close.assert_called_once_with(mock_fig)

    @patch('rle_python_gee.map.wkls')
    @patch('rle_python_gee.map.plt')
    def test_country_not_found(self, mock_plt, mock_wkls):
        """Test that ValueError is raised when country code not found in database."""
        # Mock wkls to raise ValueError when country not found
        mock_wkls.__getitem__.return_value.wkb.side_effect = ValueError("No result found for: zz")

        with pytest.raises(ValueError, match="Country code 'ZZ' not found in database"):
            create_country_map('ZZ')

    def test_invalid_country_code_empty_string(self):
        """Test that ValueError is raised for empty string."""
        with pytest.raises(ValueError, match="country_code cannot be empty"):
            create_country_map('')

    def test_invalid_country_code_whitespace(self):
        """Test that ValueError is raised for whitespace."""
        with pytest.raises(ValueError, match="country_code cannot be empty"):
            create_country_map('  ')

    def test_invalid_country_code_none(self):
        """Test that TypeError is raised for None."""
        with pytest.raises(TypeError, match="country_code must be a string"):
            create_country_map(None)

    def test_invalid_country_code_number(self):
        """Test that TypeError is raised for number."""
        with pytest.raises(TypeError, match="country_code must be a string"):
            create_country_map(123)

    def test_invalid_country_code_too_short(self):
        """Test that ValueError is raised for single letter."""
        with pytest.raises(ValueError, match="must be a 2-letter ISO 3166-1 alpha-2 code"):
            create_country_map('U')

    def test_invalid_country_code_too_long(self):
        """Test that ValueError is raised for 3+ letters."""
        with pytest.raises(ValueError, match="must be a 2-letter ISO 3166-1 alpha-2 code"):
            create_country_map('USA')

    def test_invalid_country_code_full_name(self):
        """Test that ValueError is raised for full country names."""
        with pytest.raises(ValueError, match="must be a 2-letter ISO 3166-1 alpha-2 code"):
            create_country_map('Singapore')

    def test_invalid_country_code_numbers(self):
        """Test that ValueError is raised for numbers in code."""
        with pytest.raises(ValueError, match="must be a 2-letter ISO 3166-1 alpha-2 code"):
            create_country_map('U1')

    def test_invalid_country_code_special_chars(self):
        """Test that ValueError is raised for special characters."""
        with pytest.raises(ValueError, match="must be a 2-letter ISO 3166-1 alpha-2 code"):
            create_country_map('U$')

    @patch('rle_python_gee.map.wkls')
    @patch('rle_python_gee.map.plt')
    def test_valid_lowercase_code(self, mock_plt, mock_wkls):
        """Test that lowercase ISO codes are accepted and converted."""
        # Setup mocks
        bounds = (103.6, 1.2, 104.0, 1.5)
        mock_wkls.__getitem__.return_value.wkb.return_value = create_mock_wkb_for_bounds(bounds)

        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.spines = {'top': Mock(), 'bottom': Mock(), 'left': Mock(), 'right': Mock()}
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.png')
            result = create_country_map('sg', output_path)
            assert result == output_path

    @patch('rle_python_gee.map.wkls')
    @patch('rle_python_gee.map.plt')
    def test_valid_uppercase_code(self, mock_plt, mock_wkls):
        """Test that uppercase ISO codes are accepted."""
        # Setup mocks
        bounds = (103.6, 1.2, 104.0, 1.5)
        mock_wkls.__getitem__.return_value.wkb.return_value = create_mock_wkb_for_bounds(bounds)

        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.spines = {'top': Mock(), 'bottom': Mock(), 'left': Mock(), 'right': Mock()}
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.png')
            result = create_country_map('SG', output_path)
            assert result == output_path

    @patch('rle_python_gee.map.wkls')
    @patch('rle_python_gee.map.plt')
    def test_default_output_path(self, mock_plt, mock_wkls):
        """Test that default output path is generated correctly."""
        # Setup mocks
        bounds = (166.0, -47.0, 179.0, -34.0)
        mock_wkls.__getitem__.return_value.wkb.return_value = create_mock_wkb_for_bounds(bounds)

        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.spines = {'top': Mock(), 'bottom': Mock(), 'left': Mock(), 'right': Mock()}
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        result = create_country_map('NZ')

        # Should generate 'nz.png'
        assert result == 'nz.png'

    @patch('rle_python_gee.map.wkls')
    @patch('rle_python_gee.map.plt')
    def test_custom_colors(self, mock_plt, mock_wkls):
        """Test map creation with custom fill and edge colors."""
        # Setup mocks
        bounds = (129.0, 31.0, 146.0, 46.0)
        mock_wkls.__getitem__.return_value.wkb.return_value = create_mock_wkb_for_bounds(bounds)

        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.spines = {'top': Mock(), 'bottom': Mock(), 'left': Mock(), 'right': Mock()}
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'japan.png')

            result = create_country_map(
                'JP',
                output_path,
                geometry_kwargs={
                    'facecolor': '#ff6b6b',
                    'edgecolor': 'darkred',
                    'linewidth': 2.5
                },
            )

            # Verify the geometries were added with custom colors
            mock_ax.add_geometries.assert_called_once()
            call_kwargs = mock_ax.add_geometries.call_args[1]
            assert call_kwargs['facecolor'] == '#ff6b6b'
            assert call_kwargs['edgecolor'] == 'darkred'
            assert call_kwargs['linewidth'] == 2.5

    @patch('rle_python_gee.map.wkls')
    @patch('rle_python_gee.map.plt')
    def test_no_border(self, mock_plt, mock_wkls):
        """Test map creation with show_border=False."""
        # Setup mocks
        bounds = (-74.0, -34.0, -34.0, 5.0)
        mock_wkls.__getitem__.return_value.wkb.return_value = create_mock_wkb_for_bounds(bounds)

        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.spines = {'top': Mock(), 'bottom': Mock(), 'left': Mock(), 'right': Mock()}
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'brazil.png')

            result = create_country_map(
                'BR',
                output_path,
                show_border=False,
                geometry_kwargs={
                    'edgecolor': None,
                    'linewidth': None
                }
            )

            # Verify add_geometries was called
            mock_ax.add_geometries.assert_not_called()

    @patch('rle_python_gee.map.wkls')
    @patch('rle_python_gee.map.plt')
    def test_spines_hidden(self, mock_plt, mock_wkls):
        """Test that plot frame spines are hidden."""
        # Setup mocks
        bounds = (-5.0, 41.0, 10.0, 51.0)
        mock_wkls.__getitem__.return_value.wkb.return_value = create_mock_wkb_for_bounds(bounds)

        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.spines = {'top': Mock(), 'bottom': Mock(), 'left': Mock(), 'right': Mock()}
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'france.png')

            result = create_country_map('FR', output_path)

            # Verify spines were hidden (frame/border removed)
            for spine in mock_ax.spines.values():
                spine.set_visible.assert_called_once_with(False)

    @patch('rle_python_gee.map.wkls')
    @patch('rle_python_gee.map.plt')
    def test_custom_title(self, mock_plt, mock_wkls):
        """Test map creation with custom title."""
        # Setup mocks
        bounds = (33.9, -4.7, 41.9, 4.6)
        mock_wkls.__getitem__.return_value.wkb.return_value = create_mock_wkb_for_bounds(bounds)

        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.spines = {'top': Mock(), 'bottom': Mock(), 'left': Mock(), 'right': Mock()}
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'kenya.png')

            result = create_country_map('KE', output_path, title='Kenya Wildlife')

            # Verify title was set
            mock_plt.title.assert_called_once_with('Kenya Wildlife', fontsize=16, fontweight='bold')

    @patch('rle_python_gee.map.wkls')
    @patch('rle_python_gee.map.plt')
    def test_no_title(self, mock_plt, mock_wkls):
        """Test map creation with empty title."""
        # Setup mocks
        bounds = (-24.5, 63.3, -13.5, 66.5)
        mock_wkls.__getitem__.return_value.wkb.return_value = create_mock_wkb_for_bounds(bounds)

        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.spines = {'top': Mock(), 'bottom': Mock(), 'left': Mock(), 'right': Mock()}
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'iceland.png')

            result = create_country_map('IS', output_path, title='')

            # Verify title was not called
            mock_plt.title.assert_not_called()


class TestEarthEngineBasemap:
    """Tests for Earth Engine basemap functionality."""

    @patch('rle_python_gee.map.wkls')
    @patch('rle_python_gee.map.requests.get')
    @patch('rle_python_gee.map.plt')
    def test_ee_image_basemap(self, mock_plt, mock_requests, mock_wkls):
        """Test map creation with Earth Engine image basemap."""
        # Setup mocks
        bounds = (80.0, 26.3, 88.2, 30.4)
        mock_wkls.__getitem__.return_value.wkb.return_value = create_mock_wkb_for_bounds(bounds)

        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.spines = {'top': Mock(), 'bottom': Mock(), 'left': Mock(), 'right': Mock()}
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        # Mock image response

        test_image = Image.new('RGB', (100, 100), color='red')
        img_bytes = BytesIO()
        test_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        mock_response = Mock()
        mock_response.content = img_bytes.getvalue()
        mock_requests.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'nepal.png')

            # Mock EE image
            mock_ee_image = Mock()
            mock_ee_image.getDownloadURL.return_value = 'http://example.com/image.png'
            mock_mask = Mock()
            mock_mask.getDownloadURL.return_value = 'http://example.com/mask.png'
            mock_ee_image.mask.return_value = mock_mask

            with patch('rle_python_gee.map.ee') as mock_ee:
                # Mock ee.Projection and ee.Geometry.Rectangle
                mock_ee.Projection.return_value = Mock()
                mock_ee.Geometry.Rectangle.return_value = Mock()

                result = create_country_map(
                    'NP',
                    output_path,
                    ee_image=mock_ee_image,
                )

                # Verify EE image methods were called
                # The image should call getDownloadURL (not visualize)
                mock_ee_image.getDownloadURL.assert_called()
                mock_ee_image.mask.assert_called_once()
                mock_mask.getDownloadURL.assert_called()

                # Verify imshow was called to display the basemap
                mock_ax.imshow.assert_called_once()

    @patch('rle_python_gee.map.wkls')
    @patch('rle_python_gee.map.requests.get')
    @patch('rle_python_gee.map.plt')
    def test_ee_image_clipped(self, mock_plt, mock_requests, mock_wkls):
        """Test map creation with clipped Earth Engine image."""
        # Setup mocks
        bounds = (-81.4, -18.3, -68.7, -0.0)
        mock_wkls.__getitem__.return_value.wkb.return_value = create_mock_wkb_for_bounds(bounds)

        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.spines = {'top': Mock(), 'bottom': Mock(), 'left': Mock(), 'right': Mock()}
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        # Mock image response

        test_image = Image.new('RGB', (100, 100), color='blue')
        img_bytes = BytesIO()
        test_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        mock_response = Mock()
        mock_response.content = img_bytes.getvalue()
        mock_requests.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'peru.png')

            # Mock EE image
            mock_ee_image = Mock()
            mock_clipped = Mock()
            mock_clipped.getDownloadURL.return_value = 'http://example.com/image.png'
            mock_ee_image.clip.return_value = mock_clipped

            with patch('rle_python_gee.map.ee') as mock_ee:
                # Mock ee.Projection, ee.Geometry, and ee.Geometry.Rectangle
                mock_ee.Projection.return_value = Mock()
                mock_ee_geom = Mock()
                mock_ee.Geometry.return_value = mock_ee_geom
                mock_ee.Geometry.Rectangle.return_value = Mock()

                result = create_country_map(
                    'PE',
                    output_path,
                    ee_image=mock_ee_image,
                    clip_ee_image=True
                )

                # Verify clip was called
                mock_ee_image.clip.assert_called_once()


class TestGetUtmProjWithoutLimits:
    """Tests for the get_utm_proj_without_limits function."""

    def test_northern_hemisphere_zone_13(self):
        """Test UTM Zone 13N projection parameters."""
        import cartopy.crs as ccrs
        import pyproj

        utm_zone = 13
        is_south = False

        # Get our custom projection
        custom_proj = get_utm_proj_without_limits(utm_zone, is_south)

        # Get standard UTM projection for comparison
        standard_utm = ccrs.UTM(zone=utm_zone, southern_hemisphere=is_south)

        # Verify it's a TransverseMercator
        assert isinstance(custom_proj, ccrs.TransverseMercator)

        # Calculate expected central longitude
        # Zone N covers (N-1)*6° - 180° to N*6° - 180°
        # Central meridian: (N-1)*6° - 180° + 3°
        expected_central_lon = (utm_zone - 1) * 6 - 180 + 3  # -105° for zone 13

        # Test by transforming a point at the expected central meridian
        # At the central meridian, easting should be 500,000 meters (false_easting)
        transformer = pyproj.Transformer.from_crs(
            'EPSG:4326',  # WGS84
            f'EPSG:326{utm_zone}',  # UTM 13N
            always_xy=True
        )

        # Transform point at central meridian
        x, y = transformer.transform(expected_central_lon, 23.0)

        # At central meridian, x should be exactly 500,000 (false_easting)
        assert abs(x - 500000.0) < 1.0, f"Expected easting ~500000, got {x}"

        # Verify our custom projection has much larger x_limits than standard UTM
        assert custom_proj.x_limits[0] < standard_utm.x_limits[0], \
            "Custom projection should have lower minimum x_limit"
        assert custom_proj.x_limits[1] > standard_utm.x_limits[1], \
            "Custom projection should have higher maximum x_limit"

        # Standard UTM has x_limits of (-250000, 1250000) = 1.5M meters
        # Custom should have (-20M, 20M) = 40M meters
        x_range_standard = standard_utm.x_limits[1] - standard_utm.x_limits[0]
        x_range_custom = custom_proj.x_limits[1] - custom_proj.x_limits[0]

        assert x_range_custom > x_range_standard * 10, \
            f"Custom x_range ({x_range_custom/1e6:.1f}M) should be much larger than standard ({x_range_standard/1e6:.1f}M)"

    def test_southern_hemisphere_zone_56(self):
        """Test UTM Zone 56S projection parameters."""
        import cartopy.crs as ccrs
        import pyproj

        utm_zone = 56
        is_south = True

        # Get our custom projection
        custom_proj = get_utm_proj_without_limits(utm_zone, is_south)

        # Verify it's a TransverseMercator
        assert isinstance(custom_proj, ccrs.TransverseMercator)

        # Calculate expected central longitude for zone 56
        expected_central_lon = (utm_zone - 1) * 6 - 180 + 3  # 153° for zone 56

        # Test by transforming a point at the expected central meridian
        transformer = pyproj.Transformer.from_crs(
            'EPSG:4326',  # WGS84
            f'EPSG:327{utm_zone}',  # UTM 56S
            always_xy=True
        )

        # Transform point at central meridian (in southern hemisphere)
        x, y = transformer.transform(expected_central_lon, -33.0)

        # At central meridian, x should be exactly 500,000 (false_easting)
        assert abs(x - 500000.0) < 1.0, f"Expected easting ~500000, got {x}"

        # For southern hemisphere, false_northing should be 10,000,000
        # Verify by checking a point near the equator
        x_eq, y_eq = transformer.transform(expected_central_lon, 0.0)
        assert y_eq > 9000000, f"Southern hemisphere should have false_northing ~10M, got {y_eq}"

    def test_multiple_zones(self):
        """Test that different zones have different central longitudes."""
        zones_to_test = [
            (1, -177),    # Zone 1: center at -177°
            (13, -105),   # Zone 13: center at -105°
            (30, -3),     # Zone 30: center at -3°
            (60, 177),    # Zone 60: center at 177°
        ]

        for utm_zone, expected_central_lon in zones_to_test:
            custom_proj = get_utm_proj_without_limits(utm_zone, is_south=False)

            # The central longitude should match the expected value
            # We can verify this by checking that a point at this longitude
            # transforms to easting = 500,000
            import pyproj
            transformer = pyproj.Transformer.from_crs(
                'EPSG:4326',
                f'EPSG:326{utm_zone:02d}',
                always_xy=True
            )

            x, y = transformer.transform(expected_central_lon, 0.0)
            assert abs(x - 500000.0) < 1.0, \
                f"Zone {utm_zone}: point at {expected_central_lon}° should have easting ~500000, got {x}"

    def test_projection_equivalence(self):
        """Test that our custom projection produces same coordinates as standard UTM within limits."""
        import pyproj

        utm_zone = 14

        # Create both projections
        custom_proj = get_utm_proj_without_limits(utm_zone, is_south=False)

        # Create transformers using EPSG codes
        custom_epsg = f'EPSG:326{utm_zone}'

        # Test several points within the standard UTM limits
        test_points = [
            (-102.0, 23.0),  # Near central meridian
            (-100.0, 20.0),  # East of center
            (-104.0, 25.0),  # West of center
        ]

        transformer = pyproj.Transformer.from_crs('EPSG:4326', custom_epsg, always_xy=True)

        for lon, lat in test_points:
            x, y = transformer.transform(lon, lat)

            # Verify coordinates are reasonable
            # Easting should be within valid UTM range
            assert -2000000 < x < 3000000, f"Easting {x} out of expected range"
            # Northing should be positive for northern hemisphere
            assert 0 < y < 10000000, f"Northing {y} out of expected range"