"""Tests for ee_rle module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import ee
from rle_python_gee import ee_rle
from google.auth import default


# Test geometry coordinates - region in Asia
TEST_GEOMETRY_COORDS = [[[96.90049099947396, 28.66344485978154],
                          [96.63681912447396, 28.185183529731013],
                          [97.71347928072396, 27.46620702497436],
                          [98.94340660274452, 27.72538824708764],
                          [97.93320584322396, 28.528398342301788],
                          [97.38422117062748, 28.654760045064048]]]


def get_test_geometry():
    """Get test geometry (only call after ee.Initialize())."""
    return ee.Geometry.Polygon(TEST_GEOMETRY_COORDS)


@pytest.mark.unit
class TestMakeEOO:
    """Tests for the make_eoo function."""

    @patch('rle_python_gee.ee_rle.ee')
    def test_make_eoo_basic(self, mock_ee):
        """Test that make_eoo calls the correct Earth Engine methods."""
        # Create mock objects for the chain of method calls
        mock_image = Mock()
        mock_masked = Mock()
        mock_vectors = Mock()
        mock_geometry = Mock()
        mock_hull = Mock()
        mock_projection = Mock()
        mock_nominal_scale = Mock()

        # Setup the chain of method calls
        mock_image.updateMask.return_value = mock_masked
        mock_masked.reduceToVectors.return_value = mock_vectors
        mock_vectors.geometry.return_value = mock_geometry
        mock_geometry.convexHull.return_value = mock_hull

        # Mock the projection and nominal scale
        mock_image.projection.return_value = mock_projection
        mock_projection.nominalScale.return_value = mock_nominal_scale
        mock_nominal_scale.getInfo.return_value = 100  # Return 100m scale

        # Create a mock geometry for the region
        mock_geo = Mock()

        # Call the function
        result = ee_rle.make_eoo(mock_image, mock_geo)

        # Verify the chain of calls
        mock_image.updateMask.assert_called_once_with(1)
        mock_image.projection.assert_called_once()
        mock_projection.nominalScale.assert_called_once()
        mock_nominal_scale.getInfo.assert_called_once()

        mock_masked.reduceToVectors.assert_called_once_with(
            scale=100,  # Should use the nominal scale (100m)
            geometry=mock_geo,
            geometryType='polygon',
            maxPixels=1e12,  # Default maxPixels parameter
            bestEffort=False  # Default changed from True to False
        )
        mock_vectors.geometry.assert_called_once()
        # convexHull is called twice (workaround for GEE bug), so we check it was called with maxError=1
        mock_geometry.convexHull.assert_called_with(maxError=1)

    @patch('rle_python_gee.ee_rle.ee')
    def test_make_eoo_custom_parameters(self, mock_ee):
        """Test make_eoo with custom parameters."""
        mock_image = Mock()
        mock_masked = Mock()
        mock_vectors = Mock()
        mock_geometry = Mock()
        mock_hull = Mock()
        mock_projection = Mock()
        mock_nominal_scale = Mock()

        mock_image.updateMask.return_value = mock_masked
        mock_masked.reduceToVectors.return_value = mock_vectors
        mock_vectors.geometry.return_value = mock_geometry
        mock_geometry.convexHull.return_value = mock_hull

        # Mock the projection and nominal scale (return a small scale)
        mock_image.projection.return_value = mock_projection
        mock_projection.nominalScale.return_value = mock_nominal_scale
        mock_nominal_scale.getInfo.return_value = 30  # Return 30m scale (< 50m)

        mock_geo = Mock()

        # Call with custom parameters
        result = ee_rle.make_eoo(
            mock_image,
            mock_geo,
            max_error=10,
            best_effort=True  # Test with True instead of default False
        )

        # Verify custom parameters were passed correctly
        mock_masked.reduceToVectors.assert_called_once_with(
            scale=50,  # Should use minimum of 50m (not the 30m nominal scale)
            geometry=mock_geo,
            geometryType='polygon',
            maxPixels=1e12,  # Default maxPixels parameter
            bestEffort=True  # Custom parameter
        )
        # convexHull is called twice, check it was called with custom maxError
        mock_geometry.convexHull.assert_called_with(maxError=10)

    @patch('rle_python_gee.ee_rle.ee')
    def test_make_eoo_returns_geometry(self, mock_ee):
        """Test that make_eoo returns an ee.Geometry object."""
        mock_image = Mock()
        mock_geo = Mock()
        mock_hull = Mock()
        mock_hull_final = Mock()
        mock_projection = Mock()
        mock_nominal_scale = Mock()

        # Mock the projection and nominal scale
        mock_image.projection.return_value = mock_projection
        mock_projection.nominalScale.return_value = mock_nominal_scale
        mock_nominal_scale.getInfo.return_value = 100

        # Setup the full chain - convexHull is called twice
        mock_hull.convexHull.return_value = mock_hull_final
        mock_image.updateMask.return_value.reduceToVectors.return_value.geometry.return_value.convexHull.return_value = mock_hull

        result = ee_rle.make_eoo(mock_image, mock_geo)

        assert result == mock_hull_final


@pytest.mark.unit
class TestAreaKm2:
    """Tests for the area_km2 function."""

    @patch('rle_python_gee.ee_rle.ee')
    def test_area_km2_basic(self, mock_ee):
        """Test that area_km2 calculates area correctly."""
        # Create mock geometry with area
        mock_geometry = Mock()
        mock_area = Mock()
        mock_area_km2 = Mock()

        mock_geometry.area.return_value = mock_area
        mock_area.divide.return_value = mock_area_km2

        result = ee_rle.area_km2(mock_geometry)

        # Verify area was called
        mock_geometry.area.assert_called_once()
        # Verify division by 1e6 (convert m² to km²)
        mock_area.divide.assert_called_once_with(1e6)
        # Verify result
        assert result == mock_area_km2

    @patch('rle_python_gee.ee_rle.ee')
    def test_area_km2_returns_ee_number(self, mock_ee):
        """Test that area_km2 returns an ee.Number."""
        mock_geometry = Mock()
        mock_result = Mock()
        mock_geometry.area.return_value.divide.return_value = mock_result

        result = ee_rle.area_km2(mock_geometry)

        assert result == mock_result


@pytest.mark.unit
class TestEnsureAssetFolderExists:
    """Tests for the ensure_asset_folder_exists function."""

    @patch('rle_python_gee.ee_rle.ee.data')
    def test_folder_already_exists(self, mock_data):
        """Test when folder already exists."""
        # Setup: getAsset succeeds (folder exists)
        mock_data.getAsset.return_value = {'type': 'FOLDER', 'id': 'test/folder'}

        # Call the function
        result = ee_rle.ensure_asset_folder_exists('projects/test/assets/folder')

        # Verify getAsset was called
        mock_data.getAsset.assert_called_once_with('projects/test/assets/folder')
        # Verify createFolder was NOT called
        mock_data.createFolder.assert_not_called()
        # Verify function returns False (not created)
        assert result is False

    @patch('rle_python_gee.ee_rle.ee.data')
    def test_folder_does_not_exist(self, mock_data):
        """Test when folder doesn't exist and needs to be created."""
        # Setup: getAsset raises exception (folder doesn't exist)
        mock_data.getAsset.side_effect = ee.EEException('Asset not found')
        mock_data.createFolder.return_value = {'type': 'FOLDER', 'id': 'test/folder'}

        # Call the function
        result = ee_rle.ensure_asset_folder_exists('projects/test/assets/folder')

        # Verify getAsset was called
        mock_data.getAsset.assert_called_once_with('projects/test/assets/folder')
        # Verify createFolder WAS called
        mock_data.createFolder.assert_called_once_with('projects/test/assets/folder')
        # Verify function returns True (was created)
        assert result is True

    @patch('rle_python_gee.ee_rle.ee.data')
    def test_folder_creation_with_ecosystem_code(self, mock_data):
        """Test folder creation with realistic ecosystem folder path."""
        # Setup: folder doesn't exist
        mock_data.getAsset.side_effect = ee.EEException('Asset not found')
        mock_data.createFolder.return_value = {'type': 'FOLDER'}

        folder_path = 'projects/goog-rle-assessments/assets/MMR-T1_1_1'
        result = ee_rle.ensure_asset_folder_exists(folder_path)

        # Verify createFolder was called with the correct path
        mock_data.createFolder.assert_called_once_with(folder_path)
        assert result is True


@pytest.mark.unit
class TestCreateAssetFolder:
    """Tests for the create_asset_folder function."""

    @patch('rle_python_gee.ee_rle.ee.data')
    def test_create_folder_when_not_exists(self, mock_data):
        """Test folder creation when folder doesn't exist."""
        # Setup: getAsset raises exception (folder doesn't exist)
        mock_data.getAsset.side_effect = ee.EEException('Asset not found')
        mock_data.createFolder.return_value = {'type': 'FOLDER', 'id': 'test/folder'}

        # Call the function
        result = ee_rle.create_asset_folder('projects/test/assets/folder')

        # Verify getAsset was called to check existence
        mock_data.getAsset.assert_called_once_with('projects/test/assets/folder')
        # Verify createFolder was called
        mock_data.createFolder.assert_called_once_with('projects/test/assets/folder')
        # Verify function returns True (folder was created)
        assert result is True

    @patch('rle_python_gee.ee_rle.ee.data')
    def test_create_folder_when_already_exists(self, mock_data):
        """Test folder creation when folder already exists."""
        # Setup: getAsset succeeds (folder exists)
        mock_data.getAsset.return_value = {'type': 'FOLDER', 'id': 'test/folder'}

        # Call the function
        result = ee_rle.create_asset_folder('projects/test/assets/folder')

        # Verify getAsset was called
        mock_data.getAsset.assert_called_once_with('projects/test/assets/folder')
        # Verify createFolder was NOT called
        mock_data.createFolder.assert_not_called()
        # Verify function returns False (folder already existed)
        assert result is False

    @patch('rle_python_gee.ee_rle.ee.data')
    def test_create_folder_with_ecosystem_path(self, mock_data):
        """Test folder creation with realistic ecosystem folder path."""
        # Setup: folder doesn't exist
        mock_data.getAsset.side_effect = ee.EEException('Asset not found')
        mock_data.createFolder.return_value = {'type': 'FOLDER'}

        folder_path = 'projects/goog-rle-assessments/assets/MMR-T1_1_2'
        result = ee_rle.create_asset_folder(folder_path)

        # Verify createFolder was called with the correct path
        mock_data.createFolder.assert_called_once_with(folder_path)
        assert result is True


@pytest.mark.unit
class TestMakeAOO:
    """Tests for the make_aoo function (now returns AOOGrid)."""

    def test_make_aoo_returns_aoo_grid_for_ee_image(self):
        """Test that make_aoo returns an AOOGridEEImage for ee.Image objects."""
        from rle_python_gee.aoo import AOOGridEEImage

        mock_image = MagicMock(spec=ee.Image)
        result = ee_rle.make_aoo(mock_image)
        assert isinstance(result, AOOGridEEImage)

    def test_make_aoo_returns_aoo_grid_for_parquet(self):
        """Test that make_aoo returns AOOGridGeoParquet for .parquet paths."""
        from rle_python_gee.aoo import AOOGridGeoParquet

        result = ee_rle.make_aoo("/fake/path.parquet")
        assert isinstance(result, AOOGridGeoParquet)

    def test_make_aoo_returns_aoo_grid_for_ee_fc(self):
        """Test that make_aoo returns AOOGridEEFeatureCollection for ee.FeatureCollection."""
        from rle_python_gee.aoo import AOOGridEEFeatureCollection

        mock_fc = MagicMock(spec=ee.FeatureCollection)
        result = ee_rle.make_aoo(
            mock_fc, ecosystem_column='ECO_NAME',
            asset_path='projects/test/assets/cache',
        )
        assert isinstance(result, AOOGridEEFeatureCollection)


@pytest.mark.integration
class TestIntegrationWithRealEE:
    """Integration tests using real Earth Engine objects (requires authentication)."""

    @pytest.fixture(autouse=True)
    def setup_ee(self):
        """Initialize Earth Engine before each test."""
        try:
            # ee.Initialize()
            credentials, _ = default(scopes=[
                'https://www.googleapis.com/auth/earthengine',
                'https://www.googleapis.com/auth/cloud-platform'
            ])
            ee.Initialize(credentials=credentials, project='goog-rle-assessments')
        except Exception:
            pytest.skip("Earth Engine not authenticated - skipping integration tests")

    def test_make_eoo_with_real_geometry(self):
        """Test make_eoo with real Earth Engine geometry."""
        test_geometry = get_test_geometry()

        # Create a simple binary image covering the test region
        # Using a constant image with value 1 (presence)
        test_image = ee.Image(1).clip(test_geometry)

        # Calculate EOO
        eoo_poly = ee_rle.make_eoo(test_image, test_geometry)

        # Verify result is an ee.Geometry
        assert isinstance(eoo_poly, ee.Geometry)

        # Verify the EOO is not empty (should have computed geometry)
        eoo_info = eoo_poly.getInfo()
        assert eoo_info is not None
        assert eoo_info['type'] in ['Polygon', 'MultiPolygon']

    def test_area_km2_with_real_geometry(self):
        """Test area_km2 with real Earth Engine geometry.

        Test based on:
        https://github.com/red-list-ecosystem/gee-redlist/blob/4c58f8d1adc2853dd9d1be295f9def37cbe9f4a6/Modules/functionTests

        Note: With the dynamic scale calculation update, the exact area may vary slightly
        from the original test value (12634.46 km²) depending on the reduction scale used.
        The test now uses a larger maxError to accommodate the coarser scale.
        """
        test_geometry = get_test_geometry()

        # Create a simple binary image
        elevation = ee.Image('USGS/SRTMGL1_003').clip(test_geometry)
        test_image = ee.Image(1).clip(test_geometry).updateMask(elevation.gte(4500))

        # Calculate EOO polygon using bestEffort=True
        eoo_poly = ee_rle.make_eoo(
            class_img=test_image,
            geo=test_geometry,
            scale=100,
            best_effort=True
        )

        # Calculate area using area_km2
        area = ee_rle.area_km2(eoo_poly)

        # Verify result is an ee.Number
        assert isinstance(area, ee.Number)

        # Get the actual value and verify it's reasonable
        # Area should be in a reasonable range (allowing for variation due to scale changes)
        area_val = area.getInfo()
        assert area_val > 10000, f"Expected area > 10000 km², got {area_val} km²"
        assert area_val < 15000, f"Expected area < 15000 km², got {area_val} km²"

    def test_export_fractional_coverage_on_aoo_grid(self):
        """Test export_fractional_coverage_on_aoo_grid with real Earth Engine objects."""
        import time
        test_geometry = get_test_geometry()

        # Create a simple binary image covering the test region
        test_image = ee.Image('projects/goog-rle-assessments/assets/mm_ecosys_v7b').eq(52).selfMask()

        # Use a timestamped folder to avoid conflicts
        test_folder = f'test_export_{int(time.time())}'
        asset_id = f'projects/goog-rle-assessments/assets/{test_folder}/grid'

        # Call the export function (will create the folder automatically)
        task = ee_rle.export_fractional_coverage_on_aoo_grid(
            class_img=test_image,
            asset_id=asset_id,
            export_description='integration_test_export_fractionalCoverage',
            max_pixels=65536
        )

        # Verify a task was returned
        assert task is not None
        assert isinstance(task.id, str)
        assert len(task.id) > 0
        assert task.task_type == 'EXPORT_IMAGE'
        # assert task.state in ['READY', 'RUNNING', 'COMPLETED']

        # Verify the task was created in Earth Engine
        # We can check the task status
        task_list = ee.batch.Task.list()
        task_ids = [task.id for task in task_list]
        assert task.id in task_ids

        # Cancel the task to clean up (we don't actually want to export)
        task.cancel()

        # Clean up: delete the test folder
        try:
            folder_path = f'projects/goog-rle-assessments/assets/{test_folder}'
            ee.data.deleteAsset(folder_path)
        except Exception:
            pass  # Ignore errors during cleanup

    def test_ensure_asset_folder_exists_integration(self):
        """Integration test for ensure_asset_folder_exists with real Earth Engine."""
        import time

        # Use a test folder path that we can safely create and delete
        test_folder = f'projects/goog-rle-assessments/assets/test_folder_{int(time.time())}'

        try:
            # First call should create the folder
            result = ee_rle.ensure_asset_folder_exists(test_folder)
            assert result is True, "First call should create folder and return True"

            # Verify folder was created by checking it exists
            asset_info = ee.data.getAsset(test_folder)
            assert asset_info is not None
            assert asset_info['type'] == 'FOLDER'

            # Second call should find existing folder
            result = ee_rle.ensure_asset_folder_exists(test_folder)
            assert result is False, "Second call should find existing folder and return False"

        finally:
            # Clean up: delete the test folder
            try:
                ee.data.deleteAsset(test_folder)
            except Exception:
                pass  # Ignore errors during cleanup

    def test_create_asset_folder_integration(self):
        """Integration test for create_asset_folder with real Earth Engine."""
        import time

        # Use a test folder path that we can safely create and delete
        test_folder = f'projects/goog-rle-assessments/assets/test_create_folder_{int(time.time())}'

        try:
            # First call should create the folder
            result = ee_rle.create_asset_folder(test_folder)
            assert result is True, "First call should create folder and return True"

            # Verify folder was actually created by checking it exists
            asset_info = ee.data.getAsset(test_folder)
            assert asset_info is not None, "Folder should exist after creation"
            assert asset_info['type'] == 'FOLDER', "Asset should be of type FOLDER"
            assert test_folder in asset_info['name'], "Asset name should match test folder path"

            # Second call should find existing folder
            result = ee_rle.create_asset_folder(test_folder)
            assert result is False, "Second call should find existing folder and return False"

        finally:
            # Clean up: delete the test folder
            try:
                ee.data.deleteAsset(test_folder)
            except Exception:
                pass  # Ignore errors during cleanup

    def test_make_aoo_integration(self):
        """Integration test for make_aoo with real Earth Engine."""
        from rle_python_gee.aoo import AOOGrid

        # Use an existing fractional coverage asset from the test ecosystem
        asset_id = 'projects/goog-rle-assessments/assets/MMR-T1_1_1/a00_grid'

        aoo = ee_rle.make_aoo(asset_id)
        assert isinstance(aoo, AOOGrid)

        # Verify the cell count is reasonable
        assert aoo.cell_count > 0, f"Expected cell_count > 0, got {aoo.cell_count}"
        assert aoo.cell_count < 100000, f"Expected cell_count < 100000, got {aoo.cell_count}"
        assert aoo.aoo_km2 > 0

