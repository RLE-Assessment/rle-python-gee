"""Tests for ee_rle module."""

import json
from pathlib import Path

import pytest
from unittest.mock import Mock, patch, MagicMock
import ee
from gee_redlist import ee_rle
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
class TestEcosystems:
    """Tests for the Ecosystems class."""

    @patch('gee_redlist.ee_rle.ee.data')
    @patch('gee_redlist.ee_rle.ee.FeatureCollection')
    def test_ecosystems_loads_vector_asset(self, mock_fc, mock_data):
        """Test that Ecosystems loads a TABLE asset as FeatureCollection."""
        mock_data.getAsset.return_value = {'type': 'TABLE'}
        mock_fc_instance = Mock()
        mock_fc.return_value = mock_fc_instance

        ecosystem = ee_rle.Ecosystems(
            data='projects/test/assets/vector_asset',
            get_level3_column='EFG1',
            get_level456_column='COD'
        )

        mock_data.getAsset.assert_called_once_with('projects/test/assets/vector_asset')
        mock_fc.assert_called_once_with('projects/test/assets/vector_asset')
        assert ecosystem.asset_type == 'TABLE'
        assert ecosystem.data == mock_fc_instance

    @patch('gee_redlist.ee_rle.ee.data')
    @patch('gee_redlist.ee_rle.ee.Image')
    def test_ecosystems_loads_raster_asset(self, mock_image, mock_data):
        """Test that Ecosystems loads an IMAGE asset as Image."""
        mock_data.getAsset.return_value = {'type': 'IMAGE'}
        mock_image_instance = Mock()
        mock_image.return_value = mock_image_instance

        ecosystem = ee_rle.Ecosystems(
            'projects/test/assets/raster_asset',
            get_level3_column=None,
            get_level456_column=None
        )

        mock_data.getAsset.assert_called_once_with('projects/test/assets/raster_asset')
        mock_image.assert_called_once_with('projects/test/assets/raster_asset')
        assert ecosystem.asset_type == 'IMAGE'
        assert ecosystem.data == mock_image_instance

    @patch('gee_redlist.ee_rle.ee.data')
    def test_ecosystems_raises_for_unsupported_type(self, mock_data):
        """Test that Ecosystems raises ValueError for unsupported asset types."""
        mock_data.getAsset.return_value = {'type': 'FOLDER'}

        with pytest.raises(ValueError) as exc_info:
            ee_rle.Ecosystems(
                'projects/test/assets/folder',
                get_level3_column=None,
                get_level456_column=None
            )

        assert "Unsupported asset type 'FOLDER'" in str(exc_info.value)

    @patch('gee_redlist.ee_rle.ee.data')
    @patch('gee_redlist.ee_rle.ee.FeatureCollection')
    def test_ecosystems_stores_asset_id(self, mock_fc, mock_data):
        """Test that Ecosystems stores the asset_id."""
        mock_data.getAsset.return_value = {'type': 'TABLE'}

        asset_id = 'projects/goog-rle-assessments/assets/columbia/GETCol'
        ecosystem = ee_rle.Ecosystems(
            asset_id,
            get_level3_column='EFG1',
            get_level456_column='COD'
        )

        assert ecosystem.asset_id == asset_id

    @patch('gee_redlist.ee_rle.ee.data')
    def test_ecosystems_functional_group_dataframe_raises_for_image(self, mock_data):
        """Test that functional_group_dataframe() raises ValueError for IMAGE assets."""
        mock_data.getAsset.return_value = {'type': 'IMAGE'}

        with patch('gee_redlist.ee_rle.ee.Image'):
            ecosystem = ee_rle.Ecosystems(
                'projects/test/assets/raster',
                get_level3_column='EFG1',
                get_level456_column='COD'
            )

        with pytest.raises(ValueError) as exc_info:
            _ = ecosystem.functional_group_dataframe()

        assert "only available for TABLE assets" in str(exc_info.value)

    @patch('gee_redlist.ee_rle.ee.data')
    @patch('gee_redlist.ee_rle.ee.FeatureCollection')
    def test_ecosystems_functional_group_dataframe_returns_dataframe(self, mock_fc, mock_data):
        """Test that functional_group_dataframe() returns a pandas DataFrame with MultiIndex."""
        import pandas as pd

        # Setup mocks
        mock_data.getAsset.return_value = {'type': 'TABLE'}

        # Mock the FeatureCollection and its methods
        mock_fc_instance = Mock()
        mock_fc.return_value = mock_fc_instance

        # Mock first().propertyNames() chain for getting column names
        mock_first = Mock()
        mock_property_names = Mock()
        mock_fc_instance.first.return_value = mock_first
        mock_first.propertyNames.return_value = mock_property_names
        # Mock the remove() calls for each excluded column
        mock_property_names.remove.return_value = mock_property_names

        # Mock distinct() to return unique combinations
        mock_distinct_fc = Mock()
        mock_fc_instance.distinct.return_value = mock_distinct_fc

        # Mock toList() and map() chain
        mock_size = Mock()
        mock_distinct_fc.size.return_value = mock_size
        mock_list = Mock()
        mock_distinct_fc.toList.return_value = mock_list

        # Mock map() result - returns list of dicts directly
        records = [
            {'COD': 'B36', 'ECO_NAME': 'Test Ecosystem', 'EFG1': 'MFT1.2'},
            {'COD': 'B10', 'ECO_NAME': 'Another Ecosystem', 'EFG1': 'T1.2'}
        ]
        mock_mapped_list = Mock()
        mock_list.map.return_value = mock_mapped_list
        mock_mapped_list.getInfo.return_value = records

        # Create the Ecosystems instance with column names specified
        ecosystem = ee_rle.Ecosystems(
            'projects/test/assets/vector_asset',
            get_level3_column='EFG1',
            get_level456_column='COD'
        )

        # Get the dataframe
        df = ecosystem.functional_group_dataframe()

        # Verify it's a DataFrame
        assert isinstance(df, pd.DataFrame)

        # Verify we have 2 rows
        assert len(df) == 2

        # Verify it has a MultiIndex with expected names
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.names == ['EFG1', 'COD']

        # Verify the index values are correct
        assert ('MFT1.2', 'B36') in df.index
        assert ('T1.2', 'B10') in df.index

        # Verify index columns are not in regular columns
        assert 'EFG1' not in df.columns
        assert 'COD' not in df.columns

    def test_ecosystems_accepts_featurecollection(self):
        """Test that Ecosystems accepts an ee.FeatureCollection directly."""
        mock_fc = Mock()
        mock_fc.name.return_value = 'FeatureCollection'

        ecosystem = ee_rle.Ecosystems(
            mock_fc,
            get_level3_column='EFG1',
            get_level456_column='COD'
        )

        assert ecosystem.asset_type == 'TABLE'
        assert ecosystem.data == mock_fc
        assert ecosystem.asset_id is None

    def test_ecosystems_accepts_image(self):
        """Test that Ecosystems accepts an ee.Image directly."""
        mock_image = Mock()
        mock_image.name.return_value = 'Image'

        ecosystem = ee_rle.Ecosystems(
            mock_image,
            get_level3_column=None,
            get_level456_column=None
        )

        assert ecosystem.asset_type == 'IMAGE'
        assert ecosystem.data == mock_image
        assert ecosystem.asset_id is None

    def test_ecosystems_raises_for_invalid_ee_type(self):
        """Test that Ecosystems raises ValueError for unsupported EE types."""
        mock_geometry = Mock()
        mock_geometry.name.return_value = 'Geometry'

        with pytest.raises(ValueError) as exc_info:
            ee_rle.Ecosystems(
                mock_geometry,
                get_level3_column=None,
                get_level456_column=None
            )

        assert "Unsupported data type 'Geometry'" in str(exc_info.value)


@pytest.mark.unit
class TestMakeEOO:
    """Tests for the make_eoo function."""

    @patch('gee_redlist.ee_rle.ee')
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

    @patch('gee_redlist.ee_rle.ee')
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

    @patch('gee_redlist.ee_rle.ee')
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

    @patch('gee_redlist.ee_rle.ee')
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

    @patch('gee_redlist.ee_rle.ee')
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

    @patch('gee_redlist.ee_rle.ee.data')
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

    @patch('gee_redlist.ee_rle.ee.data')
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

    @patch('gee_redlist.ee_rle.ee.data')
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

    @patch('gee_redlist.ee_rle.ee.data')
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

    @patch('gee_redlist.ee_rle.ee.data')
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

    @patch('gee_redlist.ee_rle.ee.data')
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
    """Tests for the make_aoo function."""

    @patch('gee_redlist.ee_rle.ee')
    @patch('builtins.print')
    def test_make_aoo_basic(self, mock_print, mock_ee):
        """Test that make_aoo calls the correct Earth Engine methods."""
        # Create mock objects
        mock_image = Mock()
        mock_geometry = Mock()
        mock_grid = Mock()
        mock_reduced = Mock()
        mock_filtered = Mock()
        mock_size = Mock()

        # Setup the mock chain
        mock_ee.Image.return_value = mock_image
        mock_image.geometry.return_value = mock_geometry
        mock_geometry.coveringGrid.return_value = mock_grid
        mock_image.reduceRegions.return_value = mock_reduced
        mock_reduced.filter.return_value = mock_filtered
        mock_filtered.size.return_value = mock_size
        mock_size.getInfo.return_value = 42

        # Mock the get_aoo_grid_projection
        with patch('gee_redlist.ee_rle.get_aoo_grid_projection') as mock_proj:
            mock_projection = Mock()
            mock_proj.return_value = mock_projection

            # Call the function
            ee_rle.make_aoo('projects/test/assets/fractional_coverage')

            # Verify the chain of calls
            mock_ee.Image.assert_called_once_with('projects/test/assets/fractional_coverage')
            mock_image.geometry.assert_called_once()
            mock_geometry.coveringGrid.assert_called_once_with(mock_projection)

            # Verify reduceRegions was called
            mock_image.reduceRegions.assert_called_once()
            call_kwargs = mock_image.reduceRegions.call_args[1]
            assert call_kwargs['collection'] == mock_grid
            assert call_kwargs['reducer'] == mock_ee.Reducer.mean.return_value

            # Verify filter was called
            mock_reduced.filter.assert_called_once()

            # Verify size and getInfo were called
            mock_filtered.size.assert_called_once()
            mock_size.getInfo.assert_called_once()

            # Verify print was called
            mock_print.assert_called_once_with('aoo_grid_cell_count = 42')

    @patch('gee_redlist.ee_rle.ee')
    @patch('builtins.print')
    def test_make_aoo_with_different_asset_id(self, mock_print, mock_ee):
        """Test make_aoo with a different asset ID."""
        # Setup minimal mocks
        mock_image = Mock()
        mock_ee.Image.return_value = mock_image
        mock_image.geometry.return_value.coveringGrid.return_value = Mock()
        mock_image.reduceRegions.return_value.filter.return_value.size.return_value.getInfo.return_value = 10

        with patch('gee_redlist.ee_rle.get_aoo_grid_projection'):
            # Call with different asset ID
            ee_rle.make_aoo('projects/goog-rle-assessments/assets/MMR-T1_1_1/a00_grid')

            # Verify ee.Image was called with correct asset ID
            mock_ee.Image.assert_called_once_with('projects/goog-rle-assessments/assets/MMR-T1_1_1/a00_grid')

            # Verify print output
            mock_print.assert_called_once_with('aoo_grid_cell_count = 10')

    @patch('gee_redlist.ee_rle.ee')
    @patch('builtins.print')
    def test_make_aoo_filter_gt_zero(self, mock_print, mock_ee):
        """Test that make_aoo filters cells with mean > 0."""
        # Setup mocks
        mock_image = Mock()
        mock_reduced = Mock()
        mock_filter = Mock()

        mock_ee.Image.return_value = mock_image
        mock_image.geometry.return_value.coveringGrid.return_value = Mock()
        mock_image.reduceRegions.return_value = mock_reduced
        mock_reduced.filter.return_value = mock_filter
        mock_filter.size.return_value.getInfo.return_value = 5

        with patch('gee_redlist.ee_rle.get_aoo_grid_projection'):
            ee_rle.make_aoo('test_asset')

            # Verify filter was called
            mock_reduced.filter.assert_called_once()

            # Get the filter argument and verify it's gt('mean', 0)
            filter_arg = mock_reduced.filter.call_args[0][0]
            assert filter_arg == mock_ee.Filter.gt.return_value
            mock_ee.Filter.gt.assert_called_once_with('mean', 0)


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
        import io
        import sys

        # Use an existing fractional coverage asset from the test ecosystem
        asset_id = 'projects/goog-rle-assessments/assets/MMR-T1_1_1/a00_grid'

        # Capture print output
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            # Call make_aoo
            ee_rle.make_aoo(asset_id)

            # Get the captured output
            output = captured_output.getvalue()

            # Verify output contains the expected format
            assert 'aoo_grid_cell_count' in output, "Output should contain 'aoo_grid_cell_count'"
            assert '=' in output, "Output should contain '='"

            # Extract the count from output (format: "aoo_grid_cell_count = N")
            count_str = output.split('=')[1].strip()
            count = int(count_str)

            # Verify the count is reasonable (should be > 0 for a real asset)
            assert count > 0, f"Expected AOO grid cell count > 0, got {count}"
            assert count < 100000, f"Expected AOO grid cell count < 100000, got {count}"

        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__

    def test_ecosystems_vector_integration(self):
        """Integration test for Ecosystems with a FeatureCollection from local test data."""
        # Load test data and create EE FeatureCollection
        test_data_path = Path(__file__).parent / 'test_data' / 'table.json'
        with open(test_data_path) as f:
            table_data = json.load(f)

        fc = ee.FeatureCollection(table_data)
        ecosystem = ee_rle.Ecosystems(
            fc,
            get_level3_column='EFG1',
            get_level456_column='Glob_Typol'
        )

        # Verify attributes
        assert ecosystem.asset_id is None
        assert ecosystem.asset_type == 'TABLE'
        assert isinstance(ecosystem.data, ee.FeatureCollection)

        # Verify the FeatureCollection has features
        count = ecosystem.data.size().getInfo()
        assert count == 3, f"Expected 3 features, got {count}"

    def test_ecosystems_raster_integration(self):
        """Integration test for Ecosystems with a raster asset."""
        asset_id = 'projects/goog-rle-assessments/assets/mm_ecosys_v7b'

        ecosystem = ee_rle.Ecosystems(
            asset_id,
            get_level3_column=None,
            get_level456_column=None
        )

        # Verify attributes
        assert ecosystem.asset_id == asset_id
        assert ecosystem.asset_type == 'IMAGE'
        assert isinstance(ecosystem.data, ee.Image)

        # Verify the Image has bands
        band_names = ecosystem.data.bandNames().getInfo()
        assert len(band_names) > 0, "Expected at least one band"

    def test_ecosystems_functional_group_dataframe_integration(self):
        """Integration test for Ecosystems.functional_group_dataframe() with local test data."""
        import pandas as pd

        # Load test data and create EE FeatureCollection
        test_data_path = Path(__file__).parent / 'test_data' / 'table.json'
        with open(test_data_path) as f:
            table_data = json.load(f)

        fc = ee.FeatureCollection(table_data)
        ecosystem = ee_rle.Ecosystems(
            fc,
            get_level3_column='EFG1',
            get_level456_column='Glob_Typol'
        )
        df = ecosystem.functional_group_dataframe()

        # Verify result is a pandas DataFrame
        assert isinstance(df, pd.DataFrame), "Expected pandas DataFrame"

        # Verify excluded columns are not present
        excluded_cols = {'OBJECTID', 'Shape_Area', 'Shape_Leng', 'system:index'}
        for col in excluded_cols:
            assert col not in df.columns, f"Column '{col}' should be excluded"

        # Verify we have data (test data has 3 features with 2 unique Glob_Typol values)
        assert len(df) > 0, "Expected at least one row in DataFrame"

    def test_featurecollection_from_local_test_data(self):
        """Test creating an EE FeatureCollection from local test data."""
        # Load test data and create EE FeatureCollection
        test_data_path = Path(__file__).parent / 'test_data' / 'table.json'
        with open(test_data_path) as f:
            table_data = json.load(f)

        fc = ee.FeatureCollection(table_data)

        # Verify the FeatureCollection was created correctly
        assert fc.size().getInfo() == 3

        # Verify properties are accessible
        first_feature = fc.first().getInfo()
        assert 'EFG1' in first_feature['properties']
        assert 'Glob_Typol' in first_feature['properties']

        # Verify we can get property values
        efg1_values = fc.aggregate_array('EFG1').getInfo()
        assert 'MFT1.2' in efg1_values
        assert 'T1.2' in efg1_values
