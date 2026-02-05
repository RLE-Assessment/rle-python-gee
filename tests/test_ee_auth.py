"""Tests for Earth Engine authentication utilities."""

import pytest
from unittest.mock import patch, MagicMock
import ee
from rle_python_gee.ee_auth import check_authentication, is_authenticated, print_authentication_status


class TestAuthenticationFunctions:
    """Test suite for Earth Engine authentication functions."""

    @patch('rle_python_gee.ee_auth.ee.Initialize')
    @patch('rle_python_gee.ee_auth.ee.data.getAssetRoots')
    def test_successful_authentication(self, mock_get_roots, mock_initialize):
        """Test successful authentication with project info."""
        mock_initialize.return_value = None
        mock_get_roots.return_value = [{'id': 'projects/test-project'}]

        result = check_authentication()

        assert result['authenticated'] is True
        assert 'Successfully authenticated' in result['message']
        assert result['project'] is not None
        mock_initialize.assert_called_once()

    @patch('rle_python_gee.ee_auth.ee.Initialize')
    @patch('rle_python_gee.ee_auth.ee.data.getAssetRoots')
    def test_authentication_without_project_info(self, mock_get_roots, mock_initialize):
        """Test authentication succeeds but can't get project info."""
        mock_initialize.return_value = None
        mock_get_roots.side_effect = Exception("Cannot retrieve project")

        result = check_authentication()

        assert result['authenticated'] is True
        assert 'could not retrieve project info' in result['message']
        assert result['project'] is None

    @patch('rle_python_gee.ee_auth.ee.Initialize')
    def test_authentication_ee_exception(self, mock_initialize):
        """Test authentication fails with EE exception."""
        mock_initialize.side_effect = ee.EEException("Authentication required")

        result = check_authentication()

        assert result['authenticated'] is False
        assert 'Earth Engine authentication failed' in result['message']
        assert result['project'] is None

    @patch('rle_python_gee.ee_auth.ee.Initialize')
    def test_authentication_generic_exception(self, mock_initialize):
        """Test authentication fails with generic exception."""
        mock_initialize.side_effect = RuntimeError("Network error")

        result = check_authentication()

        assert result['authenticated'] is False
        assert 'Authentication error' in result['message']
        assert result['project'] is None

    @patch('rle_python_gee.ee_auth.check_authentication')
    def test_is_authenticated_true(self, mock_test_auth):
        """Test is_authenticated returns True when authenticated."""
        mock_test_auth.return_value = {
            'authenticated': True,
            'message': 'Success',
            'project': 'test-project'
        }

        assert is_authenticated() is True

    @patch('rle_python_gee.ee_auth.check_authentication')
    def test_is_authenticated_false(self, mock_test_auth):
        """Test is_authenticated returns False when not authenticated."""
        mock_test_auth.return_value = {
            'authenticated': False,
            'message': 'Failed',
            'project': None
        }

        assert is_authenticated() is False

    @patch('rle_python_gee.ee_auth.check_authentication')
    def test_print_authentication_status_success(self, mock_test_auth, capsys):
        """Test print_authentication_status with successful auth."""
        mock_test_auth.return_value = {
            'authenticated': True,
            'message': 'Success',
            'project': 'test-project'
        }

        print_authentication_status()
        captured = capsys.readouterr()

        assert '✓ Earth Engine Authentication: SUCCESS' in captured.out
        assert 'Success' in captured.out
        assert 'test-project' in captured.out

    @patch('rle_python_gee.ee_auth.check_authentication')
    def test_print_authentication_status_failure(self, mock_test_auth, capsys):
        """Test print_authentication_status with failed auth."""
        mock_test_auth.return_value = {
            'authenticated': False,
            'message': 'Authentication required',
            'project': None
        }

        print_authentication_status()
        captured = capsys.readouterr()

        assert '✗ Earth Engine Authentication: FAILED' in captured.out
        assert 'Authentication required' in captured.out
        assert 'earthengine authenticate' in captured.out

    @patch('rle_python_gee.ee_auth.check_authentication')
    def test_print_authentication_status_no_project(self, mock_test_auth, capsys):
        """Test print_authentication_status with success but no project."""
        mock_test_auth.return_value = {
            'authenticated': True,
            'message': 'Success',
            'project': None
        }

        print_authentication_status()
        captured = capsys.readouterr()

        assert '✓ Earth Engine Authentication: SUCCESS' in captured.out
        assert 'Project:' not in captured.out


class TestAuthenticationIntegration:
    """Integration tests for Earth Engine authentication."""

    @pytest.mark.skipif(
        not hasattr(ee, 'Initialize'),
        reason="Earth Engine not available"
    )
    def test_real_authentication_attempt(self):
        """
        Test real authentication attempt (will likely fail in CI).

        This test is expected to fail in CI environments without credentials,
        but demonstrates the API usage for manual testing.
        """
        result = check_authentication()

        # Should return a valid result structure regardless of success
        assert isinstance(result, dict)
        assert 'authenticated' in result
        assert 'message' in result
        assert 'project' in result
        assert isinstance(result['authenticated'], bool)
        assert isinstance(result['message'], str)
