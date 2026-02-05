import pytest
from typer.testing import CliRunner
from unittest.mock import patch
from rle_python_gee.main import app

runner = CliRunner()


def test_main_no_command():
    """Test that main app prints the expected message when no command is provided."""
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Hello from rle-python-gee!" in result.stdout
    assert "Use --help to see available commands" in result.stdout


def test_main_help():
    """Test that --help flag shows usage information."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Google Earth Engine tools for IUCN Red List analysis" in result.stdout
    assert "test-auth" in result.stdout



@patch('rle_python_gee.main.print_authentication_status')
def test_test_auth_command(mock_print_auth):
    """Test that test-auth command calls print_authentication_status."""
    result = runner.invoke(app, ["test-auth"])
    assert result.exit_code == 0
    assert "Testing Earth Engine authentication..." in result.stdout
    mock_print_auth.assert_called_once()


def test_test_auth_help():
    """Test that test-auth command has proper help text."""
    result = runner.invoke(app, ["test-auth", "--help"])
    assert result.exit_code == 0
    assert "Test Earth Engine authentication status" in result.stdout
