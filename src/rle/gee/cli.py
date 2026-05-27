"""Command-line interface for rle-python-gee."""

import typer
from typing_extensions import Annotated

from rle.gee import __version__
from rle.gee.auth import print_authentication_status

app = typer.Typer(
    name="rle-gee",
    help="Google Earth Engine tools for IUCN Red List analysis",
    add_completion=False,
)


@app.command()
def test_auth():
    """Test Earth Engine authentication status."""
    print("Testing Earth Engine authentication...")
    print_authentication_status()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", "-v", help="Show version and exit"),
    ] = False,
):
    """Main entry point for the rle-gee CLI."""
    if version:
        print(f"rle-python-gee version {__version__}")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        print("Hello from rle-python-gee!")
        print("\nUse --help to see available commands")


if __name__ == "__main__":
    app()
