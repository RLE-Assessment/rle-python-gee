import typer
from typing_extensions import Annotated
from rle_python_gee.ee_auth import print_authentication_status
from rle_python_gee import __version__

app = typer.Typer(
    name="rle-python-gee",
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
    """Main entry point for rle-python-gee CLI."""
    if version:
        print(f"rle-python-gee version {__version__}")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        print("Hello from rle-python-gee!")
        print("\nUse --help to see available commands")


if __name__ == "__main__":
    app()
