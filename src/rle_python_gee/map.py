"""Generate PNG maps of countries using cartopy."""

import matplotlib
matplotlib.use('Agg') 

import ee
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pyproj
import requests
from rasterio.io import MemoryFile
import shapely
from shapely.ops import transform as shapely_transform
import wkls


def get_utm_epsg(lon: float, lat: float) -> int:
    """
    Determine the EPSG code for the most appropriate UTM projection.

    UTM divides the world into 60 zones, each 6 degrees of longitude wide.
    Each zone has a northern (N) and southern (S) variant based on the equator.

    Parameters
    ----------
    lon : float
        Longitude in decimal degrees (-180 to 180)
    lat : float
        Latitude in decimal degrees (-90 to 90)

    Returns
    -------
    int
        EPSG code for the UTM zone. Format:
        - Northern hemisphere: 326xx (where xx is the zone number 01-60)
        - Southern hemisphere: 327xx (where xx is the zone number 01-60)

    Examples
    --------
    >>> get_utm_epsg(-122.4, 37.8)  # San Francisco
    32610
    >>> get_utm_epsg(103.8, 1.3)  # Singapore
    32648
    >>> get_utm_epsg(-43.2, -22.9)  # Rio de Janeiro
    32723
    >>> get_utm_epsg(151.2, -33.9)  # Sydney
    32756

    Notes
    -----
    - Special zones for Norway and Svalbard are not handled (uses standard calculation)
    - For areas near zone boundaries, consider using the neighboring zone if appropriate
    """
    # Calculate UTM zone number (1-60)
    # Zone 1 starts at -180°, each zone is 6° wide
    utm_zone = int((lon + 180) / 6) + 1

    # Ensure zone is within valid range
    utm_zone = max(1, min(60, utm_zone))

    # Determine hemisphere and construct EPSG code
    if lat >= 0:
        # Northern hemisphere: EPSG:326xx
        epsg_code = 32600 + utm_zone
    else:
        # Southern hemisphere: EPSG:327xx
        epsg_code = 32700 + utm_zone

    return epsg_code


def get_utm_proj_without_limits(utm_zone: int, is_south: bool) -> ccrs.TransverseMercator:
    """Get a UTM projection without the hard-coded x_limits."""

    # Zone N covers longitudes from (N-1)*6° - 180° to N*6° - 180°
    # Central meridian is at the middle: (N-1)*6° - 180° + 3°
    central_longitude = (utm_zone - 1) * 6 - 180 + 3
    
    # Create a custom TransverseMercator projection with UTM parameters
    # This matches UTM Zone 13N but without the hard-coded x_limits
    proj = ccrs.TransverseMercator(
        central_longitude=central_longitude,
        scale_factor=0.9996,  # UTM standard scale factor
        false_easting=500000.0,  # UTM standard false easting
        false_northing=10000000.0 if is_south else 0.0,  # UTM uses 10M for southern hemisphere
    )
    return proj


def _validate_country_code(country_code: str) -> None:
    """
    Validate that the country code is a valid ISO 3166-1 alpha-2 code.

    Parameters
    ----------
    country_code : str
        The ISO 3166-1 alpha-2 country code (must be exactly 2 letters)

    Raises
    ------
    TypeError
        If country_code is not a string
    ValueError
        If country_code is not exactly 2 letters
    """
    if not isinstance(country_code, str):
        raise TypeError(
            f"country_code must be a string, got {type(country_code).__name__}"
        )

    if not country_code or country_code.isspace():
        raise ValueError("country_code cannot be empty or whitespace")

    # Strict validation: must be exactly 2 letters (ISO 3166-1 alpha-2)
    import re
    if not re.match(r'^[A-Za-z]{2}$', country_code):
        raise ValueError(
            f"country_code must be a 2-letter ISO 3166-1 alpha-2 code (e.g., 'US', 'FR', 'JP'). "
            f"Got: '{country_code}'"
        )


def create_country_map(
    country_code: str,
    output_path: str = None,
    show_stock_img: bool = False,
    show_border: bool = True,
    title: str = None,
    geometry_kwargs: dict = {},
    ee_image = None,
    clip_ee_image: bool = False,
    dpi: int = 150,
    image_cmap: str = None,
    image_vmin: float = 0,
    image_vmax: float = 1,
) -> str:
    """
    Create a PNG map of a specified country.

    Parameters
    ----------
    country_code : str
        ISO 3166-1 alpha-2 country code. Must be exactly 2 letters.
        Examples: 'SG' (Singapore), 'FR' (France), 'BR' (Brazil), 'US' (United States), 'JP' (Japan)
        See https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2 for complete list.
    output_path : str, optional
        Path where the PNG file should be saved. If None, defaults to '{country_code}.png'
    show_stock_img : bool, optional
        Whether to show the world stock image. Default is True.
    show_border : bool, optional
        Whether to show a border around the target country. Default is True.
    title : str, optional
        Custom title for the map. If None, defaults to 'Map of {country_name}'.
    geometry_kwargs : dict, optional
        Keyword arguments to pass to the add_geometries method.
        Example: {'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 0.5}
    ee_image : ee.Image, optional
        An Earth Engine image to use as a basemap under the cartopy layers.
        If provided, the image will be rendered first, then cartopy layers on top.
    clip_ee_image : bool, optional
        Whether to clip the Earth Engine image to the country geometry.
        If True, only the portion of the image within the country borders is shown.
        If False (default), the image covers the full map extent.

    Returns
    -------
    str
        Path to the saved PNG file

    Examples
    --------
    >>> create_country_map('SG')
    'sg.png'
    >>> create_country_map('FR', 'maps/france_map.png')
    'maps/france_map.png'
    >>> create_country_map('BR', show_surrounding_countries=False)
    'br.png'
    """
    # Validate country code format
    _validate_country_code(country_code)

    if output_path is None:
        output_path = f"{country_code.lower()}.png"

    # Use wkls to obtain the country boundary information by ISO 3166-1 alpha-2 code
    try:
        country_wkb = wkls[country_code.lower()].wkb()
    except ValueError as e:
        # Provide a more helpful error message
        raise ValueError(
            f"Country code '{country_code}' not found in database. "
            f"Please use a valid ISO 3166-1 alpha-2 code (e.g., 'US', 'FR', 'JP')."
        ) from e

    country_geometry = shapely.from_wkb(bytes(country_wkb))

    # Calculate the most appropriate UTM projection for this country
    # Use the centroid of the country to determine the UTM zone
    centroid = country_geometry.centroid
    utm_epsg = get_utm_epsg(centroid.x, centroid.y)
    utm_zone = utm_epsg % 100

    # Determine hemisphere for UTM projection
    is_south = centroid.y < 0

    # Use TransverseMercator instead of UTM to avoid hard-coded x_limits
    proj = get_utm_proj_without_limits(utm_zone, is_south)

    # Create a figure and axis with UTM projection
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    if show_stock_img:
        # Show the world stock image for reference
        ax.stock_img(alpha=1.0)

    # Transform the countrygeometry to UTM
    wgs84_to_utm = pyproj.Transformer.from_crs(
        "EPSG:4326",  # WGS84
        f"EPSG:{utm_epsg}",  # UTM
        always_xy=True
    )
    country_geometry_utm = shapely_transform(wgs84_to_utm.transform, country_geometry)

    # Get bounds in UTM coordinates (meters)
    bounds = country_geometry_utm.bounds

    # Calculate padding as a percentage of the extent (5% on each side)
    x_range = bounds[2] - bounds[0]
    y_range = bounds[3] - bounds[1]
    padding_x = x_range * 0.15
    padding_y = y_range * 0.15

    extent = [
        bounds[0] - padding_x, # minx
        bounds[2] + padding_x, # maxx
        bounds[1] - padding_y, # miny
        bounds[3] + padding_y # maxy
    ]
    ax.set_extent(extent, crs=proj)

    # Add Earth Engine image as basemap if provided
    if ee_image is not None:

        # Create a bounding box for the EE image request [xMin, yMin, xMax, yMax]
        ee_region = ee.Geometry.Rectangle(
            [extent[0], extent[2], extent[1], extent[3]],
            proj=ee.Projection(f'EPSG:{utm_epsg}'),
            evenOdd=False
        )

        def add_ee_image(
            ee_image: ee.Image,
            ee_region: ee.Geometry,
            scale: float = 10000,
            clip_ee_image: bool = False,
            image_cmap: str = None,
            ) -> np.ma.MaskedArray:
            """Download an Earth Engine image and return a numpy masked array."""
        
            # Clip the image to country geometry if requested
            if clip_ee_image:
                # Convert shapely geometry to GeoJSON
                # Note: Use WGS84 geometry for EE - it doesn't support MultiPolygon in projected CRS
                from shapely.geometry import mapping
                geojson = mapping(country_geometry)
                ee_geometry = ee.Geometry(geojson)
                ee_image_clipped = ee_image.clip(ee_geometry)
            else:
                ee_image_clipped = ee_image

            # Get the image URL from Earth Engine as GeoTIFF
            # Use getDownloadURL with the appropriate UTM projection for this country
            # NOTE that getDownloadURL does not return a GeoTIFF with noData, so the mask needs to be
            # requested separately
            crs = f'EPSG:{utm_epsg}'
            crs_transform = [scale, 0, extent[0], 0, scale, extent[2]]
            url = ee_image_clipped.getDownloadURL({
                'region': ee_region,
                'format': 'GEO_TIFF',
                'crs': crs,
                'crs_transform': crs_transform,
            })
            print(f"Downloading Earth Engine image...")
            response = requests.get(url, timeout=300)  # 5 minute timeout
            print(f"Downloaded image {len(response.content) / 1024 / 1024:.2f} MB")

            url_mask = ee_image_clipped.mask().getDownloadURL({
                'region': ee_region,
                'format': 'GEO_TIFF',
                'crs': crs,
                'crs_transform': crs_transform,
            })
            response_mask = requests.get(url_mask, timeout=300)  # 5 minute timeout
            print(f"Downloaded mask {len(response_mask.content) / 1024 / 1024:.2f} MB")

            # Open with rasterio from memory
            with MemoryFile(response.content) as memfile:
                with memfile.open() as dataset:
                    img_array = dataset.read()  # Shape: (1, height, width)
                    # reorder bands to be in the correct order for matplotlib
                    img_array = np.moveaxis(img_array, 0, -1)  # Shape: (height, width, bands)
                    # Get georeferencing from the raster
                    bounds = dataset.bounds

            with MemoryFile(response_mask.content) as memfile_mask:
                with memfile_mask.open() as dataset_mask:   
                    img_array_mask = dataset_mask.read()  # Shape: (1, height, width)
                    img_array_mask = img_array_mask.astype(np.uint8)
                    # reorder bands to be in the correct order for matplotlib                      
                    img_array_mask = np.moveaxis(img_array_mask, 0, -1)  # Shape: (height, width, bands)

            if image_cmap is None:
                if np.all((img_array == 0) | (img_array == 1)):
                    image_cmap='binary'
                else:
                    image_cmap='grey'

            ax.imshow(
                np.ma.masked_where(img_array_mask == 0, img_array),
                extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                origin='upper',
                transform=proj,
                vmin=image_vmin,
                vmax=image_vmax,
                cmap=image_cmap
            )
        
        image_dimension_pixels = dpi * 4
        scale = max(x_range, y_range) / image_dimension_pixels

        add_ee_image(
            ee_image,
            ee_region,
            scale=scale,
            clip_ee_image=clip_ee_image,
            image_cmap=image_cmap
        )

    # # Add map features
    # ax.add_feature(cfeature.OCEAN, facecolor='white')
    # ax.add_feature(cfeature.LAND, facecolor='white')

 
    if show_border:
        geometry_kwargs.setdefault("facecolor", 'none')
        ax.add_geometries(
            [country_geometry_utm],
            proj,
            **geometry_kwargs
        )

    # Remove the axes frame/border
    # For cartopy GeoAxes, we need to set the spines to invisible
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title:  # Only add title if not empty string
        plt.title(title, fontsize=16, fontweight='bold')

    # plt.show()

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return output_path


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        country_code = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) > 2 else None
        result = create_country_map(country_code, output)
        print(f"Map saved to: {result}")
    else:
        print("Usage: python map.py <country_code> [output_path]")
        print("Example: python map.py SG")
