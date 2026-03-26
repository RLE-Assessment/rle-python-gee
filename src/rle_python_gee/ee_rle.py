"""
Google Earth Engine functions for Red List Ecosystem (RLE) assessments.

This module provides functions for calculating spatial metrics used in
IUCN Red List of Ecosystems assessments, including Extent of Occurrence (EOO).
"""

from typing import Optional, Union
import yaml

import ee


def load_yaml(yaml_path):
    """Load YAML configuration file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def get_aoo_grid_projection(scale: float = 1e4) -> ee.Projection:
    """
    Returns the default projection to use for the AOO grid in RLE Assessments.

    Projection ESRI:54034 (World Cylindrical Equal Area)
    is used based on the grid defined in the document:
    `Global 10 x 10-km grids suitable for use in IUCN Red List of Ecosystems assessments`
    available at: https://www.iucnrle.org/rle-material-and-tools
    """

    wkt1 = """
        PROJCS["World_Cylindrical_Equal_Area",
            GEOGCS["WGS 84",
                DATUM["WGS_1984",
                    SPHEROID["WGS 84",6378137,298.257223563,
                        AUTHORITY["EPSG","7030"]],
                    AUTHORITY["EPSG","6326"]],
                PRIMEM["Greenwich",0],
                UNIT["Degree",0.0174532925199433]],
            PROJECTION["Cylindrical_Equal_Area"],
            PARAMETER["standard_parallel_1",0],
            PARAMETER["central_meridian",0],
            PARAMETER["false_easting",0],
            PARAMETER["false_northing",0],
            UNIT["metre",1,
                AUTHORITY["EPSG","9001"]],
            AXIS["Easting",EAST],
            AXIS["Northing",NORTH],
            AUTHORITY["ESRI","54034"]]
    """
    proj = ee.Projection(
        crs=wkt1,
        transform=[scale, 0, 0, 0, scale, 0]
    )
    return proj


def make_eoo(
    class_img: ee.Image,
    geo: ee.Geometry = None,
    scale: int = None,
    max_pixels: int = 1e12,
    max_error: int = 1,
    best_effort: bool = False
) -> ee.Geometry:
    """
    Calculate the Extent of Occurrence (EOO) polygon from a binary image.

    Takes a binary classification image and returns its Extent of Occurrence
    as a convex hull polygon. This is commonly used in IUCN Red List assessments
    to estimate the geographic range of a species or ecosystem.

    EOO is defined in
    Guidelines for the application of IUCN Red List of Ecosystems Categories and Criteria
    6.3.2. Methods for assessing criteria B1 and B2

    Extent of occurrence (EOO). The EOO of an ecosystem is the area (km2) of a minimum
    convex polygon – the smallest polygon in which no internal angle exceeds 180° that
    encompasses all known current spatial occurrences of the ecosystem type. The
    minimum convex polygon (also known as a convex hull) must not exclude any areas,
    discontinuities or disjunctions, regardless of whether the ecosystem can occur in those
    areas or not. Regions such as oceans (for terrestrial ecosystems), land (for coastal or
    marine ecosystems), or areas outside the study area (such as in a different country)
    must remain included within the minimum convex polygon to ensure that this
    standardized method is comparable across ecosystem types. In addition, these features
    contribute to spreading risks across the distribution of the ecosystem by making
    different parts of its distribution more spatially independent.

    Args:
        class_img: A binary ee.Image where pixels with value 1 represent presence
                   and 0/masked pixels represent absence.
        geo: The geometry to use for the reduction. Should encompass the area
             of interest for the analysis. If not provided, the geometry will be
             inferred from the class_img.
        scale: The scale (in meters) for reducing the image pixels to polygons.
               If not provided, the image's nominal scale will be used.
               If the scale is less than 50 meters per pixel, 50 meters per pixel will be used.
        max_pixels: The maximum number of pixels to process. Default is 1e12.
        max_error: The maximum error in meters for the convex hull calculation.
                   Default is 1.
        best_effort: If True, uses best effort mode which may be less accurate
                     but more likely to succeed for large areas. Default is False.

    Returns:
        An ee.Geometry representing the convex hull (EOO polygon) of all
        presence pixels in the input image.

    Example:
        >>> import ee
        >>> ee.Initialize()
        >>> # Create a binary habitat map
        >>> habitat = ee.Image(1).clip(region)
        >>> eoo_polygon = make_eoo(habitat)
        >>> print(eoo_polygon.area().getInfo())

    Note:
        The input image should be a binary classification where:
        - Value 1 indicates presence (included in EOO)
        - Value 0 or masked indicates absence (excluded from EOO)
    """

    if geo is None:
        geo = class_img.geometry()

    # Set the scale (in meters) for reducing the image pixels to polygons.
    # Use the image's nominal scale unless is is less than 50 meters per pixel.
    if scale is None:
        scale = max(class_img.projection().nominalScale().getInfo(), 50)

    # Mask the image to only include presence pixels (value = 1)
    # Then reduce to vectors to get all polygons
    eoo_poly = (
        class_img
        .updateMask(1)
        .reduceToVectors(
            scale=scale,
            geometry=geo,
            geometryType='polygon',
            maxPixels=max_pixels,
            bestEffort=best_effort,
        )
        .geometry()
        .convexHull(maxError=max_error)
        # convexHull() is called twice as a workaround for a bug
        # (https://issuetracker.google.com/issues/465490917)
        .convexHull(maxError=max_error)
    )

    return eoo_poly


def area_km2(
    eoo_poly: ee.Geometry,
) -> ee.Number:
    """
    Calculate the area of the Extent of Occurrence (EOO) in square kilometers.

    Args:
        eoo_poly: An ee.Geometry representing the EOO polygon.

    Returns:
        An ee.Number representing the EOO area in square kilometers.
    """
    return eoo_poly.area().divide(1e6)


def ensure_asset_folder_exists(folder_path: str) -> bool:
    """
    Check if an Earth Engine asset folder exists, create it if it doesn't.

    Args:
        folder_path: Full path to the asset folder
                     (e.g., 'projects/goog-rle-assessments/assets/MMR-T1_1_1')

    Returns:
        True if the folder was created, False if it already existed.

    Example:
        >>> import ee
        >>> ee.Initialize()
        >>> ensure_asset_folder_exists('projects/my-project/assets/my-folder')
        True  # Folder was created
        >>> ensure_asset_folder_exists('projects/my-project/assets/my-folder')
        False  # Folder already exists
    """
    try:
        ee.data.getAsset(folder_path)
        return False  # Folder already exists
    except ee.EEException:
        # Folder doesn't exist, create it
        ee.data.createFolder(folder_path)
        return True  # Folder was created

def create_asset_folder(folder_path: str) -> bool:
    """
    Create an Earth Engine asset folder if it doesn't already exist.

    Args:
        folder_path: Full path to the asset folder
                     (e.g., 'projects/goog-rle-assessments/assets/MMR-T1_1_1')

    Returns:
        True if the folder was created, False if it already existed.

    Example:
        >>> import ee
        >>> ee.Initialize()
        >>> create_asset_folder('projects/my-project/assets/my-folder')
        True  # Folder was created
        >>> create_asset_folder('projects/my-project/assets/my-folder')
        False  # Folder already exists
    """
    try:
        ee.data.getAsset(folder_path)
        return False  # Folder already exists
    except ee.EEException:
        # Folder doesn't exist, create it
        ee.data.createFolder(folder_path)
        return True  # Folder was created


def export_fractional_coverage_on_aoo_grid(
    class_img: ee.Image,
    asset_id: str,
    export_description: str,
    max_pixels: int = 65536,
) -> str:
    """
    Export the fractional coverage of a binary image on the AOO grid.

    Args:
        class_img: A binary ee.Image where pixels with value 1 represent presence
                   and 0/masked pixels represent absence.
        asset_id: The Earth Engine asset ID to export the fractional coverage to.
        export_description: The description to use for the export task.
        max_pixels: The maximum number of pixels to process. Default is 65536.

    Returns:
        A ee.batch.Task object.
    """

    fcov_unmasked = class_img.unmask().reduceResolution(
        reducer=ee.Reducer.mean(),
        maxPixels=max_pixels
    ).reproject(get_aoo_grid_projection())

    # Mask out zero values.
    fractionalCoverage = fcov_unmasked.mask(fcov_unmasked.gt(0))

    task = ee.batch.Export.image.toAsset(
        image=fractionalCoverage,
        description=export_description,
        assetId=asset_id,
        crs=get_aoo_grid_projection().getInfo()['wkt'],
        # Set scale to avoid errors:
        #    with no scale specified, the task fails with:
        #      "Export too large: specified 2557382439248 pixels (max: 100000000)"
        #    with scale=10000m:
        #       "Error: Reprojection output too large (14447x23745 pixels). (Error code: 3)"
        #    with scale=5000m:
        #       "Error: Reprojection output too large (14336x15603 pixels). (Error code: 3)"
        #    with scale=2000m:
        #       the task succeeds (366 EECU-seconds)
        #    with scale=1000m:
        #       the task succeeds (218 EECU-seconds)
        #    with scale=500m:
        #       the task succeeds (522 EECU-seconds)
        scale=1000,
    )
    task.start()
    return task


def make_aoo(data, **kwargs):
    """Create an AOO grid from the given data source.

    Auto-detects the data type and returns the appropriate AOOGrid subclass.
    See :func:`rle_python_gee.aoo.make_aoo` for full documentation.

    Args:
        data: Data source (EE asset ID, ee.Image, ee.FeatureCollection,
            .parquet path, or .tif/.tiff path).
        **kwargs: Additional arguments passed to the backend constructor.

    Returns:
        An AOOGrid instance.
    """
    from rle_python_gee.aoo import make_aoo as _make_aoo

    return _make_aoo(data, **kwargs)
