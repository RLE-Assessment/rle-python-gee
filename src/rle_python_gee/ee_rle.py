"""
Google Earth Engine functions for Red List Ecosystem (RLE) assessments.

This module provides functions for calculating spatial metrics used in
IUCN Red List of Ecosystems assessments, including Extent of Occurrence (EOO).
"""

from typing import Optional, Union
import yaml

import ee
import pandas as pd


def load_yaml(yaml_path):
    """Load YAML configuration file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


class Ecosystems:
    """
    Represents an ecosystems dataset from Google Earth Engine.

    Accepts either an Earth Engine asset ID (string), ee.FeatureCollection,
    or ee.Image containing ecosystem data, typically mapped to IUCN Global
    Ecosystem Typology (GET) functional groups.

    Attributes:
        asset_id: The Earth Engine asset ID (None if created from EE object directly).
        asset_type: The type of data ('TABLE' for vector, 'IMAGE' for raster).
        data: The ee.FeatureCollection or ee.Image containing ecosystem data.
        get_level3_column: Column name for GET Level 3 ecosystem functional group codes.
        get_level456_column: Column name for GET Level 4 ecosystem type codes.

    Example:
        >>> import ee
        >>> ee.Initialize()
        >>> # Load from asset ID
        >>> vec_ecosystems = Ecosystems(
        ...     'projects/goog-rle-assessments/assets/columbia/GETCol'
        ... )
        >>> print(vec_ecosystems.asset_type)
        'TABLE'
        >>> # Load from ee.FeatureCollection directly
        >>> fc = ee.FeatureCollection([...])
        >>> ecosystems = Ecosystems(fc)
        >>> # Load from ee.Image directly
        >>> img = ee.Image('projects/goog-rle-assessments/assets/mm_ecosys_v7b')
        >>> raster_ecosystems = Ecosystems(img)
    """

    def __init__(
        self,
        data: Union[str, ee.FeatureCollection, ee.Image],
        get_level3_column: Optional[str] = None,
        get_level456_column: Optional[str] = None
    ):
        """
        Initialize an Ecosystems instance.

        Accepts an Earth Engine asset ID string, ee.FeatureCollection, or ee.Image.
        Automatically detects the data type and loads appropriately.

        Args:
            data: One of:
                - Earth Engine asset ID string (e.g., 'projects/.../assets/...')
                - ee.FeatureCollection object
                - ee.Image object
            get_level3_column: Column name containing GET (Global Ecosystem Typology)
                               Level 3 ecosystem functional group codes (e.g., 'EFG1').
            get_level456_column: Column name containing GET (Global Ecosystem Typology)
                               Level 4 ecosystem type codes (e.g., 'Glob_Typol').

        Raises:
            ee.EEException: If asset_id doesn't exist or access is denied.
            ValueError: If the data type is not supported.
        """
        self.get_level3_column = get_level3_column
        self.get_level456_column = get_level456_column

        if isinstance(data, str):
            # Treat as asset_id
            self.asset_id = data
            asset_info = ee.data.getAsset(data)
            self.asset_type = asset_info['type']

            if self.asset_type == 'TABLE':
                self.data = ee.FeatureCollection(data)
            elif self.asset_type == 'IMAGE':
                self.data = ee.Image(data)
            else:
                raise ValueError(
                    f"Unsupported asset type '{self.asset_type}' for asset '{data}'. "
                    "Expected 'TABLE' (FeatureCollection) or 'IMAGE'."
                )
        else:
            # Treat as EE object
            self.asset_id = None
            type_name = data.name()

            if type_name == 'FeatureCollection':
                self.asset_type = 'TABLE'
            elif type_name == 'Image':
                self.asset_type = 'IMAGE'
            else:
                raise ValueError(
                    f"Unsupported data type '{type_name}'. "
                    "Expected ee.FeatureCollection or ee.Image."
                )
            self.data = data

    def functional_group_dataframe(self) -> pd.DataFrame:
        """Return functional groups as a pandas DataFrame with MultiIndex.

        Only available for TABLE (vector) assets. Returns a DataFrame with
        distinct combinations of GET Level 3 and Level 4 columns, using them
        as a hierarchical MultiIndex.

        Uses Earth Engine server-side grouped reduction for efficiency.

        Returns:
            pd.DataFrame: DataFrame with MultiIndex (get_level3_column, get_level456_column).

        Raises:
            ValueError: If the asset type is not TABLE or if column names are not specified.
        """
        if self.asset_type != 'TABLE':
            raise ValueError("dataframe property is only available for TABLE assets")

        if self.get_level3_column is None or self.get_level456_column is None:
            raise ValueError(
                "Both get_level3_column and get_level456_column must be specified "
                "to use the dataframe property"
            )

        # Columns to exclude from grouping
        exclude_cols = ['OBJECTID', 'Shape_Area', 'Shape_Leng', 'system:index']

        group_cols = self.data.first().propertyNames()
        for col in exclude_cols:
            group_cols = group_cols.remove(col)

        # Find distinct combinations of the L3 and L4 columns
        distinct_pairs_fc = self.data.distinct([self.get_level3_column, self.get_level456_column])

        def extract_columns(feature):
            return ee.Feature(feature).toDictionary(group_cols)

        distinct_pairs_list = distinct_pairs_fc.toList(distinct_pairs_fc.size()).map(extract_columns)

        # Get records as list of dictionaries and create DataFrame with MultiIndex
        records = distinct_pairs_list.getInfo()
        df = pd.DataFrame(records)
        return df.set_index([self.get_level3_column, self.get_level456_column])

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebook display."""
        meta_rows = []
        if self.asset_id:
            meta_rows.append(f"<tr><td><b>Asset ID</b></td><td>{self.asset_id}</td></tr>")
        meta_rows.append(f"<tr><td><b>Asset Type</b></td><td>{self.asset_type}</td></tr>")

        if self.get_level3_column:
            meta_rows.append(f"<tr><td><b>GET Level 3 Column</b></td><td>{self.get_level3_column}</td></tr>")
        if self.get_level456_column:
            meta_rows.append(f"<tr><td><b>GET Level 4 Column</b></td><td>{self.get_level456_column}</td></tr>")

        data_table = ""

        if self.asset_type == 'TABLE':
            count = self.data.size().getInfo()
            meta_rows.append(f"<tr><td><b>Feature Count</b></td><td>{count}</td></tr>")

            # Get first 5 features for preview
            head_features = self.data.limit(5).getInfo()['features']

            if head_features:
                # Get property names from first feature
                props = list(head_features[0].get('properties', {}).keys())
                if props:
                    # Build header row with highlight for GET columns
                    header_cells = []
                    for p in props:
                        if p == self.get_level3_column:
                            header_cells.append(f'<th style="padding: 4px 8px; border: 1px solid #ddd; background-color: #cce5ff; font-weight: bold;">{p}</th>')
                        elif p == self.get_level456_column:
                            header_cells.append(f'<th style="padding: 4px 8px; border: 1px solid #ddd; background-color: #d4edda; font-weight: bold;">{p}</th>')
                        else:
                            header_cells.append(f'<th style="padding: 4px 8px; border: 1px solid #ddd; background-color: #f5f5f5;">{p}</th>')
                    header_row = f'<tr>{"".join(header_cells)}</tr>'

                    # Build data rows (first 5 only) with highlight for GET columns
                    data_rows = []
                    for feature in head_features:
                        prop_values = feature.get('properties', {})
                        cells = []
                        for p in props:
                            if p == self.get_level3_column:
                                cells.append(f'<td style="padding: 4px 8px; border: 1px solid #ddd; background-color: #cce5ff;">{prop_values.get(p, "")}</td>')
                            elif p == self.get_level456_column:
                                cells.append(f'<td style="padding: 4px 8px; border: 1px solid #ddd; background-color: #d4edda;">{prop_values.get(p, "")}</td>')
                            else:
                                cells.append(f'<td style="padding: 4px 8px; border: 1px solid #ddd;">{prop_values.get(p, "")}</td>')
                        data_rows.append(f'<tr>{"".join(cells)}</tr>')

                    more_text = f"<p style='color: #666; font-style: italic;'>Showing 5 of {count} records</p>" if count > 5 else ""

                    data_table = f"""
                    <h4 style="margin-top: 16px; margin-bottom: 8px;">Records</h4>
                    <table style="border-collapse: collapse; margin-top: 8px;">
                        <thead>{header_row}</thead>
                        <tbody>{''.join(data_rows)}</tbody>
                    </table>
                    {more_text}
                    """
        elif self.asset_type == 'IMAGE':
            bands = self.data.bandNames().getInfo()
            meta_rows.append(f"<tr><td><b>Bands</b></td><td>{', '.join(bands)}</td></tr>")

        return f"""
        <table style="border-collapse: collapse;">
            <thead>
                <tr><th colspan="2" style="text-align: left; padding: 8px; background-color: #f0f0f0;">Ecosystems</th></tr>
            </thead>
            <tbody>
                {''.join(meta_rows)}
            </tbody>
        </table>
        {data_table}
        """


def get_aoo_grid_projection() -> ee.Projection:
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
    scale = 1e4
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


def make_aoo(
    asset_id: str,
):
    """
    Make the AOO grid from a fractional coverage image.

    Args:
        asset_id: The Earth Engine asset ID to the fractional coverage image.
    """

    fractional_coverage_exported = ee.Image(asset_id)

    aoo_grid_proj = get_aoo_grid_projection()

    aoo_grid_cells = fractional_coverage_exported.reduceRegions(
        collection=fractional_coverage_exported.geometry().coveringGrid(aoo_grid_proj),
        reducer=ee.Reducer.mean(),
    ).filter(ee.Filter.gt('mean', 0))

    aoo_grid_cell_count = aoo_grid_cells.size().getInfo()
    print(f'{aoo_grid_cell_count = }')