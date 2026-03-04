"""Example usage of create_country_map with Earth Engine basemaps."""

import ee
from rle_python_gee.map import create_country_map
import os
import matplotlib as mpl

def main():
    """Generate example maps with Earth Engine imagery as basemaps."""

    # Initialize Earth Engine
    print("Initializing Earth Engine...")
    google_cloud_project = 'goog-rle-assessments'
    print(f"GOOGLE_CLOUD_PROJECT = {google_cloud_project!r}")
    ee.Initialize(project=google_cloud_project)
    print("✓ Earth Engine initialized")

    # Example 1: Elevation basemap for Nepal
    print("\nCreating elevation map of Nepal...")
    elevation = ee.Image('USGS/SRTMGL1_003')
    nepal_path = create_country_map(
        'NP',
        'temp/nepal_elevation.png',
        ee_image=elevation,
        image_vmin=0,
        image_vmax=8000,
        show_border=True,
        geometry_kwargs={
            'edgecolor': 'red',
            'linewidth': 1.0,
        },
        title='Nepal - Elevation Map'
    )
    print(f"✓ Nepal elevation map saved to: {nepal_path}")

    # Example 2
    print("\nTry creating topography map...")

    dem_glo30 = ee.ImageCollection("COPERNICUS/DEM/GLO30").select('DEM').mosaic()
    rgb_image = dem_glo30

    map_path = create_country_map(
        country_code='MM',
        output_path='temp/myanmar_rgb.png',
        ee_image=rgb_image,
        image_vmin=0,
        image_vmax=1000,
        clip_ee_image=True,
        show_border=True,
        geometry_kwargs={
            'edgecolor': 'black',
            'linewidth': 2.0,
        },
    )
    print(f"✓ RGB map saved to: {map_path}")

    # Example 3
    print("\nCreating ecosystem map...")

    ee_image = (
        ee.Image('projects/goog-rle-assessments/assets/mm_ecosys_v7b')
          .eq(37).selfMask()
    )
    map_path = 'temp/myanmar_map.png'
    create_country_map(
        country_code='MM',
        output_path=map_path,
        ee_image=ee_image,
        clip_ee_image=True,
        show_border=True,
        geometry_kwargs={
            'edgecolor': 'grey',
            'linewidth': 0.5,
        },
        image_cmap=mpl.colors.ListedColormap(['red']),
        dpi=600,
    )
    print(f"✓ RGB map saved to: {map_path}")

if __name__ == "__main__":
    main()
