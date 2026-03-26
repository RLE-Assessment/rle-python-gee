import ee
from rle_python_gee.ee_rle import make_eoo, area_km2

def main():
    """Main function to calculate the EOO of a binary habitat map."""
    
    # Initialize Earth Engine
    print("Initializing Earth Engine...")
    google_cloud_project = 'goog-rle-assessments'
    ee.Initialize(project=google_cloud_project)
    print("✓ Earth Engine initialized")

    # Example 1: Calculate EOO for Myanmar ecosystem

    # Create a binary habitat map
    ee_image = (
            ee.Image('projects/goog-rle-assessments/assets/myanmar/mm_ecosys_v7b')
            .eq(37).selfMask()
        )

    eoo_polygon = make_eoo(ee_image)
    print(f'EOO area: {area_km2(eoo_polygon).getInfo()} km²')

if __name__ == "__main__":
    main()