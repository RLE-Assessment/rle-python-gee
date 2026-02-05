"""Example usage of the map.py module to create country maps."""

from rle_python_gee.map import create_country_map


def main():
    """Generate example maps for different countries with various options."""

    # Example 1: Singapore (default settings with surrounding countries)
    print("Creating map of Singapore with default settings...")
    singapore_path = create_country_map(
        'Singapore',
        'temp/singapore_default.png',
        show_surrounding_countries=True,
        show_grid=True,
        show_border=True,
        title='Singapore'
    )
    print(f"✓ Singapore map saved to: {singapore_path}")

    # Example 2: Myanmar without surrounding country labels
    print("\nCreating map of Myanmar without surrounding countries...")
    myanmar_path = create_country_map(
        'Myanmar',
        'temp/myanmar_no_context.png',
        show_surrounding_countries=False,
        show_grid=False,
        show_border=False,
        edge_color='black'
    )
    print(f"✓ Myanmar map saved to: {myanmar_path}")

    # Example 3: Brazil with custom title and no grid
    print("\nCreating map of Brazil with custom title and no grid...")
    brazil_path = create_country_map(
        'Brazil',
        'temp/brazil_custom.png',
        show_grid=False,
        title='Brazil - Biodiversity Hotspot'
    )
    print(f"✓ Brazil map saved to: {brazil_path}")

    # Example 4: Japan with minimal styling (no border, no grid, no title)
    print("\nCreating minimal map of Japan...")
    japan_path = create_country_map(
        'Japan',
        'temp/japan_minimal.png',
        show_border=False,
        show_grid=False,
        title=''
    )
    print(f"✓ Japan map saved to: {japan_path}")

    # Example 5: Kenya clean map (no context, no grid, custom title)
    print("\nCreating clean map of Kenya...")
    kenya_path = create_country_map(
        'Kenya',
        'temp/kenya_clean.png',
        show_surrounding_countries=False,
        show_grid=False,
        show_border=False,
        edge_color='black'
    )
    print(f"✓ Kenya map saved to: {kenya_path}")

    # Example 6: Custom colors - Red country with dark border
    print("\nCreating map of Vietnam with custom colors...")
    vietnam_path = create_country_map(
        'Vietnam',
        'temp/vietnam_red.png',
        fill_color='#ff6b6b',
        edge_color='#8b0000',
        edge_width=2.5,
        title='Vietnam'
    )
    print(f"✓ Vietnam map saved to: {vietnam_path}")

    # Example 7: Green country without border
    print("\nCreating map of New Zealand with green fill...")
    nz_path = create_country_map(
        'New Zealand',
        'temp/newzealand_green.png',
        fill_color='#4ecdc4',
        show_border=False,
        show_grid=False,
        title='New Zealand'
    )
    print(f"✓ New Zealand map saved to: {nz_path}")

    print("\n" + "="*50)
    print("All example maps created successfully!")
    print("="*50)
    print("\nGenerated files:")
    print("  - singapore_default.png (with context)")
    print("  - myanmar_no_context.png (isolated)")
    print("  - brazil_custom.png (custom title, no grid)")
    print("  - japan_minimal.png (minimal styling)")
    print("  - kenya_clean.png (clean, simple)")
    print("  - vietnam_red.png (custom red colors)")
    print("  - newzealand_green.png (custom green, no border)")
    print("="*50)


if __name__ == "__main__":
    main()
