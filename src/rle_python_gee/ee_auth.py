"""Earth Engine authentication utilities."""

import ee
from google.auth import default


def initialize_ee(project: str):
    """Initialize Earth Engine.

    Args:
        project: The Earth Engine project to initialize.

    Returns:
        None
    """
    credentials, _ = default(scopes=[
        'https://www.googleapis.com/auth/earthengine',
        'https://www.googleapis.com/auth/cloud-platform'
    ])
    ee.Initialize(credentials=credentials, project=project)


def check_authentication() -> dict[str, bool | str]:
    """
    Test authentication to the Earth Engine Python API.

    Returns:
        dict: A dictionary containing:
            - 'authenticated' (bool): Whether authentication was successful
            - 'message' (str): A descriptive message about the authentication status
            - 'project' (str | None): The authenticated project ID if available

    Examples:
        >>> result = check_authentication()
        >>> if result['authenticated']:
        ...     print(f"Authenticated with project: {result['project']}")
        ... else:
        ...     print(f"Authentication failed: {result['message']}")
    """
    try:
        # Try to initialize Earth Engine
        ee.Initialize()

        # If we get here, initialization succeeded
        # Try to get the current project
        try:
            # Attempt a simple operation to verify authentication works
            asset_roots = ee.data.getAssetRoots()
            project_id = None
            if asset_roots and len(asset_roots) > 0:
                # Extract project ID from the first asset root
                # Asset roots look like: [{'id': 'projects/my-project', ...}]
                root_id = asset_roots[0].get('id', '')
                if root_id.startswith('projects/'):
                    project_id = root_id.split('/', 1)[1]
                else:
                    project_id = root_id
            return {
                'authenticated': True,
                'message': 'Successfully authenticated to Earth Engine',
                'project': project_id
            }
        except Exception as e:
            # Initialization succeeded but we can't get project info
            return {
                'authenticated': True,
                'message': f'Authenticated but could not retrieve project info: {str(e)}',
                'project': None
            }

    except ee.EEException as e:
        # Earth Engine specific exception
        return {
            'authenticated': False,
            'message': f'Earth Engine authentication failed: {str(e)}',
            'project': None
        }

    except Exception as e:
        # Generic exception
        return {
            'authenticated': False,
            'message': f'Authentication error: {str(e)}',
            'project': None
        }


def is_authenticated() -> bool:
    """
    Check if Earth Engine is authenticated.

    Returns:
        bool: True if authenticated, False otherwise

    Examples:
        >>> if is_authenticated():
        ...     print("Ready to use Earth Engine")
        ... else:
        ...     print("Please authenticate first")
    """
    result = check_authentication()
    return result['authenticated']


def print_authentication_status() -> None:
    """
    Print the current Earth Engine authentication status to stdout.

    This is useful for debugging and CLI usage.
    """
    result = check_authentication()

    if result['authenticated']:
        print(f"✓ Earth Engine Authentication: SUCCESS")
        print(f"  Message: {result['message']}")
        if result['project']:
            print(f"  Project: {result['project']}")
    else:
        print(f"✗ Earth Engine Authentication: FAILED")
        print(f"  Message: {result['message']}")
        print(f"\nTo authenticate, run:")
        print(f"  earthengine authenticate")
        print(f"Or use service account authentication with:")
        print(f"  ee.Initialize(credentials=ee.ServiceAccountCredentials(email, key_file))")
