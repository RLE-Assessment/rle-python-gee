"""Upload GeoDataFrames to Earth Engine table assets.

``upload_gdf_to_ee_asset`` is called (via lazy import) by the core
``Ecosystems.to_ee_feature_collection`` / ``AOOGrid.to_ee_feature_collection``
methods, so installing ``rle-python-gee`` enables Earth Engine export from
local objects.
"""


def _geodataframe_to_ee_fc(gdf):
    """Convert a GeoDataFrame to an ee.FeatureCollection (small datasets only)."""
    import json

    import ee

    geojson = json.loads(gdf.to_json())
    return ee.FeatureCollection(geojson)


# Max features to upload inline (larger datasets go via GCS ingestion)
_INLINE_UPLOAD_LIMIT = 1000


def upload_gdf_to_ee_asset(gdf, asset_id: str, *,
                           gcs_bucket: str | None = None,
                           description: str = "upload"):
    """Upload a GeoDataFrame to an EE table asset.

    For small datasets (<= _INLINE_UPLOAD_LIMIT features), uses inline upload.
    For larger datasets, writes a temp shapefile to GCS and uses
    ee.data.startTableIngestion(). Requires ``gcs_bucket`` for large uploads.

    Returns the task dict (for ingestion) or started Task (for inline export),
    or None if the asset already exists.
    """
    import ee
    import logging
    import tempfile
    from pathlib import Path

    logger = logging.getLogger(__name__)

    # Check if asset already exists
    try:
        ee.data.getAsset(asset_id)
        logger.info("Asset already exists: %s", asset_id)
        return None
    except ee.EEException:
        pass

    if len(gdf) <= _INLINE_UPLOAD_LIMIT:
        logger.info("Uploading %d features inline to %s", len(gdf), asset_id)
        fc = _geodataframe_to_ee_fc(gdf)
        task = ee.batch.Export.table.toAsset(
            collection=fc,
            assetId=asset_id,
            description=description,
        )
        task.start()
        return task

    # Large dataset: write shapefile to temp dir, upload to GCS, ingest
    if gcs_bucket is None:
        raise ValueError(
            f"Dataset has {len(gdf):,} features (> {_INLINE_UPLOAD_LIMIT}). "
            f"A gcs_bucket parameter is required for large uploads. "
            f"Example: .to_ee_feature_collection(asset_id, gcs_bucket='my-bucket')"
        )

    logger.info("Uploading %d features via GCS ingestion to %s", len(gdf), asset_id)

    # Derive a staging path from the asset_id
    parts = asset_id.split("/")
    asset_name = "/".join(parts[3:])  # strip projects/<proj>/assets/

    with tempfile.TemporaryDirectory() as tmpdir:
        shp_name = "upload"
        shp_path = Path(tmpdir) / f"{shp_name}.shp"

        # Ensure EPSG:4326 for EE
        if gdf.crs is not None and not gdf.crs.equals("EPSG:4326"):
            gdf = gdf.to_crs("EPSG:4326")
        elif gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")

        # Repair invalid geometries and drop degenerate slivers (< 1 m²)
        import shapely
        gdf = gdf.copy()
        gdf.geometry = shapely.make_valid(gdf.geometry)
        n_before = len(gdf)
        gdf_ea = gdf.to_crs("ESRI:54034")
        gdf = gdf[gdf_ea.geometry.area >= 1.0].reset_index(drop=True)
        n_dropped = n_before - len(gdf)
        if n_dropped:
            logger.info("Dropped %d degenerate features (area < 1 m²)", n_dropped)

        # Drop columns with names too long for shapefile (10-char limit)
        # and any non-essential columns to keep the upload lean
        long_cols = [c for c in gdf.columns
                     if c != "geometry" and len(c) > 10]
        if long_cols:
            logger.info("Dropping %d columns with names > 10 chars for "
                        "shapefile compatibility: %s", len(long_cols), long_cols)
            gdf = gdf.drop(columns=long_cols)

        logger.info("Writing shapefile to temp directory...")
        gdf.to_file(shp_path)

        # Upload all shapefile components to GCS
        logger.info("Uploading shapefile to gs://%s/_ee_uploads/%s/ ...",
                     gcs_bucket, asset_name)
        import gcsfs
        fs = gcsfs.GCSFileSystem(token="google_default")

        gcs_prefix = f"{gcs_bucket}/_ee_uploads/{asset_name}/{shp_name}"
        try:
            for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                local_file = Path(tmpdir) / f"{shp_name}{ext}"
                if local_file.exists():
                    gcs_path = f"{gcs_prefix}{ext}"
                    fs.put(str(local_file), gcs_path)
                    logger.debug("Uploaded %s", gcs_path)
        except OSError as e:
            if "Forbidden" in str(e) or "billing" in str(e).lower():
                raise RuntimeError(
                    f"GCS upload to gs://{gcs_bucket} failed: permission denied.\n"
                    f"This usually means your credentials have expired or the "
                    f"ADC quota project has billing disabled.\n\n"
                    f"To fix, run:\n"
                    f"  gcloud auth application-default login "
                    f"--project=<your-gcp-project>\n\n"
                    f"Then restart the notebook kernel."
                ) from e
            raise

    # Start ingestion from GCS
    logger.info("Starting EE table ingestion from gs://%s.shp", gcs_prefix)
    request_id = ee.data.newTaskId()[0]
    params = {
        "name": asset_id,
        "sources": [{
            "uris": [f"gs://{gcs_prefix}.shp"],
            "charset": "UTF-8",
        }],
    }
    result = ee.data.startTableIngestion(request_id, params)
    logger.info("Ingestion task started: %s", result.get("id", result.get("name", "unknown")))
    return result
