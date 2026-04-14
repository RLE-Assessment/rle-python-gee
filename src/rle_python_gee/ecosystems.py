"""Ecosystem distribution data sources for RLE assessments.

Provides the Ecosystems class hierarchy and make_ecosystems() factory
for loading ecosystem data from multiple backends (GeoJSON, GeoParquet,
Earth Engine FeatureCollections, Earth Engine Images, COGs).
"""

import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


def _natural_key(s: str) -> list:
    """Sort key that orders numeric parts numerically (e.g. T1.1.2 before T1.1.10)."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', s)]


def _write_parquet(gdf, path) -> None:
    """Write a GeoDataFrame to a GeoParquet file (local or gs://)."""
    path_str = str(path)
    if path_str.startswith("gs://"):
        bucket = path_str.split("/")[2]
        try:
            gdf.to_parquet(path_str)
        except FileNotFoundError as exc:
            msg = str(exc)
            if "does not exist" in msg:
                raise FileNotFoundError(
                    f"GCS bucket '{bucket}' not found. "
                    f"Create it with:  gcloud storage buckets create gs://{bucket}"
                ) from None
            raise FileNotFoundError(
                f"Failed to write to {path_str!r}: {msg}"
            ) from None
        except ImportError:
            raise ImportError(
                "The 'gcsfs' package is required to write to GCS. "
                "Install it with:  pip install gcsfs"
            ) from None
    else:
        from pathlib import Path
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(path_str)


class EcosystemKind(Enum):
    VECTOR_LOCAL = "vector_local"
    RASTER_LOCAL = "raster_local"
    EE_FEATURE_COLLECTION = "ee_fc"
    EE_IMAGE = "ee_image"


def _geodataframe_to_ee_fc(gdf):
    """Convert a GeoDataFrame to an ee.FeatureCollection (small datasets only)."""
    import json

    import ee

    geojson = json.loads(gdf.to_json())
    return ee.FeatureCollection(geojson)


# Max features to upload inline (larger datasets go via GCS ingestion)
_INLINE_UPLOAD_LIMIT = 1000


def _upload_gdf_to_ee_asset(gdf, asset_id: str, *,
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


class Ecosystems(ABC):
    """Base class for ecosystem distribution datasets."""

    def __init__(self, data, *, ecosystem_column: str | None = None,
                 ecosystem_name_column: str | None = None,
                 functional_group_column: str | None = None):
        self._data = data
        self._cached = None
        self.ecosystem_column = ecosystem_column
        self.ecosystem_name_column = ecosystem_name_column
        self.functional_group_column = functional_group_column

    @property
    @abstractmethod
    def kind(self) -> EcosystemKind: ...

    @abstractmethod
    def _load(self) -> Any: ...

    def load(self) -> Any:
        """Load and cache the ecosystem data. Returns the native object."""
        if self._cached is None:
            self._cached = self._load()
        return self._cached

    def head(self, n: int = 5):
        """Return the first n rows of the loaded data."""
        data = self.load()
        if hasattr(data, 'head'):
            return data.head(n)
        return data

    def size(self) -> int:
        """Return the number of features."""
        data = self.load()
        if hasattr(data, '__len__'):
            return len(data)
        raise NotImplementedError(
            f"size not supported for {self.kind.value}"
        )

    def limit(self, n: int) -> "Ecosystems":
        """Return a new Ecosystems with only the first n features."""
        data = self.load()
        if hasattr(data, 'iloc'):
            return EcosystemsGeoDataFrame(data.iloc[:n], ecosystem_column=self.ecosystem_column,
                                         ecosystem_name_column=self.ecosystem_name_column,
                                         functional_group_column=self.functional_group_column)
        raise NotImplementedError(
            f"limit not supported for {self.kind.value}"
        )

    def unique_ecosystems(self) -> list[str]:
        """Return a naturally sorted list of unique ecosystem values."""
        if self.ecosystem_column is None:
            raise ValueError("ecosystem_column is not set")
        data = self.load()
        if hasattr(data, '__getitem__'):
            return sorted(data[self.ecosystem_column].unique(), key=_natural_key)
        raise NotImplementedError(
            f"unique_ecosystems not supported for {self.kind.value}"
        )

    def unique_functional_groups(self) -> list[str]:
        """Return a naturally sorted list of unique functional group values."""
        if self.functional_group_column is None:
            raise ValueError("functional_group_column is not set")
        data = self.load()
        if hasattr(data, '__getitem__'):
            return sorted(data[self.functional_group_column].unique(), key=_natural_key)
        raise NotImplementedError(
            f"unique_functional_groups not supported for {self.kind.value}"
        )

    def ecosystem_name(self, code: str) -> str:
        """Look up the human-readable name for an ecosystem code.

        Requires ``ecosystem_name_column`` to be set.
        """
        if self.ecosystem_name_column is None:
            raise ValueError("ecosystem_name_column is not set")
        if self.ecosystem_column is None:
            raise ValueError("ecosystem_column is not set")
        data = self.load()
        match = data.loc[data[self.ecosystem_column] == code, self.ecosystem_name_column]
        if match.empty:
            raise KeyError(f"Ecosystem code {code!r} not found")
        return match.iloc[0]

    def ecosystem_names(self) -> dict[str, str]:
        """Return a mapping of ecosystem codes to their names, sorted naturally by code.

        Requires ``ecosystem_name_column`` to be set.
        """
        if self.ecosystem_name_column is None:
            raise ValueError("ecosystem_name_column is not set")
        if self.ecosystem_column is None:
            raise ValueError("ecosystem_column is not set")
        data = self.load()
        pairs = data.drop_duplicates(subset=self.ecosystem_column)
        mapping = dict(zip(
            pairs[self.ecosystem_column],
            pairs[self.ecosystem_name_column],
        ))
        return {k: mapping[k] for k in sorted(mapping, key=_natural_key)}

    def filter(self, pattern: str, *, regex: bool = False) -> "Ecosystems":
        """Return a new Ecosystems containing only features matching the given value.

        Args:
            pattern: Exact value or regex pattern to match against the ecosystem column.
            regex: If True, treat pattern as a regular expression.

        Returns:
            A new Ecosystems object with only the matching features.
        """
        if self.ecosystem_column is None:
            raise ValueError("ecosystem_column is not set")
        data = self.load()
        if not hasattr(data, '__getitem__'):
            raise NotImplementedError(
                f"filter not supported for {self.kind.value}"
            )
        if regex:
            mask = data[self.ecosystem_column].str.match(pattern)
        else:
            mask = data[self.ecosystem_column] == pattern
        return EcosystemsGeoDataFrame(data[mask], ecosystem_column=self.ecosystem_column,
                                     ecosystem_name_column=self.ecosystem_name_column,
                                     functional_group_column=self.functional_group_column)

    def calculate_aoo(self, *, threshold: float = 0.01) -> int:
        """Calculate the Area of Occupancy (AOO) cell count for this ecosystem.

        Computes an AOO grid, filters by the ecosystem, sorts cells by
        fractional coverage, and counts cells whose cumulative proportion
        exceeds *threshold* (Section 6.3.2 of IUCN RLE Guidelines 2024).

        Args:
            threshold: Minimum cumulative proportion to include a cell.
                Default 0.01 (exclude cells accounting for up to 1% of
                total mapped extent).

        Returns:
            Number of AOO grid cells.
        """
        from rle_python_gee.aoo import make_aoo_grid, slugify_ecosystem_name

        ecosystems = self.unique_ecosystems()
        if len(ecosystems) != 1:
            raise ValueError(
                f"calculate_aoo requires exactly one ecosystem, "
                f"but found {len(ecosystems)}. Filter first with "
                f".filter('ecosystem_name')."
            )
        ecosystem_code = ecosystems[0]
        column = slugify_ecosystem_name(ecosystem_code)

        aoo_grid = make_aoo_grid(self).compute()
        filtered = aoo_grid.filter_by_ecosystem(ecosystem_code)
        gdf = filtered.grid_cells.sort_values(by=column)
        gdf["cumulative_fraction"] = gdf[column].cumsum()
        total = gdf["cumulative_fraction"].iloc[-1]
        gdf["cumulative_proportion"] = gdf["cumulative_fraction"] / total
        return int(len(gdf[gdf["cumulative_proportion"] > threshold]))

    def calculate_eoo(self) -> "EOO":
        """Calculate the Extent of Occurrence (EOO) for this ecosystem.

        Computes the convex hull of all ecosystem geometries and returns
        an EOO object with the area in km².

        Returns:
            A computed EOO instance. Access area via ``.area_km2``.
        """
        from rle_python_gee.eoo import make_eoo
        return make_eoo(self).compute()

    def to_raster(
        self,
        path,
        *,
        crs,
        scale,
        mode: str = "index",
        oversampling: int = 10,
        nodata=None,
    ) -> dict[int, str]:
        """Rasterize ecosystem polygons to a Cloud Optimized GeoTIFF.

        Two modes are supported:

        * ``mode="index"`` (default): single-band integer raster where each
          pixel holds the 1-based index of the ecosystem covering the
          pixel's center (rasterio's default ``all_touched=False``
          semantics). Indices follow the natural-sort order of
          ``unique_ecosystems()``. Where polygons overlap at a pixel
          center, the naturally-later code wins (rasterio's default
          ``MergeAlg.replace`` combined with our deterministic iteration
          order). Default nodata is the maximum value of the chosen
          output dtype (255 / 65535 / 4294967295).

        * ``mode="fraction"``: multi-band float32 raster with one band per
          ecosystem (also in natural-sort order). Each band stores the
          fraction of the pixel covered by that ecosystem in
          ``[0.0, 1.0]``, computed by rasterizing a binary mask at
          ``oversampling`` × per axis and averaging the resulting
          sub-pixels. Each band's description tag is set to its ecosystem
          code. Default nodata is ``NaN``.

        Args:
            path: Output COG path.
            crs: Target CRS (EPSG code, WKT, or pyproj CRS).
            scale: Pixel size in CRS units (meters for projected CRS).
            mode: ``"index"`` or ``"fraction"``.
            oversampling: Sub-pixel factor per axis for ``"fraction"``
                mode (1..255). Ignored in ``"index"`` mode.
            nodata: Nodata sentinel. Defaults to dtype-max in ``"index"``
                mode and ``NaN`` in ``"fraction"`` mode.

        Returns:
            Mapping of 1-based index (= band number in fraction mode) ->
            ecosystem code.
        """
        if self.kind != EcosystemKind.VECTOR_LOCAL:
            raise NotImplementedError(
                f"to_raster not supported for {self.kind.value}"
            )
        if mode not in ("index", "fraction"):
            raise ValueError(
                f"mode must be 'index' or 'fraction', got {mode!r}"
            )
        if mode == "fraction" and not (1 <= oversampling <= 255):
            raise ValueError("oversampling must be in [1, 255]")

        import json
        import math
        from pathlib import Path
        import numpy as np
        import rasterio
        from rasterio.features import rasterize as _rio_rasterize
        from rasterio.transform import from_origin
        import shapely

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # ---- Load + validate + reproject ----
        gdf = self.to_geodataframe()
        if gdf.empty:
            raise ValueError("Cannot rasterize empty ecosystem dataset")
        if gdf.crs is None:
            raise ValueError("Input GeoDataFrame has no CRS set")
        gdf = gdf.copy()
        gdf["geometry"] = shapely.make_valid(gdf.geometry)
        gdf = gdf.to_crs(crs)
        gdf = gdf[~(gdf.geometry.is_empty | gdf.geometry.isna())]
        if gdf.empty:
            raise ValueError("All geometries dropped after reprojection")

        codes = self.unique_ecosystems()  # naturally sorted
        n = len(codes)
        if n == 0:
            raise ValueError("No ecosystems to rasterize")
        code_to_index = {c: i for i, c in enumerate(codes, start=1)}
        mapping: dict[int, str] = {i: c for i, c in enumerate(codes, start=1)}
        ecosystem_col = self.ecosystem_column

        # ---- Snap target grid to pixel boundaries ----
        minx, miny, maxx, maxy = gdf.total_bounds
        minx = float(np.floor(minx / scale) * scale)
        miny = float(np.floor(miny / scale) * scale)
        maxx = float(np.ceil(maxx / scale) * scale)
        maxy = float(np.ceil(maxy / scale) * scale)
        W = int(round((maxx - minx) / scale))
        H = int(round((maxy - miny) / scale))
        transform = from_origin(minx, maxy, scale, scale)

        # ---- Mode dispatch ----
        if mode == "index":
            # dtype: indices 1..n + reserved nodata at dtype_max
            if n < 255:
                dtype = np.uint8
            elif n < 65535:
                dtype = np.uint16
            elif n < 4294967295:
                dtype = np.uint32
            else:
                raise ValueError(
                    f"Too many ecosystems ({n}) for to_raster"
                )
            dtype_str = np.dtype(dtype).name
            dtype_max = int(np.iinfo(dtype).max)
            if nodata is None:
                resolved_nodata = dtype_max
            else:
                resolved_nodata = int(nodata)
                if not (0 <= resolved_nodata <= dtype_max):
                    raise ValueError(
                        f"nodata={nodata} out of range for {dtype_str}"
                    )
                if 1 <= resolved_nodata <= n:
                    raise ValueError(
                        f"nodata={nodata} collides with valid ecosystem "
                        f"index 1..{n}"
                    )

            # All shapes in one rasterize call. Natural-sort order combined
            # with the default MergeAlg.replace means later codes win at
            # overlapping pixel centers.
            shapes = []
            for code in codes:
                idx = code_to_index[code]
                for geom in gdf.loc[gdf[ecosystem_col] == code, "geometry"]:
                    if geom is None or geom.is_empty:
                        continue
                    shapes.append((geom, idx))
            arr = _rio_rasterize(
                shapes=shapes,
                out_shape=(H, W),
                transform=transform,
                fill=resolved_nodata,
                dtype=dtype_str,
                all_touched=False,
            )
            count = 1
        else:  # mode == "fraction"
            if nodata is None:
                resolved_nodata = float("nan")
            else:
                resolved_nodata = float(nodata)
                if not (math.isnan(resolved_nodata)
                        or math.isfinite(resolved_nodata)):
                    raise ValueError(
                        f"nodata={nodata} must be finite or NaN"
                    )
            dtype_str = "float32"

            N = oversampling
            over_transform = from_origin(minx, maxy, scale / N, scale / N)
            over_shape = (H * N, W * N)
            inv_n2 = 1.0 / (N * N)

            arr = np.zeros((n, H, W), dtype=np.float32)
            for i, code in enumerate(codes, start=1):
                subset = gdf.loc[gdf[ecosystem_col] == code, "geometry"]
                if subset.empty:
                    continue
                mask = _rio_rasterize(
                    shapes=(
                        (geom, 1) for geom in subset if not geom.is_empty
                    ),
                    out_shape=over_shape,
                    transform=over_transform,
                    fill=0,
                    dtype="uint8",
                    all_touched=False,
                )
                cov = mask.reshape(H, N, W, N).sum(axis=(1, 3))
                arr[i - 1, :, :] = cov.astype(np.float32) * inv_n2
            count = n

        profile = {
            "driver": "COG",
            "dtype": dtype_str,
            "count": count,
            "height": H,
            "width": W,
            "crs": crs,
            "transform": transform,
            "nodata": resolved_nodata,
            "compress": "deflate",
            "predictor": 2 if mode == "index" else 3,
            "blocksize": 512,
            "overview_resampling": "nearest" if mode == "index" else "average",
            "BIGTIFF": "IF_SAFER",
        }
        with rasterio.open(path, "w", **profile) as dst:
            if mode == "index":
                dst.write(arr, 1)
            else:
                dst.write(arr)
                for i, code in mapping.items():
                    dst.set_band_description(i, code)
            dst.update_tags(
                ECOSYSTEM_COLUMN=ecosystem_col,
                ECOSYSTEM_INDEX_JSON=json.dumps(mapping),
                RASTERIZE_MODE=mode,
            )
            dst.update_tags(1, **{f"ECO_{i}": c for i, c in mapping.items()})

        return mapping

    def _feature_count(self) -> int | None:
        """Return the number of features, or None if not applicable."""
        if hasattr(self._cached, '__len__'):
            return len(self._cached)
        return None

    # -- export / write -------------------------------------------------------

    def to_geodataframe(self) -> "gpd.GeoDataFrame":
        """Convert to a GeoDataFrame.

        For vector local backends, returns the loaded GeoDataFrame directly.
        For EE FeatureCollection, downloads via the EE API.
        """
        import geopandas as gpd

        if self.kind == EcosystemKind.VECTOR_LOCAL:
            return self.load()
        raise NotImplementedError(
            f"to_geodataframe not supported for {self.kind.value}"
        )

    def to_parquet(self, path) -> None:
        """Write ecosystem data as a GeoParquet file."""
        _write_parquet(self.to_geodataframe(), path)

    def to_geojson(self, path) -> None:
        """Write ecosystem data as a GeoJSON file."""
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        gdf = self.to_geodataframe()
        gdf.to_file(path, driver="GeoJSON")

    def to_ee_feature_collection(self, asset_id: str, *,
                                  gcs_bucket: str | None = None):
        """Upload ecosystem data as an Earth Engine asset.

        Small datasets are uploaded inline. Large datasets (> 1000 features)
        are written as a shapefile to GCS and ingested (requires gcs_bucket).

        Returns the task/result, or None if asset already exists.
        """
        gdf = self.to_geodataframe()
        return _upload_gdf_to_ee_asset(
            gdf, asset_id, gcs_bucket=gcs_bucket, description="ecosystem_export"
        )

    # -- visualization -------------------------------------------------------

    def to_layer(self, *, get_fill_color=None, get_line_color=None, max_features: int = 1000):
        """Return lonboard layer(s) for this ecosystem dataset.

        Args:
            get_fill_color: Fill color for polygons.
            get_line_color: Line color for polygons.
            max_features: Maximum number of features to display. Default 1000.
        """
        if self.kind != EcosystemKind.VECTOR_LOCAL:
            raise NotImplementedError(
                f"Visualization not yet supported for {self.kind.value}"
            )
        try:
            from lonboard import PolygonLayer
        except ImportError:
            raise ImportError(
                "lonboard is required for visualization. "
                "Install it with: pip install lonboard"
            ) from None

        if get_fill_color is None:
            get_fill_color = [0, 255, 0, 128]
        if get_line_color is None:
            get_line_color = [0, 0, 0, 255]

        gdf = self.load()
        if gdf.empty:
            return []
        if len(gdf) > max_features:
            raise ValueError(
                f"Dataset has {len(gdf):,} features, exceeding max_features={max_features:,}. "
                f"Use .limit() or .filter() to reduce, increase max_features, "
                f"or upload to Earth Engine for tile-based visualization."
            )
        return [PolygonLayer.from_geopandas(
            gdf,
            get_fill_color=get_fill_color,
            get_line_color=get_line_color,
            line_width_min_pixels=1,
        )]

    def to_map(self, *, max_features: int = 1000, **kwargs):
        """Return a lonboard Map showing the ecosystem polygons.

        Args:
            max_features: Maximum number of features to display. Default 1000.
            **kwargs: Additional arguments passed to lonboard.Map.
        """
        try:
            from lonboard import Map
        except ImportError:
            raise ImportError(
                "lonboard is required for visualization. "
                "Install it with: pip install lonboard"
            ) from None

        try:
            layers = self.to_layer(max_features=max_features)
        except ValueError as e:
            from IPython.display import HTML, display
            display(HTML(f"<div style='padding:12px;background:#fff3cd;border:1px solid #ffc107;border-radius:4px'>"
                         f"<b>Cannot display map:</b> {e}</div>"))
            return None
        return Map(layers=layers, **kwargs)

    # -- display -------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{type(self).__name__}(data={self._data!r})"

    def _repr_html_(self) -> str:
        parts = [
            f"<b>{type(self).__name__}</b>",
            f"Kind: {self.kind.value}",
            f"Source: {self._data!r}",
        ]
        if self._cached is not None:
            count = self._feature_count()
            if count is not None:
                parts.append(f"Features: {count:,}")
        return "<br>".join(parts)

    # -- factory classmethods -------------------------------------------------

    @classmethod
    def from_file(cls, path, *, ecosystem_column: str, **kwargs) -> "Ecosystems":
        """Create from a vector file (Shapefile, GeoJSON, etc.)."""
        return EcosystemsFile(path, ecosystem_column=ecosystem_column, **kwargs)

    @classmethod
    def from_parquet(cls, path, *, ecosystem_column: str, **kwargs) -> "Ecosystems":
        """Create from a GeoParquet file."""
        return EcosystemsGeoParquet(path, ecosystem_column=ecosystem_column, **kwargs)

    @classmethod
    def from_gee_feature_collection(cls, data, *,
                                    ecosystem_column: str,
                                    **kwargs) -> "Ecosystems":
        """Create from an Earth Engine FeatureCollection or asset ID."""
        return EcosystemsEEFeatureCollection(
            data, ecosystem_column=ecosystem_column, **kwargs
        )

    @classmethod
    def from_gee_image(cls, data, **kwargs) -> "Ecosystems":
        """Create from an Earth Engine Image or asset ID."""
        return EcosystemsEEImage(data, **kwargs)

    @classmethod
    def from_cog(cls, data, **kwargs) -> "Ecosystems":
        """Create from a Cloud Optimized GeoTIFF."""
        return EcosystemsCOG(data, **kwargs)


# ---------------------------------------------------------------------------
# Vector local backends
# ---------------------------------------------------------------------------


class EcosystemsFile(Ecosystems):
    """Ecosystem polygons from a vector file (Shapefile, GeoJSON, etc.)."""

    kind = EcosystemKind.VECTOR_LOCAL

    def __init__(self, data, *, ecosystem_column: str, ecosystem_name_column: str | None = None,
                 functional_group_column: str | None = None):
        super().__init__(data, ecosystem_column=ecosystem_column,
                         ecosystem_name_column=ecosystem_name_column,
                         functional_group_column=functional_group_column)

    def _load(self):
        import geopandas as gpd

        return gpd.read_file(self._data)


class EcosystemsGeoParquet(Ecosystems):
    """Ecosystem polygons from a GeoParquet file."""

    kind = EcosystemKind.VECTOR_LOCAL

    def __init__(self, data, *, ecosystem_column: str, ecosystem_name_column: str | None = None,
                 functional_group_column: str | None = None):
        super().__init__(data, ecosystem_column=ecosystem_column,
                         ecosystem_name_column=ecosystem_name_column,
                         functional_group_column=functional_group_column)

    def _load(self):
        import geopandas as gpd

        return gpd.read_parquet(self._data)


class EcosystemsGeoDataFrame(Ecosystems):
    """Ecosystem polygons from an in-memory GeoDataFrame."""

    kind = EcosystemKind.VECTOR_LOCAL

    def __init__(self, data, *, ecosystem_column: str, ecosystem_name_column: str | None = None,
                 functional_group_column: str | None = None):
        super().__init__(data, ecosystem_column=ecosystem_column,
                         ecosystem_name_column=ecosystem_name_column,
                         functional_group_column=functional_group_column)
        self._cached = data

    def _load(self):
        return self._data


# ---------------------------------------------------------------------------
# Earth Engine backends
# ---------------------------------------------------------------------------


class EcosystemsEEFeatureCollection(Ecosystems):
    """Ecosystem polygons from an Earth Engine FeatureCollection."""

    kind = EcosystemKind.EE_FEATURE_COLLECTION

    def __init__(self, data, *, ecosystem_column: str, ecosystem_name_column: str | None = None,
                 functional_group_column: str | None = None):
        super().__init__(data, ecosystem_column=ecosystem_column,
                         ecosystem_name_column=ecosystem_name_column,
                         functional_group_column=functional_group_column)

    def _load(self):
        import ee

        if isinstance(self._data, str):
            return ee.FeatureCollection(self._data)
        return self._data

    def _feature_count(self) -> int | None:
        if self._cached is not None:
            return self._cached.size().getInfo()
        return None

    def to_geodataframe(self) -> "gpd.GeoDataFrame":
        """Download the FeatureCollection as a GeoDataFrame."""
        import ee

        fc = self.load()
        gdf = ee.data.computeFeatures({
            "expression": fc,
            "fileFormat": "GEOPANDAS_GEODATAFRAME",
        })
        return gdf.set_crs("EPSG:4326")

    def to_ee_feature_collection(self, asset_id: str):
        """Export to an Earth Engine table asset. Returns the started Task."""
        import ee

        fc = self.load()
        task = ee.batch.Export.table.toAsset(
            collection=fc,
            assetId=asset_id,
            description="ecosystem_export",
        )
        task.start()
        return task

    def to_layer(self):
        """Return a BitmapTileLayer rendering the FeatureCollection via EE tiles."""
        try:
            from lonboard import BitmapTileLayer
        except ImportError:
            raise ImportError(
                "lonboard is required for visualization. "
                "Install it with: pip install lonboard"
            ) from None

        fc = self.load()
        styled = fc.style(color='0080FF', fillColor='0080FF40')
        map_id = styled.getMapId()
        tile_url = map_id['tile_fetcher'].url_format
        return [BitmapTileLayer(data=tile_url)]


class EcosystemsEEImage(Ecosystems):
    """Ecosystem coverage from an Earth Engine Image."""

    kind = EcosystemKind.EE_IMAGE

    def _load(self):
        import ee

        if isinstance(self._data, str):
            return ee.Image(self._data)
        return self._data


# ---------------------------------------------------------------------------
# Raster local backend
# ---------------------------------------------------------------------------


class EcosystemsCOG(Ecosystems):
    """Ecosystem coverage from a Cloud Optimized GeoTIFF."""

    kind = EcosystemKind.RASTER_LOCAL

    def _load(self):
        import rioxarray  # noqa: F401

        return rioxarray.open_rasterio(self._data)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def _is_file_path(data) -> bool:
    """Check if data looks like a file path."""
    if not isinstance(data, str):
        return False
    return (
        data.endswith((".parquet", ".geojson", ".tif", ".tiff"))
        or data.startswith(("gs://", "/", "."))
    )


def make_ecosystems(data, **kwargs) -> Ecosystems:
    """Auto-detect and create an Ecosystems instance.

    Args:
        data: Data source. One of:
            - Path to a GeoJSON file (.geojson)
            - Path to a GeoParquet file (.parquet)
            - Path to a COG file (.tif, .tiff)
            - ee.Image object
            - ee.FeatureCollection object
            - Earth Engine asset ID string
        **kwargs: Additional arguments passed to the backend constructor.

    Returns:
        An Ecosystems instance.
    """
    # Accept Path objects
    from pathlib import PurePath
    if isinstance(data, PurePath):
        data = str(data)

    # File paths
    if isinstance(data, str):
        if data.endswith(".geojson"):
            return EcosystemsFile(data, **kwargs)
        if data.endswith(".parquet"):
            return EcosystemsGeoParquet(data, **kwargs)
        if data.endswith((".tif", ".tiff")):
            return EcosystemsCOG(data, **kwargs)

    # Earth Engine objects
    try:
        import ee

        if isinstance(data, ee.Image):
            return EcosystemsEEImage(data, **kwargs)
        if isinstance(data, ee.FeatureCollection):
            return EcosystemsEEFeatureCollection(data, **kwargs)
    except ImportError:
        pass

    # String asset IDs — detect via EE API
    if isinstance(data, str) and not _is_file_path(data):
        try:
            import ee

            asset_info = ee.data.getAsset(data)
            asset_type = asset_info.get("type", "")
            if asset_type in ("IMAGE", "IMAGE_COLLECTION"):
                return EcosystemsEEImage(data, **kwargs)
            if asset_type == "TABLE":
                return EcosystemsEEFeatureCollection(data, **kwargs)
        except Exception:
            pass

    raise ValueError(
        f"Cannot determine ecosystem backend for data: {data!r}. "
        f"Supported types: .geojson, .parquet, .tif/.tiff, "
        f"ee.Image, ee.FeatureCollection"
    )
