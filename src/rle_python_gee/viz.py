"""Visualization helpers that gracefully fall back to static images.

The main entry point is :func:`smart_map`, which attempts to build an
interactive :class:`lonboard.Map`. If any layer construction raises a
``max_features`` / "too many" :class:`ValueError`, it falls back to rendering
the same data as a static matplotlib PNG so that documents (e.g. Quarto
notebooks) still render instead of failing.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np


_FALLBACK_MARKERS = ("max_features", "too many")


def _is_fallback_error(exc: BaseException) -> bool:
    msg = str(exc)
    return any(marker in msg for marker in _FALLBACK_MARKERS)


def _normalize_specs(specs: Iterable[Any]) -> list[tuple[str, Any, dict | None]]:
    """Normalize heterogeneous spec entries to a uniform internal shape.

    Each returned entry is one of:
      - ("source", datasource, kwargs_dict) — has .to_layer() / .to_gdf_for_viz()
      - ("layer", prebuilt_lonboard_layer, None)
    """
    normalized: list[tuple[str, Any, dict | None]] = []
    for entry in specs:
        if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], dict):
            normalized.append(("source", entry[0], entry[1]))
        elif hasattr(entry, "to_layer"):
            normalized.append(("source", entry, {}))
        else:
            normalized.append(("layer", entry, None))
    return normalized


def smart_map(
    specs: Sequence[Any],
    *,
    view_state: dict | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 8),
):
    """Build an interactive ``lonboard.Map`` or fall back to a static PNG.

    Args:
        specs: Iterable of layer specs. Each entry may be:
            - a datasource with ``.to_layer()`` (uses default styling)
            - a ``(datasource, kwargs)`` tuple forwarded to ``.to_layer(**kwargs)``
            - a pre-built lonboard layer (passed through as-is)
        view_state: Optional initial view state forwarded to ``lonboard.Map``.
        title: Optional title rendered on the static fallback image.
        figsize: Matplotlib figsize for the static fallback.

    Returns:
        ``lonboard.Map`` on success, or ``IPython.display.Image`` (PNG) on
        fallback. Both render inline in Jupyter / Quarto.
    """
    normalized = _normalize_specs(specs)

    try:
        layers: list = []
        for kind, obj, kwargs in normalized:
            if kind == "source":
                layers.extend(obj.to_layer(**(kwargs or {})))
            else:
                layers.append(obj)
    except ValueError as e:
        if not _is_fallback_error(e):
            raise
        return _static_fallback(normalized, title=title, figsize=figsize)

    from lonboard import Map

    map_kwargs = {}
    if view_state is not None:
        map_kwargs["view_state"] = view_state
    return Map(layers=layers, **map_kwargs)


def _static_fallback(
    normalized: list[tuple[str, Any, dict | None]],
    *,
    title: str | None,
    figsize: tuple[float, float],
):
    import io

    import matplotlib.pyplot as plt
    from IPython.display import Image

    fig, ax = plt.subplots(figsize=figsize)

    skipped = 0
    for kind, obj, kwargs in normalized:
        if kind != "source":
            skipped += 1
            continue
        if not hasattr(obj, "to_gdf_for_viz"):
            skipped += 1
            continue
        gdf, style = obj.to_gdf_for_viz(**(kwargs or {}))
        if gdf is None or len(gdf) == 0:
            continue
        _plot_gdf(ax, gdf, style)

    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if title:
        ax.set_title(title)
    if skipped:
        ax.text(
            0.99, 0.01,
            f"{skipped} pre-built layer(s) omitted in static view",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="gray",
        )

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return Image(data=buf.getvalue(), format="png")


def _plot_gdf(ax, gdf, style):
    """Plot a GeoDataFrame onto ``ax`` using lonboard-style color kwargs."""
    from pyproj import CRS

    if gdf.crs is not None and CRS.from_user_input(gdf.crs).to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    fill = _lonboard_color_to_mpl(style.get("fill"), len(gdf))
    edge = _lonboard_color_to_mpl(style.get("edge"), len(gdf))

    plot_kwargs: dict[str, Any] = {"ax": ax, "linewidth": 0.5}
    if fill is not None:
        plot_kwargs["color"] = fill
    if edge is not None:
        plot_kwargs["edgecolor"] = edge
    gdf.plot(**plot_kwargs)


def _lonboard_color_to_mpl(color, n_rows: int):
    """Convert lonboard-style RGB(A) colors to matplotlib-friendly floats.

    Accepts:
      - None (returned as-is)
      - list/tuple of 3 or 4 ints (0-255) → single matplotlib RGBA tuple
      - ndarray shape (N, 3 or 4) uint8/int → array of RGBA tuples (0-1 floats)
    """
    if color is None:
        return None
    arr = np.asarray(color)
    if arr.ndim == 1:
        rgba = _rgba_from_bytes(arr)
        return rgba
    if arr.ndim == 2:
        if arr.shape[0] != n_rows:
            return None
        if np.issubdtype(arr.dtype, np.integer) or arr.max() > 1.0:
            arr = arr.astype(float) / 255.0
        if arr.shape[1] == 3:
            alpha = np.ones((arr.shape[0], 1))
            arr = np.concatenate([arr, alpha], axis=1)
        return arr
    return None


def _rgba_from_bytes(arr: np.ndarray) -> tuple[float, float, float, float]:
    vals = [float(v) for v in arr.tolist()]
    if len(vals) == 3:
        vals.append(255.0)
    r, g, b, a = vals[:4]
    if max(vals) > 1.0:
        r, g, b, a = r / 255.0, g / 255.0, b / 255.0, a / 255.0
    return (r, g, b, a)
