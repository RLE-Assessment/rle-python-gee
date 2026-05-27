"""Entry-point registration for the Earth Engine backends."""

from __future__ import annotations

from rle.core.registry import BackendInfo


def _is_ee_asset(data) -> bool:
    """Best-effort check for an Earth Engine asset ID string."""
    return isinstance(data, str) and data.startswith("projects/")


def register() -> list[BackendInfo]:
    """Advertise the Earth Engine backends provided by rle-python-gee."""
    from rle.gee.ecosystems import (
        EcosystemsEEFeatureCollection,
        EcosystemsEEImage,
    )
    from rle.gee.aoo import (
        AOOGridEEFeatureCollection,
        AOOGridEEImage,
    )

    dist = "rle-python-gee"
    return [
        BackendInfo(
            name="gee-feature-collection",
            cls=EcosystemsEEFeatureCollection,
            capability="ecosystems",
            distribution=dist,
            can_handle=_is_ee_asset,
        ),
        BackendInfo(
            name="gee-image",
            cls=EcosystemsEEImage,
            capability="ecosystems",
            distribution=dist,
            can_handle=_is_ee_asset,
        ),
        BackendInfo(
            name="gee-aoo-feature-collection",
            cls=AOOGridEEFeatureCollection,
            capability="aoo",
            distribution=dist,
        ),
        BackendInfo(
            name="gee-aoo-image",
            cls=AOOGridEEImage,
            capability="aoo",
            distribution=dist,
        ),
    ]
