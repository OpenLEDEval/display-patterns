"""
Chart production: colorimetric patch lists rendered to display RGB
(§spec:catalog).

``color_types`` (numpy-only) loads eagerly; the conversion, renderer,
and TIFF modules load on first attribute access, so importing this
package pays only for what the caller uses — colour-science and
Pillow arrive with the ``charts`` extra, tifffile with ``io``
(§spec:package-shape).
"""

import importlib
from typing import TYPE_CHECKING

from display_patterns.charts.color_types import ChartLayout, ColorValue, Patch

if TYPE_CHECKING:
    from display_patterns.charts.conversion import xyz_to_display_rgb
    from display_patterns.charts.renderer import render_chart
    from display_patterns.charts.tiff_reader import TiffMetadata, load_chart_tiff
    from display_patterns.charts.tiff_writer import write_chart_tiff

# name -> (submodule, extra supplying its dependencies). The runtime
# source of truth for the lazy exports; __all__ derives from it, and
# the TYPE_CHECKING block above mirrors it for static analysis.
_LAZY_EXPORTS = {
    "xyz_to_display_rgb": ("display_patterns.charts.conversion", "charts"),
    "render_chart": ("display_patterns.charts.renderer", "charts"),
    "TiffMetadata": ("display_patterns.charts.tiff_reader", "io"),
    "load_chart_tiff": ("display_patterns.charts.tiff_reader", "io"),
    "write_chart_tiff": ("display_patterns.charts.tiff_writer", "io"),
}

# Literal so ruff recognizes the re-exports (F401); a test asserts every
# name resolves, keeping this list and _LAZY_EXPORTS from drifting.
__all__ = [
    "ChartLayout",
    "ColorValue",
    "Patch",
    "TiffMetadata",
    "load_chart_tiff",
    "render_chart",
    "write_chart_tiff",
    "xyz_to_display_rgb",
]


def __getattr__(name: str) -> object:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, extra = _LAZY_EXPORTS[name]
    try:
        return getattr(importlib.import_module(module), name)
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            f"{name} needs the {extra!r} extra: pip install 'display-patterns[{extra}]'"
        ) from error
