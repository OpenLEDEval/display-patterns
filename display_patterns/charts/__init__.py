"""
Chart generation module for BMD signal generator.

This module provides tools for creating display-ready test charts from
colorimetric data (XYZ, RGB, or built-in definitions like SMPTE bars).
Charts can include optional text labels for measurement validation
with spectroradiometers and colorimeters.
"""

import importlib
from typing import TYPE_CHECKING

from display_patterns.charts.color_types import ChartLayout, ColorValue, Patch
from display_patterns.charts.conversion import xyz_to_display_rgb
from display_patterns.charts.renderer import render_chart

if TYPE_CHECKING:
    from display_patterns.charts.tiff_reader import TiffMetadata, load_chart_tiff
    from display_patterns.charts.tiff_writer import write_chart_tiff

# TIFF export depends on tifffile, priced separately by the ``io`` extra
# (§spec:package-shape). Import lazily so ``display-patterns[charts]``
# imports without it.
_IO_EXPORTS = {
    "TiffMetadata": "display_patterns.charts.tiff_reader",
    "load_chart_tiff": "display_patterns.charts.tiff_reader",
    "write_chart_tiff": "display_patterns.charts.tiff_writer",
}

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
    if name in _IO_EXPORTS:
        return getattr(importlib.import_module(_IO_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
