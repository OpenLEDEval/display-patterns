"""
display-patterns: deterministic display test patterns, device-free and exact.

The core catalog lives in ``display_patterns.patterns`` and re-exports
here: pure functions of parameters and a frame index, rendered through
a caller-supplied array namespace (numpy default). The core depends on
numpy alone. Chart authoring and TIFF export live in the
``display_patterns.charts`` subpackage behind the ``charts`` and ``io``
extras.
"""

from display_patterns.patterns import (
    ROI,
    ColorRangeError,
    checkerboard,
)

__all__ = [
    "ROI",
    "ColorRangeError",
    "checkerboard",
]
