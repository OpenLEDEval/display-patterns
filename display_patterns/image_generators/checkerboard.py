"""Extraction-era checkerboard surface, kept for existing consumers.

``PatternGenerator`` predates the frame-indexed rendering model
(§spec:render-model) and now delegates to
:func:`display_patterns.patterns.checkerboard`; ``ROI`` and
``ColorRangeError`` re-export from their catalog home. New code should
call the pure entry points in :mod:`display_patterns.patterns`.
"""

import numpy as np
from numpy.typing import ArrayLike

from display_patterns.patterns import ROI, ColorRangeError, checkerboard

__all__ = [
    "DEFAULT_PATTERN_BUFFER",
    "DEFAULT_PATTERN_GENERATOR",
    "ROI",
    "ColorRangeError",
    "PatternGenerator",
]


class PatternGenerator:
    """Generates image patterns with validation and ROI support.

    A stateful wrapper over :func:`display_patterns.patterns.checkerboard`
    that holds frame geometry, bit depth, and region of interest across
    ``generate`` calls.

    Parameters
    ----------
    bit_depth : int
        Bit depth for color values (e.g., 8, 10, 12).
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    roi : ROI, optional
        Region of interest for pattern generation. If None, uses full image.

    Examples
    --------
    Create a pattern generator for 1080p 12-bit output:

    >>> generator = PatternGenerator(bit_depth=12, width=1920, height=1080)
    >>> colors = [[4095, 0, 0], [0, 4095, 0], [0, 0, 4095], [4095, 4095, 4095]]
    >>> pattern = generator.generate(colors)
    >>> print(f"Pattern shape: {pattern.shape}")
    Pattern shape: (1080, 1920, 3)
    """

    def __init__(
        self,
        *,
        bit_depth: int,
        width: int,
        height: int,
        roi: ROI | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.bit_depth = bit_depth

        if roi is None:
            self.roi = ROI(0, 0, self.width, self.height)
        else:
            self.roi = roi

    def generate(self, colors: ArrayLike) -> np.ndarray:
        """Generate a checkerboard pattern with the specified colors.

        Accepts 1-4 colors with the established expansion (one color
        solid, two alternating, four mapped to the 2x2 tile).

        Parameters
        ----------
        colors : ArrayLike
            Color array; shape (3,), or (N, 3) with N in 1..4.

        Returns
        -------
        np.ndarray
            Generated pattern image with shape (height, width, 3),
            dtype uint16.

        Raises
        ------
        RuntimeError
            If colors array has invalid shape.
        ColorRangeError
            If color values exceed bit depth limits.
        """
        return checkerboard(
            colors,
            width=self.width,
            height=self.height,
            bit_depth=self.bit_depth,
            roi=self.roi,
        )


# Default pattern generator for common 1080p 12-bit use case
# Uses 100-pixel border ROI to create centered pattern (1720x880 active area)
DEFAULT_PATTERN_GENERATOR = PatternGenerator(
    bit_depth=12,
    width=1920,
    height=1080,
    roi=ROI(x=100, y=100, width=1920 - 200, height=1080 - 200),
)

# Pre-generated pattern with 12-bit white/black checkerboard
# White: 2000/4095 ≈ 49% of 12-bit range (conservative for HDR displays)
# Black: 0/4095 = 0% (minimum luminance)
DEFAULT_PATTERN_BUFFER = DEFAULT_PATTERN_GENERATOR.generate(
    ((2000, 2000, 2000), (0, 0, 0))
)
