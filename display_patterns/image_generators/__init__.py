"""
Extraction-era pattern surface, kept for existing consumers.

Re-exports the checkerboard class surface bmd-signal-gen adopted at the
extraction (§spec:extraction). New code should use the frame-indexed
catalog in :mod:`display_patterns.patterns`.
"""

from display_patterns.image_generators.checkerboard import (
    DEFAULT_PATTERN_GENERATOR,
    ROI,
    PatternGenerator,
)

__all__ = [
    "DEFAULT_PATTERN_GENERATOR",
    "ROI",
    "PatternGenerator",
]
