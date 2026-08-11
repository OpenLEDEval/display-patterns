"""
Pytest fixtures for display-patterns tests.

Moved from bmd-signal-gen's test fixtures, trimmed to the pattern
fixtures the extracted modules use (the DeckLink fixtures stayed
behind with the device tool).
"""

import pytest

from display_patterns.image_generators.checkerboard import ROI, PatternGenerator


@pytest.fixture
def pattern_generator_12bit() -> PatternGenerator:
    """
    Create a 12-bit pattern generator for testing.

    Returns
    -------
    PatternGenerator
        Generator configured for 1920x1080 at 12-bit depth.
    """
    return PatternGenerator(
        bit_depth=12,
        width=1920,
        height=1080,
        roi=ROI(x=0, y=0, width=1920, height=1080),
    )


@pytest.fixture
def pattern_generator_8bit() -> PatternGenerator:
    """
    Create an 8-bit pattern generator for testing.

    Returns
    -------
    PatternGenerator
        Generator configured for 1920x1080 at 8-bit depth.
    """
    return PatternGenerator(
        bit_depth=8,
        width=1920,
        height=1080,
        roi=ROI(x=0, y=0, width=1920, height=1080),
    )


@pytest.fixture
def sample_colors_12bit() -> list[list[int]]:
    """
    Sample 12-bit color values for testing.

    Returns
    -------
    list[list[int]]
        Four colors: white, black, red, green (12-bit range).
    """
    return [
        [4095, 4095, 4095],  # White
        [0, 0, 0],  # Black
        [4095, 0, 0],  # Red
        [0, 4095, 0],  # Green
    ]
