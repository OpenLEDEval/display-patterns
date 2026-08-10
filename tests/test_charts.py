"""
Chart workflow tests for the ``charts`` and ``io`` extras.

Covers the review workflow from §req:user-stories: author a chart as a
YAML patch list, render it to a uint16 array, export a 16-bit TIFF, and
read the identical array back (§spec:catalog, §spec:file-export).
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("colour")
pytest.importorskip("PIL")
pytest.importorskip("yaml")
pytest.importorskip("tifffile")

from display_patterns.charts import (
    load_chart_tiff,
    render_chart,
    write_chart_tiff,
)
from display_patterns.charts.loaders import load_chart

CHART_YAML_PATH = Path(__file__).parent / "data" / "two_patch.yaml"


def test_yaml_chart_renders() -> None:
    """A YAML-authored chart loads and renders to a uint16 array."""
    layout = load_chart(CHART_YAML_PATH)

    assert layout.name == "Two Patch Chart"
    assert [patch.name for patch in layout.patches] == ["GS 1", "Red"]

    image = render_chart(layout, bit_depth=12)

    assert image.shape == (360, 640, 3)
    assert image.dtype == np.uint16
    assert image.max() <= 4095
    # The white patch center lands near full code value.
    assert image[180, 160].min() > 3900


def test_tiff_roundtrip(tmp_path: Path) -> None:
    """Writing a rendered chart to TIFF and reading it back is lossless."""
    layout = load_chart(CHART_YAML_PATH)
    image = render_chart(layout, bit_depth=12)

    tiff_path = tmp_path / "two_patch.tiff"
    write_chart_tiff(tiff_path, image, layout, bit_depth=12)

    loaded, metadata = load_chart_tiff(tiff_path)

    np.testing.assert_array_equal(loaded, image)
    assert metadata.chart_name == "Two Patch Chart"
    assert metadata.bit_depth == 12
    assert metadata.patches is not None
    assert len(metadata.patches) == 2
