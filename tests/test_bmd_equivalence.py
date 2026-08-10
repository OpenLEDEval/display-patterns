"""
Verbatim-move equivalence against bmd-signal-gen (§spec:extraction).

For identical parameters, the moved modules render arrays equal to the
bmd-signal-gen in-tree implementation, array for array. The source repo
is located via ``BMD_SIGNAL_GEN_REPO`` or as a sibling checkout; the
tests skip when it is absent (for example in CI).

The bmd_sg package initializer imports the DeckLink layer, which needs
device dependencies this repo does not install. A stub package entry
with the real ``__path__`` lets the pattern and chart submodules import
without executing that initializer.
"""

import importlib
import os
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

CHART_YAML_PATH = Path(__file__).parent / "data" / "two_patch.yaml"


def _bmd_repo() -> Path:
    env = os.environ.get("BMD_SIGNAL_GEN_REPO")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "bmd-signal-gen"


pytestmark = pytest.mark.skipif(
    not (_bmd_repo() / "bmd_sg").is_dir(),
    reason="bmd-signal-gen source checkout not available",
)


@pytest.fixture(scope="module")
def bmd_sg_on_path() -> Iterator[None]:
    """Make bmd_sg importable without running its package initializer."""
    stub = types.ModuleType("bmd_sg")
    stub.__path__ = [str(_bmd_repo() / "bmd_sg")]
    sys.modules["bmd_sg"] = stub
    try:
        yield
    finally:
        for name in [m for m in sys.modules if m.split(".")[0] == "bmd_sg"]:
            del sys.modules[name]


CHECKERBOARD_CASES = [
    pytest.param(
        12, 1920, 1080, (100, 100, 1720, 880), [[2000, 2000, 2000]], id="solid"
    ),
    pytest.param(
        12,
        1920,
        1080,
        (100, 100, 1720, 880),
        [[2000, 2000, 2000], [0, 0, 0]],
        id="two",
    ),
    pytest.param(
        12,
        1920,
        1080,
        (0, 0, 1920, 1080),
        [[4095, 0, 0], [0, 4095, 0], [0, 0, 4095]],
        id="three",
    ),
    pytest.param(
        8,
        640,
        480,
        (17, 23, 200, 100),
        [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]],
        id="four-8bit",
    ),
]


@pytest.mark.parametrize(
    ("bit_depth", "width", "height", "roi", "colors"), CHECKERBOARD_CASES
)
def test_checkerboard_arrays_match(
    bmd_sg_on_path: None,
    bit_depth: int,
    width: int,
    height: int,
    roi: tuple[int, int, int, int],
    colors: list[list[int]],
) -> None:
    """Checkerboards render array-for-array equal to bmd-signal-gen's."""
    theirs = importlib.import_module("bmd_sg.image_generators.checkerboard")

    from display_patterns.image_generators import checkerboard as ours

    their_generator = theirs.PatternGenerator(
        bit_depth=bit_depth, width=width, height=height, roi=theirs.ROI(*roi)
    )
    our_generator = ours.PatternGenerator(
        bit_depth=bit_depth, width=width, height=height, roi=ours.ROI(*roi)
    )

    their_pattern = their_generator.generate(colors)
    our_pattern = our_generator.generate(colors)

    assert our_pattern.dtype == their_pattern.dtype
    np.testing.assert_array_equal(our_pattern, their_pattern)


def test_chart_render_arrays_match(bmd_sg_on_path: None) -> None:
    """A YAML chart renders array-for-array equal to bmd-signal-gen's."""
    pytest.importorskip("colour")
    pytest.importorskip("PIL")
    pytest.importorskip("yaml")
    pytest.importorskip("tifffile")

    their_loader = importlib.import_module("bmd_sg.charts.loaders.yaml_chart")
    their_renderer = importlib.import_module("bmd_sg.charts.renderer")

    from display_patterns.charts import renderer as our_renderer
    from display_patterns.charts.loaders import yaml_chart as our_loader

    their_image = their_renderer.render_chart(
        their_loader.load_chart(CHART_YAML_PATH), bit_depth=12, include_labels=True
    )
    our_image = our_renderer.render_chart(
        our_loader.load_chart(CHART_YAML_PATH), bit_depth=12, include_labels=True
    )

    assert our_image.dtype == their_image.dtype
    np.testing.assert_array_equal(our_image, their_image)
