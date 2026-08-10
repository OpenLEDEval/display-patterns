"""
Core footprint tests (§spec:package-shape).

Core pattern rendering works with numpy alone: importing and using
``display_patterns.image_generators`` never pulls in the chart or IO
dependencies (colour-science, Pillow, PyYAML, tifffile).
"""

import subprocess
import sys
import textwrap

_CORE_RENDER_SCRIPT = textwrap.dedent(
    """
    import sys

    import numpy as np

    from display_patterns.image_generators import ROI, PatternGenerator

    generator = PatternGenerator(
        bit_depth=12, width=64, height=64, roi=ROI(0, 0, 64, 64)
    )
    pattern = generator.generate([[4095, 2048, 0]])
    assert pattern.shape == (64, 64, 3)
    assert pattern.dtype == np.uint16
    assert set(np.unique(pattern)) == {0, 2048, 4095}

    heavy = sorted(
        module
        for module in sys.modules
        if module.split(".")[0] in {"colour", "PIL", "yaml", "tifffile"}
    )
    assert not heavy, f"core rendering imported heavy deps: {heavy}"
    """
)


def test_core_render_pulls_no_heavy_deps() -> None:
    """Render a checkerboard in a clean interpreter; only numpy loads."""
    subprocess.run([sys.executable, "-c", _CORE_RENDER_SCRIPT], check=True)


def test_charts_subpackage_is_not_imported_by_core() -> None:
    """Importing the top-level package leaves charts (and its deps) unloaded."""
    script = textwrap.dedent(
        """
        import sys

        import display_patterns

        assert display_patterns.__name__ == "display_patterns"
        assert "display_patterns.charts" not in sys.modules
        heavy = sorted(
            module
            for module in sys.modules
            if module.split(".")[0] in {"colour", "PIL", "yaml", "tifffile"}
        )
        assert not heavy, f"package import loaded heavy deps: {heavy}"
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)
