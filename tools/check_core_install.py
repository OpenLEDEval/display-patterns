"""Verify the numpy-only core footprint (§spec:package-shape).

Renders a checkerboard and asserts the import path pulled in no chart
or IO dependency. By default the environment is first required to be
core-only (no extras installed) — the wheel-install CI job's mode.
``--skip-env-check`` drops that gate so the same probe runs in an
all-extras environment (the test suite's mode), where the post-render
module sweep is the live assertion.
"""

import importlib.util
import sys

# Import names of every dependency priced into an extra
# ([project.optional-dependencies] in pyproject.toml). The probe fails
# if importing or rendering the core loads any of them.
FORBIDDEN_MODULES = ("colour", "PIL", "scipy", "tifffile", "yaml")


def main() -> None:
    if "--skip-env-check" not in sys.argv[1:]:
        for name in FORBIDDEN_MODULES:
            if importlib.util.find_spec(name) is not None:
                sys.exit(f"{name} is installed; this check requires a core-only env")

    import numpy as np

    from display_patterns import (
        PanelGeometry,
        checkerboard,
        decode_counter,
        render_counter_panel,
    )
    from display_patterns.image_generators import ROI, PatternGenerator

    pattern = checkerboard([[4095, 2048, 0]], width=64, height=64, bit_depth=12)
    assert pattern.shape == (64, 64, 3)
    assert pattern.dtype == np.uint16
    assert set(np.unique(pattern)) == {0, 2048, 4095}

    geometry = PanelGeometry.for_frame(width=64, height=16, bits=8)
    overlay, _mask = render_counter_panel(42, geometry)
    assert decode_counter(overlay, geometry) == 42

    # Extraction-era class surface still renders identically.
    legacy = PatternGenerator(
        bit_depth=12, width=64, height=64, roi=ROI(0, 0, 64, 64)
    ).generate([[4095, 2048, 0]])
    assert (legacy == pattern).all()

    assert "display_patterns.charts" not in sys.modules, (
        "core rendering imported the charts subpackage"
    )
    loaded = sorted(m for m in sys.modules if m.split(".")[0] in FORBIDDEN_MODULES)
    assert not loaded, f"core rendering imported heavy deps: {loaded}"

    print("core rendering loads numpy alone")


if __name__ == "__main__":
    main()
