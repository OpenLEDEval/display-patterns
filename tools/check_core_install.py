"""Verify a numpy-only install of display-patterns (§spec:package-shape).

Run with the package installed WITHOUT extras. Confirms the core imports
and renders a checkerboard using numpy alone, and that no chart or IO
dependency is present in the environment.
"""

import importlib.util
import sys

FORBIDDEN_DISTS = ("colour", "PIL", "yaml", "tifffile")


def main() -> None:
    for name in FORBIDDEN_DISTS:
        if importlib.util.find_spec(name) is not None:
            sys.exit(f"{name} is installed; this check requires a core-only env")

    import numpy as np

    from display_patterns.image_generators import ROI, PatternGenerator

    generator = PatternGenerator(
        bit_depth=12, width=64, height=64, roi=ROI(0, 0, 64, 64)
    )
    pattern = generator.generate([[4095, 2048, 0]])
    assert pattern.shape == (64, 64, 3)
    assert pattern.dtype == np.uint16
    assert set(np.unique(pattern)) == {0, 2048, 4095}

    loaded = [m for m in sys.modules if m.split(".")[0] in FORBIDDEN_DISTS]
    assert not loaded, f"core import pulled in heavy deps: {loaded}"

    sys.stdout.write("core-only install renders a checkerboard with numpy alone\n")


if __name__ == "__main__":
    main()
