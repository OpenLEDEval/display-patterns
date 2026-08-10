"""Verify a numpy-only install of display-patterns (§spec:package-shape).

Run with the package installed WITHOUT extras. Confirms the core imports
using numpy alone and that no chart or IO dependency is present in the
environment.
"""

import importlib.util
import sys

FORBIDDEN_DISTS = ("colour", "PIL", "yaml", "tifffile")


def main() -> None:
    for name in FORBIDDEN_DISTS:
        if importlib.util.find_spec(name) is not None:
            sys.exit(f"{name} is installed; this check requires a core-only env")

    import display_patterns  # noqa: F401 - core import must succeed

    loaded = [m for m in sys.modules if m.split(".")[0] in FORBIDDEN_DISTS]
    assert not loaded, f"core import pulled in heavy deps: {loaded}"

    sys.stdout.write("core-only install imports with numpy alone\n")


if __name__ == "__main__":
    main()
