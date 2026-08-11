"""
Core footprint test (§spec:package-shape).

Core pattern rendering works with numpy alone: importing and using
``display_patterns.image_generators`` never pulls in the chart or IO
dependencies. The probe body lives in ``tools/check_core_install.py``
so this test, the wheel-install CI job, and any hand run execute the
same assertions; ``--skip-env-check`` adapts it to this all-extras
environment, where the post-render module sweep is the live check.
"""

import subprocess
import sys
from pathlib import Path

_PROBE = Path(__file__).resolve().parents[1] / "tools" / "check_core_install.py"


def test_core_render_pulls_no_heavy_deps() -> None:
    """Render a checkerboard in a clean interpreter; only numpy loads."""
    subprocess.run([sys.executable, str(_PROBE), "--skip-env-check"], check=True)
