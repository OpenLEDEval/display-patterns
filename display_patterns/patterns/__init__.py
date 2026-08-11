"""Core pattern catalog on the frame-indexed rendering model.

Every entry point is a pure function (§spec:render-model): parameters
and a frame index in, an array out, rendered through a caller-supplied
array namespace ``xp`` (numpy default) with optional ``device``
placement. Stills ignore the frame index; temporal patterns use it as
their only time source. The render path performs no I/O and holds no
state.
"""

from display_patterns.patterns._counter_panel import (
    PanelGeometry,
    decode_counter,
    render_counter_panel,
)
from display_patterns.patterns._fills import (
    ROI,
    ColorRangeError,
    checkerboard,
)

__all__ = [
    "ROI",
    "ColorRangeError",
    "PanelGeometry",
    "checkerboard",
    "decode_counter",
    "render_counter_panel",
]
