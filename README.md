# display-patterns

**Deterministic display test patterns, device-free and exact.**

Test-pattern math tends to live trapped inside device tools. This
library is the importable version: solid fields, checkerboards,
charts, and machine-readable temporal-alignment patterns, rendered as
pure functions of their parameters and a frame index. Integer
patterns deliver exact code values at a stated bit depth — the
property measurement work depends on — and the render path never
touches a device, a clock, or a file.

The core depends on numpy alone and renders into a caller-supplied
array namespace (numpy or torch), so it serves CLI signal generators
and GPU-resident render graphs alike. Chart authoring and TIFF export
install as extras.

Extracted from
[bmd-signal-gen](https://github.com/OpenDisplayEval/bmd-signal-gen),
which consumes it back over SDI. Sibling consumers and context:
[color-wrangler](https://github.com/Fuse-Technical-Group/color-wrangler)
(LED-surface characterization umbrella) and its component repos.

## Installation

```sh
pip install display-patterns              # core: numpy only
pip install "display-patterns[charts,io]" # chart authoring + TIFF export
```

Until the first PyPI release, install from a checkout:
`pip install .` (or `uv pip install .`).

## Usage

Render a 12-bit checkerboard at exact code values. Every catalog entry
is a pure function of its parameters and a frame index (stills ignore
the index), rendered through a caller-supplied array namespace —
numpy by default, torch on GPU hosts:

```python
from display_patterns import checkerboard

frame = checkerboard(
    [[4095, 2048, 0], [0, 0, 0]], width=1920, height=1080, bit_depth=12
)  # uint16 (1080, 1920, 3)

# GPU-resident render, no host round trip:
# frame = checkerboard(..., xp=torch, device="cuda")
```

Encode a frame counter as machine-readable bit-cells and decode it back
— render one end of a video chain, measure latency and frame skew at
the other:

```python
from display_patterns import PanelGeometry, decode_counter, render_counter_panel

geometry = PanelGeometry.for_frame(width=1920, height=1080, bits=16)
overlay, mask = render_counter_panel(1234, geometry)  # float32 [0, 1], HWC
assert decode_counter(overlay, geometry) == 1234
```

With the `charts` and `io` extras, author a chart in YAML, render it,
and write a 16-bit TIFF:

```python
from display_patterns.charts import render_chart, write_chart_tiff
from display_patterns.charts.loaders import load_chart

layout = load_chart("my_chart.yaml")
image = render_chart(layout, bit_depth=12)
write_chart_tiff("my_chart.tiff", image, layout)
```

## API

- `display_patterns` / `display_patterns.patterns` — the core catalog:
  `checkerboard` (one color renders a solid), `ROI`, `ColorRangeError`,
  and the temporal-alignment counter panel (`PanelGeometry`,
  `render_counter_panel`, `decode_counter`). The root is the canonical
  import surface; no dependency beyond numpy — torch renders through
  the `xp` parameter without ever being required.
- `display_patterns.image_generators` — extraction-era class surface
  (`PatternGenerator`), kept for existing consumers; delegates to the
  catalog.
- `display_patterns.charts` — chart types, colorimetric conversion, and
  rendering (`charts` extra); TIFF read/write (`io` extra).
- `display_patterns.charts.loaders` — YAML chart definitions
  (`charts` extra).

All public functions and classes carry NumPy-style docstrings; the
package ships `py.typed` for static type checking.

## Governance

- [REQUIREMENTS.md](REQUIREMENTS.md) — the problem space
- [SPEC.md](SPEC.md) — design and rationale
- [ROADMAP.md](ROADMAP.md) — work remaining

## License

BSD 3-Clause.
